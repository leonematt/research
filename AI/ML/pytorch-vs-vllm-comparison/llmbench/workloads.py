"""Workload grids, plus the prompt generator both engines share.

Workload kinds
--------------
reference  the headline shape; always present
grid       input_len x output_len x concurrency (closed loop, `vllm bench serve` random dataset)
prefix     a shared system-prompt prefix + a unique suffix (prefix caching)
mixed      variable input AND output lengths (continuous batching; static batching wastes
           slots waiting for the longest request in a batch)
text       real English instructions (speculative decoding needs predictable text to accept)
rate       open-loop Poisson arrivals at fractions of vLLM-default capacity (large-scale serving)
sharegpt   optional real conversations (--sharegpt-path)

`grid`, `reference`, `prefix`, `rate` and `sharegpt` go through `vllm bench serve` on the vLLM
side. `mixed` and `text` go through the built-in client (client.py) with token ids generated
here, so vLLM and PyTorch see token-identical prompts and output lengths.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import asdict, dataclass, field
from typing import Optional

# Each grid: full factorial of input_lens x output_lens x concurrencies, plus
#   reference = (input_len, output_len, concurrency)            must lie on the grid
#   prefix    = (prefix_len, unique_input_len, output_len, concurrency)
#   mixed     = (mean_input_len, mean_output_len, concurrency)
#   text      = (output_len, [concurrencies])
#   rates     = fractions of vLLM-default capacity at the reference shape
POW2 = [1, 2, 4, 8, 16, 32, 64, 128]       # concurrency (requests in flight) sweep

GRIDS: dict[str, dict] = {
    "check": dict(input_lens=[256], output_lens=[64], concurrencies=[1, 4],
                  reference=(256, 64, 4), prefix=(512, 128, 32, 4),
                  mixed=(256, 64, 4), text=(64, [1]), rates=[0.5]),
    "quick": dict(input_lens=[256, 2048], output_lens=[256], concurrencies=POW2,
                  reference=(2048, 256, 16), prefix=(2048, 256, 128, 16),
                  mixed=(1024, 256, 32), text=(256, [1, 16]), rates=[0.5, 1.0]),
    "standard": dict(input_lens=[256, 1024, 4096], output_lens=[256, 1024], concurrencies=POW2,
                     reference=(1024, 256, 32), prefix=(4096, 256, 128, 32),
                     mixed=(1024, 512, 64), text=(256, [1, 8, 32]), rates=[0.25, 0.5, 0.75, 1.0, 1.25]),
    "long": dict(input_lens=[1024, 4096, 16384], output_lens=[256, 1024], concurrencies=POW2,
                 reference=(4096, 256, 32), prefix=(8192, 512, 256, 32),
                 mixed=(4096, 512, 32), text=(512, [1, 8]), rates=[0.5, 1.0]),
}

NATIVE_KINDS = ("mixed", "text")          # built-in client on the vLLM side


# Per-run work budget: enough requests to reach steady state, not so many that long-output,
# high-concurrency shapes take forever. A run sends max(8, 4 x C) requests (4 "waves" of the
# concurrency cap, so ramp-up and drain are a small part of the window), capped by an
# output-token budget but never below 2 x C (the window must contain a full batch turnover).
WAVES_PER_RUN = 4
MIN_PROMPTS = 8
MAX_PROMPTS = 512
OUTPUT_TOKEN_BUDGET = 131_072


def num_prompts_for(concurrency: int, output_len: int = 256) -> int:
    n = max(MIN_PROMPTS, WAVES_PER_RUN * concurrency)
    budget = max(2 * concurrency, OUTPUT_TOKEN_BUDGET // max(1, output_len))
    return int(min(n, budget, MAX_PROMPTS))


@dataclass
class Workload:
    name: str
    input_len: int
    output_len: int
    concurrency: int                 # closed-loop cap; 0 = unlimited (open loop)
    num_prompts: int
    prefix_len: int = 0
    kind: str = "grid"
    dataset: str = "random"          # random | text | sharegpt
    rate_frac: float = 0.0           # rate workloads: fraction of the vLLM-default capacity
    request_rate: float = 0.0        # resolved req/s (0 = closed loop / inf)
    dataset_path: Optional[str] = None

    @property
    def label(self) -> str:
        if self.kind == "sharegpt":
            return f"sharegpt/out{self.output_len}/c{self.concurrency}"
        if self.kind == "text":
            return f"text/out{self.output_len}/c{self.concurrency}"
        if self.kind == "mixed":
            return f"mixed/in~{self.input_len}/out~{self.output_len}/c{self.concurrency}"
        if self.kind == "rate":
            r = f"{self.request_rate:g}rps" if self.request_rate else f"{self.rate_frac:g}xcap"
            return f"in{self.input_len}/out{self.output_len}/{r}"
        s = f"in{self.input_len}/out{self.output_len}/c{self.concurrency}"
        if self.prefix_len:
            s += f"/pfx{self.prefix_len}"
        return s

    @property
    def total_len(self) -> int:
        """Longest sequence this workload can produce (mixed: up to 1.5x input, 2x output)."""
        if self.kind == "mixed":
            return int(self.input_len * 1.5) + 2 * self.output_len
        if self.kind == "text":
            return 160 + self.output_len
        return self.prefix_len + self.input_len + self.output_len

    @property
    def closed_loop(self) -> bool:
        return self.kind != "rate"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label
        return d

    @staticmethod
    def from_dict(d: dict) -> "Workload":
        keys = Workload.__dataclass_fields__
        return Workload(**{k: v for k, v in d.items() if k in keys})

    def bench_args(self) -> list[str]:
        """Dataset arguments for `vllm bench serve`."""
        if self.kind == "sharegpt":
            return ["--dataset-name", "sharegpt", "--dataset-path", str(self.dataset_path),
                    "--sharegpt-output-len", str(self.output_len)]
        return ["--dataset-name", "random",
                "--random-input-len", str(self.input_len),
                "--random-output-len", str(self.output_len),
                "--random-range-ratio", "0.0",
                "--random-prefix-len", str(self.prefix_len)]


def build_workloads(grid: str, max_model_len: int, sharegpt_path: Optional[str] = None,
                    only: Optional[set[str]] = None, kinds: Optional[set[str]] = None,
                    quiet: bool = False) -> list[Workload]:
    g = GRIDS[grid]
    wls: list[Workload] = []
    seen = set()

    def add(w: Workload):
        kind = "grid" if w.kind in ("grid", "reference") else w.kind     # the reference IS a grid point
        key = (kind, w.input_len, w.output_len, w.concurrency, w.prefix_len, w.rate_frac)
        if key not in seen:
            seen.add(key)
            wls.append(w)

    ri, ro, rc = g["reference"]
    assert ri in g["input_lens"] and ro in g["output_lens"] and rc in g["concurrencies"], \
        f"grid {grid}: reference {g['reference']} must be a point on the grid"
    add(Workload("reference", ri, ro, rc, num_prompts_for(rc, ro), kind="reference"))
    for inp in g["input_lens"]:
        for out in g["output_lens"]:
            for conc in g["concurrencies"]:
                add(Workload(f"i{inp}_o{out}_c{conc}", inp, out, conc, num_prompts_for(conc, out)))
    p_len, p_in, p_out, p_c = g["prefix"]
    add(Workload("shared_prefix", p_in, p_out, p_c, num_prompts_for(p_c, p_out), prefix_len=p_len, kind="prefix"))
    m_in, m_out, m_c = g["mixed"]
    add(Workload("mixed", m_in, m_out, m_c, max(num_prompts_for(m_c, m_out), 2 * m_c), kind="mixed"))
    t_out, t_cs = g["text"]
    for c in t_cs:
        add(Workload(f"text_c{c}", 0, t_out, c, num_prompts_for(c, t_out), kind="text", dataset="text"))
    for f in g["rates"]:
        add(Workload(f"rate_{f:g}x", ri, ro, 0, 0, kind="rate", rate_frac=f))
    if sharegpt_path:
        add(Workload("sharegpt", 0, ro, rc, num_prompts_for(rc, ro), kind="sharegpt", dataset="sharegpt",
                     dataset_path=sharegpt_path))

    kept = []
    for w in wls:
        if only is not None and w.name not in only and w.name != "reference":
            continue
        if kinds is not None and w.kind not in kinds and w.kind != "reference":
            continue
        if w.kind != "sharegpt" and w.total_len > max_model_len:
            if not quiet:
                print(f"[grid] dropping {w.label}: needs {w.total_len} > max_model_len {max_model_len}")
            continue
        kept.append(w)
    return kept


def resolve_rates(wls: list[Workload], base_rps: Optional[float]) -> None:
    """Turn rate fractions into req/s once vLLM-default capacity is known."""
    for w in wls:
        if w.kind == "rate" and base_rps:
            w.request_rate = round(max(0.05, w.rate_frac * base_rps), 3)
            # ~60 s of arrivals, bounded so slow boxes still finish
            w.num_prompts = int(max(32, min(1024, w.request_rate * 60)))


# ----------------------------------------------------------------------------
# prompts (identical token ids for vLLM's built-in client and the PyTorch worker)
# ----------------------------------------------------------------------------

_TOPICS = [
    "how a refrigerator keeps food cold", "the causes of the French Revolution", "how vaccines train the immune system",
    "the life cycle of a star", "how compilers turn source code into machine code", "the water cycle",
    "why the sky is blue", "how bridges carry heavy loads", "the history of the printing press",
    "how neural networks learn", "the rules of chess for a beginner", "how coffee is grown and roasted",
    "plate tectonics and earthquakes", "how a bill becomes a law", "the basics of personal budgeting",
    "how solar panels generate electricity", "the plot of a classic adventure novel", "how bees make honey",
    "the difference between weather and climate", "how GPS determines your location", "photosynthesis",
    "the Roman Empire's road network", "how to train for a first marathon", "how the internet routes packets",
    "the invention of the telephone", "how airplanes stay in the air", "the human digestive system",
    "how to bake sourdough bread", "the causes of inflation", "how a database index speeds up queries",
    "volcanoes and how they erupt", "the history of jazz", "how batteries store energy",
    "the structure of DNA", "how to write a clear technical report", "the moon landing missions",
    "how recycling plants sort materials", "ocean currents and climate", "how elevators work safely",
    "the basics of probability", "the industrial revolution", "how noise-cancelling headphones work",
]
_TEMPLATES = [
    "Explain {t} in detail. Start with the basic idea, then walk through each step, and finish with "
    "a few practical examples that a curious student would find useful.",
    "Write a thorough, well-organized article about {t}. Use clear paragraphs, define any technical "
    "terms, and include a short summary at the end.",
    "You are a patient teacher. Describe {t} to a high-school class, covering the history, the key "
    "principles, and common misconceptions.",
    "Give a step-by-step explanation of {t}, followed by a list of frequently asked questions with "
    "detailed answers.",
]


def text_prompts(n: int, seed: int) -> list[str]:
    rng = random.Random(seed ^ 0x7E57)
    return [rng.choice(_TEMPLATES).format(t=rng.choice(_TOPICS)) for _ in range(n)]


def _rng(w: Workload, seed: int) -> random.Random:
    return random.Random(seed ^ zlib.crc32(w.name.encode()))


def output_lengths(w: Workload, n: int, seed: int) -> list[int]:
    """Per-request output lengths. Fixed except for `mixed`, where they're spread uniformly over
    [out/8, 2*out - out/8] (mean = out): a static batch waits for its longest member."""
    if w.kind != "mixed":
        return [w.output_len] * n
    rng = _rng(w, seed + 1)
    lo = max(1, w.output_len // 8)
    return [rng.randint(lo, 2 * w.output_len - lo) for _ in range(n)]


@dataclass
class PromptSet:
    ids: list[list[int]] = field(default_factory=list)     # token ids (empty lists when text-only)
    texts: list[str] = field(default_factory=list)         # text prompts (text workloads)
    out_lens: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.out_lens)


def make_prompts(w: Workload, n: int, vocab: list[int], seed: int, tokenizer=None) -> PromptSet:
    """n prompts for workload w. `vocab` = regular (non-special) token ids to sample from.
    Text prompts are tokenized with `tokenizer` when given, so both engines get the same ids."""
    rng = _rng(w, seed)
    outs = output_lengths(w, n, seed)
    if w.kind == "text":
        texts = text_prompts(n, seed)
        ids = [tokenizer.encode(t, add_special_tokens=False) for t in texts] if tokenizer is not None else []
        return PromptSet(ids=ids, texts=texts, out_lens=outs)
    if w.kind == "mixed":
        lens = [rng.randint(max(1, w.input_len // 2), w.input_len * 3 // 2) for _ in range(n)]
        return PromptSet(ids=[[rng.choice(vocab) for _ in range(L)] for L in lens], out_lens=outs)
    prefix = [rng.choice(vocab) for _ in range(w.prefix_len)]
    return PromptSet(ids=[prefix + [rng.choice(vocab) for _ in range(w.input_len)] for _ in range(n)],
                     out_lens=outs)


def regular_vocab(tokenizer=None, model_vocab: Optional[int] = None) -> list[int]:
    """Token ids safe to sample: regular tokens only (like vllm bench's random dataset)."""
    vocab_n, special = None, set()
    if tokenizer is not None:
        try:
            vocab_n = tokenizer.vocab_size
            special = set(tokenizer.all_special_ids)
        except Exception:  # noqa: BLE001
            vocab_n = None
    if not vocab_n or vocab_n < 1000:          # no real tokenizer (v5 can return a 1-token stub)
        vocab_n, special = (model_vocab or 32000), set()
    if model_vocab:
        vocab_n = min(vocab_n, model_vocab)
    return [i for i in range(min(100, vocab_n // 2), vocab_n) if i not in special]
