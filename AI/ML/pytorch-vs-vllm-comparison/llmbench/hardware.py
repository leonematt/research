"""GPU detection, model auto-sizing, toolkit discovery and the clean-machine preflight."""
from __future__ import annotations

import datetime as dt
import functools
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .models import FAMILIES
from .util import GIB, drop_page_cache, http_get, meminfo_gib, sh

# Run in a subprocess so this process never holds a CUDA context. torch respects
# CUDA_VISIBLE_DEVICES and knows whether a device is integrated (unified memory).
_TORCH_PROBE = r"""
import json
try:
    import torch
    out = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append(dict(name=p.name, total_gib=p.total_memory / 2**30,
                            cc=float(f"{p.major}.{p.minor}"),
                            integrated=bool(getattr(p, "is_integrated", False)),
                            uuid=str(getattr(p, "uuid", "") or "")))
    print(json.dumps(out))
except Exception as e:
    print(json.dumps({"error": repr(e)}))
"""

UNIFIED_NAME_HINTS = ("GB10", "THOR", "ORIN", "XAVIER", "TEGRA")


@dataclass
class Hardware:
    gpus: list[dict]            # visible devices in CUDA order: name, total_gib, cc, uuid, smi_index
    unified: bool               # CPU and GPU share one memory pool (GB10, Jetson)
    mem_total_gib: float        # system RAM
    source: str                 # torch | nvidia-smi | fake | none

    @property
    def n(self) -> int:
        return len(self.gpus)

    @property
    def min_cc(self) -> float:
        return min((g["cc"] for g in self.gpus if g.get("cc")), default=0.0)

    @property
    def per_gpu_gib(self) -> float:
        return min((g["total_gib"] for g in self.gpus if g.get("total_gib")), default=0.0)

    @property
    def name(self) -> str:
        return self.gpus[0]["name"] if self.gpus else "none"

    @property
    def slug(self) -> str:
        if not self.gpus:
            return "nogpu"
        drop = {"nvidia", "geforce", "tesla", "quadro"}
        toks = [t for t in re.split(r"[\s\-_]+", self.gpus[0]["name"].lower())
                if t and t not in drop and not re.fullmatch(r"\d+gb|hbm\d*e?|pcie|sxm\d*|nvl\d*|oam", t)]
        s = re.sub(r"[^a-z0-9]", "", "".join(toks)) or "gpu"
        return f"{self.n}x{s}" if self.n > 1 else s

    def describe(self) -> str:
        if not self.gpus:
            return "no CUDA GPU detected"
        g = self.gpus[0]
        mem = (f"{self.mem_total_gib:.0f} GiB unified" if self.unified
               else f"{g['total_gib']:.0f} GiB each" if self.n > 1 else f"{g['total_gib']:.0f} GiB")
        return f"{self.n}x {g['name']} (sm_{str(g['cc']).replace('.', '')}, {mem})"

    def smi_ids(self, k: Optional[int] = None) -> list[str]:
        return [str(g["smi_index"]) for g in self.gpus[:k] if g.get("smi_index") is not None]

    def to_dict(self) -> dict:
        return dict(gpus=self.gpus, unified=self.unified, mem_total_gib=self.mem_total_gib,
                    source=self.source, slug=self.slug, describe=self.describe())


def _smi_rows() -> list[dict]:
    out = sh(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,compute_cap",
              "--format=csv,noheader,nounits"])
    rows = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 6:
            continue
        num = lambda s: float(s) if re.fullmatch(r"[\d.]+", s) else None  # noqa: E731
        rows.append(dict(smi_index=int(p[0]), uuid=p[1], name=p[2],
                         total_mib=num(p[3]), used_mib=num(p[4]), cc=num(p[5])))
    return rows


def detect_hardware() -> Hardware:
    fake = os.environ.get("LLMBENCH_FAKE_HW") or os.environ.get("VLLM_ABLATION_FAKE_HW")
    if fake:   # preview another box: '{"name": "H100", "gib": 80, "n": 2, "cc": 9.0}'
        d = json.loads(fake)
        gpus = [dict(name=d["name"], total_gib=float(d["gib"]), cc=float(d.get("cc", 9.0)),
                     integrated=bool(d.get("unified")), uuid="", smi_index=None if d.get("no_smi") else i)
                for i in range(int(d.get("n", 1)))]
        return Hardware(gpus=gpus, unified=bool(d.get("unified")),
                        mem_total_gib=float(d.get("ram_gib", d["gib"] if d.get("unified") else 256)),
                        source="fake")
    mem_total = meminfo_gib().get("mem_total_gib", 0.0)
    smi = _smi_rows()
    gpus, source = [], "none"
    raw = sh([sys.executable, "-c", _TORCH_PROBE])
    try:
        probed = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        probed = None
    if isinstance(probed, list) and probed:
        gpus, source = probed, "torch"
    elif smi:
        # no torch: honour CUDA_VISIBLE_DEVICES ourselves (indices or UUIDs)
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        sel = smi
        if cvd is not None:
            want = [x.strip() for x in cvd.split(",") if x.strip()]
            sel = [r for w in want for r in smi if w == str(r["smi_index"]) or r["uuid"].endswith(w)]
        gpus = [dict(name=r["name"], total_gib=(r["total_mib"] or 0) / 1024, cc=r["cc"] or 0.0,
                     integrated=False, uuid=r["uuid"]) for r in sel]
        source = "nvidia-smi"
    # attach nvidia-smi index (for memory/process queries) by UUID, else by position
    cvd_idx = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip().isdigit()]
    for i, g in enumerate(gpus):
        u = g.get("uuid", "").lower().replace("gpu-", "")
        hit = next((r for r in smi if u and r["uuid"].lower().replace("gpu-", "") == u), None)
        if hit is None and smi:
            j = int(cvd_idx[i]) if i < len(cvd_idx) else i
            hit = next((r for r in smi if r["smi_index"] == j), None)
        g["smi_index"] = hit["smi_index"] if hit else None
        if hit and not g.get("total_gib") and hit["total_mib"]:
            g["total_gib"] = hit["total_mib"] / 1024
    unified = any(g.get("integrated") for g in gpus) \
        or any(h in g["name"].upper() for g in gpus for h in UNIFIED_NAME_HINTS) \
        or (len(gpus) == 1 and mem_total > 0 and gpus[0].get("total_gib")
            and abs(gpus[0]["total_gib"] - mem_total) / mem_total < 0.06)
    for g in gpus:
        if unified and not g.get("total_gib"):
            g["total_gib"] = mem_total
        g["total_gib"] = round(float(g.get("total_gib") or 0.0), 1)
    return Hardware(gpus=gpus, unified=bool(unified), mem_total_gib=mem_total, source=source)


# ----------------------------------------------------------------------------
# auto-sizing
# ----------------------------------------------------------------------------

_DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
                "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def weights_gib(model: str) -> Optional[float]:
    """Best-effort size of the checkpoint's weights: local safetensors, the family table,
    or the Hub's safetensors metadata. None if unknown."""
    p = Path(model).expanduser()
    if p.is_dir():
        sz = sum(f.stat().st_size for f in p.glob("*.safetensors"))
        return sz / GIB if sz else None
    for spec in FAMILIES.values():
        for m, w in spec["ladder"] + spec["extra"]:
            if m == model:
                return w
    try:
        from huggingface_hub import get_safetensors_metadata
        md = get_safetensors_metadata(model)
        return sum(n * _DTYPE_BYTES.get(d, 2) for d, n in md.parameter_count.items()) / GIB
    except Exception:  # noqa: BLE001
        return None


def kv_and_overhead(w: float) -> float:
    """Minimum non-weight memory to start sensibly: a few GiB of KV + graphs/activations."""
    return max(2.0, 0.10 * w) + 1.5


def fits(w: float, hw: Hardware, gpu_mem: float, tp: int) -> bool:
    return w + kv_and_overhead(w) <= hw.per_gpu_gib * gpu_mem * tp


def autosize(hw: Hardware, family: str, model: Optional[str], gpu_mem: float,
             tp: Optional[int]) -> dict:
    """Pick model (if not given), TP (if not given) and a max_model_len that leaves KV room."""
    tps = [t for t in (1, 2, 4, 8) if t <= max(hw.n, 1)]
    why = []
    if model is None:
        ladder = FAMILIES[family]["ladder"]
        for m, w in ladder:
            t = tp or next((t for t in tps if fits(w, hw, gpu_mem, t)), None)
            if t and fits(w, hw, gpu_mem, t):
                model, tp = m, t
                why.append(f"{m} (~{w:.0f} GiB BF16 weights) is the largest {family} model that fits "
                           f"{hw.per_gpu_gib:.0f} GiB x {gpu_mem} x TP{t}")
                break
        else:
            model, tp = ladder[-1][0], tp or tps[-1]
            why.append(f"nothing fits comfortably; falling back to {model} (expect OOM / tiny KV)")
    w = weights_gib(model)
    if tp is None:
        tp = next((t for t in tps if w is None or fits(w, hw, gpu_mem, t)), tps[-1])
        if w is not None and tp > 1:
            why.append(f"TP{tp}: ~{w:.0f} GiB of weights needs >1 GPU at {hw.per_gpu_gib:.0f} GiB each")
    if w is None:
        mml = 16384
        why.append("weights size unknown; max_model_len 16384")
    else:
        kv_room = hw.per_gpu_gib * gpu_mem * tp - w - 1.5
        mml = 32768 if kv_room >= 16 else 16384 if kv_room >= 6 else 8192
        why.append(f"~{kv_room:.0f} GiB left for KV -> max_model_len {mml}")
    return dict(model=model, tp=tp, max_model_len=mml, weights_gib=w, why=why)


@functools.lru_cache(maxsize=1)
def vllm_version() -> Optional[tuple[int, int]]:
    """(major, minor) of the vllm on PATH, or None if it can't be read."""
    forced = os.environ.get("LLMBENCH_VLLM_VERSION")
    if forced:
        a, b = forced.split(".")[:2]
        return int(a), int(b)
    out = sh(["vllm", "--version"]) or sh([sys.executable, "-c", "import vllm;print(vllm.__version__)"])
    m = re.search(r"^\s*v?(\d+)\.(\d+)\.\d+", out or "", flags=re.M)   # a bare version line, not log noise
    return (int(m.group(1)), int(m.group(2))) if m else None


def vllm_default_batching(hw: Optional[Hardware]) -> tuple[int, int]:
    """vLLM's `vllm serve` defaults for (max_num_batched_tokens, max_num_seqs) on this GPU
    (EngineArgs._set_default_max_num_seqs_and_batched_tokens_args, v0.31): >=160 GiB -> 16384/1024,
    >=70 GiB and not A100 -> 8192/1024, else 2048/256. A knob set to the default is a no-op."""
    mem = hw.per_gpu_gib if hw and hw.gpus else 0.0
    name = str(hw.gpus[0].get("name", "")).lower() if hw and hw.gpus else ""
    if mem >= 160:
        return 16384, 1024
    if mem >= 70 and "a100" not in name:
        return 8192, 1024
    return 2048, 256


def environment(hw: Hardware) -> dict:
    return dict(
        timestamp=dt.datetime.now().isoformat(timespec="seconds"),
        host=os.uname().nodename, arch=os.uname().machine,
        python=sys.version.split()[0],
        vllm=sh(["vllm", "--version"]) or sh([sys.executable, "-c", "import vllm;print(vllm.__version__)"]),
        torch=sh([sys.executable, "-c", "import torch;print(torch.__version__)"]),
        transformers=sh([sys.executable, "-c", "import transformers;print(transformers.__version__)"]),
        cuda=sh([sys.executable, "-c", "import torch;print(torch.version.cuda)"]),
        driver=sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).split("\n")[0],
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        hardware=hw.to_dict(),
    )


# ----------------------------------------------------------------------------
# CUDA toolkit
# ----------------------------------------------------------------------------

def find_nvcc() -> Optional[str]:
    """Where FlashInfer's JIT will look: $CUDA_HOME, PATH, then /usr/local/cuda."""
    home = os.environ.get("CUDA_HOME")
    for c in (os.path.join(home, "bin", "nvcc") if home else None,
              sh(["which", "nvcc"]) or None, "/usr/local/cuda/bin/nvcc"):
        if c and os.path.exists(c):
            return c
    return None


def nvcc_env(env: dict) -> dict:
    """PATH/CUDA_HOME additions so vLLM sees the toolkit. FlashInfer's JIT falls back to
    /usr/local/cuda on its own, but vLLM's has_flashinfer() checks `which nvcc`: with nvcc off
    PATH it treats FlashInfer as missing, and anything routed to it (FP8 KV cache on sm_120,
    --attention-backend FLASHINFER) dies with "FlashInfer backend is not available"."""
    nvcc = find_nvcc()
    if not nvcc:
        return {}
    bindir = os.path.dirname(nvcc)
    path = env.get("PATH", "")
    out = {}
    if bindir not in path.split(os.pathsep):
        out["PATH"] = bindir + os.pathsep + path
    if not env.get("CUDA_HOME"):
        out["CUDA_HOME"] = os.path.dirname(bindir)
    return out


# ----------------------------------------------------------------------------
# GPU occupancy
# ----------------------------------------------------------------------------

# Processes that block a clean benchmark: anything ML-shaped, or anything holding real
# memory. Desktop clients (file manager, browser compositor, ...) at tens of MiB don't.
_ML_PROC = re.compile(r"python|vllm|ollama|llama|sglang|triton|text-generation|comfy|torch|"
                      r"jupyter|lm-?studio|kobold|exllama|tensorrt|trtllm|nemo", re.I)
BUSY_MIB = 1024
# vLLM's startup check (free >= gpu_mem x total) runs after its own CUDA context exists
# (~0.5-0.8 GiB), and a GPU that drives a desktop drifts by a few hundred MiB over a sweep.
CUDA_CTX_MIB = 768
DISCRETE_HEADROOM_MIB = 1536
# memory.used per GPU when the sweep started (preflight): "settled" means back near this
_IDLE_USED_MIB: dict[int, float] = {}

_VLLM_PROC = re.compile(
    r"^(?:\S*/)?(?:python[\d.]*\s+)?(?:\S*/)?vllm\s+serve\b"   # vllm serve / python .../vllm serve
    r"|^VLLM::\w+"                                               # VLLM::EngineCore / workers
    r"|^(?:\S*/)?python[\d.]*\s+(?:-\S+\s+)*-m\s+vllm\.entrypoints")


def other_vllm_processes() -> list[str]:
    """Running vLLM servers/engines, matched on the executable, not on any mention of it."""
    out = sh(["ps", "-eo", "pid=,user=,etime=,args="])
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and _VLLM_PROC.search(parts[3]):
            rows.append(line.strip()[:220])
    return rows


def gpu_compute_apps(smi_ids: list[str]) -> list[dict]:
    """Processes holding memory on the GPUs we're about to use (discrete GPUs)."""
    if not smi_ids:
        return []
    uuids = sh(["nvidia-smi", "-i", ",".join(smi_ids), "--query-gpu=uuid",
                "--format=csv,noheader"]).split()
    out = sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
              "--format=csv,noheader,nounits"])
    rows = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4 and p[0] in uuids:
            mib = float(p[3]) if re.fullmatch(r"[\d.]+", p[3]) else None
            rows.append(dict(pid=p[1], name=p[2], mib=mib))
    return rows


def gpu_mem_state(smi_ids: Optional[list[str]]) -> dict[int, tuple[float, float]]:
    """{smi_index: (used_mib, total_mib)} for the given GPUs (all when smi_ids is None)."""
    want = {int(i) for i in smi_ids} if smi_ids else None
    return {r["smi_index"]: (r["used_mib"], r["total_mib"]) for r in _smi_rows()
            if r["used_mib"] is not None and r["total_mib"] and (want is None or r["smi_index"] in want)}


def _fmt_app(r: dict) -> str:
    mib = "?" if r["mib"] is None else "%.0f" % r["mib"]
    return "pid %s %s (%s MiB)" % (r["pid"], r["name"], mib)


def wait_gpu_settle(max_s: float, smi_ids: Optional[list[str]] = None) -> None:
    """Best-effort: wait for the GPUs we used to drain before the next launch. On unified
    memory nvidia-smi usage is not meaningful, so this degrades to a fixed settle delay."""
    t0 = time.time()
    while time.time() - t0 < max_s:
        st = gpu_mem_state(smi_ids)
        # settled = back within 512 MiB of what was in use before the sweep (desktop etc.)
        if st and all(used <= _IDLE_USED_MIB.get(i, 2488) + 512 for i, (used, _) in st.items()):
            time.sleep(min(3, max_s))
            return
        time.sleep(min(3, max_s))


def wait_gpu_free(gpu_mem: float, smi_ids: list[str], max_s: float = 60) -> Optional[str]:
    """Before a launch on discrete GPUs: wait until each GPU has room for gpu_mem x total plus
    vLLM's CUDA context. Returns a warning if it never does (something else grew)."""
    t0, short = time.time(), {}
    while True:
        st = gpu_mem_state(smi_ids)
        short = {i: (t - u, gpu_mem * t + CUDA_CTX_MIB) for i, (u, t) in st.items()
                 if t - u < gpu_mem * t + CUDA_CTX_MIB}
        if not short or time.time() - t0 >= max_s:
            break
        time.sleep(3)
    if not short:
        return None
    return "; ".join(f"GPU {i}: {f / 1024:.1f} GiB free < {n / 1024:.1f} GiB needed (gpu_mem {gpu_mem} "
                     f"+ CUDA context)" for i, (f, n) in short.items()) + \
        ". Something else grew on the GPU since the sweep started (browser/desktop?); starting anyway."


# ----------------------------------------------------------------------------
# Hub access + preflight
# ----------------------------------------------------------------------------

def hf_access(repo: str) -> tuple[str, Optional[str]]:
    """("ok"|"local"|"unchecked"|"fail", reason). Only "fail" blocks."""
    if Path(repo).expanduser().exists():
        return "local", None
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return "unchecked", "HF_HUB_OFFLINE=1"
    try:
        from huggingface_hub import auth_check
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except Exception:  # noqa: BLE001
        return "unchecked", "huggingface_hub too old for auth_check"
    try:
        auth_check(repo)
        return "ok", None
    except GatedRepoError:
        return "fail", (f"{repo} is gated: accept its license at https://huggingface.co/{repo} "
                        f"and `export HF_TOKEN=hf_...` (or `hf auth login`)")
    except RepositoryNotFoundError:
        return "fail", f"{repo} not found on the Hub (or private)"
    except Exception as e:  # noqa: BLE001
        return "unchecked", f"{type(e).__name__}"   # offline / transient: let vllm try


def preflight(hw: Hardware, *, model: str, port: Optional[int], n_gpus: int, gpu_mem: float,
              drop_caches: bool, allow_shared: bool, nvcc_check: bool = True) -> None:
    """Refuse to benchmark on a busy machine and verify the main model is reachable. Another
    server on the box both steals memory and shares memory bandwidth, contaminating every
    decode number. (Extra checkpoints are checked per config in configs.drop_inaccessible.)"""
    if not hw.gpus:
        raise SystemExit("[preflight] no CUDA GPU visible (check nvidia-smi / CUDA_VISIBLE_DEVICES)")

    # Blackwell (sm_100+) has FlashInfer kernels that are JIT-compiled at startup: no nvcc,
    # every config dies ~2 min into its server start. Older archs mostly ship prebuilt.
    nvcc = find_nvcc()
    if nvcc:
        added = " (not on PATH: adding its directory for vLLM, whose FlashInfer check uses PATH)" \
            if "PATH" in nvcc_env(dict(os.environ)) else ""
        print(f"[preflight] nvcc: {nvcc}{added}")
    elif nvcc_check and hw.min_cc >= 10.0:
        raise SystemExit(
            f"[preflight] no CUDA toolkit (nvcc) found, and this GPU (sm_{str(hw.min_cc).replace('.', '')}) "
            "needs it: FlashInfer JIT-compiles its Blackwell kernels at server start.\n"
            "  `./run.sh cuda` installs it, or set CUDA_HOME to an existing toolkit. --no-nvcc-check to try anyway.")
    elif nvcc_check:
        print("[preflight] warning: no nvcc found; fine unless a backend needs to JIT kernels")

    status, why = hf_access(model)
    if status == "fail":
        raise SystemExit(f"[preflight] {why}")
    print(f"[preflight] model {model}: " + {"ok": "HF access ok", "local": "local path"}.get(
        status, f"access not verified ({why}); the engine will find out"))

    problems = []
    others = other_vllm_processes()
    if others:
        problems.append("another vLLM server is running:\n" + "\n".join("      " + r for r in others))
    if port:
        st, _ = http_get(f"http://127.0.0.1:{port}/health", timeout=2)
        if st:
            problems.append(f"something is already answering on port {port} (use --port / PORT=)")
    if hw.unified:
        if drop_caches:
            drop_page_cache()
        m = meminfo_gib()
        if m and gpu_mem > 0:
            need = gpu_mem * hw.per_gpu_gib
            if m["mem_available_gib"] < need:
                problems.append(f"only {m['mem_available_gib']:.0f} GiB available but "
                                f"--gpu-mem {gpu_mem} asks for {need:.0f} GiB of {hw.per_gpu_gib:.0f}")
    else:
        ids = hw.smi_ids(n_gpus)
        apps = gpu_compute_apps(ids)
        busy = [r for r in apps if _ML_PROC.search(r["name"]) or (r["mib"] or 0) >= BUSY_MIB]
        minor = [r for r in apps if r not in busy]
        if busy:
            problems.append(f"processes already on GPU(s) {','.join(ids)}:\n"
                            + "\n".join("      " + _fmt_app(r) for r in busy))
        if minor:
            print(f"[preflight] ignoring {len(minor)} small non-ML GPU process(es) (desktop etc.): "
                  + ", ".join(_fmt_app(r) for r in minor[:4]))
        for idx, (used, total) in gpu_mem_state(ids).items():
            _IDLE_USED_MIB[idx] = used
            if gpu_mem <= 0:
                continue
            free = (total - used) / 1024
            need = (gpu_mem * total + CUDA_CTX_MIB) / 1024
            if free < need:
                fit = math.floor((total - used - DISCRETE_HEADROOM_MIB) / total * 100) / 100
                problems.append(f"GPU {idx}: {free:.1f} GiB free, but --gpu-mem {gpu_mem} needs "
                                f"{need:.1f} GiB ({gpu_mem} x {total / 1024:.1f} GiB + ~{CUDA_CTX_MIB} MiB "
                                f"CUDA context)" + (f"; GPU_MEM={fit} would fit" if fit >= 0.3 else ""))
    if problems:
        print("\n[preflight] machine isn't free for benchmarking:")
        for p_ in problems:
            print("  - " + p_)
        print("  The sweep needs these GPUs to itself: anything else running there shares memory and\n"
              "  bandwidth and skews every number. Stop it for the sweep (`docker ps` -> `docker stop`),\n"
              "  pick other GPUs with CUDA_VISIBLE_DEVICES, or rerun with --allow-shared.")
        if not allow_shared:
            raise SystemExit(2)
        print("  --allow-shared set: continuing; results are not clean.\n")


def default_gpu_mem(hw: Hardware) -> tuple[float, Optional[str]]:
    """(--gpu-memory-utilization, why it differs from the default)."""
    if hw.unified:
        return (0.75 if hw.per_gpu_gib >= 64 else 0.6), None
    gpu_mem = 0.90
    # A GPU that also drives a desktop: stay under what's actually free. vLLM refuses to start
    # if free < gpu_mem x total, measured AFTER its own CUDA context exists, and the budget must
    # also survive desktop drift over a long sweep -> DISCRETE_HEADROOM_MIB.
    ids = {g["smi_index"] for g in hw.gpus if g.get("smi_index") is not None}
    rows = [r for r in _smi_rows() if r["smi_index"] in ids and r["total_mib"] and r["used_mib"] is not None]
    fracs = [(r["total_mib"] - r["used_mib"] - DISCRETE_HEADROOM_MIB) / r["total_mib"] for r in rows]
    if fracs and 0.5 <= min(fracs) < 0.90:
        gpu_mem = math.floor(min(fracs) * 100) / 100
        r = rows[fracs.index(min(fracs))]
        return gpu_mem, (f"GPU {r['smi_index']} has {r['used_mib'] / 1024:.1f} GiB in use by other "
                         f"processes; leaving {DISCRETE_HEADROOM_MIB / 1024:.1f} GiB more for vLLM's "
                         f"CUDA context and desktop drift -> gpu_mem {gpu_mem}")
    return gpu_mem, None
