"""Built-in load generator for the OpenAI-compatible /v1/completions endpoint (vLLM side).

Used for the two things `vllm bench serve` can't do:
  * token-exact prompts: `mixed` and `text` workloads send the same token ids (and per-request
    output lengths) that the PyTorch worker generates, so both engines do identical work;
  * static-batch admission ("waves"): C requests are sent, and the next C only after all of
    them finished. Same server, same prompts; only admission differs from continuous batching.

Closed-loop mode matches `vllm bench serve --max-concurrency C --request-rate inf`: C worker
threads, each sends its next request as soon as its previous one completes; latency is measured
from send time. Standard library only (http.client + threads), so it runs in any environment.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .workloads import PromptSet


class RequestError(RuntimeError):
    pass


def _one(host: str, port: int, model: str, prompt, out_len: int, timeout: float) -> dict:
    payload = {"model": model, "prompt": prompt, "max_tokens": out_len, "min_tokens": out_len,
               "ignore_eos": True, "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True}}
    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        t_submit = time.perf_counter()
        conn.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RequestError(f"HTTP {resp.status}: {resp.read()[:200]!r}")
        times: list[float] = []
        n_out = n_in = None
        for raw in resp:                       # server-sent events, one `data:` line per chunk
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            obj = json.loads(data)
            usage = obj.get("usage")
            if usage:
                n_out = usage.get("completion_tokens", n_out)
                n_in = usage.get("prompt_tokens", n_in)
            if obj.get("choices"):
                times.append(time.perf_counter())
        t_end = time.perf_counter()
    finally:
        conn.close()
    if not times:
        raise RequestError("no tokens streamed")
    n_out = int(n_out if n_out is not None else len(times))
    n_in = int(n_in if n_in is not None else (len(prompt) if isinstance(prompt, list) else 0))
    return dict(submit=t_submit, first=times[0], token_times=times, end=t_end, n_in=n_in, n_out=n_out)


def run(host: str, port: int, model: str, prompts: PromptSet, concurrency: int, mode: str = "closed",
        timeout: float = 1800.0, on_progress=None) -> dict:
    """Send every prompt; returns dict(reqs, duration, n_errors, errors)."""
    n = len(prompts)
    items = [(prompts.ids[i] if prompts.ids else prompts.texts[i], prompts.out_lens[i]) for i in range(n)]
    reqs: list[dict] = []
    errors: list[str] = []
    lock = threading.Lock()

    def work(item):
        prompt, out_len = item
        try:
            r = _one(host, port, model, prompt, out_len, timeout)
            with lock:
                reqs.append(r)
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(f"{type(e).__name__}: {e}"[:200])
        if on_progress:
            on_progress()

    c = max(1, concurrency)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=c) as pool:
        if mode == "waves":
            for i in range(0, n, c):
                list(pool.map(work, items[i:i + c]))       # the whole wave must finish first
        else:
            list(pool.map(work, items))
    wall = time.perf_counter() - t0
    if not reqs:
        raise RequestError(f"all {n} requests failed; first error: {errors[0] if errors else '?'}")
    duration = max(r["end"] for r in reqs) - min(r["submit"] for r in reqs)
    return dict(reqs=reqs, duration=duration or wall, n_errors=len(errors), errors=errors[:5])


def warmup(host: str, port: int, model: str, prompts: PromptSet, k: int, timeout: float = 600.0) -> None:
    """A few requests to settle the server (graph capture, allocator) before measuring."""
    for i in range(min(k, len(prompts))):
        try:
            _one(host, port, model, prompts.ids[i] if prompts.ids else prompts.texts[i],
                 min(prompts.out_lens[i], 32), timeout)
        except Exception:  # noqa: BLE001
            pass


def load_tokenizer(model: str):
    """The model's tokenizer when transformers is importable, else None (random/mixed prompts
    then sample from a generic id range; text prompts are sent as text)."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model)
    except Exception:  # noqa: BLE001
        return None


def model_vocab_size(model: str) -> Optional[int]:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model)
        tc = getattr(cfg, "text_config", cfg)
        return int(tc.vocab_size)
    except Exception:  # noqa: BLE001
        return None
