"""Per-request timings -> the same result fields `vllm bench serve` writes, so rows from the
built-in client, the PyTorch worker and `vllm bench serve` share one schema."""
from __future__ import annotations

import statistics
from typing import Optional

from .util import NAN, pctl


def summarize(reqs: list[dict], duration: float, slo: Optional[dict] = None) -> dict:
    """reqs: dicts with submit, first, token_times, end (perf_counter seconds), n_in, n_out.

    TTFT = first - submit; TPOT = (end - first) / (n_out - 1); ITL = gaps between token times;
    E2E = end - submit. Throughputs divide by the measured window `duration`."""
    ttft = [(r["first"] - r["submit"]) * 1e3 for r in reqs]
    e2el = [(r["end"] - r["submit"]) * 1e3 for r in reqs]
    tpot = [(r["end"] - r["first"]) / (r["n_out"] - 1) * 1e3 for r in reqs if r["n_out"] > 1]
    itl: list[float] = []
    for r in reqs:
        t = [r["first"]] + list(r["token_times"][1:])
        itl += [(b - a) * 1e3 for a, b in zip(t, t[1:])]
    tin = sum(r["n_in"] for r in reqs)
    tout = sum(r["n_out"] for r in reqs)
    ok = duration and duration > 0
    row = dict(duration=duration, completed=float(len(reqs)), total_input_tokens=float(tin),
               total_output_tokens=float(tout),
               request_throughput=len(reqs) / duration if ok else NAN,
               output_throughput=tout / duration if ok else NAN,
               total_token_throughput=(tin + tout) / duration if ok else NAN)
    for name, xs in (("ttft", ttft), ("tpot", tpot), ("itl", itl), ("e2el", e2el)):
        row[f"mean_{name}_ms"] = statistics.fmean(xs) if xs else NAN
        row[f"median_{name}_ms"] = pctl(xs, 50)
        row[f"std_{name}_ms"] = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        for q in (50, 90, 99):
            row[f"p{q}_{name}_ms"] = pctl(xs, q)
    if slo and ok:
        good = sum(1 for r, a in zip(reqs, ttft)
                   if a <= slo.get("ttft", 1e18)
                   and (r["n_out"] <= 1 or (r["end"] - r["first"]) / (r["n_out"] - 1) * 1e3 <= slo.get("tpot", 1e18))
                   and (r["end"] - r["submit"]) * 1e3 <= slo.get("e2el", 1e18))
        row["request_goodput"] = good / duration
    return row


def parse_slo(s: Optional[str]) -> Optional[dict]:
    """'ttft:500 tpot:50' -> {'ttft': 500.0, 'tpot': 50.0} (ms)."""
    if not s:
        return None
    out = {}
    for kv in s.split():
        k, v = kv.split(":")
        out[k] = float(v)
    return out
