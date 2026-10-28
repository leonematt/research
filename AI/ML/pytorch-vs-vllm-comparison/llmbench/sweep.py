"""Concurrency sweeps stop where the card does.

Every grid shape (input_len x output_len) is swept over concurrency 1, 2, 4, ..., 128 in
ascending order, and each config stops a shape as soon as going higher can't tell us anything new:

  out of memory   a concurrency ran out of memory (PyTorch static batching): every larger one
                  would too, so they're skipped;
  saturated       the last step up gained < SATURATION_GAIN throughput while the engine was
                  preempting requests (vLLM queues instead of failing): larger concurrencies only
                  measure a longer waiting line, so they're skipped.

Skipped points are written as rows with status "skipped" and the reason, so tables and charts show
where each config hit its limit instead of a silent gap.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Optional

from .util import is_num, read_row

SATURATION_GAIN = 0.05
OOM_TAG = 2            # rows carrying oom_v == OOM_TAG were measured on a clean GPU (trustworthy OOMs)


def is_sweep(w) -> bool:
    return w.kind in ("grid", "reference") and w.closed_loop


def _shape(w) -> tuple:
    return (w.input_len, w.output_len, w.prefix_len)


def _good(r: dict) -> bool:
    return not r.get("bench_rc") and is_num(r.get("output_throughput"))


def is_skip_row(r: dict) -> bool:
    return bool(r.get("skipped"))


def _trusted_oom(r: dict) -> bool:
    return str(r.get("bench_error", "")).startswith("OOM") and r.get("oom_v") == OOM_TAG


def skip_reason(w, rows_by_wl: dict[str, list[dict]], wls_by_name: dict) -> Optional[str]:
    """Why this sweep point should be skipped, given what the same config measured at smaller
    concurrencies of the same shape; None to run it."""
    if not is_sweep(w):
        return None
    pts = []
    for name, rows in rows_by_wl.items():
        w2 = wls_by_name.get(name)
        if w2 is None or not is_sweep(w2) or _shape(w2) != _shape(w) or w2.concurrency >= w.concurrency:
            continue
        pts.append((w2.concurrency, rows))
    pts.sort(key=lambda p: p[0])
    for c, rows in pts:
        if rows and not any(_good(r) for r in rows) and any(_trusted_oom(r) or is_skip_row(r) for r in rows):
            return f"c={c} ran out of memory" if any(_trusted_oom(r) for r in rows) else f"c={c} was already past the limit"
    good_pts = []
    for c, rows in pts:
        g = [r for r in rows if _good(r)]
        if g:
            pre = [r.get("preemptions") for r in g if is_num(r.get("preemptions"))]
            good_pts.append((c, statistics.median(r["output_throughput"] for r in g),
                             max(pre) if pre else None))      # None: the engine reports no preemptions
    if len(good_pts) >= 2:
        (c1, t1, _), (c2, t2, p2) = good_pts[-2], good_pts[-1]
        if t1 > 0 and t2 < (1 + SATURATION_GAIN) * t1:
            if p2 is None:       # transformers CB queues internally and exposes no counter: flat is enough
                return f"saturated: c={c1}->{c2} gained {t2 / t1 - 1:+.0%} throughput (requests queue for KV space)"
            if p2 > 0:
                return f"saturated: c={c1}->{c2} gained {t2 / t1 - 1:+.0%} throughput with {p2:.0f} preemptions"
    return None


def load_rows(raw: Path, cfg_name: str, wls: list) -> dict[str, list[dict]]:
    out = {}
    for w in wls:
        rows = []
        for f in raw.glob(f"{cfg_name}__{w.name}__r*.row.json"):
            try:
                r = read_row(f)
                if r is not None:
                    rows.append(r)
            except Exception:  # noqa: BLE001
                pass
        out[w.name] = rows
    return out
