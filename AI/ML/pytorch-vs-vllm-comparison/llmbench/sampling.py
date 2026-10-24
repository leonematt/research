"""How much to measure: adaptive repeats and a running ETA.

Every (config, workload) gets at least `min_repeats` measured runs. More are added only while
the runs disagree: when the coefficient of variation (stdev / mean) of output throughput is
above `target_cv`, another repeat is run, up to `max_repeats`. Quiet measurements stop at the
minimum; noisy ones get the samples they need. The reported value is the median of the good
repeats, so one outlier can't drag it.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from .util import is_num


@dataclass
class Policy:
    min_repeats: int = 1
    max_repeats: int = 3
    target_cv: float = 0.05        # 5% run-to-run spread on output throughput

    def __post_init__(self):
        self.min_repeats = max(1, self.min_repeats)
        self.max_repeats = max(self.min_repeats, self.max_repeats)


def good(row: dict) -> bool:
    return not row.get("bench_rc") and is_num(row.get("output_throughput"))


def cv(vals: list[float]) -> float:
    vals = [v for v in vals if is_num(v)]
    if len(vals) < 2:
        return float("nan")
    m = statistics.fmean(vals)
    return statistics.stdev(vals) / m if m else float("nan")


def need_more(rows: list[dict], attempted: int, p: Policy, force_extra: int = 0) -> bool:
    """rows = every repeat so far (good or failed); attempted = runs made in this session."""
    ok = [r for r in rows if good(r)]
    if attempted >= p.max_repeats + force_extra:
        return False
    if len(ok) < p.min_repeats + force_extra:
        return True
    if p.min_repeats == 1 and p.max_repeats == 1:
        return False
    c = cv([r["output_throughput"] for r in ok])
    if len(ok) == 1:                          # can't judge noise from one sample
        return p.max_repeats > 1 and p.min_repeats > 1
    return c == c and c > p.target_cv


QUIET_PROBES = 2   # workloads with >= 2 repeats needed before a config counts as quiet


def is_quiet(cvs: list[float], p: Policy) -> bool:
    """A config whose first QUIET_PROBES measured workloads all agreed to within half the target
    spread gets 1 repeat for the rest: a second repeat would only re-measure the same number.
    (On a stable box this halves the run; noisy configs keep the full adaptive treatment.)"""
    return len(cvs) >= QUIET_PROBES and all(c <= p.target_cv / 2 for c in cvs[:QUIET_PROBES])


class ETA:
    """Remaining-time estimate from the average wall time of the runs done so far."""

    def __init__(self, planned_runs: int):
        self.planned = max(1, planned_runs)
        self.done = 0
        self.t0 = time.time()

    def tick(self) -> None:
        self.done += 1

    def add(self, n: int) -> None:
        self.planned += n

    def __str__(self) -> str:
        if not self.done:
            return "ETA –"
        per = (time.time() - self.t0) / self.done
        left = max(0, self.planned - self.done) * per
        h, m = divmod(int(left / 60), 60)
        return f"ETA ~{h}h{m:02d}m" if h else f"ETA ~{m}m"
