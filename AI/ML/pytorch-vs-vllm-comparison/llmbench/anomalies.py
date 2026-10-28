"""Walk the data after a sweep, flag anomalies, and plan targeted retries.

Checks (per config x workload unless noted)          action
------------------------------------------------------ -------------------------------
failed        every repeat failed (not OOM)              retry
partial       some repeats failed                        retry (failed ones are replaced)
oom           a PyTorch static batch didn't fit           report (deterministic)
errors        some requests inside a run errored          retry
incomplete    fewer requests completed than were sent     retry
impossible    TTFT > E2E, TPOT <= 0, zero throughput      retry
noisy         repeats disagree (CV above --noisy-cv)      retry with 2 more repeats
dip           throughput falls > --dip when concurrency   retry both points
              rises, without preemptions to explain it
prefix_reset  the prefix cache couldn't be reset          retry
compiling     torch.compile ran inside the measured       retry (now warm)
              window (PyTorch)
ladder        a ladder step is > --dip slower than the    retry both steps once, to confirm
              step before it at the same workload
no_spec_stats a speculative config reported no            report
              acceptance counters on the text workload
server        a vLLM server failed to start for a         retry the config once
              transient-looking reason (timeout, memory)

Retries are capped (--max-retry-pairs), ordered by severity, and each round is re-checked; a
finding that survives its retry is marked `persisting` (likely real behaviour, or a genuine
bug in that config), one that disappears is `resolved` (the first measurement was the outlier).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .util import is_num, write_json

SEVERITY = {"server": 0, "failed": 1, "impossible": 2, "incomplete": 3, "errors": 4, "partial": 5,
            "prefix_reset": 6, "compiling": 7, "noisy": 8, "dip": 9, "ladder": 10, "oom": 20, "no_spec_stats": 21}
TRANSIENT_SERVER = re.compile(r"startup timeout|probe failed|Free memory on device|exited with -9|exited with 137",
                              re.I)


def _f(x):
    return float(x) if is_num(x) else float("nan")


def detect(run_dir: Path, noisy_cv: float = 0.10, dip: float = 0.25) -> list[dict]:
    import pandas as pd

    if not (run_dir / "results.csv").exists():
        return []
    df = pd.read_csv(run_dir / "results.csv")
    summary = json.loads((run_dir / "summary.json").read_text())
    out: list[dict] = []

    def add(r, check, detail, action="retry", extra=1):
        out.append(dict(config=r["config"], engine=r.get("engine"), workload=r["workload"],
                        workload_label=r.get("workload_label"), check=check, action=action, extra_repeats=extra,
                        detail=detail))

    for _, r in df.iterrows():
        r = r.to_dict()
        status = r.get("status")
        err = str(r.get("bench_error") or "")
        if status == "skipped":
            continue                  # past this config's limit (sweep.py): expected, not an anomaly
        if status == "oom":
            add(r, "oom", err[:120], action="report", extra=0)
            continue
        if status != "ok":
            add(r, "failed", err[:160] or "no successful repeat")
            continue
        if int(r.get("n_failed") or 0) > 0:
            add(r, "partial", f"{int(r['n_failed'])} of {int(r['n_failed']) + int(r['n_repeats'])} repeats failed")
        if _f(r.get("n_errors")) > 0:
            add(r, "errors", f"{int(r['n_errors'])} request errors")
        comp, sent = _f(r.get("completed")), _f(r.get("num_prompts"))
        if is_num(comp) and is_num(sent) and r.get("kind") != "rate" and comp < sent * 0.98:
            add(r, "incomplete", f"{comp:.0f} of {sent:.0f} requests completed")
        ttft, e2e, tpot, thr = (_f(r.get(k)) for k in ("median_ttft_ms", "median_e2el_ms", "median_tpot_ms",
                                                       "output_throughput"))
        bad = []
        if is_num(ttft) and is_num(e2e) and ttft > e2e * 1.01:
            bad.append(f"TTFT p50 {ttft:.0f} > E2E p50 {e2e:.0f} ms")
        if is_num(tpot) and tpot <= 0:
            bad.append(f"TPOT {tpot}")
        if not is_num(thr) or thr <= 0:
            bad.append(f"throughput {thr}")
        if bad:
            add(r, "impossible", "; ".join(bad))
        c = _f(r.get("cv_output_throughput"))
        if is_num(c) and c > noisy_cv:
            add(r, "noisy", f"run-to-run CV {c:.0%} over {int(r['n_repeats'])} repeats (limit {noisy_cv:.0%})", extra=2)
        if r.get("kind") == "prefix" and r.get("engine") == "vllm" and _f(r.get("prefix_reset_ok")) == 0:
            add(r, "prefix_reset", "prefix cache was not reset before the run")
        if _f(r.get("compile_s_in_measure")) >= 1.0:
            add(r, "compiling", f"{_f(r.get('compile_s_in_measure')):.1f}s of torch.compile inside the measured window")
        if "speculative_decoding" in str(r.get("categories")) and r.get("engine") == "vllm" \
                and r.get("kind") == "text" and not is_num(r.get("spec_accept_rate")):
            add(r, "no_spec_stats", "no spec-decode acceptance counters scraped", action="report", extra=0)

    # throughput dips as concurrency rises (closed-loop grid points of one shape)
    ok = df[df["status"] == "ok"]
    grid = ok[ok["kind"].isin(["grid", "reference"]) & (ok["prefix_len"].fillna(0) == 0)]
    for (cfg, _i, _o), g in grid.groupby(["config", "input_len", "output_len"]):
        g = g.drop_duplicates("concurrency").sort_values("concurrency")
        pts = list(g.to_dict("records"))
        for lo, hi in zip(pts, pts[1:]):
            a, b = _f(lo.get("output_throughput")), _f(hi.get("output_throughput"))
            if is_num(a) and is_num(b) and b < (1 - dip) * a and not _f(hi.get("preemptions")) > 0:
                why = f"c={int(lo['concurrency'])}: {a:,.0f} -> c={int(hi['concurrency'])}: {b:,.0f} tok/s"
                add(hi, "dip", why)
                add(lo, "dip", why)

    # ladder regressions
    idx = ok.set_index(["config", "workload"])
    for lname, steps in summary.get("ladders", {}).items():
        present = [s["config"] for s in steps if s["config"] in set(ok["config"])]
        for prev, cur in zip(present, present[1:]):
            if cur == "vllm_optimized_spec":       # spec decoding at high concurrency is expected to lose
                continue
            for wl in set(idx.xs(cur, level="config").index) & set(idx.xs(prev, level="config").index):
                a, b = _f(idx.loc[(prev, wl)].get("output_throughput")), _f(idx.loc[(cur, wl)].get("output_throughput"))
                if is_num(a) and is_num(b) and b < (1 - dip) * a:
                    rp, rc = idx.loc[(prev, wl)].to_dict(), idx.loc[(cur, wl)].to_dict()
                    rp.update(config=prev, workload=wl)
                    rc.update(config=cur, workload=wl)
                    why = f"{lname}: {prev} {a:,.0f} -> {cur} {b:,.0f} tok/s"
                    add(rc, "ladder", why)
                    add(rp, "ladder", why)

    # vLLM servers that failed for a transient-looking reason
    for name, f in summary.get("server_facts", {}).items():
        if name.startswith("pt_") or not isinstance(f, dict) or f.get("status") == "ok":
            continue
        err = str(f.get("error") or "") + " " + str(f.get("log_tail") or "")[-400:]
        if TRANSIENT_SERVER.search(err) and not f.get("retry_of"):
            out.append(dict(config=name, engine="vllm", workload="*", workload_label="(server start)", check="server",
                            action="retry", extra_repeats=1, detail=str(f.get("error"))[:160]))

    # one finding per (config, workload, check)
    seen, uniq = set(), []
    for x in sorted(out, key=lambda x: SEVERITY.get(x["check"], 50)):
        k = (x["config"], x["workload"], x["check"])
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


def plan_retries(findings: list[dict], max_pairs: int) -> dict[str, dict[str, dict[str, int]]]:
    """{engine: {config: {workload ('*' = all its workloads): extra good repeats}}}, most severe first."""
    plan: dict[str, dict[str, dict[str, int]]] = {}
    pairs = 0
    for f in findings:
        if f["action"] != "retry":
            continue
        eng = "pytorch" if str(f["config"]).startswith("pt_") else "vllm"
        cur = plan.setdefault(eng, {}).setdefault(f["config"], {})
        if f["workload"] not in cur:
            if pairs >= max_pairs:
                f["action"] = "report (retry budget used)"
                continue
            pairs += 1
        cur[f["workload"]] = max(cur.get(f["workload"], 0), int(f.get("extra_repeats") or 1))
    return plan


def write(run_dir: Path, rounds: list[dict]) -> None:
    """rounds: [{round, findings, after}] -> anomalies.csv with each finding's outcome."""
    import pandas as pd

    rows = []
    for rd in rounds:
        after = {(x["config"], x["workload"], x["check"]) for x in rd.get("after", [])} \
            if rd.get("after") is not None else None
        for f in rd["findings"]:
            k = (f["config"], f["workload"], f["check"])
            if not str(f["action"]).startswith("retry"):
                outcome = "reported"
            elif after is None:
                outcome = "pending"
            else:
                outcome = "persisting" if k in after else "resolved"
            rows.append(dict(round=rd["round"], **f, outcome=outcome))
    pd.DataFrame(rows, columns=["round", "config", "engine", "workload", "workload_label", "check", "action",
                                "extra_repeats", "detail", "outcome"]).to_csv(run_dir / "anomalies.csv", index=False)
    write_json(run_dir / "anomalies.json", rounds)


def print_findings(findings: list[dict], title: str) -> None:
    if not findings:
        print(f"[anomalies] {title}: none")
        return
    n_retry = sum(1 for f in findings if f["action"] == "retry")
    print(f"[anomalies] {title}: {len(findings)} finding(s), {n_retry} to retry")
    for f in findings[:40]:
        print(f"  {f['check']:<13} {f['action'][:6]:<6} {f['config']:<26} {str(f['workload_label'])[:34]:<34} {f['detail'][:70]}")
    if len(findings) > 40:
        print(f"  ... {len(findings) - 40} more in anomalies.csv")
