"""Functional check: PASS / WARN / FAIL / NOOP / SKIP per config, from a check-grid run.

PASS  started, every workload produced a row, and the engine reported the knob as applied
WARN  ran fine, but the knob couldn't be read back to confirm it (still included)
FAIL  didn't start, a workload failed, or the engine contradicts the knob: the flag was accepted
      but did nothing, so its numbers would be the default's under another name (excluded)
NOOP  ran, but here it ends up with the same engine settings as another config (excluded)
SKIP  not applicable on this machine/model, with the reason

Composite configs (vllm_minimal, the ladder, vllm_optimized*) are rebuilt for the full run from
the components that passed, so their own check status is advisory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .util import md_table, write_json

COMPILE_WARN_S = 1.0


def _check(op: str, want: str, got) -> bool:
    g = str(got).lower()
    if op == "is":
        return g == str(want).lower()
    if op == "has":
        return str(want).lower() in g
    if op == "not_has":
        return str(want).lower() not in g
    if op == "set":
        return g not in ("none", "", "nan", "false")
    return True


def harness_fingerprint() -> str:
    """Hash of the harness code, stored with a check so `suite` never reuses a check made with
    different config definitions."""
    h = hashlib.sha1()
    for p in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def write_checklist(run_dir: Path) -> dict:
    import pandas as pd

    from .vllm_runner import parse_server_log

    if not (run_dir / "summary.json").exists():
        raise SystemExit(f"[check] {run_dir} has no summary.json: nothing ran")
    summary = json.loads((run_dir / "summary.json").read_text())
    facts_all = summary.get("server_facts", {})
    for name, f in facts_all.items():             # re-parse logs with the current patterns
        log = run_dir / "server_logs" / f"{name}.log"
        if not name.startswith("pt_") and isinstance(f, dict) and log.exists():
            f.update(parse_server_log(log))
    df = pd.read_csv(run_dir / "results_all.csv") if (run_dir / "results_all.csv").exists() else None
    rows, passed, failed = [], {"vllm": [], "pytorch": []}, []
    cfgs = {c["name"]: c for c in summary["configs"]}
    for name in list(cfgs) + [n for n in summary.get("skipped", {}) if n not in cfgs]:
        c = cfgs.get(name, {})
        engine = c.get("engine") or ("pytorch" if name.startswith("pt_") else "vllm")
        composite = bool(c.get("components")) or c.get("tier") in ("ladder", "optimized", "baseline") and engine == "vllm"
        f = facts_all.get(name, {})
        skip = summary.get("skipped", {}).get(name)
        sub = df[df["config"] == name] if df is not None and len(df) else None
        n_ok = int(((sub["bench_rc"].fillna(0) == 0) & sub["output_throughput"].notna()).sum()) \
            if sub is not None and len(sub) and "output_throughput" in sub else 0
        n_all = int(sub["workload"].nunique()) if sub is not None and len(sub) else 0
        n_ok_wl = int(sub[(sub["bench_rc"].fillna(0) == 0)]["workload"].nunique()) if n_all else 0
        knob, detail = "–", ""
        if skip:
            status, detail = "SKIP", skip
        elif f.get("status") != "ok":
            status, detail = "FAIL", str(f.get("error") or "no facts (did it run?)")[:140]
        elif n_all == 0 or n_ok_wl < n_all:
            bad = sub[sub["bench_rc"].fillna(0) != 0] if n_all else None
            status = "FAIL"
            detail = str(bad["bench_error"].iloc[0])[:140] if bad is not None and len(bad) else "no workload rows"
            if bad is not None and len(bad) and bad["bench_error"].astype(str).str.startswith("OOM").all():
                status, detail = "WARN", f"OOM on {len(bad)} workload(s) (a result, not a failure): {detail[:80]}"
        else:
            status = "PASS"
            verify = c.get("verify") or []
            bad = [(k, want, f.get(k), what) for k, op, want, what in verify
                   if f.get(k) is not None and not _check(op, want, f.get(k))]
            unread = [(k, what) for k, op, want, what in verify if f.get(k) is None]
            dup = c.get("same_as")
            other = facts_all.get(dup[0], {}) if dup else {}
            if f.get("cb_dropped"):
                status, knob = "FAIL", "✗"
                detail = f"transformers doesn't accept {', '.join(f['cb_dropped'])}: the knob can't apply"
            elif bad:
                k, want, got, what = bad[0]
                status, knob, detail = "FAIL", "✗", f"knob not applied: expected {what}, engine reported {k}={got}"
            elif dup and other.get("status") == "ok" and all(
                    f.get(k) is not None and f.get(k) == other.get(k) for k in dup[1]):
                status, knob = "NOOP", "="
                detail = (f"same engine settings as {dup[0]} here ({', '.join(f'{k}={f.get(k)}' for k in dup[1])}): "
                          f"its numbers would duplicate {dup[0]}'s")
            elif verify and unread:
                status, knob = "WARN", "?"
                detail = "couldn't read back " + ", ".join(f"`{k}` ({w})" for k, w in unread[:2])
            elif verify:
                knob = "✓ " + "; ".join(v[3] for v in verify[:3])
            elif f.get("resolved"):
                knob = f.get("resolved")
            if status == "PASS" and (f.get("compile_s_in_measure") or 0) >= COMPILE_WARN_S:
                status, detail = "WARN", (f"torch.compile still compiling inside the measured window "
                                          f"({f.get('compiles_in_measure')} compiles, {f.get('compile_s_in_measure', 0):.0f}s)")
        if status in ("PASS", "WARN"):
            passed[engine].append(name)
        elif status in ("FAIL", "NOOP") and not composite:
            failed.append(name)
        rows.append(dict(config=name, engine=engine, tier=c.get("tier", "–"), status=status,
                         workloads=f"{n_ok_wl}/{n_all}" if n_all else "–", knob=knob, detail=detail))
        _ = n_ok

    counts = {s: sum(1 for r in rows if r["status"] == s) for s in ("PASS", "WARN", "FAIL", "NOOP", "SKIP")}
    t = pd.DataFrame(rows).set_index("config")
    md = [f"# Functional check — `{run_dir.name}`\n",
          f"{summary.get('model')} · {summary.get('environment', {}).get('hardware', {}).get('describe', '')}\n",
          f"**{counts['PASS']} pass · {counts['WARN']} warn · {counts['FAIL']} fail · {counts['NOOP']} no-op · "
          f"{counts['SKIP']} skip**\n", __doc__.split("\n\n", 1)[1].replace("\n", "  \n"), md_table(t, index_name="config")]
    (run_dir / "check.md").write_text("\n".join(md))
    write_json(run_dir / "check_passed.json", {**passed, "failed_components": failed, "harness": harness_fingerprint()})
    icon = {"PASS": "ok  ", "WARN": "warn", "FAIL": "FAIL", "NOOP": "noop", "SKIP": "skip"}
    print(f"\n[check] {run_dir.name}: {counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail, "
          f"{counts['NOOP']} no-op, {counts['SKIP']} skip")
    for r in rows:
        print(f"  {icon[r['status']]}  {r['config']:<28} {r['workloads']:>5}  "
              f"{(r['detail'] or (r['knob'] if r['knob'] != '–' else ''))[:100]}")
    print(f"[check] wrote {run_dir / 'check.md'}")
    return passed
