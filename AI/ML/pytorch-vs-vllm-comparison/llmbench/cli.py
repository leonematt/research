"""`python -m llmbench <command>`.

Commands
--------
list       what this box would run: hardware, model, configs by tier/category, workloads
dry        print every server / benchmark command without running anything
check      functional test: every config on a tiny grid -> check.md (PASS/WARN/FAIL/NOOP/SKIP)
run        the benchmark (both engines by default), then the anomaly walk + retries, CSVs, report
suite      check (reused if fresh), then `run` with only what passed; composites rebuilt from it
retry      anomaly walk + targeted retries on an existing run
anomalies  just the anomaly walk (no retries) on an existing run
report     rebuild every CSV + report.md for an existing run
collect    merge every run's CSVs into results/all_*.csv (cross-hardware charts)

Presets (--preset) set the grid and how much work each measurement gets:
  check     tiny grid, 1 repeat, no retries                                (~1 h for everything)
  quick     quick grid, 1 repeat, 1 retry round                            (a few hours)
  standard  standard grid, 2-4 adaptive repeats (CV 5%), 1 retry round     (overnight)
  full      standard grid, every config on every workload, 2-5 repeats, 2 retry rounds
  long      long-context grid, 2-4 adaptive repeats, 1 retry round
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from . import anomalies, configs as C, hardware as H, tailor
from .checklist import harness_fingerprint, write_checklist
from .models import EPILOG, FAMILIES, family
from .report import write_report
from .results import aggregate, collect, write_all
from .taxonomy import CATEGORY_LABEL, CATEGORY_ORDER
from .util import exit_on_term, write_json
from .workloads import GRIDS, Workload, build_workloads

PRESETS = {
    "check": dict(grid="check", min_repeats=1, max_repeats=1, retry_rounds=0, warmup_prompts=2, settle_s=5,
                  max_batches=1),
    "quick": dict(grid="quick", min_repeats=1, max_repeats=1, retry_rounds=1),
    "standard": dict(grid="standard", min_repeats=2, max_repeats=4, retry_rounds=1),
    "full": dict(grid="standard", min_repeats=2, max_repeats=5, retry_rounds=2, all_workloads=True),
    "long": dict(grid="long", min_repeats=2, max_repeats=4, retry_rounds=1),
}
DEFAULT_SLO = "ttft:2000 tpot:100"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="llmbench", description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["list", "dry", "check", "run", "suite", "retry", "anomalies", "report",
                                       "collect", "probe"])
    p.add_argument("run_id_pos", nargs="?", help="run id for retry / anomalies / report")
    g = p.add_argument_group("what to run")
    g.add_argument("--preset", choices=list(PRESETS), help="grid + repeats + retries (default quick)")
    g.add_argument("--engines", default="vllm,pytorch", help="vllm, pytorch or both (default)")
    g.add_argument("--configs", help="comma-separated config names")
    g.add_argument("--categories", help="comma-separated categories: " + ",".join(CATEGORY_ORDER[:-1]))
    g.add_argument("--tiers", help="comma-separated tiers: baseline,default,ladder,optimized,component")
    g.add_argument("--skip", help="comma-separated config names to skip")
    g.add_argument("--exclude-components", default="", help="vLLM components to leave out of composites")
    g.add_argument("--grid", choices=list(GRIDS))
    g.add_argument("--workloads", help="comma-separated workload names (reference is always kept)")
    g.add_argument("--kinds", help="comma-separated workload kinds: grid,prefix,mixed,text,rate,sharegpt")
    g.add_argument("--all-workloads", action="store_true", default=None,
                   help="every config runs every workload (default: components run the core set + their focus)")
    g.add_argument("--sharegpt-path", help="ShareGPT json; adds a real-conversation workload")
    g = p.add_argument_group("model / hardware")
    g.add_argument("--family", choices=list(FAMILIES), default="qwen3")
    g.add_argument("--model", help="HF id or local path (default: largest family model that fits)")
    g.add_argument("--tp", type=int, help="tensor-parallel size (default: smallest that fits)")
    g.add_argument("--gpu-mem", type=float, help="--gpu-memory-utilization (default 0.90 discrete, 0.75 unified)")
    g.add_argument("--max-model-len", type=int)
    g.add_argument("--dtype")
    g.add_argument("--mm-off", choices=["language-model-only", "limit", "none"])
    g.add_argument("--spec-config", help="JSON for spec_draft's --speculative-config")
    g.add_argument("--eagle-config", help="JSON for spec_eagle3's --speculative-config")
    g.add_argument("--w4a16-model", help="4-bit checkpoint for the w4a16 config")
    g = p.add_argument_group("how much work")
    g.add_argument("--min-repeats", type=int)
    g.add_argument("--max-repeats", type=int)
    g.add_argument("--target-cv", type=float, default=0.05, help="add repeats while CV of throughput exceeds this")
    g.add_argument("--retry-rounds", type=int, help="anomaly walk + retry rounds after the sweep")
    g.add_argument("--max-retry-pairs", type=int, default=40, help="cap on (config, workload) pairs retried per round")
    g.add_argument("--noisy-cv", type=float, default=0.10, help="anomaly: repeats disagree by more than this")
    g.add_argument("--dip", type=float, default=0.25, help="anomaly: throughput drop that counts as a dip")
    g.add_argument("--max-hours", type=float, help="skip retry rounds once the run has taken this long")
    g.add_argument("--max-batches", type=int, help="PyTorch static batching: measured batches per run (default 4)")
    g.add_argument("--warmup-prompts", type=int)
    g.add_argument("--rate-base", type=float, help="req/s that rate workloads are fractions of (default: measured)")
    g.add_argument("--kv-oversub", type=float, default=4.0,
                   help="drop shapes needing more than this x the estimated KV capacity in flight (0 = keep all)")
    g = p.add_argument_group("output / runtime")
    g.add_argument("--out", default="results")
    g.add_argument("--run-id")
    g.add_argument("--run-tag")
    g.add_argument("--resume", action="store_true", help="keep existing rows; only run what's missing")
    g.add_argument("--profile", action="store_true", help="torch-profiler pass per config -> kernel_breakdown.csv")
    g.add_argument("--no-plots", dest="plots", action="store_false")
    g.add_argument("--port", type=int, default=8031)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--slo", default=DEFAULT_SLO, help=f'goodput SLO in ms (default "{DEFAULT_SLO}")')
    g.add_argument("--startup-timeout", type=int, default=1800)
    g.add_argument("--bench-timeout", type=int, default=3600)
    g.add_argument("--settle-s", type=float)
    g.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    g.add_argument("--drop-caches", dest="drop_caches", action="store_true", default=None)
    g.add_argument("--no-drop-caches", dest="drop_caches", action="store_false")
    g.add_argument("--allow-shared", action="store_true", help="run even if the GPUs are busy (results not clean)")
    g.add_argument("--no-nvcc-check", dest="nvcc_check", action="store_false")
    g.add_argument("--extra-server-args", default="")
    g.add_argument("--extra-bench-args", default="")
    g.add_argument("--kv-tokens", type=int, default=0, help=argparse.SUPPRESS)
    g.add_argument("--device", default="cuda", help=argparse.SUPPRESS)     # cpu: plumbing tests
    g.add_argument("--fresh-check", action="store_true", help="suite: always redo the check")
    a = p.parse_args(argv)
    preset = PRESETS[a.preset or ("check" if a.command == "check" else "quick")]
    for k, v in preset.items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    defaults = dict(min_repeats=1, max_repeats=1, retry_rounds=0, warmup_prompts=8, settle_s=20, max_batches=4,
                    all_workloads=False)
    for k, v in defaults.items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    a.max_repeats = max(a.max_repeats, a.min_repeats)
    a.run_id = a.run_id or a.run_id_pos
    a.dry_run = a.command == "dry"
    return a


# ----------------------------------------------------------------------------
# resolution
# ----------------------------------------------------------------------------

def resolve(a: argparse.Namespace) -> H.Hardware:
    hw = H.detect_hardware()
    a.hw = hw
    why_mem = None
    if a.gpu_mem is None:
        a.gpu_mem, why_mem = H.default_gpu_mem(hw)
    auto = H.autosize(hw, a.family, a.model, a.gpu_mem, a.tp)
    a.model, a.tp, a.weights_gib = auto["model"], auto["tp"], auto["weights_gib"]
    a.max_model_len = a.max_model_len or auto["max_model_len"]
    a.autosize_why = ([why_mem] if why_mem else []) + auto["why"]
    a.dtype = a.dtype or ("float16" if 0 < hw.min_cc < 8.0 else "bfloat16")
    fam = family(a.model)
    a.mm_off = a.mm_off or (fam["mm_off"] if fam else "none")
    if a.drop_caches is None:
        a.drop_caches = hw.unified
    return hw


def apply_summary(a: argparse.Namespace, s: dict) -> None:
    """An existing run fixes model, TP, grid, memory and context so everything stays comparable."""
    for k in ("model", "tp", "dtype", "max_model_len", "gpu_mem", "grid", "mm_off", "weights_gib"):
        if s.get(k) is not None:
            setattr(a, k, s[k])
    a.autosize_why = s.get("autosize_why", [])
    a.hw = H.detect_hardware()
    if a.drop_caches is None:
        a.drop_caches = a.hw.unified


def print_setup(a: argparse.Namespace, hw: H.Hardware) -> None:
    print(f"[hw]   {hw.describe()}  (via {hw.source}"
          + (f", CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}" if os.environ.get("CUDA_VISIBLE_DEVICES") else "")
          + ")")
    for w in a.autosize_why:
        print(f"[auto] {w}")
    print(f"[cfg]  model={a.model} tp={a.tp} dtype={a.dtype} max_model_len={a.max_model_len} gpu_mem={a.gpu_mem} "
          f"grid={a.grid} repeats={a.min_repeats}..{a.max_repeats} (CV {a.target_cv:.0%}) retries={a.retry_rounds}")
    if 0 < hw.min_cc < 8.0:
        print(f"[warn] sm_{str(hw.min_cc).replace('.', '')} is below sm_80: no BF16/FP8; numbers may not be representative.")


def build_plan(a: argparse.Namespace, exclude: Optional[set[str]] = None, quiet: bool = False):
    """(vllm configs, pytorch configs, ladders, workloads, skipped) after filters and skips."""
    from .torch_runner import apply_skips

    engines = set(a.engines.split(","))
    wls = build_workloads(a.grid, a.max_model_len, a.sharegpt_path,
                          only=set(a.workloads.split(",")) if a.workloads else None,
                          kinds=set(a.kinds.split(",")) if a.kinds else None, quiet=quiet)
    exclude = set(exclude or set()) | {x for x in a.exclude_components.split(",") if x}
    vcfgs, ladders = C.vllm_catalog(a, a.hw, exclude)
    pcfgs, pl = C.pytorch_catalog()
    ladders.update(pl)
    apply_skips(pcfgs, a, a.hw, any(w.concurrency == 1 for w in wls))
    vsel = C.filter_configs(vcfgs, a.configs, a.categories, a.tiers, a.skip) if "vllm" in engines else []
    psel = C.filter_configs(pcfgs, a.configs, a.categories, a.tiers, a.skip) if "pytorch" in engines else []
    if vsel and not any(c.name == "vllm_default" for c in vsel):
        vsel.insert(0, next(c for c in vcfgs if c.name == "vllm_default"))
        if not quiet:
            print("[select] vllm_default added: component deltas and rate workloads need it")
    skipped = {}
    for c in vsel + psel:
        if c.skip:
            skipped[c.name] = c.skip
    vsel = [c for c in vsel if not c.skip]
    psel = [c for c in psel if not c.skip]
    # fit the run to the card: memory reservations + workload shapes its KV cache can hold
    card = tailor.profile(a, a.hw)
    a.card = card
    if not quiet:
        for line in card.lines():
            print(f"[card] {line}")
    tailor.apply_headroom(vsel, a, card, quiet=quiet)
    wls, dropped = tailor.fit_workloads(wls, card, a.kv_oversub, quiet=quiet)
    a.dropped_workloads = [w.label for w in dropped]
    return vsel, psel, ladders, wls, skipped


def new_summary(a: argparse.Namespace, run_id: str, wls: list[Workload], ladders: dict) -> dict:
    ref = GRIDS[a.grid]["reference"]
    return dict(run_id=run_id, model=a.model, family=None, tp=a.tp, dtype=a.dtype, weights_gib=a.weights_gib,
                autosize_why=a.autosize_why, grid=a.grid, max_model_len=a.max_model_len, gpu_mem=a.gpu_mem,
                mm_off=a.mm_off, min_repeats=a.min_repeats, max_repeats=a.max_repeats, target_cv=a.target_cv,
                slo=a.slo, reference=list(ref), reference_label=next(w.label for w in wls if w.kind == "reference"),
                configs=[], workloads=[w.to_dict() for w in wls], server_facts={}, skipped={}, ladders=ladders,
                rate_base_rps=a.rate_base, harness=harness_fingerprint(),
                card=dict(vars(a.card)) if getattr(a, "card", None) else None,
                dropped_workloads=getattr(a, "dropped_workloads", []),
                environment=H.environment(a.hw) if not a.dry_run and a.device == "cuda" else dict(hardware=a.hw.to_dict()))


def merge_configs(summary: dict, cfgs: list[C.Config]) -> None:
    by = {c["name"]: i for i, c in enumerate(summary["configs"])}
    for c in cfgs:
        if c.name in by:
            summary["configs"][by[c.name]] = c.to_dict()
        else:
            summary["configs"].append(c.to_dict())
    order = {t: i for i, t in enumerate(C.TIER_ORDER)}
    summary["configs"].sort(key=lambda c: (c.get("engine") != "pytorch", order.get(c.get("tier"), 9)))
    summary["configs"].sort(key=lambda c: c.get("engine") != "pytorch")


def drop_inaccessible(cfgs: list[C.Config], skipped: dict) -> list[C.Config]:
    kept = []
    for c in cfgs:
        bad = next((r for s, r in (H.hf_access(m) for m in c.extra_models) if s == "fail"), None)
        if bad:
            print(f"[preflight] dropping {c.name}: {bad}")
            skipped[c.name] = bad
            continue
        kept.append(c)
    return kept


# ----------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------

def cmd_list(a: argparse.Namespace) -> None:
    hw = resolve(a)
    print()
    print_setup(a, hw)
    vsel, psel, ladders, wls, skipped = build_plan(a)
    ref = tuple(GRIDS[a.grid]["reference"])
    by_name = {}
    for c in vsel + psel:
        by_name[c.name] = c
    print("\nCONFIGS (tier · categories · workloads it runs · knob)\n")
    for eng, cs in (("vLLM", vsel), ("PyTorch", psel)):
        if not cs:
            continue
        print(f"  {eng}")
        for c in sorted(cs, key=lambda c: C.TIER_ORDER.index(c.tier)):
            n = len(C.select_workloads(c, wls, ref, a.all_workloads))
            cats = ",".join(CATEGORY_LABEL.get(x, x) for x in c.categories) or "–"
            print(f"    {c.name:<28}{c.tier:<10}{cats[:30]:<31}{n:>3} wl  {c.knob[:70]}")
            if c.components:
                print(f"    {'':<28}composed of: {', '.join(c.components)}")
    if skipped:
        print("\n  skipped here:")
        for k, v in skipped.items():
            print(f"    {k:<28}{v}")
    print(f"\nWORKLOADS `{a.grid}` ({len(wls)}, max_model_len={a.max_model_len})\n")
    for w in wls:
        print(f"  {w.name:<16}{w.kind:<10}{w.label:<38}num_prompts={w.num_prompts}")
    print("\nCATEGORIES -> configs")
    for cat in CATEGORY_ORDER[:-1]:
        names = [c.name for c in vsel + psel if cat in c.categories]
        print(f"  {CATEGORY_LABEL[cat]:<24}{', '.join(names) or '(none here)'}")
    n_runs = sum(len(C.select_workloads(c, wls, ref, a.all_workloads)) for c in vsel + psel)
    print(f"\n{len(vsel)} vLLM + {len(psel)} PyTorch configs, {n_runs} (config, workload) pairs x "
          f"{a.min_repeats}..{a.max_repeats} repeats")


def install_rebuilder(a: argparse.Namespace, exclude: Optional[set[str]], summary: dict, vsel: list) -> None:
    """Lets the vLLM runner rebuild a composite without a component whose flag got rejected."""
    excluded = set(exclude or set()) | {x for x in a.exclude_components.split(",") if x}
    comps = {c.name: c for c in C.vllm_components(a, a.hw)}

    def lookup(cfg, flag):
        return next((n for n in cfg.components if n in comps and any(
            tok == flag or tok.startswith(flag + "=") or (flag.isidentifier() and flag in tok)
            for tok in comps[n].args)), None)

    def rebuild(name, more):
        excluded.update(more)
        cat, _ = C.vllm_catalog(a, a.hw, excluded)
        new = next((c for c in cat if c.name == name), None)
        if new is not None:
            merge_configs(summary, [new])
            summary.setdefault("rebuilt_without", []).extend(sorted(more))
            for i, c in enumerate(vsel):            # retries and reports use the rebuilt definition
                if c.name == name:
                    vsel[i] = new
        return new

    a.rebuild_composite, a.rebuild_composite_lookup = rebuild, lookup


def _run_dir(a) -> Path:
    return Path(a.out) / a.run_id


def execute(a: argparse.Namespace, exclude: Optional[set[str]] = None, tag: Optional[str] = None) -> Path:
    """Resolve, preflight, run both engines, then the anomaly rounds, CSVs and report."""
    t_start = time.time()
    existing = a.run_id and (_run_dir(a) / "summary.json").exists()
    if existing:
        summary = json.loads((_run_dir(a) / "summary.json").read_text())
        apply_summary(a, summary)
        a.resume = True
    hw = a.hw if existing else resolve(a)
    run_id = a.run_id or f"{hw.slug}_{tag or a.run_tag or a.preset or 'run'}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    a.run_id = run_id
    run_dir = _run_dir(a)
    print()
    print_setup(a, hw)
    vsel, psel, ladders, wls, skipped = build_plan(a, exclude)
    if not existing:
        summary = new_summary(a, run_id, wls, ladders)
    else:
        known = {w["name"]: w for w in summary.get("workloads", [])}
        for w in wls:
            if w.name in known and known[w.name].get("request_rate"):
                w.request_rate, w.num_prompts = known[w.name]["request_rate"], known[w.name]["num_prompts"]
        summary.setdefault("ladders", {}).update(ladders)
    if a.dry_run:
        from .vllm_runner import run_vllm
        run_dir.mkdir(parents=True, exist_ok=True)
        run_vllm(a, run_dir, vsel, wls, summary, lambda: None)
        for c in psel:
            print(f"=== {c.name}  (pytorch {c.tier}: {c.knob})")
        return run_dir

    if vsel or psel:
        vsel = drop_inaccessible(vsel, skipped)
        n_gpus = max([c.gpus(a.tp) for c in vsel] + [a.tp])
        if a.device == "cuda":
            H.preflight(hw, model=a.model, port=a.port if vsel else None, n_gpus=n_gpus,
                        gpu_mem=a.gpu_mem if vsel else 0.0, drop_caches=a.drop_caches,
                        allow_shared=a.allow_shared, nvcc_check=a.nvcc_check and bool(vsel))
    run_dir.mkdir(parents=True, exist_ok=True)
    summary.setdefault("skipped", {}).update(skipped)
    merge_configs(summary, vsel + psel)

    def save_summary():
        summary["workloads"] = [w.to_dict() for w in wls]
        write_json(run_dir / "summary.json", summary)

    save_summary()
    if vsel:
        from .vllm_runner import run_vllm
        install_rebuilder(a, exclude, summary, vsel)
        run_vllm(a, run_dir, vsel, wls, summary, save_summary)
    if psel:
        from .torch_runner import run_pytorch
        run_pytorch(a, run_dir, psel, wls, summary, save_summary)
    aggregate(run_dir)
    finish(a, run_dir, summary, vsel, psel, wls, save_summary, t_start)
    return run_dir


def finish(a, run_dir: Path, summary: dict, vsel, psel, wls, save_summary, t_start: float) -> None:
    """Anomaly walk -> targeted retries -> re-check, then every CSV and the report."""
    rounds = []
    for rnd in range(1, (a.retry_rounds or 0) + 1):
        if a.max_hours and (time.time() - t_start) / 3600 > a.max_hours:
            print(f"[anomalies] --max-hours {a.max_hours} reached; skipping retry round {rnd}")
            break
        aggregate(run_dir)
        found = anomalies.detect(run_dir, a.noisy_cv, a.dip)
        anomalies.print_findings(found, f"round {rnd}")
        plan = anomalies.plan_retries(found, a.max_retry_pairs)
        rounds.append(dict(round=rnd, findings=found, after=None))
        if not plan:
            break
        ref_t = tuple(summary["reference"])
        for eng, cfgs, targets in (("vllm", vsel, plan.get("vllm", {})), ("pytorch", psel, plan.get("pytorch", {}))):
            sel = [c for c in cfgs if c.name in targets]
            if not sel:
                continue
            for c in sel:          # '*' = every workload of that config (server failed to start)
                if "*" in targets[c.name]:
                    targets[c.name] = {w.name: 1 for w in C.select_workloads(c, wls, ref_t, a.all_workloads)}
            if eng == "vllm":
                from .vllm_runner import run_vllm
                run_vllm(a, run_dir, sel, wls, summary, save_summary, targets=targets)
            else:
                from .torch_runner import run_pytorch
                run_pytorch(a, run_dir, sel, wls, summary, save_summary, targets=targets)
        aggregate(run_dir)
        rounds[-1]["after"] = anomalies.detect(run_dir, a.noisy_cv, a.dip)
        res = [f for f in found if f["action"] == "retry"]
        still = {(x["config"], x["workload"], x["check"]) for x in rounds[-1]["after"]}
        n_res = sum(1 for f in res if (f["config"], f["workload"], f["check"]) not in still)
        print(f"[anomalies] round {rnd}: {n_res} of {len(res)} retried findings resolved")
    if not rounds:
        aggregate(run_dir)
        found = [dict(f, action="report") if f["action"] == "retry" else f
                 for f in anomalies.detect(run_dir, a.noisy_cv, a.dip)]
        anomalies.print_findings(found, "walk (retries off; `retry <run_id>` re-runs these)")
        rounds.append(dict(round=0, findings=found, after=None))
    anomalies.write(run_dir, rounds)
    summary["elapsed_h"] = round((time.time() - t_start) / 3600, 2)
    save_summary()
    write_all(run_dir)
    write_report(run_dir, a.plots)
    print(f"\n[done] {run_dir}  ({summary['elapsed_h']:.1f} h)\n"
          f"       graph-ready: results_long.csv, optimizations.csv, category_summary.csv, headline.csv, ladder.csv")


def latest_check(out: Path) -> Optional[Path]:
    cs = sorted([d for d in out.glob("*_check_*") if (d / "check_passed.json").exists()],
                key=lambda d: d.stat().st_mtime, reverse=True)
    return cs[0] if cs else None


def cmd_check(a: argparse.Namespace) -> Path:
    a.retry_rounds = 0
    run_dir = execute(a, tag="check")
    write_checklist(run_dir)
    return run_dir


def cmd_suite(a: argparse.Namespace) -> None:
    out = Path(a.out)
    chk = latest_check(out)
    fresh = None
    if chk and not a.fresh_check:
        cp = json.loads((chk / "check_passed.json").read_text())
        age_h = (time.time() - (chk / "check_passed.json").stat().st_mtime) / 3600
        if cp.get("harness") == harness_fingerprint() and age_h < 6 and "vllm_default" in cp.get("vllm", []):
            fresh = chk
            print(f"=== suite 1/2: reusing the check from {chk} (same harness, {age_h:.1f} h old; --fresh-check redoes it)")
    if fresh is None:
        print("=== suite 1/2: functional check (every config, tiny grid)")
        ca = parse_args(["check"] + _passthrough(a))
        fresh = cmd_check(ca)
    cp = json.loads((fresh / "check_passed.json").read_text())
    s = json.loads((fresh / "summary.json").read_text())
    if "vllm" in a.engines and "vllm_default" not in cp.get("vllm", []):
        raise SystemExit(f"suite: vllm_default failed the check, so every vLLM delta would be meaningless; "
                         f"see {fresh / 'check.md'}")
    # run with exactly what the check validated: same model / TP / memory fraction / context
    for k in ("model", "tp", "gpu_mem", "max_model_len"):
        if getattr(a, k) is None and s.get(k) is not None:
            setattr(a, k, s[k])
    composites = {c["name"] for c in s["configs"] if c.get("components") or c.get("tier") in ("ladder", "optimized", "baseline")}
    keep = set(cp.get("vllm", [])) | set(cp.get("pytorch", [])) | composites
    if a.configs:
        keep &= set(a.configs.split(","))
    a.configs = ",".join(sorted(keep))
    print(f"=== suite 2/2: {a.preset or 'quick'} run with {len(keep)} configs that passed "
          f"(composites rebuilt without: {', '.join(cp.get('failed_components', [])) or 'nothing'})")
    run_dir = execute(a, exclude=set(cp.get("failed_components", [])))
    print(f"=== suite done: {run_dir / 'report.md'}   (checklist: {fresh / 'check.md'})")


def _passthrough(a: argparse.Namespace) -> list[str]:
    out = []
    for flag, val in (("--family", a.family), ("--model", a.model), ("--tp", a.tp), ("--gpu-mem", a.gpu_mem),
                      ("--max-model-len", a.max_model_len), ("--out", a.out), ("--port", a.port),
                      ("--engines", a.engines), ("--sharegpt-path", a.sharegpt_path), ("--device", a.device)):
        if val is not None:
            out += [flag, str(val)]
    for flag, on in (("--allow-shared", a.allow_shared), ("--profile", False), ("--no-plots", not a.plots)):
        if on:
            out.append(flag)
    return out


def existing_run(a: argparse.Namespace) -> tuple[Path, dict]:
    if not a.run_id:
        runs = sorted([d for d in Path(a.out).glob("*") if (d / "summary.json").exists()],
                      key=lambda d: d.stat().st_mtime, reverse=True)
        if not runs:
            raise SystemExit(f"no runs under {a.out}")
        a.run_id = runs[0].name
    run_dir = _run_dir(a)
    if not (run_dir / "summary.json").exists():
        raise SystemExit(f"{run_dir} has no summary.json")
    return run_dir, json.loads((run_dir / "summary.json").read_text())


def cmd_retry(a: argparse.Namespace, detect_only: bool = False) -> None:
    run_dir, summary = existing_run(a)
    apply_summary(a, summary)
    aggregate(run_dir)
    if detect_only:
        found = anomalies.detect(run_dir, a.noisy_cv, a.dip)
        anomalies.print_findings(found, run_dir.name)
        anomalies.write(run_dir, [dict(round=0, findings=found, after=None)])
        return
    cfgs = [C.Config.from_dict(c) for c in summary["configs"]]
    wls = [Workload.from_dict(w) for w in summary["workloads"]]
    a.min_repeats = summary.get("min_repeats", a.min_repeats)
    a.max_repeats = summary.get("max_repeats", a.max_repeats)
    a.resume = True
    a.retry_rounds = a.retry_rounds or 1

    def save_summary():
        summary["workloads"] = [w.to_dict() for w in wls]
        write_json(run_dir / "summary.json", summary)

    if a.device == "cuda":
        H.preflight(a.hw, model=a.model, port=a.port, n_gpus=max([c.gpus(a.tp) for c in cfgs] + [a.tp]),
                    gpu_mem=a.gpu_mem, drop_caches=a.drop_caches, allow_shared=a.allow_shared, nvcc_check=a.nvcc_check)
    finish(a, run_dir, summary, [c for c in cfgs if c.engine == "vllm"], [c for c in cfgs if c.engine == "pytorch"],
           wls, save_summary, time.time())


def cmd_report(a: argparse.Namespace) -> None:
    run_dir, _ = existing_run(a)
    aggregate(run_dir)
    write_all(run_dir)
    write_report(run_dir, a.plots)


def main(argv=None) -> None:
    exit_on_term()
    a = parse_args(argv)
    if a.command == "probe":          # for run.sh: run id + unified flag, cheaply
        hw = H.detect_hardware()
        print(a.run_id or f"{hw.slug}_{a.run_tag or a.preset or 'run'}_{dt.datetime.now():%Y%m%d_%H%M%S}",
              int(hw.unified))
        return
    if a.command == "list":
        cmd_list(a)
    elif a.command in ("dry", "run"):
        execute(a)
    elif a.command == "check":
        cmd_check(a)
    elif a.command == "suite":
        cmd_suite(a)
    elif a.command == "retry":
        cmd_retry(a)
    elif a.command == "anomalies":
        cmd_retry(a, detect_only=True)
    elif a.command == "report":
        cmd_report(a)
    elif a.command == "collect":
        files = collect(Path(a.out))
        print("\n".join(f"[collect] {f}" for f in files) or "[collect] nothing to merge")


if __name__ == "__main__":
    main(sys.argv[1:])
