#!/usr/bin/env python3
"""Tests that run anywhere: no GPU, no vLLM, no torch.

Unit tests cover the pure logic (argument merging, run sizing, metrics, adaptive repeats, trace
parsing, anomaly detection). The end-to-end test runs the real orchestrator against the stand-in
`vllm` CLI in tests/fake_vllm and a synthetic PyTorch worker (LLMBENCH_FAKE_TORCH=1), with injected
noise, a failing workload and a flag the "engine" rejects, then checks every CSV and that the
anomaly walk caught and retried what it should.

    python3 tests/run_tests.py            # everything (~2-3 min)
    python3 tests/run_tests.py unit       # unit tests only (seconds)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILS: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        FAILS.append(what)


def unit() -> None:
    from llmbench import anomalies, profiling, sampling
    from llmbench.configs import merge_args, pytorch_catalog, vllm_catalog
    from llmbench.metrics import summarize
    from llmbench.taxonomy import CATEGORIES, all_comparisons
    from llmbench.workloads import build_workloads, make_prompts, num_prompts_for, output_lengths

    print("unit: configs")
    m = merge_args(["--enforce-eager", "--max-num-batched-tokens", "8192"],
                   ["--compilation-config", '{"cudagraph_mode": "NONE"}'],
                   ["--compilation-config", '{"pass_config": {"fuse_norm_quant": false}}', "-O1"], ["-O3"],
                   ["--enable-prefix-caching"], ["--no-enable-prefix-caching"])
    check(m.count("-O3") == 1 and "-O1" not in m, "merge_args keeps only the last -O level")
    cc = json.loads(m[m.index("--compilation-config") + 1])
    check(cc == {"cudagraph_mode": "NONE", "pass_config": {"fuse_norm_quant": False}}, "compilation-config deep-merged")
    check("--no-enable-prefix-caching" in m and "--enable-prefix-caching" not in m, "later --no-x overrides --x")

    print("unit: workloads")
    check(num_prompts_for(1, 256) == 8 and num_prompts_for(32, 256) == 128, "num_prompts: max(8, 4*C)")
    check(num_prompts_for(128, 1024) == 256, "num_prompts: output-token budget caps long outputs (>= 2*C)")
    wls = build_workloads("standard", 16384, quiet=True)
    labels = [w.label for w in wls]
    check(len(labels) == len(set(labels)), "no duplicate workloads (reference is a grid point)")
    mixed = next(w for w in wls if w.kind == "mixed")
    outs = output_lengths(mixed, 200, 42)
    check(min(outs) < mixed.output_len < max(outs) and abs(sum(outs) / 200 - mixed.output_len) < 0.15 * mixed.output_len,
          "mixed workload spreads output lengths around the mean")
    a = make_prompts(mixed, 10, list(range(100, 1000)), 7)
    b = make_prompts(mixed, 10, list(range(100, 1000)), 7)
    check(a.ids == b.ids and a.out_lens == b.out_lens, "prompts are deterministic (same ids for both engines)")

    print("unit: metrics + sampling")
    reqs = [dict(submit=0, first=0.1, token_times=[0.1, 0.2, 0.3], end=0.3, n_in=10, n_out=3)] * 4
    s = summarize(reqs, 1.0, {"ttft": 50, "tpot": 200})
    check(abs(s["median_ttft_ms"] - 100) < 1e-6 and abs(s["median_tpot_ms"] - 100) < 1e-6, "TTFT / TPOT")
    check(s["output_throughput"] == 12 and s["request_goodput"] == 0, "throughput and SLO goodput")
    p = sampling.Policy(2, 4, 0.05)
    quiet = [dict(output_throughput=100.0), dict(output_throughput=101.0)]
    noisy = [dict(output_throughput=100.0), dict(output_throughput=130.0)]
    check(not sampling.need_more(quiet, 2, p), "adaptive: quiet pair stops at min repeats")
    check(sampling.need_more(noisy, 2, p), "adaptive: noisy pair gets another repeat")
    check(not sampling.need_more(noisy + noisy, 4, p), "adaptive: stops at max repeats")
    check(sampling.need_more([dict(bench_rc=1.0)], 1, p), "adaptive: a failed repeat doesn't count")

    print("unit: profiling")
    tmp = Path(tempfile.mkdtemp())
    (tmp / "rank0.trace.json").write_text(json.dumps({"traceEvents": [
        {"ph": "X", "cat": "kernel", "name": "sm90_xmma_gemm_bf16bf16", "ts": 0, "dur": 600},
        {"ph": "X", "cat": "kernel", "name": "flash_fwd_splitkv_kernel", "ts": 600, "dur": 300},
        {"ph": "X", "cat": "kernel", "name": "ncclDevKernel_AllReduce", "ts": 900, "dur": 100},
    ]}))
    bd = profiling.breakdown(profiling.find_traces(tmp)[0], window_s=0.002)
    check(set(bd["classes"]) == {"gemm", "attention", "communication"}, "kernel classes")
    check(abs(bd["classes"]["gemm"]["pct"] - 60) < 1e-6 and abs(bd["gpu_busy_pct"] - 50) < 1e-6, "pct + GPU busy")

    print("unit: taxonomy covers all 11 categories")

    class Ctx:
        model, tp, max_model_len, gpu_mem, weights_gib = "Qwen/Qwen3-30B-A3B", 1, 16384, 0.9, 57.0
        spec_config = eagle_config = w4a16_model = None
    from llmbench.hardware import Hardware
    hw = Hardware(gpus=[dict(name="H100", total_gib=80, cc=9.0, smi_index=None)] * 2, unified=False,
                  mem_total_gib=512, source="fake")
    os.environ["LLMBENCH_VLLM_VERSION"] = "0.31"
    v, _ = vllm_catalog(Ctx, hw)
    pt, _ = pytorch_catalog()
    names = [c.name for c in v + pt if not c.skip]
    cats = {c.category for c in all_comparisons(names, 1)}
    missing = [c for c, _ in CATEGORIES if c not in cats]
    check(not missing, f"every category has comparisons on a 2-GPU MoE box (missing: {missing})")

    print("unit: card tailoring")
    from llmbench import tailor
    from llmbench.configs import Config
    os.environ["LLMBENCH_KV_BYTES_PER_TOKEN"] = "147456"          # Qwen3-4B: 36 layers x 8 KV heads x 128 x 2 x bf16

    class Small:
        model, tp, gpu_mem, weights_gib = "Qwen/Qwen3-4B", 1, 0.79, 7.45
    hw16 = Hardware(gpus=[dict(name="RTX 5070 Ti", total_gib=15.4, cc=12.0, smi_index=None)], unified=False,
                    mem_total_gib=64, source="fake")
    card = tailor.profile(Small, hw16)
    check(card.tier == "small" and card.tight and 20000 < card.kv_tokens_est < 27000, "16 GB card: small tier, ~23k KV tokens")
    spec = Config("spec_draft", "vllm", "component", [], "", args=["--speculative-config",
                  '{"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 4}'])
    plain = Config("vllm_default", "vllm", "default", [], "")
    tailor.apply_headroom([spec, plain], Small, card, quiet=True)
    frac = float(spec.args[spec.args.index("--gpu-memory-utilization") + 1])
    check(0.70 <= frac <= 0.73 and "--gpu-memory-utilization" not in plain.args,
          "draft config reserves only the post-profiling warmup (~1 GiB), default untouched")
    check(tailor.oom_kind("ValueError: No available memory for the cache blocks") == "starved"
          and tailor.oom_kind("torch.OutOfMemoryError: CUDA out of memory") == "device", "memory failures classified")
    sc = Config("x", "vllm", "component", [], "", args=["--gpu-memory-utilization", "0.62"])
    check(tailor.lower_gpu_mem(sc, 0.78, raise_=True) == 0.66, "cache-starved retry raises the fraction")
    check(not sampling.is_quiet([0.0], sampling.Policy(2, 4, 0.05)) and sampling.is_quiet([0.0, 0.01], sampling.Policy(2, 4, 0.05))
          and not sampling.is_quiet([0.0, 0.04], sampling.Policy(2, 4, 0.05)), "quiet config detected after 2 calm workloads")
    kept, dropped = tailor.fit_workloads(build_workloads("standard", 8192, quiet=True), card, 4.0, quiet=True,
                                         drop_grid=True)
    keep_all, none = tailor.fit_workloads(build_workloads("standard", 8192, quiet=True), card, 4.0, quiet=True)
    check(not none and any(w.concurrency == 128 for w in keep_all), "by default every grid shape stays in the plan")
    check(any(w.concurrency == 128 for w in dropped) and any(w.kind == "reference" for w in kept),
          "shapes far beyond KV capacity dropped, reference kept")
    kinds = {w.kind for w in kept}
    check({"prefix", "mixed"} <= kinds, "shared-prefix and mixed workloads are never dropped")
    pfx = next(w for w in kept if w.kind == "prefix")
    check(pfx.concurrency == 32, "shared prefix counted once in KV demand (fits at c=32)")
    mx = next(w for w in kept if w.kind == "mixed")
    check(mx.concurrency < 64 and tailor.kv_demand(mx) <= 4 * card.kv_tokens_est, "mixed concurrency scaled to fit")
    check(tailor.lower_gpu_mem(plain, 0.79) == 0.75 and tailor.lower_gpu_mem(plain, 0.79) == 0.71, "OOM retry steps down")
    del os.environ["LLMBENCH_KV_BYTES_PER_TOKEN"]

    print("unit: context fit for draft-model configs")
    log = "ValueError: To serve at least one request with the model's max seq len (8192), (2.0 GiB KV cache is needed"
    check(tailor.oom_kind(log) == "starved", "context-starved error recognized")
    sd = Config("spec_draft", "vllm", "component", [], "", args=["--speculative-config",
                '{"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 4}'])
    wl = build_workloads("standard", 8192, quiet=True)
    small = [w for w in wl if w.kind in ("reference", "text")]
    tailor.fit_context([sd, Config("vllm_default", "vllm", "default", [], "")], lambda c: small, 8192, card, quiet=True)
    check(sd.args[sd.args.index("--max-model-len") + 1] == "2048", "draft config's context fitted to its workloads (2048)")
    check(tailor.shrink_context(sd, 8192, tailor.context_needed(small)) is None, "context never shrinks below what workloads need")
    big = Config("x", "vllm", "component", [], "")
    check(tailor.shrink_context(big, 8192, 2048) == 4096 and tailor.shrink_context(big, 8192, 2048) == 2048
          and tailor.shrink_context(big, 8192, 2048) is None, "starved retry halves the context down to the floor")

    print("unit: sweep stops at the card's limit")
    from llmbench import sweep
    from llmbench.workloads import Workload
    ws = [Workload(f"c{c}", 1024, 256, c, 8) for c in (1, 2, 4, 8, 16, 32, 64, 128)]
    by = {w.name: w for w in ws}
    ok = lambda thr, pre=0: [dict(output_throughput=thr, preemptions=pre)]
    rows = {"c1": ok(90), "c2": ok(170), "c4": ok(320), "c8": ok(500), "c16": ok(800), "c32": ok(835, 21)}
    check(sweep.skip_reason(by["c64"], rows, by) and sweep.skip_reason(by["c128"], rows, by)
          and "saturated" in sweep.skip_reason(by["c64"], rows, by), "vLLM: flat throughput + preemptions -> skip larger")
    rows2 = {"c1": ok(90), "c2": ok(170), "c4": ok(320), "c8": ok(500), "c16": ok(800), "c32": ok(1400, 5)}
    check(sweep.skip_reason(by["c64"], rows2, by) is None, "still scaling -> keep going even with some preemption")
    cb = lambda thr: [dict(output_throughput=thr, preemptions=float("nan"))]      # transformers CB: no counter
    check("queue" in (sweep.skip_reason(by["c128"], {"c16": cb(560), "c32": cb(640), "c64": cb(654)}, by) or ""),
          "continuous batching without a preemption counter: flat throughput alone -> skip")
    check(sweep.skip_reason(by["c64"], {"c16": cb(400), "c32": cb(640)}, by) is None, "...but not while still scaling")
    tmpd = Path(tempfile.mkdtemp())
    (tmpd / "x__w__r0.row.json").write_text("")
    from llmbench.util import read_row
    check(read_row(tmpd / "x__w__r0.row.json") is None and (tmpd / "corrupt" / "x__w__r0.row.json").exists(),
          "empty row file (power loss) is set aside, not fatal")
    shutil.rmtree(tmpd)
    oom = [dict(bench_rc=1.0, bench_error="OOM: CUDA out of memory", oom_v=sweep.OOM_TAG)]
    rows3 = {"c1": ok(20), "c2": ok(38), "c4": ok(70), "c8": oom}
    check("c=8 ran out of memory" in (sweep.skip_reason(by["c16"], rows3, by) or ""), "PyTorch: OOM at c=8 -> skip c>=16")
    old_oom = [dict(bench_rc=1.0, bench_error="OOM: CUDA out of memory")]          # from before the leak fix
    check(sweep.skip_reason(by["c16"], {"c8": old_oom}, by) is None, "untrusted (pre-fix) OOMs don't stop the sweep")

    print("unit: retry repeats supersede suspect ones")
    import pandas as pd
    from llmbench.results import aggregate
    rd = Path(tempfile.mkdtemp()); (rd / "raw").mkdir()
    import json as _j
    for rep, thr, retry in ((0, 150.0, 0), (1, 600.0, 0), (2, 160.0, 0), (3, 434.0, 1), (4, 422.0, 1), (5, 436.0, 1)):
        (rd / "raw" / f"c__w__r{rep}.row.json").write_text(_j.dumps(dict(
            config="c", workload="w", repeat=rep, output_throughput=thr, bench_rc=0.0, retry=float(retry))))
    aggregate(rd)
    agg = pd.read_csv(rd / "results.csv").iloc[0]
    check(agg["output_throughput"] == 434.0 and agg["n_repeats"] == 3 and agg["superseded_repeats"] == 3
          and agg["cv_output_throughput"] < 0.02, "median/CV come from the 3 consistent retry repeats only")
    shutil.rmtree(rd)

    print("unit: anomaly walk")
    import pandas as pd
    run = Path(tempfile.mkdtemp())
    rows = []
    for cfg, thr in (("vllm_default", [1000, 2000, 1200, 3000]), ("enforce_eager", [500, 900, 1700, 2100])):
        for c, t in zip([1, 8, 32, 128], thr):
            rows.append(dict(config=cfg, workload=f"w{c}", workload_label=f"in1/out1/c{c}", kind="grid", input_len=1,
                             output_len=1, prefix_len=0, concurrency=c, status="ok", n_repeats=2, n_failed=0,
                             output_throughput=t, median_ttft_ms=10, median_e2el_ms=100, median_tpot_ms=1,
                             cv_output_throughput=0.2 if (cfg, c) == ("enforce_eager", 8) else 0.01, engine="vllm",
                             completed=8, num_prompts=8, preemptions=0))
    pd.DataFrame(rows).to_csv(run / "results.csv", index=False)
    (run / "summary.json").write_text(json.dumps({"ladders": {}, "server_facts": {}}))
    found = anomalies.detect(run)
    checks = {(f["config"], f["workload"], f["check"]) for f in found}
    check(("vllm_default", "w32", "dip") in checks and ("vllm_default", "w8", "dip") in checks, "dip flags both points")
    check(("enforce_eager", "w8", "noisy") in checks, "noisy repeats flagged")
    plan = anomalies.plan_retries(found, max_pairs=2)
    check(sum(len(w) for w in plan.get("vllm", {}).values()) == 2, "retry budget respected")
    shutil.rmtree(run)
    shutil.rmtree(tmp)


def e2e() -> None:
    print("e2e: fake vLLM + synthetic PyTorch, injected anomalies")
    out = Path(tempfile.mkdtemp(prefix="llmbench_e2e_"))
    env = dict(os.environ, PATH=f"{ROOT / 'tests' / 'fake_vllm'}{os.pathsep}{os.environ['PATH']}",
               LLMBENCH_ROOT=str(ROOT), LLMBENCH_FAKE_TORCH="1",
               LLMBENCH_FAKE_HW='{"name": "NVIDIA H100 80GB HBM3", "gib": 80, "n": 1, "cc": 9.0, "no_smi": true}',
               FAKE_VLLM_SLOW="--block-size", FAKE_VLLM_REJECT="--no-enable-chunked-prefill",
               LLMBENCH_FAKE_FAIL="pt_cb:mixed", PYTHONPATH=str(ROOT), FAKE_VLLM_STARVE_BELOW="0.90")
    env.pop("LLMBENCH_VLLM_VERSION", None)
    cmd = [sys.executable, "-m", "llmbench", "run", "--preset", "check", "--min-repeats", "2", "--max-repeats", "4",
           "--retry-rounds", "1", "--profile", "--out", str(out), "--port", "18099", "--settle-s", "0",
           "--startup-timeout", "30", "--no-plots", "--configs",
           "vllm_default,vllm_minimal,vllm_lad_prefix,vllm_optimized,block_size_32,no_prefix_cache,enforce_eager,"
           "vllm_static_waves,spec_ngram,spec_draft,pt_eager,pt_static,pt_cb,pt_compile,pt_cb_noprefix"]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    log = r.stdout + r.stderr
    (out / "e2e.log").write_text(log)
    check(r.returncode == 0, f"run exits 0 (log: {out / 'e2e.log'})")
    runs = [d for d in out.iterdir() if d.is_dir()]
    check(len(runs) == 1, "one run directory")
    if not runs:
        return
    rd = runs[0]
    import pandas as pd
    for f in ("results_long.csv", "optimizations.csv", "category_summary.csv", "headline.csv", "ladder.csv",
              "kernel_breakdown.csv", "anomalies.csv", "report.md", "dashboard.html"):
        check((rd / f).exists() and (rd / f).stat().st_size > 50, f"{f} written")
    check("[rebuild] vllm_minimal" in log, "composite rebuilt after a rejected flag")
    sd = s_ = json.loads((rd / "summary.json").read_text())["server_facts"].get("spec_draft", {})
    check(sd.get("status") == "ok" and sd.get("oom_retries"), "cache-starved spec config recovered by raising its fraction")
    s = json.loads((rd / "summary.json").read_text())
    vm = next(c for c in s["configs"] if c["name"] == "vllm_minimal")
    check("mnbt_max_model_len" in vm["components"] and "rebuilt" in vm["note"], "vllm_minimal uses the fallback component")
    an = pd.read_csv(rd / "anomalies.csv")
    got = set(zip(an["config"], an["check"]))
    check(("pt_cb", "partial") in got, "failed repeat flagged as partial")
    check(an[(an["config"] == "pt_cb")]["outcome"].eq("resolved").all(), "pt_cb retry resolved it")
    res = pd.read_csv(rd / "results.csv")
    # adaptive rule: a workload whose first two repeats differ by > 5% gets more (the stub's noise is
    # random, so check the rule itself rather than assume a particular draw was noisy)
    import statistics
    allr = pd.read_csv(rd / "results_all.csv")
    ok_rule = True
    for _, g in allr[(allr["config"] == "block_size_32") & (allr["status"] == "ok")].groupby("workload"):
        thr = list(g.sort_values("repeat")["output_throughput"])
        if len(thr) >= 2:
            spread = statistics.stdev(thr[:2]) / statistics.fmean(thr[:2])
            ok_rule &= (len(thr) > 2) == (spread > 0.05) or len(thr) >= 4
    check(ok_rule, "adaptive repeats follow the CV rule on a noisy config")
    check((rd / "raw" / "retired").exists(), "failed repeats retired, not deleted")
    long = pd.read_csv(rd / "results_long.csv")
    check({"engine", "tier", "categories", "metric", "value", "unit"} <= set(long.columns), "long CSV has labels")
    cs = pd.read_csv(rd / "category_summary.csv")
    for cat in ("prefix_caching", "continuous_batching", "speculative_decoding", "cuda_graphs", "paged_attention"):
        check(cat in set(cs["category"]), f"category_summary has {cat}")
    sp = cs[cs["comparison"] == "no_prefix_cache -> vllm_default"]
    check(len(sp) == 1 and sp["ttft_x_focus"].iloc[0] > 1.2, "prefix caching cuts TTFT on the shared-prefix workload")
    kb = pd.read_csv(rd / "kernel_breakdown.csv")
    check(set(kb["kernel_class"]) >= {"gemm", "attention"}, "kernel breakdown classified")
    dash = (rd / "dashboard.html").read_text() if (rd / "dashboard.html").exists() else ""
    check("/*__DATA__*/null" not in dash and '"views"' in dash and '"points"' in dash, "dashboard has its data embedded")
    # rebuild from raw files
    r2 = subprocess.run([sys.executable, "-m", "llmbench", "report", rd.name, "--out", str(out), "--no-plots"],
                        cwd=ROOT, env=env, capture_output=True, text=True)
    check(r2.returncode == 0, "report rebuilds from raw/")
    check((rd / "engine_matrix.csv").exists(), "engine_matrix.csv written")
    check('"matrix"' in dash, "dashboard carries the 2x2 matrix")
    r3 = subprocess.run([sys.executable, "-m", "llmbench", "report", rd.name, "--out", str(out)],
                        cwd=ROOT, env=env, capture_output=True, text=True)
    pngs = {p.name for p in (rd / "plots").glob("*.png")} if (rd / "plots").exists() else set()
    check(r3.returncode == 0 and "skipped" not in r3.stdout and "frontier_vllm.png" in pngs
          and "frontier_vllm_first_vs_last.png" in pngs and "stages_vllm.png" in pngs,
          f"static images written (got {sorted(pngs)}; {r3.stdout[-300:]})")
    if not FAILS:
        shutil.rmtree(out)


if __name__ == "__main__":
    unit()
    if sys.argv[1:] != ["unit"]:
        e2e()
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: ' + '; '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
