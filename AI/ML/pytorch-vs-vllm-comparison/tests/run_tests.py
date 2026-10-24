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
    check(frac < 0.70 and "--gpu-memory-utilization" not in plain.args, "draft-model config gets a memory reservation, default doesn't")
    kept, dropped = tailor.fit_workloads(build_workloads("standard", 8192, quiet=True), card, 4.0, quiet=True)
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
               LLMBENCH_FAKE_FAIL="pt_cb:mixed", PYTHONPATH=str(ROOT))
    env.pop("LLMBENCH_VLLM_VERSION", None)
    cmd = [sys.executable, "-m", "llmbench", "run", "--preset", "check", "--min-repeats", "2", "--max-repeats", "4",
           "--retry-rounds", "1", "--profile", "--out", str(out), "--port", "18099", "--settle-s", "0",
           "--startup-timeout", "30", "--no-plots", "--configs",
           "vllm_default,vllm_minimal,vllm_lad_prefix,vllm_optimized,block_size_32,no_prefix_cache,enforce_eager,"
           "vllm_static_waves,spec_ngram,pt_eager,pt_static,pt_cb,pt_compile,pt_cb_noprefix"]
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
              "kernel_breakdown.csv", "anomalies.csv", "report.md"):
        check((rd / f).exists() and (rd / f).stat().st_size > 50, f"{f} written")
    check("[rebuild] vllm_minimal" in log, "composite rebuilt after a rejected flag")
    s = json.loads((rd / "summary.json").read_text())
    vm = next(c for c in s["configs"] if c["name"] == "vllm_minimal")
    check("mnbt_max_model_len" in vm["components"] and "rebuilt" in vm["note"], "vllm_minimal uses the fallback component")
    an = pd.read_csv(rd / "anomalies.csv")
    got = set(zip(an["config"], an["check"]))
    check(("pt_cb", "partial") in got, "failed repeat flagged as partial")
    check(an[(an["config"] == "pt_cb")]["outcome"].eq("resolved").all(), "pt_cb retry resolved it")
    res = pd.read_csv(rd / "results.csv")
    bs = res[(res["config"] == "block_size_32") & (res["workload"] == "reference")]
    check(len(bs) == 1 and bs["n_repeats"].iloc[0] > 2, "noisy config got extra adaptive repeats")
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
    # rebuild from raw files
    r2 = subprocess.run([sys.executable, "-m", "llmbench", "report", rd.name, "--out", str(out), "--no-plots"],
                        cwd=ROOT, env=env, capture_output=True, text=True)
    check(r2.returncode == 0, "report rebuilds from raw/")
    if not FAILS:
        shutil.rmtree(out)


if __name__ == "__main__":
    unit()
    if sys.argv[1:] != ["unit"]:
        e2e()
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: ' + '; '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
