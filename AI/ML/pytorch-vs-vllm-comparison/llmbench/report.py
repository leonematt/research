"""report.md (+ plots/) from the CSVs. Everything here reads files written by results.py, so
`python -m llmbench report <run_id>` can rebuild it at any time."""
from __future__ import annotations

import json
from pathlib import Path

from .taxonomy import CATEGORY_LABEL, CATEGORY_ORDER
from .util import is_num, md_table


def _read(run_dir: Path, name: str):
    import pandas as pd
    p = run_dir / name
    if not p.exists() or p.stat().st_size < 2:
        return None
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return None


def write_report(run_dir: Path, make_plots: bool = True) -> None:
    import pandas as pd

    summary = json.loads((run_dir / "summary.json").read_text())
    res = _read(run_dir, "results.csv")
    if res is None:
        (run_dir / "report.md").write_text(f"# `{run_dir.name}`\n\nNo results yet: every config failed before "
                                           "producing a row. See server_logs/.\n")
        print(f"[report] no results yet; wrote stub {run_dir / 'report.md'}")
        return
    env = summary.get("environment", {})
    hwd = env.get("hardware", {})
    ref = summary.get("reference_label")
    out = [f"# PyTorch vs vLLM — `{run_dir.name}`\n",
           f"Model `{summary['model']}` · {hwd.get('describe', '?')} · TP={summary.get('tp', 1)} · "
           f"dtype={summary.get('dtype', '?')} · max_model_len={summary.get('max_model_len')}\n",
           f"vLLM {env.get('vllm', '?')} · torch {env.get('torch', '?')} · transformers {env.get('transformers', '?')} "
           f"· CUDA {env.get('cuda', '?')} · driver {env.get('driver', '?')}\n",
           f"grid `{summary.get('grid')}` · reference workload `{ref}` · repeats {summary.get('min_repeats')}.."
           f"{summary.get('max_repeats')} (target CV {summary.get('target_cv', 0):.0%})"
           + (f" · rate workloads at fractions of {summary['rate_base_rps']:.2f} req/s" if summary.get("rate_base_rps") else "")
           + "\n"]
    if summary.get("autosize_why"):
        out.append("Auto-sizing: " + "; ".join(summary["autosize_why"]) + "\n")
    out.append("Graph-ready CSVs in this folder: `results_long.csv` (every metric, tidy), `optimizations.csv` "
               "(every off→on comparison), `category_summary.csv` (one line per optimization), `headline.csv` "
               "(baseline vs optimized anchors), `ladder.csv` (cumulative build-up), `kernel_breakdown.csv` "
               "(with --profile), `anomalies.csv`.\n")

    # 1. headline
    head = _read(run_dir, "headline.csv")
    out.append("## 1. Headline: baselines vs fully optimized\n")
    if head is not None and len(head):
        h = head[head["workload_label"] == ref]
        if not len(h):
            h = head[head["workload"] == head["workload"].iloc[0]]
        t = h.set_index("anchor")[["config", "output_throughput", "median_ttft_ms", "median_tpot_ms",
                                   "median_e2el_ms", "x_vs_pt_eager", "x_vs_vllm_default"]]
        out.append(f"At `{h['workload_label'].iloc[0]}`. `pytorch_best` / `vllm_best` = the fastest config of "
                   "that engine at this workload.\n")
        out.append(md_table(t, "{:,.2f}"))
        p = head[head["anchor"].isin(["pt_eager", "pt_optimized", "pt_serve_optimized", "vllm_minimal",
                                      "vllm_default", "vllm_optimized"])]
        if len(p):
            pv = p.pivot_table(index="workload_label", columns="anchor", values="output_throughput", aggfunc="first")
            cols = [c for c in ["pt_eager", "pt_optimized", "pt_serve_optimized", "vllm_minimal", "vllm_default",
                                "vllm_optimized"] if c in pv.columns]
            out.append("### Output throughput (tok/s) per workload\n")
            out.append(md_table(pv[cols], "{:,.0f}"))
    else:
        out.append("_no anchors measured_\n")

    # 1b. the 2x2: PyTorch vs vLLM, not optimized vs optimized
    em = _read(run_dir, "engine_matrix.csv")
    if em is not None and len(em):
        out += engine_matrix_section(em, summary)

    # 2. breakdown by category
    cs = _read(run_dir, "category_summary.csv")
    out.append("## 2. Optimization breakdown by category\n")
    out.append("Speedup = on/off for throughput, off/on for latency, so **> 1.00 means the optimization helps**. "
               "`focus` = the workloads where that optimization should matter (e.g. the shared-prefix workload "
               "for prefix caching), else the reference workload. `ablation` = vLLM default with that one "
               "thing switched off; `ladder` = one cumulative build-up step; `variant` = an alternative "
               "implementation; `engine` = end-to-end anchors.\n")
    if cs is not None and len(cs):
        for cat in CATEGORY_ORDER:
            sub = cs[cs["category"] == cat]
            if not len(sub):
                continue
            t = sub.set_index("comparison")[["engine", "method", "what", "throughput_x_focus", "ttft_x_focus",
                                             "tpot_x_focus", "throughput_x_geomean_all", "n_workloads"]].copy()
            t["what"] = t["what"].str.slice(0, 60)
            out.append(f"### {CATEGORY_LABEL.get(cat, cat)}\n")
            out.append(md_table(t, "{:,.2f}"))
            notes = sub[sub["confounded_with"].fillna("") != ""]
            for _, n in notes.iterrows():
                out.append(f"- `{n['comparison']}` is confounded with "
                           f"{CATEGORY_LABEL.get(n['confounded_with'], n['confounded_with'])}"
                           + (f": {n['note']}" if isinstance(n["note"], str) and n["note"] else "") + "\n")
    else:
        out.append("_no comparable config pairs measured_\n")

    # 3. ladders
    lad = _read(run_dir, "ladder.csv")
    if lad is not None and len(lad):
        out.append("## 3. Cumulative ladders (waterfall)\n")
        out.append("Each step adds one feature to the step before. `share` = that step's part of the total "
                   "speedup (log-scale, so the shares add up to 100%).\n")
        for lname, g in lad.groupby("ladder", sort=False):
            s = g[(g["metric"] == "output_throughput") & (g["workload_label"] == ref)]
            if not len(s):
                continue
            t = s.set_index("config")[["added", "value", "step_x", "cumulative_x", "share_of_total_gain"]].copy()
            t["share_of_total_gain"] = t["share_of_total_gain"] * 100
            t = t.rename(columns={"value": "out tok/s", "share_of_total_gain": "share %"})
            out.append(f"### {lname} at `{ref}`\n")
            out.append(md_table(t, "{:,.2f}"))

    # 4. anomalies
    an = _read(run_dir, "anomalies.csv")
    out.append("## 4. Anomalies and retries\n")
    if an is not None and len(an):
        last = an.sort_values("round").groupby(["config", "workload", "check"]).tail(1)
        cnt = last["outcome"].value_counts().to_dict()
        out.append(", ".join(f"{v} {k}" for k, v in cnt.items()) + ". `resolved` = the retry disagreed with the "
                   "first measurement (it was the outlier); `persisting` = the retry reproduced it (likely real).\n")
        t = last.set_index("config")[["workload_label", "check", "outcome", "detail"]].copy()
        t["detail"] = t["detail"].astype(str).str.slice(0, 80)
        out.append(md_table(t.head(60)))
    else:
        out.append("_none found_\n")

    # 5. configs and engine facts
    out.append("## 5. Configurations and engine facts\n")
    rows = []
    for c in summary["configs"]:
        f = summary.get("server_facts", {}).get(c["name"], {})
        kv = f.get("kv_cache_tokens")
        cold = f.get("cold_start") or {}
        comp = f.get("compile_s")
        comp_txt = f"{cold['compile_s']:.1f} cold / {comp or 0:.1f} warm" if cold.get("compile_s") else comp
        resolved = f.get("resolved") or " ".join(
            f"{lbl}={f[k]}" for k, lbl in (("cfg_compile_mode", "compile"), ("cfg_cudagraph_mode", "graphs"),
                                          ("cfg_kv_cache_dtype", "kv"), ("cfg_prefix_caching", "apc"),
                                          ("cfg_chunked_prefill", "chunked"), ("cfg_quantization", "quant"),
                                          ("cfg_speculative", "spec"), ("attention_backend", "attn"))
            if f.get(k) is not None)
        rows.append(dict(config=c["name"], engine=c.get("engine"), tier=c.get("tier"),
                         categories=", ".join(CATEGORY_LABEL.get(x, x) for x in c.get("categories", [])),
                         knob=c.get("knob", "")[:70], status=f.get("status", "skipped"), resolved=resolved[:80] or None,
                         gpus=f.get("gpu_count"), startup_s=f.get("startup_s"), compile_s=comp_txt,
                         kv_tokens=f"{int(kv):,}" if is_num(kv) else None, peak_gib=f.get("peak_mem_gib"),
                         note=(f.get("error") or c.get("note") or "")[:80]))
    out.append(md_table(pd.DataFrame(rows).set_index("config")))
    if summary.get("skipped"):
        out.append("Skipped here: " + "; ".join(f"`{k}` ({v})" for k, v in summary["skipped"].items()) + "\n")

    # 6. pivots
    cfg_order = [c["name"] for c in summary["configs"]]
    wl_order = [w["label"] for w in summary["workloads"]]
    ok = res[res["status"] == "ok"]
    out.append("## 6. Full grid (rows = config, columns = workload; medians of good repeats)\n")
    for col, title, ff in [("output_throughput", "Output throughput (tok/s)", "{:,.0f}"),
                           ("median_ttft_ms", "TTFT p50 (ms)", "{:,.1f}"),
                           ("median_tpot_ms", "TPOT p50 (ms)", "{:,.2f}"),
                           ("p99_itl_ms", "ITL p99 (ms)", "{:,.2f}")]:
        if col not in ok.columns:
            continue
        p = ok.pivot_table(index="config", columns="workload_label", values=col, aggfunc="first")
        p = p.reindex([c for c in cfg_order if c in p.index])[[w for w in wl_order if w in p.columns]]
        if p.notna().any().any():
            out.append(f"### {title}\n")
            out.append(md_table(p, ff))

    # 7. kernels
    kb = _read(run_dir, "kernel_breakdown.csv")
    if kb is not None and len(kb):
        out.append("## 7. Where the GPU time goes (profiled pass at the reference shape)\n")
        p = kb.pivot_table(index="config", columns="kernel_class", values="pct_of_gpu_time", aggfunc="first").fillna(0)
        busy = kb.drop_duplicates("config").set_index("config")["gpu_busy_pct"]
        p.insert(0, "GPU busy %", busy)
        out.append(md_table(p, "{:,.1f}"))

    # 8. failures
    adf = _read(run_dir, "results_all.csv")
    fails = [(n, f) for n, f in summary.get("server_facts", {}).items() if f.get("status") != "ok"]
    bench_fail = adf[~adf["status"].isin(["ok", "skipped"])] if adf is not None and "status" in adf else []
    if fails or len(bench_fail):
        out.append("## 8. Failures\n")
        for n, f in fails:
            out.append(f"- **{n}**: {f.get('error', f.get('status'))} — see `server_logs/{n}.log`")
        for _, rw in (bench_fail.iterrows() if len(bench_fail) else []):
            out.append(f"- `{rw['config']}` / `{rw['workload_label']}` r{int(rw.get('repeat') or 0)}: "
                       f"{str(rw.get('bench_error', ''))[:160]}")
        out.append("")

    (run_dir / "report.md").write_text("\n".join(out))
    print(f"[report] wrote {run_dir / 'report.md'}")
    try:
        from .dashboard import write_dashboard
        d = write_dashboard(run_dir)
        if d:
            print(f"[dashboard] wrote {d}  (open it in a browser)")
    except Exception as e:  # noqa: BLE001
        print(f"[dashboard] skipped: {type(e).__name__}: {e}")
    if make_plots:
        try:
            plots(run_dir, summary)
        except Exception as e:  # noqa: BLE001
            print(f"[plots] skipped: {type(e).__name__}: {e}")


MATRIX_SHOW = [("output_throughput_per_gpu", "Output tok/s per GPU", "{:,.0f}"),
               ("median_ttft_ms", "TTFT p50 (ms)", "{:,.0f}"),
               ("median_tpot_ms", "TPOT p50 (ms)", "{:,.1f}"),
               ("median_e2el_ms", "E2E p50 (ms)", "{:,.0f}"),
               ("peak_mem_gib", "Peak memory (GiB)", "{:,.1f}")]


def _cell(r, side, ff):
    v = r.get(side)
    if is_num(v):
        return ff.format(float(v))
    st = str(r.get(side + "_status") or "")
    return "OOM" if st == "oom" else "saturated" if st == "skipped" else "–"


def _x(v):
    return f"×{float(v):.2f}" if is_num(v) else "–"


def engine_matrix_section(em, summary: dict) -> list[str]:
    """2x2 tables (engine x optimization) at a few representative operating points."""
    rin, rout, rc = summary["reference"]
    pts = []
    grid = em[em["kind"].isin(["grid", "reference"]) & (em["input_len"] == rin) & (em["output_len"] == rout)]
    for c in sorted({1, 8, int(rc)}):
        if (grid["concurrency"] == c).any():
            pts.append(grid[grid["concurrency"] == c])
    text1 = em[(em["kind"] == "text") & (em["concurrency"] == 1)]
    if len(text1):
        pts.append(text1)
    out = ["## 1b. PyTorch vs vLLM: not optimized vs optimized\n",
           "Baseline = `pt_eager` / `vllm_minimal`. Optimized = the better of each engine's optimized configs at that "
           "point (named under the table): `pt_optimized` (static batching) or `pt_serve_optimized` (continuous "
           "batching); `vllm_optimized` or `vllm_optimized_spec`. Gains are oriented so ×>1 is better (for "
           "latencies, how many times lower). Every shape and concurrency is in `engine_matrix.csv`.\n"]
    for sub in pts:
        sub = sub.set_index("metric")
        lab = sub["workload_label"].iloc[0]
        lines = ["| metric | PyTorch baseline | PyTorch optimized | vLLM baseline | vLLM optimized | "
                 "PyTorch gain | vLLM gain | vLLM ÷ PyTorch (optimized) |", "|---|---|---|---|---|---|---|---|"]
        for m, label, ff in MATRIX_SHOW:
            if m not in sub.index:
                continue
            r = sub.loc[m]
            lines.append(f"| {label} | {_cell(r, 'pytorch_baseline', ff)} | {_cell(r, 'pytorch_optimized', ff)} | "
                         f"{_cell(r, 'vllm_baseline', ff)} | {_cell(r, 'vllm_optimized', ff)} | "
                         f"{_x(r.get('pytorch_optimization_x'))} | {_x(r.get('vllm_optimization_x'))} | "
                         f"{_x(r.get('vllm_vs_pytorch_optimized_x'))} |")
        r0 = sub.iloc[0]
        who = f"PyTorch optimized = `{r0.get('pytorch_optimized_config')}`, vLLM optimized = `{r0.get('vllm_optimized_config')}`"
        out += [f"### `{lab}`\n", "\n".join(lines) + "\n", who + "\n"]
    return out


def plots(run_dir: Path, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = run_dir / "plots"
    pdir.mkdir(exist_ok=True)
    ref = summary.get("reference_label")
    n = 0
    cs = _read(run_dir, "category_summary.csv")
    if cs is not None and len(cs):
        d = cs[cs["method"].isin(["ablation", "ladder"]) & cs["throughput_x_focus"].notna()].copy()
        if len(d):
            d["label"] = d["category_label"] + " · " + d["engine"] + " · " + d["what"].str.slice(0, 40)
            d = d.iloc[::-1]
            fig, ax = plt.subplots(figsize=(10, max(3, 0.32 * len(d))))
            colors = ["#2a78d6" if x >= 1 else "#d03b3b" for x in d["throughput_x_focus"]]
            ax.barh(d["label"], d["throughput_x_focus"], color=colors)
            ax.axvline(1.0, color="#333", lw=0.8)
            ax.set_xscale("log")
            ax.set_xlabel("throughput speedup at focus workload (log, >1 = helps)")
            ax.set_title("Optimization breakdown")
            ax.tick_params(axis="y", labelsize=7)
            fig.tight_layout()
            fig.savefig(pdir / "optimization_breakdown.png", dpi=160)
            plt.close(fig)
            n += 1
    from .figures import write_figures
    try:
        n += write_figures(run_dir, summary)
    except Exception as e:  # noqa: BLE001
        print(f"[plots] frontier figures skipped: {type(e).__name__}: {e}")
    kb = _read(run_dir, "kernel_breakdown.csv")
    if kb is not None and len(kb):
        p = kb.pivot_table(index="config", columns="kernel_class", values="pct_of_gpu_time", aggfunc="first").fillna(0)
        ax = p.plot(kind="barh", stacked=True, figsize=(10, max(3, 0.35 * len(p))), colormap="tab20")
        ax.set_xlabel("% of GPU kernel time")
        ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(pdir / "kernel_breakdown.png", dpi=130)
        plt.close("all")
        n += 1
    print(f"[plots] wrote {n} plot(s) to {pdir}")
