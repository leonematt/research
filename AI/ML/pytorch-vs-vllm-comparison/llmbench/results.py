"""Aggregation and every CSV.

On disk the source of truth is raw/<config>__<workload>__r<repeat>.row.json: one file per
measured run, written by the vLLM runner and the PyTorch worker alike. Everything below is
rebuilt from those files, so a run can be resumed, retried or extended at any time.

CSV outputs (results/<run_id>/)
-------------------------------
results_all.csv      one row per measured run (every repeat, including failed ones)
results.csv          one row per (config, workload): median of the GOOD repeats, n_repeats, cv
results_long.csv     tidy/long: one row per (config, workload, metric) with every label you'd
                     filter or color by. The file to load into a plotting tool.
optimizations.csv    one row per (comparison, workload, metric): off/on values, delta %, speedup
                     (>1 = the optimization helps), noise and a significance flag
category_summary.csv one row per comparison: speedup at its focus workload + geomean over all
headline.csv         the anchors (PyTorch eager/optimized, vLLM minimal/default/optimized) per workload
ladder.csv           cumulative build-up steps per engine with each step's share of the total gain
kernel_breakdown.csv GPU time by kernel class per config (only with --profile)
anomalies.csv        what the anomaly walk found and what the retry did about it
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from .taxonomy import CATEGORY_LABEL, CATEGORY_ORDER, COMPARE_METRICS, METRICS, all_comparisons
from .util import is_num

LONG_METRICS = list(METRICS)
# columns that identify a (config, workload) pair; everything else in a row is a measurement
ID_COLS = ["config", "engine", "tier", "categories", "knob", "model", "workload", "workload_label", "kind",
           "dataset", "input_len", "output_len", "prefix_len", "concurrency", "request_rate", "num_prompts",
           "gpu_count", "client"]


def base_row(cfg, wl, model: str, rep: int, gpu_count: int) -> dict:
    return dict(config=cfg.name, engine=cfg.engine, tier=cfg.tier, categories=";".join(cfg.categories),
                knob=cfg.knob, model=model, workload=wl.name, workload_label=wl.label, kind=wl.kind,
                dataset=wl.dataset, input_len=wl.input_len, output_len=wl.output_len, prefix_len=wl.prefix_len,
                concurrency=wl.concurrency, request_rate=wl.request_rate, num_prompts=wl.num_prompts,
                repeat=rep, gpu_count=gpu_count)


def write_row(raw_dir: Path, row: dict) -> None:
    name = f"{row['config']}__{row['workload']}__r{int(row.get('repeat') or 0)}.row.json"
    tmp = raw_dir / (name + ".tmp")
    tmp.write_text(json.dumps(row, default=float))
    tmp.replace(raw_dir / name)


def load_rows(run_dir: Path) -> list[dict]:
    rows = []
    for f in sorted((run_dir / "raw").glob("*.row.json")):
        try:
            rows.append(json.loads(f.read_text()))
        except Exception:  # noqa: BLE001
            pass
    return rows


def _status(r: dict) -> str:
    if not r.get("bench_rc") and is_num(r.get("output_throughput")):
        return "ok"
    err = str(r.get("bench_error") or "")
    return "oom" if err.startswith("OOM") else "failed"


def aggregate(run_dir: Path) -> None:
    """results_all.csv (every run) and results.csv (median of the good repeats per pair)."""
    import pandas as pd

    rows = load_rows(run_dir)
    if not rows:
        return
    for r in rows:
        r["status"] = _status(r)
        g = r.get("gpu_count") or 1
        if is_num(r.get("output_throughput")):
            r["output_throughput_per_gpu"] = r["output_throughput"] / g
    all_df = pd.DataFrame(rows)
    all_df.to_csv(run_dir / "results_all.csv", index=False)

    key = ["config", "workload"]
    out = []
    for (_cfg, _wl), g in all_df.groupby(key, sort=False):
        ok = g[g["status"] == "ok"]
        src = ok if len(ok) else g
        rec = {}
        for c in g.columns:
            if c in key:
                continue
            col = src[c]
            if pd.api.types.is_numeric_dtype(col) and c not in ("repeat",):
                rec[c] = col.median() if col.notna().any() else float("nan")
            else:
                nn = col.dropna()
                rec[c] = nn.iloc[-1] if len(nn) else None
        rec["config"], rec["workload"] = _cfg, _wl
        rec["n_repeats"] = int(len(ok))
        rec["n_failed"] = int(len(g) - len(ok))
        rec["status"] = "ok" if len(ok) else src["status"].iloc[-1]
        thr = ok["output_throughput"].dropna() if "output_throughput" in ok else []
        rec["cv_output_throughput"] = float(thr.std() / thr.mean()) if len(thr) >= 2 and thr.mean() else float("nan")
        rec["retried"] = int(bool(g.get("retry", pd.Series(dtype=float)).fillna(0).astype(float).sum()))
        out.append(rec)
    agg = pd.DataFrame(out)
    front = [c for c in ["config", "workload", "status", "n_repeats", "n_failed", "cv_output_throughput"] if c in agg]
    agg = agg[front + [c for c in agg.columns if c not in front]]
    agg.to_csv(run_dir / "results.csv", index=False)


# ----------------------------------------------------------------------------
# derived CSVs
# ----------------------------------------------------------------------------

def _summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


def _run_labels(summary: dict) -> dict:
    env = summary.get("environment", {})
    hwd = env.get("hardware", {})
    gpus = hwd.get("gpus") or [{}]
    return dict(run_id=summary.get("run_id"), hardware=hwd.get("describe"), gpu_name=gpus[0].get("name"),
                gpus_visible=len(hwd.get("gpus") or []), vllm_version=env.get("vllm"), torch_version=env.get("torch"),
                base_model=summary.get("model"), base_tp=summary.get("tp"), grid=summary.get("grid"))


def write_long(run_dir: Path, df, summary: dict) -> None:
    import pandas as pd

    labels = _run_labels(summary)
    facts = summary.get("server_facts", {})
    id_cols = [c for c in ID_COLS if c in df.columns] + [c for c in ("status", "n_repeats", "cv_output_throughput",
                                                                    "retried") if c in df.columns]
    metrics = [m for m in LONG_METRICS if m in df.columns]
    long = df.melt(id_vars=id_cols, value_vars=metrics, var_name="metric", value_name="value")
    long = long[long["value"].notna()]
    long["metric_label"] = long["metric"].map(lambda m: METRICS[m][0])
    long["unit"] = long["metric"].map(lambda m: METRICS[m][2])
    long["higher_is_better"] = long["metric"].map(lambda m: METRICS[m][1])
    # engine-level facts that are per config, not per workload
    extra = []
    for name, f in facts.items():
        for k, unit in (("kv_cache_tokens", "tokens"), ("startup_s", "s"), ("compile_s", "s"),
                        ("cudagraph_capture_s", "s"), ("gpu_busy_pct", "%"), ("weights_gib", "GiB")):
            if is_num(f.get(k)):
                extra.append(dict(config=name, workload="(server)", metric=k, value=float(f[k]), unit=unit,
                                  metric_label=k.replace("_", " "), higher_is_better=k in ("kv_cache_tokens",)))
    if extra:
        meta = df.drop_duplicates("config").set_index("config")
        ex = pd.DataFrame(extra)
        for c in ("engine", "tier", "categories", "knob", "model", "gpu_count"):
            if c in meta.columns:
                ex[c] = ex["config"].map(meta[c])
        long = pd.concat([long, ex], ignore_index=True)
    for k, v in reversed(list(labels.items())):
        long.insert(0, k, v)
    long["category_labels"] = long["categories"].fillna("").map(
        lambda s: ";".join(CATEGORY_LABEL.get(c, c) for c in str(s).split(";") if c))
    long.to_csv(run_dir / "results_long.csv", index=False)


def _better(metric: str) -> bool:
    return METRICS.get(metric, ("", True, ""))[1]


def _speedup(metric: str, off: float, on: float) -> float:
    if not (is_num(off) and is_num(on)) or off <= 0 or on <= 0:
        return float("nan")
    return on / off if _better(metric) else off / on


def _noise_floor(*cvs) -> float:
    """Smallest change we call real: 2x the measured run-to-run spread, at least 3%
    (5% when there's only one repeat and no spread to measure)."""
    c = [x for x in cvs if is_num(x)]
    return max(0.03, 2 * max(c)) if c else 0.05


def _ladder_comparisons(ladders: dict, names: set[str], known: set[tuple[str, str]] = frozenset()) -> list[dict]:
    out = []
    seen = set(known)
    for lname, steps in ladders.items():
        present = [s for s in steps if s["config"] in names]
        for prev, cur in zip(present, present[1:]):
            if (prev["config"], cur["config"]) in seen:      # already an ablation / another ladder
                continue
            seen.add((prev["config"], cur["config"]))
            out.append(dict(category=cur["category"] or "end_to_end", engine=lname.split("_")[0], off=prev["config"],
                            on=cur["config"], what=cur["added"], method="ladder", focus="", confounded_with="",
                            note=f"ladder {lname}"))
    return out


def write_optimizations(run_dir: Path, df, summary: dict) -> None:
    import pandas as pd

    labels = _run_labels(summary)
    ok = df[df["status"] == "ok"].set_index(["config", "workload"])
    names = sorted(set(df["config"]))
    comps = [c.__dict__ for c in all_comparisons(names, int(summary.get("tp") or 1))]
    comps += _ladder_comparisons(summary.get("ladders", {}), set(names), {(c["off"], c["on"]) for c in comps})
    rows = []
    for cmp_ in comps:
        try:
            a = ok.xs(cmp_["off"], level="config")
            b = ok.xs(cmp_["on"], level="config")
        except KeyError:
            continue
        for wl in [w for w in b.index if w in a.index]:
            ra, rb = a.loc[wl], b.loc[wl]
            noise = _noise_floor(ra.get("cv_output_throughput"), rb.get("cv_output_throughput"))
            for m in COMPARE_METRICS:
                if m not in ok.columns:
                    continue
                vo, vn = ra.get(m), rb.get(m)
                if not (is_num(vo) and is_num(vn)):
                    continue
                sp = _speedup(m, vo, vn)
                rows.append(dict(**labels, category=cmp_["category"],
                                 category_label=CATEGORY_LABEL.get(cmp_["category"], cmp_["category"]),
                                 engine=cmp_["engine"], method=cmp_["method"], comparison=f"{cmp_['off']} -> {cmp_['on']}",
                                 what=cmp_["what"], config_off=cmp_["off"], config_on=cmp_["on"], workload=wl,
                                 workload_label=rb.get("workload_label"), kind=rb.get("kind"),
                                 input_len=rb.get("input_len"), output_len=rb.get("output_len"),
                                 concurrency=rb.get("concurrency"), request_rate=rb.get("request_rate"),
                                 is_focus=bool(cmp_["focus"]) and rb.get("kind") == cmp_["focus"],
                                 metric=m, metric_label=METRICS[m][0], unit=METRICS[m][2],
                                 higher_is_better=_better(m), value_off=vo, value_on=vn,
                                 delta_pct=(vn - vo) / vo * 100 if vo else float("nan"),
                                 speedup_x=sp, improves=bool(sp > 1) if is_num(sp) else None,
                                 noise_floor_pct=noise * 100,
                                 significant=bool(is_num(sp) and abs(math.log(sp)) > math.log(1 + noise)),
                                 confounded_with=cmp_["confounded_with"], note=cmp_["note"]))
    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "optimizations.csv", index=False)
    write_category_summary(run_dir, out, summary)


def _geomean(xs) -> float:
    xs = [x for x in xs if is_num(x) and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def write_category_summary(run_dir: Path, opt, summary: dict) -> None:
    import pandas as pd

    if opt is None or not len(opt):
        pd.DataFrame().to_csv(run_dir / "category_summary.csv", index=False)
        return
    ref_label = summary.get("reference_label")
    rows = []
    for (cat, comparison, method), g in opt.groupby(["category", "comparison", "method"], sort=False):
        thr = g[g["metric"] == "output_throughput"]
        focus = thr[thr["is_focus"]]
        at_ref = thr[thr["workload_label"] == ref_label]
        pick = focus if len(focus) else at_ref if len(at_ref) else thr

        def at(metric, sub_labels):
            s = g[(g["metric"] == metric) & (g["workload_label"].isin(sub_labels))]["speedup_x"]
            return _geomean(s)

        labels_ = list(pick["workload_label"])
        first = g.iloc[0]
        rows.append(dict(category=cat, category_label=CATEGORY_LABEL.get(cat, cat), engine=first["engine"],
                         method=method, comparison=comparison, what=first["what"],
                         focus_workloads=";".join(labels_[:4]) + ("…" if len(labels_) > 4 else ""),
                         throughput_x_focus=_geomean(pick["speedup_x"]),
                         ttft_x_focus=at("median_ttft_ms", labels_), tpot_x_focus=at("median_tpot_ms", labels_),
                         e2e_x_focus=at("median_e2el_ms", labels_),
                         throughput_x_geomean_all=_geomean(thr["speedup_x"]),
                         throughput_x_min=thr["speedup_x"].min() if len(thr) else float("nan"),
                         throughput_x_max=thr["speedup_x"].max() if len(thr) else float("nan"),
                         significant_share=float(thr["significant"].mean()) if len(thr) else float("nan"),
                         n_workloads=int(thr["workload"].nunique()),
                         confounded_with=first["confounded_with"], note=first["note"]))
    out = pd.DataFrame(rows)
    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    m_order = {"ablation": 0, "ladder": 1, "variant": 2, "engine": 3}
    out["_o"] = out["category"].map(lambda c: order.get(c, 99)) * 10 + out["method"].map(lambda m: m_order.get(m, 9))
    out.sort_values("_o").drop(columns="_o").to_csv(run_dir / "category_summary.csv", index=False)


HEADLINE_ANCHORS = ["pt_eager", "pt_optimized", "pt_serve_optimized", "pytorch_best",
                    "vllm_minimal", "vllm_default", "vllm_optimized", "vllm_optimized_spec", "vllm_best"]


def write_headline(run_dir: Path, df, summary: dict) -> None:
    import pandas as pd

    labels = _run_labels(summary)
    ok = df[df["status"] == "ok"]
    rows = []
    for wl, g in ok.groupby("workload", sort=False):
        g = g.set_index("config")
        picks = {}
        for name in HEADLINE_ANCHORS:
            if name in g.index:
                picks[name] = name
        for eng, label in (("pytorch", "pytorch_best"), ("vllm", "vllm_best")):
            sub = g[g["engine"] == eng]["output_throughput"].dropna()
            if len(sub):
                picks[label] = sub.idxmax()
        eager = g.loc["pt_eager", "output_throughput"] if "pt_eager" in g.index else float("nan")
        vdef = g.loc["vllm_default", "output_throughput"] if "vllm_default" in g.index else float("nan")
        for anchor in HEADLINE_ANCHORS:
            if anchor not in picks:
                continue
            r = g.loc[picks[anchor]]
            thr = r.get("output_throughput")
            rows.append(dict(**labels, workload=wl, workload_label=r.get("workload_label"), kind=r.get("kind"),
                             input_len=r.get("input_len"), output_len=r.get("output_len"),
                             concurrency=r.get("concurrency"), request_rate=r.get("request_rate"),
                             anchor=anchor, config=picks[anchor], engine=r.get("engine"),
                             output_throughput=thr, output_throughput_per_gpu=r.get("output_throughput_per_gpu"),
                             request_throughput=r.get("request_throughput"),
                             median_ttft_ms=r.get("median_ttft_ms"), p99_ttft_ms=r.get("p99_ttft_ms"),
                             median_tpot_ms=r.get("median_tpot_ms"), median_e2el_ms=r.get("median_e2el_ms"),
                             peak_mem_gib=r.get("peak_mem_gib"),
                             x_vs_pt_eager=thr / eager if is_num(eager) and eager and is_num(thr) else float("nan"),
                             x_vs_vllm_default=thr / vdef if is_num(vdef) and vdef and is_num(thr) else float("nan")))
    pd.DataFrame(rows).to_csv(run_dir / "headline.csv", index=False)


def write_ladder(run_dir: Path, df, summary: dict) -> None:
    import pandas as pd

    labels = _run_labels(summary)
    ok = df[df["status"] == "ok"].set_index(["config", "workload"])
    rows = []
    for lname, steps in summary.get("ladders", {}).items():
        present = [s for s in steps if s["config"] in ok.index.get_level_values(0)]
        if len(present) < 2:
            continue
        wls = set(ok.xs(present[0]["config"], level="config").index)
        for wl in sorted(wls):
            for m in ("output_throughput", "median_ttft_ms", "median_tpot_ms", "median_e2el_ms"):
                vals = []
                for s in present:
                    v = ok.loc[(s["config"], wl)].get(m) if (s["config"], wl) in ok.index else float("nan")
                    vals.append(v)
                start = next((v for v in vals if is_num(v)), float("nan"))
                end = next((v for v in reversed(vals) if is_num(v)), float("nan"))
                total = _speedup(m, start, end)
                prev = float("nan")
                for i, (s, v) in enumerate(zip(present, vals)):
                    if not is_num(v):
                        continue
                    step_x = _speedup(m, prev, v) if is_num(prev) else 1.0
                    share = (math.log(step_x) / math.log(total) if is_num(total) and abs(math.log(total)) > 1e-9
                             and is_num(step_x) else float("nan"))
                    r0 = ok.loc[(s["config"], wl)]
                    rows.append(dict(**labels, ladder=lname, step=i, config=s["config"], added=s["added"],
                                     category=s["category"], category_label=CATEGORY_LABEL.get(s["category"], ""),
                                     workload=wl, workload_label=r0.get("workload_label"), kind=r0.get("kind"),
                                     concurrency=r0.get("concurrency"), metric=m, metric_label=METRICS[m][0],
                                     value=v, step_x=step_x, cumulative_x=_speedup(m, start, v),
                                     share_of_total_gain=share))
                    prev = v
    pd.DataFrame(rows).to_csv(run_dir / "ladder.csv", index=False)


def write_kernel_breakdown(run_dir: Path, summary: dict) -> None:
    import pandas as pd

    rows = []
    labels = _run_labels(summary)
    cfg_meta = {c["name"]: c for c in summary.get("configs", [])}
    for f in sorted((run_dir / "raw").glob("*__profile.json")):
        bd = json.loads(f.read_text())
        c = cfg_meta.get(bd.get("config"), {})
        for cls, d in bd.get("classes", {}).items():
            rows.append(dict(**labels, config=bd.get("config"), engine=bd.get("engine"), tier=c.get("tier"),
                             categories=";".join(c.get("categories", [])), workload=bd.get("workload"),
                             kernel_class=cls, gpu_ms=d["ms"], pct_of_gpu_time=d["pct"], n_kernels=d["n"],
                             top_kernel=d.get("top", ""), gpu_busy_pct=bd.get("gpu_busy_pct"),
                             window_ms=bd.get("window_ms"), device=bd.get("device")))
    if rows:
        pd.DataFrame(rows).to_csv(run_dir / "kernel_breakdown.csv", index=False)


def write_all(run_dir: Path) -> None:
    """Every derived CSV from results.csv + summary.json."""
    import pandas as pd

    if not (run_dir / "results.csv").exists():
        return
    summary = _summary(run_dir)
    df = pd.read_csv(run_dir / "results.csv")
    if "status" not in df.columns:
        df["status"] = "ok"
    write_long(run_dir, df, summary)
    write_optimizations(run_dir, df, summary)
    write_headline(run_dir, df, summary)
    write_ladder(run_dir, df, summary)
    write_kernel_breakdown(run_dir, summary)


CSV_FILES = ["results_long.csv", "optimizations.csv", "category_summary.csv", "headline.csv", "ladder.csv",
             "kernel_breakdown.csv", "anomalies.csv", "results.csv"]


def collect(out_dir: Path, run_ids: Optional[list[str]] = None) -> list[Path]:
    """Concatenate every run's CSVs into out_dir/all_<name>.csv, for cross-hardware charts."""
    import pandas as pd

    runs = [d for d in sorted(out_dir.iterdir()) if d.is_dir() and (d / "summary.json").exists()
            and (run_ids is None or d.name in run_ids)]
    written = []
    for name in CSV_FILES:
        parts = []
        for d in runs:
            p = d / name
            if p.exists() and p.stat().st_size > 1:
                try:
                    t = pd.read_csv(p)
                except Exception:  # noqa: BLE001  (empty file)
                    continue
                if "run_id" not in t.columns:
                    t.insert(0, "run_id", d.name)
                parts.append(t)
        if parts:
            dst = out_dir / f"all_{name}"
            pd.concat(parts, ignore_index=True).to_csv(dst, index=False)
            written.append(dst)
    return written
