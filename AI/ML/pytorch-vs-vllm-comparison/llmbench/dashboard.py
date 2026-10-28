"""dashboard.html: an InferenceX-style interactive view of how the optimization stages move the
throughput / interactivity frontier, from the first stages of a ladder to the last.

The main chart is throughput per GPU (y) vs interactivity in tok/s/user (x): one curve per stage,
one point per concurrency, so a better stage sits up and to the right. A focus control dims the
middle of a ladder to compare its beginning stages with its end stages, and a "compare at" bar
chart reads each stage at the same concurrency or at a matched interactivity (interpolated along
each curve). Self-contained: one HTML file with the data embedded, no network needed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .util import is_num

ENGINE_VIEW = [  # (config, label, engine, stage) for the head-to-head view
    ("pt_eager", "PyTorch eager", "pytorch", "begin"),
    ("pt_optimized", "PyTorch optimized (static)", "pytorch", "end"),
    ("pt_serve_optimized", "PyTorch optimized (serving)", "pytorch", "end2"),
    ("vllm_minimal", "vLLM minimal", "vllm", "begin"),
    ("vllm_default", "vLLM default", "vllm", "mid"),
    ("vllm_optimized", "vLLM optimized", "vllm", "end"),
]
LADDER_TITLES = {"vllm": "vLLM ladder", "pytorch_static": "PyTorch static ladder",
                 "pytorch_serving": "PyTorch serving ladder"}


def _num(v):
    return float(v) if is_num(v) else None


def build_data(run_dir: Path) -> dict:
    import pandas as pd

    summary = json.loads((run_dir / "summary.json").read_text())
    res = pd.read_csv(run_dir / "results.csv")
    if "status" not in res.columns:
        res["status"] = "ok"
    sweep = res[res["kind"].isin(["grid", "reference"]) & (res["prefix_len"].fillna(0) == 0)]
    views = []
    for lname, steps in summary.get("ladders", {}).items():
        views.append(dict(id=lname, title=LADDER_TITLES.get(lname, lname), kind="ladder",
                          series=[dict(config=s["config"], label=s["added"].replace("(= ", "(").strip(),
                                       category=s.get("category", "")) for s in steps]))
    views.append(dict(id="engines", title="Engines head-to-head", kind="engines",
                      series=[dict(config=c, label=l, engine=e, stage=st) for c, l, e, st in ENGINE_VIEW]))
    wanted = {s["config"] for v in views for s in v["series"]}
    points = []
    for _, r in sweep[sweep["config"].isin(wanted)].iterrows():
        g = r.get("gpu_count")
        g = float(g) if is_num(g) and g else 1.0
        tot = _num(r.get("total_token_throughput"))
        tpot = _num(r.get("median_tpot_ms"))
        points.append(dict(
            config=r["config"], inp=int(r["input_len"]), out=int(r["output_len"]), c=int(r["concurrency"]),
            status=str(r.get("status")),
            reason=str(r.get("bench_error") or "") if str(r.get("status")) != "ok" else "",
            out_gpu=_num(r.get("output_throughput_per_gpu")) or (
                _num(r.get("output_throughput")) / g if _num(r.get("output_throughput")) else None),
            tot_gpu=tot / g if tot else None,
            inter=1000.0 / tpot if tpot and tpot > 0 else None,
            ttft=_num(r.get("median_ttft_ms")), e2e=_num(r.get("median_e2el_ms")),
            tpot=tpot, pre=_num(r.get("preemptions"))))
    env = summary.get("environment", {})
    ref = summary.get("reference") or [None, None, None]
    return dict(run_id=summary.get("run_id"), model=summary.get("model"),
                hardware=env.get("hardware", {}).get("describe"), vllm=env.get("vllm"),
                torch=env.get("torch"), ref_shape=[ref[0], ref[1]], views=views, points=points,
                matrix=_matrix(run_dir))


MATRIX_COLS = ["pytorch_baseline", "pytorch_optimized", "vllm_baseline", "vllm_optimized"]
MATRIX_X = ["pytorch_optimization_x", "vllm_optimization_x", "vllm_vs_pytorch_baseline_x",
            "vllm_vs_pytorch_optimized_x", "end_to_end_x"]


def _matrix(run_dir: Path) -> list[dict]:
    """engine_matrix.csv rows (2x2: engine x optimized) for the dashboard's table."""
    import pandas as pd

    p = run_dir / "engine_matrix.csv"
    if not p.exists() or p.stat().st_size < 2:
        return []
    em = pd.read_csv(p)
    em = em[em["kind"].isin(["grid", "reference"])]
    out = []
    for _, r in em.iterrows():
        d = dict(inp=int(r["input_len"]), out=int(r["output_len"]), c=int(r["concurrency"]), metric=r["metric"])
        for k in MATRIX_COLS:
            d[k] = _num(r.get(k))
            d[k + "_status"] = str(r.get(k + "_status") or "")
            d[k + "_config"] = str(r.get(k + "_config") or "") if is_num(r.get(k)) else ""
        for k in MATRIX_X:
            d[k] = _num(r.get(k))
        out.append(d)
    return out


def write_dashboard(run_dir: Path) -> Path | None:
    if not (run_dir / "results.csv").exists():
        return None
    data = build_data(run_dir)
    blob = json.dumps(data, allow_nan=False, default=lambda o: None).replace("</", "<\\/")
    out = run_dir / "dashboard.html"
    out.write_text(TEMPLATE.replace("/*__DATA__*/null", blob))
    return out


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Optimization Frontier</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10); --dim: #d4d3cc;
  --s0: #86b6ef; --s1: #5598e7; --s2: #2a78d6; --s3: #1c5cab; --s4: #104281;
  --c1: #2a78d6; --c2: #eb6834; --c3: #1baf7a; --c4: #eda100; --c5: #e87ba4; --c6: #008300; --c7: #6250d6; --c8: #e34948;
  --vllm: #2a78d6; --pytorch: #eb6834; --good: #006300; --bad: #d03b3b; --chip: #efeee9;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --dim: #3a3a37;
    --s0: #184f95; --s1: #256abf; --s2: #3987e5; --s3: #6da7ec; --s4: #b7d3f6;
  --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --c5: #d55181; --c6: #008300; --c7: #9085e9; --c8: #e66767;
    --vllm: #3987e5; --pytorch: #d95926; --good: #0ca30c; --bad: #e66767; --chip: #262624;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10); --dim: #3a3a37;
  --s0: #184f95; --s1: #256abf; --s2: #3987e5; --s3: #6da7ec; --s4: #b7d3f6;
  --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --c5: #d55181; --c6: #008300; --c7: #9085e9; --c8: #e66767;
  --vllm: #3987e5; --pytorch: #d95926; --good: #0ca30c; --bad: #e66767; --chip: #262624;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }
header { display: flex; gap: 16px; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; }
h1 { font-size: 20px; margin: 0 0 4px; font-weight: 650; }
.meta { color: var(--ink-2); font-size: 13px; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 16px; margin-top: 16px; }
.controls { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: flex-end; }
.ctl { display: flex; flex-direction: column; gap: 4px; }
.ctl > span { font-size: 12px; color: var(--muted); }
select, button, input[type=range] { font: inherit; color: var(--ink); }
select { background: var(--surface); border: 1px solid var(--axis); border-radius: 8px; padding: 6px 8px; }
.seg { display: inline-flex; border: 1px solid var(--axis); border-radius: 8px; overflow: hidden; }
.seg button { background: transparent; border: 0; padding: 6px 10px; cursor: pointer; color: var(--ink-2); }
.seg button + button { border-left: 1px solid var(--axis); }
.seg button[aria-pressed=true] { background: var(--chip); color: var(--ink); font-weight: 600; }
.toggle { display: inline-flex; gap: 6px; align-items: center; color: var(--ink-2); cursor: pointer; }
#themeBtn { background: transparent; border: 1px solid var(--axis); border-radius: 8px; padding: 6px 10px; cursor: pointer; color: var(--ink-2); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 16px; }
.tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 12px 14px; }
.tile .k { color: var(--muted); font-size: 12px; }
.tile .v { font-size: 26px; font-weight: 650; margin-top: 2px; }
.tile .s { color: var(--ink-2); font-size: 12px; }
.chart-title { font-weight: 600; margin: 0 0 2px; }
.chart-sub { color: var(--ink-2); font-size: 12px; margin: 0 0 8px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 8px; font-size: 12px; }
.legend button { display: inline-flex; gap: 6px; align-items: center; background: none; border: 0; padding: 2px 0;
  color: var(--ink-2); cursor: pointer; font: inherit; }
.legend button[aria-pressed=false] { opacity: .45; }
.sw { width: 18px; height: 0; border-top: 3px solid; border-radius: 2px; }
.tip { position: fixed; pointer-events: none; z-index: 10; background: var(--surface); color: var(--ink);
  border: 1px solid var(--ring); border-radius: 8px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,.18); max-width: 300px; display: none; }
.tip b { font-size: 14px; }
.tip .row { display: flex; justify-content: space-between; gap: 12px; }
.tip .lab { color: var(--muted); }
.note { color: var(--muted); font-size: 12px; margin-top: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 600; }
td.na { color: var(--muted); }
.tablewrap { overflow-x: auto; }
details summary { cursor: pointer; color: var(--ink-2); }
.dl { display: flex; gap: 6px; justify-content: flex-end; margin-top: 8px; }
.dl button { background: transparent; border: 1px solid var(--axis); border-radius: 6px; padding: 3px 8px;
  font-size: 12px; color: var(--ink-2); cursor: pointer; }
.dl button:hover { color: var(--ink); }
td.gain { font-weight: 600; }
td.up { color: var(--good); } td.down { color: var(--bad); }
th.grp { text-align: center; border-bottom: 0; }
.mx-who { color: var(--muted); font-size: 12px; margin-top: 6px; }
.two { display: grid; grid-template-columns: 1fr; gap: 16px; }
@media (min-width: 980px) { .two { grid-template-columns: 3fr 2fr; } }
</style>
</head>
<body>
<main>
<header>
  <div>
    <h1>Optimization frontier</h1>
    <div class="meta" id="meta"></div>
  </div>
  <button id="themeBtn" type="button">Theme</button>
</header>

<div class="card controls" role="group" aria-label="Filters">
  <label class="ctl"><span>Ladder</span><select id="view"></select></label>
  <label class="ctl"><span>Shape (input / output tokens)</span><select id="shape"></select></label>
  <div class="ctl"><span>Focus</span><div class="seg" id="focus"></div></div>
  <label class="ctl"><span>X axis</span><select id="xm">
    <option value="inter">Interactivity (tok/s/user)</option>
    <option value="ttft">TTFT p50 (ms)</option>
    <option value="e2e">End-to-end latency p50 (s)</option></select></label>
  <label class="ctl"><span>Y axis</span><select id="ym">
    <option value="out_gpu">Output tok/s per GPU</option>
    <option value="tot_gpu">Total tok/s per GPU</option></select></label>
  <label class="toggle"><input type="checkbox" id="logs"> Log scale</label>
  <label class="toggle"><input type="checkbox" id="pareto"> Pareto frontier</label>
</div>

<div class="tiles" id="tiles"></div>

<div class="two">
  <div class="card">
    <p class="chart-title" id="mainTitle"></p>
    <p class="chart-sub" id="mainSub"></p>
    <svg id="main" viewBox="0 0 760 460" role="img" aria-labelledby="mainTitle"></svg>
    <div class="legend" id="legend"></div>
    <div class="note" id="stops"></div>
    <div class="dl"><button type="button" data-dl="main" data-fmt="png">Download PNG</button>
      <button type="button" data-dl="main" data-fmt="svg">SVG</button></div>
  </div>
  <div class="card">
    <p class="chart-title">Compare stages at one operating point</p>
    <div class="controls" style="margin-bottom:8px">
      <div class="ctl"><span>Compare at</span><div class="seg" id="atMode"></div></div>
      <label class="ctl" id="atConcWrap"><span>Concurrency</span><select id="atConc"></select></label>
      <label class="ctl" id="atInterWrap"><span>Interactivity: <b id="atInterVal"></b> tok/s/user</span>
        <input type="range" id="atInter" min="0" max="100" step="1"></label>
    </div>
    <p class="chart-sub" id="barSub"></p>
    <svg id="bars" viewBox="0 0 460 360" role="img" aria-label="Stage comparison"></svg>
    <div class="dl"><button type="button" data-dl="bars" data-fmt="png">Download PNG</button>
      <button type="button" data-dl="bars" data-fmt="svg">SVG</button></div>
  </div>
</div>

<div class="card" id="mxCard">
  <p class="chart-title">PyTorch vs vLLM: not optimized vs optimized</p>
  <p class="chart-sub">Baseline = <code>pt_eager</code> / <code>vllm_minimal</code>. Optimized = the better of each engine's
    optimized configs at that point. Gains are ×&gt;1 = better (for latencies, how many times lower).</p>
  <div class="controls" style="margin-bottom:8px">
    <label class="ctl"><span>Shape</span><select id="mxShape"></select></label>
    <label class="ctl"><span>Concurrency</span><select id="mxConc"></select></label>
  </div>
  <div class="tablewrap"><table id="mx"></table></div>
  <div class="mx-who" id="mxWho"></div>
</div>

<div class="card">
  <details>
    <summary>Table view</summary>
    <div class="tablewrap"><table id="tbl"></table></div>
  </details>
</div>
</main>
<div class="tip" id="tip" role="tooltip"></div>

<script>
const DATA = /*__DATA__*/null;
const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (v, d = 0) => v == null ? "–" : Number(v).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d});
const fmtX = (v) => v == null ? "–" : v === 0 ? "0" : (v >= 100 ? fmt(v, 0) : v >= 10 ? fmt(v, 1) : fmt(v, 2));
const el = (tag, attrs = {}, parent) => { const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); if (parent) parent.appendChild(n); return n; };
const text = (parent, x, y, s, attrs = {}) => { const t = el("text", {x, y, ...attrs}, parent); t.textContent = s; return t; };

const state = { view: null, shape: null, focus: "all", xm: "inter", ym: "out_gpu", pareto: false, logBy: {},
  hidden: new Set(), atMode: "conc", atConc: null, atInter: null };
try { const s = JSON.parse(localStorage.getItem("llmbench-dash-v2") || "{}"); Object.assign(state, s, {hidden: new Set()}); } catch (e) {}
const save = () => { try { localStorage.setItem("llmbench-dash-v2", JSON.stringify({view: state.view, focus: state.focus,
  xm: state.xm, ym: state.ym, pareto: state.pareto, atMode: state.atMode})); } catch (e) {} };

// ------------------------------------------------------------------ data helpers
const views = DATA.views.filter(v => v.series.some(s => DATA.points.some(p => p.config === s.config && p.status === "ok")));
const byView = Object.fromEntries(views.map(v => [v.id, v]));
function seriesOf(view) { return view.series.filter(s => DATA.points.some(p => p.config === s.config)); }
function pts(cfg, shape) { return DATA.points.filter(p => p.config === cfg && p.inp === shape[0] && p.out === shape[1])
  .sort((a, b) => a.c - b.c); }
function okPts(cfg, shape) { return pts(cfg, shape).filter(p => p.status === "ok" && p[state.ym] != null && xval(p) != null); }
function xval(p) { return state.xm === "inter" ? p.inter : state.xm === "ttft" ? p.ttft : (p.e2e == null ? null : p.e2e / 1000); }
function shapesFor(view) { const s = new Map(); for (const ser of seriesOf(view)) for (const p of DATA.points)
  if (p.config === ser.config && p.status === "ok") s.set(p.inp + "/" + p.out, [p.inp, p.out]);
  return [...s.values()].sort((a, b) => a[0] - b[0] || a[1] - b[1]); }
function stopReason(cfg, shape) {
  const all = pts(cfg, shape); const bad = all.find(p => p.status !== "ok");
  if (!bad) return null;
  const r = bad.reason.replace(/^SKIP:\s*/, "");
  const kind = /memory|OOM/i.test(r) ? "out of memory" : /saturat|past the limit/i.test(r) ? "saturated" : bad.status;
  return {c: bad.c, kind, detail: r};
}

// ------------------------------------------------------------------ colors & focus
const CAT = ["--c1", "--c2", "--c3", "--c4", "--c5", "--c6", "--c7", "--c8"];
function stageColor(view, i, n) {   // each stage keeps its own hue in every focus mode; a 9th+ stage reuses a hue, dashed
  // head-to-head: hue = engine (matches the 2x2), line style = baseline / default / optimized
  if (view.kind === "engines") return css(view.series[i].engine === "vllm" ? "--vllm" : "--pytorch");
  return css(CAT[i % CAT.length]);
}
function mix(h1, h2, f) { const p = (h) => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const a = p(h1), b = p(h2); return "#" + a.map((v, i) => Math.round(v + (b[i] - v) * f).toString(16).padStart(2, "0")).join(""); }
function dashOf(view, s) { if (view.kind !== "engines") { const i = view.series.indexOf(s); return i >= CAT.length ? "7 4" : null; }
  return s.stage === "begin" ? "6 5" : s.stage === "mid" ? "2 4" : s.stage === "end2" ? "10 3 2 3" : null; }
function inFocus(view, i, n) {
  if (view.kind === "engines") { const st = view.series[i].stage;
    return state.focus === "all" || state.focus === "both" ? st !== "mid" || state.focus === "all"
      : state.focus === "begin" ? st === "begin" : st !== "begin" && st !== "mid"; }
  const k = Math.min(3, Math.max(1, Math.floor(n / 3)));
  if (state.focus === "all") return true;
  if (state.focus === "begin") return i < k;
  if (state.focus === "end") return i >= n - k;
  return i === 0 || i === n - 1;            // first vs last
}

// ------------------------------------------------------------------ controls
function seg(container, opts, key, after) {
  container.textContent = "";
  for (const [v, label] of opts) { const b = document.createElement("button"); b.type = "button";
    b.textContent = label; b.setAttribute("aria-pressed", String(state[key] === v));
    b.onclick = () => { state[key] = v; save(); render(); }; container.appendChild(b); }
}
function initControls() {
  const vs = $("view"); vs.textContent = "";
  for (const v of views) { const o = document.createElement("option"); o.value = v.id; o.textContent = v.title; vs.appendChild(o); }
  if (!byView[state.view]) state.view = byView.vllm ? "vllm" : views[0].id;
  vs.value = state.view;
  vs.onchange = () => { state.view = vs.value; state.hidden.clear(); state.shape = null; save(); render(); };
  for (const id of ["xm", "ym"]) { $(id).value = state[id]; $(id).onchange = () => { state[id] = $(id).value; state.atInter = null; save(); render(); }; }
  $("pareto").checked = !!state.pareto; $("pareto").onchange = () => { state.pareto = $("pareto").checked; save(); render(); };
  $("logs").onchange = () => { state.logBy[state.view] = $("logs").checked; render(); };
  $("shape").onchange = () => { state.shape = $("shape").value.split("/").map(Number); state.atConc = null; state.atInter = null; render(); };
  $("atConc").onchange = () => { state.atConc = Number($("atConc").value); render(); };
  $("atInter").oninput = () => { state.atInter = Number($("atInter").value); render(); };
  $("themeBtn").onclick = () => { const cur = document.documentElement.dataset.theme ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.dataset.theme = cur === "dark" ? "light" : "dark"; render(); };
  $("meta").textContent = [DATA.model, DATA.hardware, DATA.vllm ? "vLLM " + DATA.vllm : null,
    DATA.torch ? "torch " + DATA.torch : null, DATA.run_id].filter(Boolean).join(" · ");
}

// ------------------------------------------------------------------ scales & axes
function niceTicks(lo, hi, n = 6) {   // rounds OUTWARD so every value lies inside the axis
  const span = hi - lo || 1; const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0))); const r = step0 / mag;
  const step = (r > 5 ? 10 : r > 2 ? 5 : r > 1 ? 2 : 1) * mag;
  const a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step;
  const out = []; for (let v = a; v <= b + step * 1e-6; v += step) out.push(+v.toPrecision(12)); return out; }
function logTicks(lo, hi) {   // 1-2-5 ticks spanning [lo, hi], rounded outward
  const out = []; let e = Math.floor(Math.log10(lo)); const top = Math.ceil(Math.log10(hi));
  for (; e <= top; e++) for (const m of [1, 2, 5]) out.push(+(m * Math.pow(10, e)).toPrecision(12));
  const a = out.filter(v => v <= lo).pop() ?? out[0], b = out.find(v => v >= hi) ?? out[out.length - 1];
  return out.filter(v => v >= a && v <= b); }
function axes(svg, x, y, W, H, M, xs, ys, xlab, ylab) {
  const g = el("g", {}, svg);
  for (const t of ys) { el("line", {x1: M.l, x2: W - M.r, y1: y(t), y2: y(t), stroke: css("--grid"), "stroke-width": 1}, g);
    text(g, M.l - 8, y(t) + 4, fmtX(t), {"text-anchor": "end", fill: css("--muted"), "font-size": 11, "font-variant-numeric": "tabular-nums"}); }
  for (const t of xs) { text(g, x(t), H - M.b + 18, fmtX(t), {"text-anchor": "middle", fill: css("--muted"), "font-size": 11}); }
  el("line", {x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, stroke: css("--axis")}, g);
  text(g, (M.l + W - M.r) / 2, H - 6, xlab, {"text-anchor": "middle", fill: css("--ink-2"), "font-size": 12});
  text(g, 14, (M.t + H - M.b) / 2, ylab, {"text-anchor": "middle", fill: css("--ink-2"), "font-size": 12,
    transform: `rotate(-90 14 ${(M.t + H - M.b) / 2})`});
}
const XLAB = {inter: "Interactivity (tok/s/user) →  better", ttft: "← better   TTFT p50 (ms)", e2e: "← better   End-to-end latency p50 (s)"};
const YLAB = {out_gpu: "Output tok/s per GPU", tot_gpu: "Total tok/s per GPU"};

// ------------------------------------------------------------------ tooltip
const tip = $("tip");
function showTip(ev, lines) { tip.textContent = "";
  for (const [a, b, strong] of lines) { const r = document.createElement("div"); r.className = "row";
    const l = document.createElement("span"); l.className = "lab"; l.textContent = a;
    const v = document.createElement(strong ? "b" : "span"); v.textContent = b; r.append(l, v); tip.appendChild(r); }
  tip.style.display = "block";
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(innerWidth - w - 8, ev.clientX + 14) + "px";
  tip.style.top = Math.max(8, ev.clientY - h - 12) + "px"; }
const hideTip = () => { tip.style.display = "none"; };

// ------------------------------------------------------------------ main chart
function renderMain(view, shape) {
  const svg = $("main"); svg.textContent = "";
  const W = 760, H = 460, M = {l: 64, r: 24, t: 16, b: 44};
  const ser = seriesOf(view); const n = view.series.length;
  const lines = ser.map(s => { const i = view.series.indexOf(s);
    return {s, i, color: stageColor(view, i, n), focus: inFocus(view, i, n), hidden: state.hidden.has(s.config),
            p: okPts(s.config, shape)}; }).filter(l => l.p.length && !l.hidden);
  const all = lines.flatMap(l => l.p);
  if (!all.length) { text(svg, W / 2, H / 2, "No measurements for this shape", {"text-anchor": "middle", fill: css("--muted")}); return; }
  const xv = all.map(xval), yv = all.map(p => p[state.ym]);
  const xmin = Math.min(...xv), xmax = Math.max(...xv), ymin = Math.min(...yv), ymax = Math.max(...yv);
  const log = !!state.logBy[view.id];
  let x, y, xs, ys;
  if (log) {
    xs = logTicks(xmin, xmax); ys = logTicks(ymin, ymax);
    const lx0 = Math.log10(xs[0]), lx1 = Math.log10(xs[xs.length - 1]), ly0 = Math.log10(ys[0]), ly1 = Math.log10(ys[ys.length - 1]);
    x = v => M.l + (Math.log10(v) - lx0) / ((lx1 - lx0) || 1) * (W - M.l - M.r);
    y = v => H - M.b - (Math.log10(v) - ly0) / ((ly1 - ly0) || 1) * (H - M.t - M.b);
  } else {
    const xpad = (xmax - xmin) * 0.06 || 1;
    xs = niceTicks(Math.max(0, xmin - xpad), xmax + xpad); ys = niceTicks(0, ymax * 1.06);
    const x0 = xs[0], x1 = xs[xs.length - 1], y1 = ys[ys.length - 1];
    x = v => M.l + (v - x0) / (x1 - x0) * (W - M.l - M.r);
    y = v => H - M.b - v / y1 * (H - M.t - M.b);
  }
  axes(svg, x, y, W, H, M, xs, ys, XLAB[state.xm], YLAB[state.ym]);
  // context (out of focus) first, focused on top
  const order = [...lines].sort((a, b) => (a.focus - b.focus) || (a.i - b.i));
  const labelYs = [];
  for (const L of order) {
    const g = el("g", {}, svg);
    const col = L.color; if (!L.focus) g.setAttribute("opacity", "0.28");
    const d = L.p.map((p, k) => (k ? "L" : "M") + x(xval(p)).toFixed(1) + " " + y(p[state.ym]).toFixed(1)).join(" ");
    const attrs = {d, fill: "none", stroke: col, "stroke-width": L.focus ? 2 : 1.5, "stroke-linejoin": "round", "stroke-linecap": "round"};
    const dash = dashOf(view, L.s); if (dash) attrs["stroke-dasharray"] = dash;
    el("path", attrs, g);
    for (const p of L.p) {
      const cx = x(xval(p)), cy = y(p[state.ym]);
      el("circle", {cx, cy, r: L.focus ? 4.5 : 3, fill: col, stroke: css("--surface"), "stroke-width": 2}, g);
      const hit = el("circle", {cx, cy, r: 11, fill: "transparent", tabindex: L.focus ? 0 : -1}, g);
      const on = (ev) => showTip(ev, [[L.s.label, "", false], ["concurrency", String(p.c), true],
        [YLAB[state.ym], fmt(p[state.ym]), true], ["interactivity", fmt(p.inter, 1) + " tok/s/user"],
        ["TTFT p50", fmt(p.ttft) + " ms"], ["TPOT p50", fmt(p.tpot, 2) + " ms"],
        ["E2E p50", fmt(p.e2e == null ? null : p.e2e / 1000, 2) + " s"]].concat(p.pre ? [["preemptions", fmt(p.pre)]] : []));
      hit.addEventListener("pointermove", on); hit.addEventListener("pointerleave", hideTip);
      hit.addEventListener("focus", (e) => { const r = hit.getBoundingClientRect(); on({clientX: r.x, clientY: r.y}); });
      hit.addEventListener("blur", hideTip);
    }
    if (L.focus) {   // concurrency tags at the curve ends (not where the stage label sits)
      const f = L.p[0], l = L.p[L.p.length - 1];
      const top = L.p.reduce((a, b) => (b[state.ym] > a[state.ym] ? b : a));
      const tagThis = view.kind === "engines" || state.focus !== "all" || L.i === 0 || L.i === n - 1;
      if (tagThis) {
        text(g, x(xval(f)) + 7, y(f[state.ym]) + 14, "c" + f.c, {fill: css("--muted"), "font-size": 10});
        if (L.p.length > 1 && l !== top) text(g, x(xval(l)) - 6, y(l[state.ym]) + 16, "c" + l.c,
          {fill: css("--muted"), "font-size": 10, "text-anchor": "end"});
      }
      // label each curve at its highest-throughput point, where the curves are furthest apart
      labelYs.push({y: y(top[state.ym]), x: x(xval(top)), s: L.s.label});
    }
  }
  // direct labels just above each curve's peak, nudged apart so they never overlap; with more than
  // four highlighted lines the legend carries identity instead (labels would pile up)
  if (labelYs.length > (view.kind === "engines" ? 6 : 4)) labelYs.length = 0;
  labelYs.sort((a, b) => a.y - b.y); const placed = [];
  for (const lb of labelYs) {
    const short = lb.s.length > 26 ? lb.s.slice(0, 25) + "…" : lb.s;
    const w = short.length * 6.2 + 4;
    let lx = Math.min(Math.max(lb.x + 8, M.l + 4), W - M.r - w), ly = Math.max(M.t + 10, lb.y - 10);
    for (let guard = 0; guard < 40 && placed.some(q => Math.abs(q.y - ly) < 13 && lx < q.x + q.w && q.x < lx + w); guard++) ly += 13;
    placed.push({x: lx, y: ly, w});
    const t = text(svg, lx, ly, short, {fill: css("--ink-2"), "font-size": 11, "paint-order": "stroke",
      stroke: css("--surface"), "stroke-width": 3, "stroke-linejoin": "round"}); }
  if (state.pareto) {
    const cand = lines.filter(l => l.focus).flatMap(l => l.p);
    const better = state.xm === "inter" ? (a, b) => a >= b : (a, b) => a <= b;
    const front = cand.filter(p => !cand.some(q => q !== p && better(xval(q), xval(p)) && q[state.ym] >= p[state.ym]
      && (xval(q) !== xval(p) || q[state.ym] !== p[state.ym]))).sort((a, b) => xval(a) - xval(b));
    if (front.length > 1) el("path", {d: front.map((p, k) => (k ? "L" : "M") + x(xval(p)) + " " + y(p[state.ym])).join(" "),
      fill: "none", stroke: css("--ink"), "stroke-width": 1.5, "stroke-dasharray": "4 4", opacity: .7}, svg);
  }
  // legend
  const lg = $("legend"); lg.textContent = "";
  ser.forEach((s) => { const i = view.series.indexOf(s); const b = document.createElement("button"); b.type = "button";
    b.setAttribute("aria-pressed", String(!state.hidden.has(s.config)));
    const sw = document.createElement("span"); sw.className = "sw";
    sw.style.borderTopColor = stageColor(view, i, view.series.length);
    if (!inFocus(view, i, view.series.length)) sw.style.opacity = "0.4";
    const dash = dashOf(view, s); if (dash) sw.style.borderTopStyle = dash === "2 4" ? "dotted" : "dashed";
    const t = document.createElement("span"); t.textContent = (view.kind === "ladder" ? (i + 1) + ". " : "") + s.label;
    b.append(sw, t); b.onclick = () => { state.hidden.has(s.config) ? state.hidden.delete(s.config) : state.hidden.add(s.config); render(); };
    lg.appendChild(b); });
  const stops = ser.map(s => { const r = stopReason(s.config, shape); return r ? `${s.label}: stops at c=${r.c} (${r.kind})` : null; })
    .filter(Boolean);
  $("stops").textContent = stops.length ? "Where curves end — " + stops.join(" · ") : "";
}

// ------------------------------------------------------------------ compare-at bars
function interpAt(p, target) {   // throughput along a curve at a given interactivity (linear between neighbours)
  const q = p.filter(r => r.inter != null).sort((a, b) => a.inter - b.inter);
  if (!q.length || target < q[0].inter || target > q[q.length - 1].inter) return null;
  for (let k = 0; k < q.length - 1; k++) { const a = q[k], b = q[k + 1];
    if (target >= a.inter && target <= b.inter) { const f = b.inter === a.inter ? 0 : (target - a.inter) / (b.inter - a.inter);
      return a[state.ym] + f * (b[state.ym] - a[state.ym]); } }
  return q[q.length - 1][state.ym];
}
function renderBars(view, shape) {
  const ser = seriesOf(view); const n = view.series.length;
  const concs = [...new Set(ser.flatMap(s => okPts(s.config, shape).map(p => p.c)))].sort((a, b) => a - b);
  seg($("atMode"), [["conc", "Same concurrency"], ["inter", "Same interactivity"]], "atMode");
  $("atConcWrap").style.display = state.atMode === "conc" ? "" : "none";
  $("atInterWrap").style.display = state.atMode === "inter" ? "" : "none";
  const cs = $("atConc"); cs.textContent = "";
  for (const c of concs) { const o = document.createElement("option"); o.value = c; o.textContent = "c = " + c; cs.appendChild(o); }
  // default: the highest concurrency where most stages still have a point
  if (!concs.includes(state.atConc)) { let best = concs[0], bestN = -1;
    for (const c of concs) { const k = ser.filter(s => okPts(s.config, shape).some(p => p.c === c)).length; if (k >= bestN) { best = c; bestN = k; } }
    state.atConc = best; }
  cs.value = state.atConc;
  const inters = ser.flatMap(s => okPts(s.config, shape).map(p => p.inter)).filter(v => v != null);
  const lo = Math.floor(Math.min(...inters)), hi = Math.ceil(Math.max(...inters));
  const r = $("atInter"); r.min = lo; r.max = hi;
  if (state.atInter == null || state.atInter < lo || state.atInter > hi) state.atInter = Math.round(lo + (hi - lo) * 0.5);
  r.value = state.atInter; $("atInterVal").textContent = fmt(state.atInter);
  const rows = ser.map(s => { const i = view.series.indexOf(s); const p = okPts(s.config, shape);
    const v = state.atMode === "conc" ? (p.find(q => q.c === state.atConc) || {})[state.ym] ?? null : interAtSafe(p);
    const why = v == null ? (state.atMode === "conc" ? (stopReason(s.config, shape) && stopReason(s.config, shape).c <= state.atConc
      ? stopReason(s.config, shape).kind : "not measured") : "outside its range") : "";
    return {s, i, v, why, color: stageColor(view, i, n), focus: inFocus(view, i, n)}; });
  function interAtSafe(p) { const v = interpAt(p, state.atInter); return v == null ? null : v; }
  $("barSub").textContent = (state.atMode === "conc" ? `${YLAB[state.ym]} at concurrency ${state.atConc}`
    : `${YLAB[state.ym]} at ${fmt(state.atInter)} tok/s/user, interpolated along each curve`) +
    (view.kind === "ladder" ? " · ×prev = gain over the stage before" : "");
  const svg = $("bars"); svg.textContent = "";
  const W = 460, rowH = 30, M = {l: 168, r: 92, t: 6};
  const H = M.t + rows.length * rowH + 10; svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const vmax = Math.max(1, ...rows.map(r => r.v || 0)); const xs = (v) => (W - M.l - M.r) * v / vmax;
  let prev = null; const first = rows.find(r => r.v != null);
  rows.forEach((r, k) => { const yy = M.t + k * rowH;
    const lab = (view.kind === "ladder" ? (r.i + 1) + ". " : "") + r.s.label;
    text(svg, M.l - 8, yy + 19, lab.length > 24 ? lab.slice(0, 23) + "…" : lab, {"text-anchor": "end", fill: css(r.focus ? "--ink" : "--muted"), "font-size": 12});
    if (r.v == null) { text(svg, M.l + 4, yy + 19, r.why, {fill: css("--muted"), "font-size": 11, "font-style": "italic"}); return; }
    const w = Math.max(2, xs(r.v));
    el("path", {d: `M${M.l} ${yy + 6} h${w - 4} a4 4 0 0 1 4 4 v10 a4 4 0 0 1 -4 4 h${-(w - 4)} z`,
      fill: r.color, opacity: r.focus ? 1 : 0.35}, svg);
    text(svg, M.l + w + 6, yy + 19, fmt(r.v), {fill: css("--ink"), "font-size": 12, "font-weight": 600});
    if (view.kind === "ladder" && prev != null && prev.v) { const g = r.v / prev.v;
      text(svg, W - 4, yy + 19, "×" + g.toFixed(2), {"text-anchor": "end", "font-size": 11, fill: css(g >= 1 ? "--good" : "--bad")}); }
    prev = r; });
  // tiles: first vs last stage at this operating point
  const tiles = $("tiles"); tiles.textContent = "";
  const valid = rows.filter(r => r.v != null);
  const add = (k, v, s) => { const d = document.createElement("div"); d.className = "tile";
    for (const [cls, t] of [["k", k], ["v", v], ["s", s]]) { const e = document.createElement("div"); e.className = cls; e.textContent = t; d.appendChild(e); }
    tiles.appendChild(d); };
  if (valid.length >= 2) {
    const a = valid[0], b = valid[valid.length - 1];
    const at = state.atMode === "conc" ? `at c=${state.atConc}` : `at ${fmt(state.atInter)} tok/s/user`;
    add("First → last stage", "×" + (b.v / a.v).toFixed(2), `${a.s.label} → ${b.s.label}, ${at}`);
    const beg = valid.filter(r => inFocus(view, r.i, n) && r.i < n / 2), endr = valid.filter(r => inFocus(view, r.i, n) && r.i >= n / 2);
    let bestStep = null; for (let k = 1; k < valid.length; k++) { const g = valid[k].v / valid[k - 1].v;
      if (!bestStep || g > bestStep.g) bestStep = {g, r: valid[k]}; }
    if (bestStep && view.kind === "ladder") add("Biggest single step", "×" + bestStep.g.toFixed(2), bestStep.r.s.label + ", " + at);
    add(YLAB[state.ym], fmt(b.v), `${b.s.label}, ${at}`);
  }
}

// ------------------------------------------------------------------ table
function renderTable(view, shape) {
  const ser = seriesOf(view);
  const concs = [...new Set(ser.flatMap(s => pts(s.config, shape).map(p => p.c)))].sort((a, b) => a - b);
  const t = $("tbl"); t.textContent = "";
  const hr = t.insertRow(); const h0 = document.createElement("th"); h0.textContent = "stage — " + YLAB[state.ym]; hr.appendChild(h0);
  for (const c of concs) { const th = document.createElement("th"); th.textContent = "c=" + c; hr.appendChild(th); }
  for (const s of ser) { const row = t.insertRow(); const c0 = row.insertCell(); c0.textContent = s.label;
    for (const c of concs) { const p = pts(s.config, shape).find(q => q.c === c); const cell = row.insertCell();
      if (!p) { cell.textContent = ""; continue; }
      if (p.status !== "ok") { cell.textContent = /memory|OOM/i.test(p.reason) ? "OOM" : /saturat|past the limit/i.test(p.reason) ? "sat." : p.status;
        cell.className = "na"; cell.title = p.reason; continue; }
      cell.textContent = fmt(p[state.ym]); } }
}

// ------------------------------------------------------------------ render
function render() {
  const view = byView[state.view] || views[0]; state.view = view.id;
  if (!(view.id in state.logBy)) state.logBy[view.id] = view.kind === "engines";   // engines differ ~10x
  $("logs").checked = !!state.logBy[view.id];
  const shapes = shapesFor(view); const ss = $("shape"); ss.textContent = "";
  for (const sh of shapes) { const o = document.createElement("option"); o.value = sh.join("/");
    o.textContent = `${sh[0]} in / ${sh[1]} out` + (DATA.ref_shape[0] === sh[0] && DATA.ref_shape[1] === sh[1] ? " (reference)" : "");
    ss.appendChild(o); }
  if (!state.shape || !shapes.some(s => s[0] === state.shape[0] && s[1] === state.shape[1])) {
    // default: the shape where the most stages have points (the reference shape wins ties)
    let best = shapes[0], bestN = -1;
    for (const sh of shapes) { const k = seriesOf(view).filter(s => okPts(s.config, sh).length >= 2).length
      + (DATA.ref_shape[0] === sh[0] && DATA.ref_shape[1] === sh[1] ? 0.5 : 0); if (k > bestN) { best = sh; bestN = k; } }
    state.shape = best; }
  ss.value = state.shape.join("/");
  const focusOpts = view.kind === "engines"
    ? [["both", "Baselines vs optimized"], ["begin", "Baselines"], ["end", "Optimized"], ["all", "All"]]
    : [["both", "First vs last"], ["begin", "Beginning stages"], ["end", "End stages"], ["all", "All stages"]];
  if (!focusOpts.some(o => o[0] === state.focus)) state.focus = "both";
  seg($("focus"), focusOpts, "focus");
  $("mainTitle").textContent = `${view.title}: ${YLAB[state.ym].replace(/^O/, "o")} vs ${state.xm === "inter" ? "interactivity" : state.xm === "ttft" ? "TTFT" : "end-to-end latency"}`;
  $("mainSub").textContent = `${state.shape[0]} input / ${state.shape[1]} output tokens · one point per concurrency (1 → 128) · ` +
    (view.kind === "ladder" ? "one color per stage (numbered in the legend); up and to the right is better"
      : "orange = PyTorch, blue = vLLM; dashed = baseline, dotted = default, solid = optimized");
  renderMain(view, state.shape); renderBars(view, state.shape); renderTable(view, state.shape);
}
// ------------------------------------------------------------------ 2x2 engine matrix
const MX = DATA.matrix || [];
const MX_ROWS = [["output_throughput_per_gpu", "Output tok/s per GPU", 0], ["output_throughput", "Output tok/s", 0],
  ["request_throughput", "Requests/s", 2], ["median_ttft_ms", "TTFT p50 (ms)", 0], ["p99_ttft_ms", "TTFT p99 (ms)", 0],
  ["median_tpot_ms", "TPOT p50 (ms)", 1], ["median_e2el_ms", "E2E p50 (ms)", 0], ["peak_mem_gib", "Peak memory (GiB)", 1]];
const MX_SIDES = ["pytorch_baseline", "pytorch_optimized", "vllm_baseline", "vllm_optimized"];
const mxState = {shape: null, c: null};
function mxFill(sel, opts, cur) { sel.textContent = ""; for (const [v, t] of opts) { const o = document.createElement("option");
  o.value = v; o.textContent = t; sel.appendChild(o); } sel.value = cur; }
function renderMatrix() {
  if (!MX.length) { $("mxCard").style.display = "none"; return; }
  const shapes = [...new Set(MX.map(r => r.inp + "/" + r.out))].sort((a, b) => { const [ai, ao] = a.split("/").map(Number),
    [bi, bo] = b.split("/").map(Number); return ai - bi || ao - bo; });
  const ref = DATA.ref_shape.join("/");
  if (!shapes.includes(mxState.shape)) mxState.shape = shapes.includes(ref) ? ref : shapes[0];
  mxFill($("mxShape"), shapes.map(s => [s, s.replace("/", " in / ") + " out" + (s === ref ? " (reference)" : "")]), mxState.shape);
  const rows = MX.filter(r => r.inp + "/" + r.out === mxState.shape);
  const concs = [...new Set(rows.map(r => r.c))].sort((a, b) => a - b);
  if (!concs.includes(mxState.c)) mxState.c = concs.includes(8) ? 8 : concs[0];
  mxFill($("mxConc"), concs.map(c => [c, "c=" + c]), mxState.c);
  const at = rows.filter(r => r.c === mxState.c);
  const t = $("mx"); t.textContent = "";
  const g = t.insertRow();
  for (const [txt, span] of [["", 1], ["PyTorch", 2], ["vLLM", 2], ["Gain from optimizing", 2], ["vLLM ÷ PyTorch", 2]]) {
    const th = document.createElement("th"); th.textContent = txt; th.colSpan = span; th.className = "grp"; g.appendChild(th); }
  const h = t.insertRow();
  for (const txt of ["metric", "baseline", "optimized", "baseline", "optimized", "PyTorch", "vLLM", "baseline", "optimized"]) {
    const th = document.createElement("th"); th.textContent = txt; h.appendChild(th); }
  for (const [m, label, dp] of MX_ROWS) {
    const r = at.find(q => q.metric === m); if (!r) continue;
    const row = t.insertRow(); row.insertCell().textContent = label;
    for (const s of MX_SIDES) { const cell = row.insertCell(); const v = r[s];
      if (v != null) cell.textContent = v.toLocaleString(undefined, {maximumFractionDigits: dp, minimumFractionDigits: dp});
      else { cell.textContent = r[s + "_status"] === "oom" ? "OOM" : r[s + "_status"] === "skipped" ? "saturated" : "–";
        cell.className = "na"; } }
    for (const k of ["pytorch_optimization_x", "vllm_optimization_x", "vllm_vs_pytorch_baseline_x", "vllm_vs_pytorch_optimized_x"]) {
      const cell = row.insertCell(); const v = r[k];
      if (v == null) { cell.textContent = "–"; cell.className = "na"; continue; }
      cell.textContent = "×" + v.toFixed(2); cell.className = "gain " + (v >= 1.02 ? "up" : v <= 0.98 ? "down" : ""); } }
  const r0 = at[0] || {};
  $("mxWho").textContent = `Optimized configs at this point: PyTorch = ${r0.pytorch_optimized_config || "none ran"}, ` +
    `vLLM = ${r0.vllm_optimized_config || "none ran"}. All shapes and concurrencies are in engine_matrix.csv.`;
}
$("mxShape").addEventListener("change", e => { mxState.shape = e.target.value; mxState.c = null; renderMatrix(); });
$("mxConc").addEventListener("change", e => { mxState.c = +e.target.value; renderMatrix(); });

// ------------------------------------------------------------------ chart export (PNG / SVG)
const STYLE_PROPS = ["fill", "fill-opacity", "stroke", "stroke-width", "stroke-dasharray", "stroke-opacity", "stroke-linecap",
  "opacity", "font-size", "font-family", "font-weight", "text-anchor", "dominant-baseline", "paint-order", "visibility", "display"];
function exportSvg(id) {
  const src = $(id); const vb = src.viewBox.baseVal; const W = vb.width, H = vb.height;
  const clone = src.cloneNode(true);
  const a = src.querySelectorAll("*"), b = clone.querySelectorAll("*");
  for (let i = 0; i < a.length; i++) { const cs = getComputedStyle(a[i]); // CSS variables don't survive outside the page
    for (const p of STYLE_PROPS) { const v = cs.getPropertyValue(p); if (v) b[i].style.setProperty(p, v); } }
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", W); clone.setAttribute("height", H);
  clone.style.fontFamily = getComputedStyle(document.body).fontFamily;
  const bgCol = getComputedStyle(src.closest(".card")).backgroundColor;
  const pad = 16, head = 44;
  const card = src.closest(".card");
  const title = card.querySelector(".chart-title")?.textContent || "";
  const sub = (id === "main" ? $("mainSub") : $("barSub")).textContent || "";
  const ns = "http://www.w3.org/2000/svg";
  const out = document.createElementNS(ns, "svg");
  out.setAttribute("xmlns", ns); out.setAttribute("width", W + 2 * pad); out.setAttribute("height", H + head + 2 * pad);
  out.setAttribute("viewBox", `0 0 ${W + 2 * pad} ${H + head + 2 * pad}`);
  out.style.fontFamily = clone.style.fontFamily;
  const bg = document.createElementNS(ns, "rect"); bg.setAttribute("width", "100%"); bg.setAttribute("height", "100%");
  bg.setAttribute("fill", bgCol); out.appendChild(bg);
  const txt = (s, y, size, col, weight) => { const t = document.createElementNS(ns, "text"); t.textContent = s;
    t.setAttribute("x", pad); t.setAttribute("y", y); t.setAttribute("font-size", size); t.setAttribute("fill", col);
    if (weight) t.setAttribute("font-weight", weight); out.appendChild(t); };
  const css = getComputedStyle(document.documentElement);
  txt(title, pad + 14, 15, css.getPropertyValue("--ink").trim(), 600);
  txt(sub, pad + 34, 11, css.getPropertyValue("--ink-2").trim());
  const gr = document.createElementNS(ns, "g"); gr.setAttribute("transform", `translate(${pad},${pad + head})`);
  clone.setAttribute("x", 0); clone.setAttribute("y", 0); gr.appendChild(clone); out.appendChild(gr);
  return {text: new XMLSerializer().serializeToString(out), w: W + 2 * pad, h: H + head + 2 * pad};
}
function fileStem(id) { const v = (byView[state.view] || views[0]).id;
  return `${DATA.run_id || "run"}_${id === "main" ? "frontier" : "stages"}_${v}_${state.shape.join("x")}_${state.focus}`; }
function saveBlob(blob, name) { const u = URL.createObjectURL(blob); const a = document.createElement("a");
  a.href = u; a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(u), 1000); }
document.querySelectorAll(".dl button").forEach(btn => btn.addEventListener("click", () => {
  const id = btn.dataset.dl; const {text, w, h} = exportSvg(id); const stem = fileStem(id);
  if (btn.dataset.fmt === "svg") { saveBlob(new Blob([text], {type: "image/svg+xml"}), stem + ".svg"); return; }
  const img = new Image(); const scale = 2;
  img.onload = () => { const cv = document.createElement("canvas"); cv.width = w * scale; cv.height = h * scale;
    const ctx = cv.getContext("2d"); ctx.scale(scale, scale); ctx.drawImage(img, 0, 0);
    cv.toBlob(b => saveBlob(b, stem + ".png"), "image/png"); };
  img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(text);
}));

initControls(); render(); renderMatrix();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
</script>
</body>
</html>
"""
