"""Static images of the dashboard's charts (plots/*.png and *.svg), for slides, papers and READMEs.

  frontier_<ladder>.png                 every stage of a ladder: tok/s per GPU vs tok/s per user
  frontier_<ladder>_first_vs_last.png   the same with only the first and last stages in color
  frontier_engines.png                  PyTorch vs vLLM, baseline (dashed) vs optimized (solid), log scale
  stages_<ladder>.png                   every stage at one concurrency, with each stage's gain
  engine_matrix.png                     the 2x2 (engine x optimized) across concurrency, log scale
  overview.png                          the headline charts above stitched into one image

Same data, colors and conventions as dashboard.html (light theme).
"""
from __future__ import annotations

from pathlib import Path

from .dashboard import build_data

# validated categorical order (fixed, never cycled past 8; a 9th+ series reuses a hue with a dashed line)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _cat(i: int) -> tuple[str, str]:
    return CAT[i % len(CAT)], ("-" if i < len(CAT) else (0, (6, 3)))
ENGINE = {"vllm": "#2a78d6", "pytorch": "#eb6834"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def _style(ax, xlab: str, ylab: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlabel(xlab, color=INK2, fontsize=10)
    ax.set_ylabel(ylab, color=INK2, fontsize=10)


def _plain_log(axis) -> None:
    """Log axis with plain tick labels (2,000 instead of 2x10^3)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _pts(data, cfg, shape):
    return sorted([p for p in data["points"] if p["config"] == cfg and p["inp"] == shape[0] and p["out"] == shape[1]
                   and p["status"] == "ok" and p["out_gpu"] and p["inter"]], key=lambda p: p["c"])


def _best_shape(data, view) -> tuple[int, int] | None:
    shapes = {(p["inp"], p["out"]) for p in data["points"] if p["status"] == "ok"}
    if not shapes:
        return None
    ref = tuple(data["ref_shape"])

    def score(sh):
        return sum(len(_pts(data, s["config"], sh)) >= 2 for s in view["series"]) + (0.5 if sh == ref else 0)
    return max(sorted(shapes), key=score)


def _save(fig, pdir: Path, name: str) -> None:
    fig.savefig(pdir / f"{name}.png", dpi=160, facecolor=SURFACE)
    fig.savefig(pdir / f"{name}.svg", facecolor=SURFACE)


def _frontier(plt, data, view, shape, pdir: Path, focus: str) -> bool:
    series = [s for s in view["series"] if _pts(data, s["config"], shape)]
    if len(series) < 2:
        return False
    n = len(view["series"])
    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.patch.set_facecolor(SURFACE)
    engines = view["kind"] == "engines"
    labels = []
    for s in sorted(series, key=lambda s: view["series"].index(s)):
        i = view["series"].index(s)
        col, ls = _cat(i)
        if engines:   # hue = engine (as in the 2x2), line style = baseline / default / optimized
            on = True
            col = ENGINE[s["engine"]]
            ls = {"begin": "--", "mid": ":", "end2": (0, (8, 3, 2, 3))}.get(s["stage"], "-")
        else:
            on = focus == "all" or i in (0, n - 1)
        a = 1.0 if on else 0.28
        p = _pts(data, s["config"], shape)
        xs, ys = [q["inter"] for q in p], [q["out_gpu"] for q in p]
        ax.plot(xs, ys, linestyle=ls, color=col, lw=2 if on else 1.4, alpha=a, zorder=3 if on else 2,
                label=(f"{i + 1}. " if not engines else "") + s["label"])
        ax.scatter(xs, ys, s=28 if on else 14, color=col, edgecolor=SURFACE, linewidth=1.5, alpha=a, zorder=4 if on else 2)
        if on and (engines or focus != "all" or i in (0, n - 1)):
            k = max(range(len(ys)), key=lambda j: ys[j])
            labels.append((ys[k], xs[k], (f"{i + 1}. " if not engines else "") + s["label"]))
            ax.annotate(f"c{p[0]['c']}", (xs[0], ys[0]), textcoords="offset points", xytext=(5, -11),
                        fontsize=8, color=MUTED)
    # direct labels at each curve's peak, nudged apart vertically
    labels.sort(reverse=True)
    placed = []
    if (not engines and (len(labels) > 4 or focus == "all")):
        labels = []   # many lines: the legend carries identity, labels would collide
    for yv, xv, txt in labels:
        dy = 8
        while any(abs(dy - q) < 11 and abs(xv - qx) < (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.25 for q, qx in placed):
            dy -= 11
        placed.append((dy, xv))
        ax.annotate(txt, (xv, yv), textcoords="offset points", xytext=(6, dy), fontsize=8.5, color=INK2,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    if len(series) > 2 or engines:   # with many lines, a legend backs up the direct labels
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0), labelcolor=INK2,
                  handlelength=2.6)
    if engines:
        ax.set_xscale("log")
        ax.set_yscale("log")
        _plain_log(ax.xaxis)
        _plain_log(ax.yaxis)
    else:
        ax.set_ylim(bottom=0)
    _style(ax, "Interactivity (tok/s/user) → better", "Output tok/s per GPU")
    title = view["title"] if not engines else "PyTorch (orange) vs vLLM (blue): baseline dashed, default dotted, optimized solid"
    sub = f"{shape[0]} in / {shape[1]} out tokens · one point per concurrency"
    if not engines:
        sub += " · one color per stage" if focus == "all" else " · first vs last stage (others faded)"
    ax.set_title(f"{title}\n", loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK2)
    fig.tight_layout()
    name = "frontier_engines" if engines else f"frontier_{view['id']}" + ("" if focus == "all" else "_first_vs_last")
    _save(fig, pdir, name)
    plt.close(fig)
    return True


def _stage_bars(plt, data, view, shape, pdir: Path) -> bool:
    series = [s for s in view["series"] if _pts(data, s["config"], shape)]
    if len(series) < 2:
        return False
    concs = sorted({q["c"] for s in series for q in _pts(data, s["config"], shape)})
    c = max(concs, key=lambda c: (sum(any(q["c"] == c for q in _pts(data, s["config"], shape)) for s in series), c))
    n = len(view["series"])
    rows = []
    for s in view["series"]:
        p = [q for q in _pts(data, s["config"], shape) if q["c"] == c]
        rows.append((view["series"].index(s), s["label"], p[0]["out_gpu"] if p else None))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.4))
    fig.patch.set_facecolor(SURFACE)
    vmax = max(v for _, _, v in rows if v) or 1
    prev = None
    for k, (i, label, v) in enumerate(rows):
        yy = len(rows) - 1 - k
        if v is None:
            ax.text(vmax * 0.01, yy, "not measured / out of memory at this concurrency", va="center",
                    fontsize=8.5, color=MUTED, style="italic")
        else:
            ax.barh(yy, v, height=0.6, color=_cat(i)[0], hatch="///" if i >= len(CAT) else None,
                    edgecolor=SURFACE if i >= len(CAT) else None)
            ax.text(v + vmax * 0.01, yy, f"{v:,.0f}", va="center", fontsize=9, color=INK)
            if prev:
                g = v / prev
                ax.text(vmax * 1.22, yy, f"×{g:.2f}", va="center", ha="right", fontsize=9,
                        color="#006300" if g >= 1 else "#d03b3b")
            prev = v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{i + 1}. {label}" for i, label, _ in reversed(rows)], fontsize=9, color=INK2)
    ax.set_xlim(0, vmax * 1.25)
    _style(ax, f"Output tok/s per GPU at concurrency {c}", "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.text(0.01, 1 - 0.3 / fig.get_figheight(), f"{view['title']}: each stage at concurrency {c}",
             fontsize=12, color=INK, fontweight="bold", va="center")
    fig.text(0.01, 1 - 0.58 / fig.get_figheight(),
             f"{shape[0]} in / {shape[1]} out tokens · ×N = gain over the stage before (green helps, red hurts)",
             fontsize=9, color=INK2, va="center")
    _save(fig, pdir, f"stages_{view['id']}")
    plt.close(fig)
    return True


def _engine_matrix(plt, run_dir: Path, summary: dict, pdir: Path) -> bool:
    import pandas as pd

    p = run_dir / "engine_matrix.csv"
    if not p.exists() or p.stat().st_size < 2:
        return False
    em = pd.read_csv(p)
    rin, rout, _ = summary["reference"]
    sub = em[(em["metric"] == "output_throughput_per_gpu") & em["kind"].isin(["grid", "reference"])
             & (em["input_len"] == rin) & (em["output_len"] == rout)].drop_duplicates("concurrency")
    sub = sub.sort_values("concurrency")
    if not len(sub):
        return False
    sides = [("pytorch_baseline", "PyTorch baseline", "pytorch", True), ("pytorch_optimized", "PyTorch optimized", "pytorch", False),
             ("vllm_baseline", "vLLM baseline", "vllm", True), ("vllm_optimized", "vLLM optimized", "vllm", False)]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(SURFACE)
    from matplotlib.patches import Patch
    w = 0.2
    xs = list(range(len(sub)))
    handles = []
    for k, (col, label, eng, base) in enumerate(sides):
        vals = list(sub[col])
        pos = [x + (k - 1.5) * w for x in xs]
        for x, v in zip(pos, vals):
            if pd.notna(v) and v > 0:
                ax.bar(x, v, width=w * 0.9, color=SURFACE if base else ENGINE[eng], edgecolor=ENGINE[eng],
                       hatch="///" if base else None, linewidth=1.4)
            else:
                ax.text(x, 0.01, "×", ha="center", va="bottom", fontsize=10, color=MUTED,
                        transform=ax.get_xaxis_transform())
        handles.append(Patch(facecolor=SURFACE if base else ENGINE[eng], edgecolor=ENGINE[eng],
                             hatch="///" if base else None, label=label))
    ax.set_yscale("log")
    _plain_log(ax.yaxis)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"c={int(c)}" for c in sub["concurrency"]])
    _style(ax, "Concurrency", "Output tok/s per GPU (log)")
    ax.legend(handles=handles, frameon=False, fontsize=9, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.12), labelcolor=INK2)
    ax.set_title(f"PyTorch vs vLLM, not optimized (hatched) vs optimized (solid) · {rin} in / {rout} out   "
                 "(× = out of memory / not run)", loc="left", fontsize=11, color=INK, pad=28)
    fig.tight_layout()
    _save(fig, pdir, "engine_matrix")
    plt.close(fig)
    return True


def write_figures(run_dir: Path, summary: dict) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["hatch.linewidth"] = 0.8
    pdir = run_dir / "plots"
    pdir.mkdir(exist_ok=True)
    data = build_data(run_dir)
    n = 0
    for view in data["views"]:
        shape = _best_shape(data, view)
        if not shape:
            continue
        if view["kind"] == "engines":
            n += _frontier(plt, data, view, shape, pdir, "both")
        else:
            n += _frontier(plt, data, view, shape, pdir, "all")
            n += _frontier(plt, data, view, shape, pdir, "both")
            n += _stage_bars(plt, data, view, shape, pdir)
    n += _engine_matrix(plt, run_dir, summary, pdir)
    n += _overview(plt, data, pdir)
    return n


def _overview(plt, data, pdir: Path) -> bool:
    """One image with the headline charts: the 2x2, engines head-to-head, and each ladder's first vs last stage."""
    names = ["engine_matrix", "frontier_engines"] + [f"frontier_{v['id']}" for v in data["views"]
                                                     if v["kind"] == "ladder"]
    files = [pdir / f"{n}.png" for n in names if (pdir / f"{n}.png").exists()]
    if len(files) < 2:
        return False
    imgs = [plt.imread(f) for f in files]
    cols = 2
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 9, rows * 5.6 + 0.8))
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.axis("off")
    for ax, im in zip(axes.flat, imgs):
        ax.imshow(im)
    sub = " · ".join(x for x in [data.get("model"), data.get("hardware")] if x)
    fig.suptitle(f"PyTorch vs vLLM optimization overview  ({data.get('run_id')})\n{sub}", x=0.01, ha="left",
                 fontsize=15, color=INK)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=1 - 0.8 / (rows * 5.6 + 0.8), wspace=0.01, hspace=0.02)
    fig.savefig(pdir / "overview.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return True
