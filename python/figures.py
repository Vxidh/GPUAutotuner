"""Report figures. Static PNGs (embedded in the report) using the validated
reference palette: categorical slots 1-3 only (all-pairs CVD-safe), muted gray
for de-emphasised marks, single-hue blue ramp for magnitude, solid hairline
grid, one y-axis per plot."""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9.5,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10.5,
    "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2.0,
    "figure.dpi": 150,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def residual_panel(rdf, fit, cooks, title, path):
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.4))
    ax[0, 0].scatter(rdf.fitted, rdf.resid, s=14, color=S1, edgecolor=SURFACE, linewidth=0.6)
    ax[0, 0].axhline(0, color=AXIS, lw=1)
    ax[0, 0].set(title="Residuals vs fitted", xlabel="fitted", ylabel="residual")
    (osm, osr), (slope, icpt, _) = stats.probplot(rdf.resid, dist="norm")
    ax[0, 1].scatter(osm, osr, s=14, color=S1, edgecolor=SURFACE, linewidth=0.6)
    ax[0, 1].plot(osm, slope * osm + icpt, color=MUTED, lw=1.2)
    ax[0, 1].set(title="Normal Q-Q", xlabel="theoretical quantile", ylabel="ordered residual")
    ax[1, 0].plot(np.arange(len(rdf)), rdf.resid.values, color=S1, lw=1, marker="o", ms=3)
    ax[1, 0].axhline(0, color=AXIS, lw=1)
    ax[1, 0].set(title="Residuals vs run order", xlabel="run order", ylabel="residual")
    ax[1, 1].vlines(np.arange(len(cooks)), 0, cooks, color=S1, lw=1.2)
    ax[1, 1].axhline(4 / len(cooks), color=MUTED, lw=1)
    ax[1, 1].text(len(cooks) - 1, 4 / len(cooks), "4/n", ha="right", va="bottom", color=INK2, fontsize=8)
    ax[1, 1].set(title="Cook's distance", xlabel="observation", ylabel="D")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    _save(fig, path)


def response_surfaces(surface, best_cfg, path,
                      title="Fitted response surface: strongest interactions (other factors at model optimum)"):
    """Contours over the strongest interactions, other factors at the best
    enumerated configuration."""
    R = surface.retained
    p = surface.fit.params
    inter = sorted(((abs(p[f"x_{a}:x_{b}"]), a, b) for a, b in itertools.combinations(R, 2)), reverse=True)[:3]
    fig, axes = plt.subplots(1, len(inter), figsize=(4.4 * len(inter), 3.9))
    from designs import C, H
    for ax, (_, a, b) in zip(np.atleast_1d(axes), inter):
        xs = np.linspace(-1, 1, 41)
        Z = np.zeros((41, 41))
        for i, ya in enumerate(xs):
            for j, xb in enumerate(xs):
                cfg = {f: best_cfg[f] for f in best_cfg}
                cfg[a] = 2 ** (C[a] + xb * H[a])
                cfg[b] = 2 ** (C[b] + ya * H[b])
                Z[i, j] = surface.predict_tflops([cfg])[0]
        cs = ax.contourf(xs, xs, Z, levels=14, cmap=BLUE_RAMP)
        ax.contour(xs, xs, Z, levels=14, colors=SURFACE, linewidths=0.4)
        ax.grid(False)
        fig.colorbar(cs, ax=ax, label="predicted TFLOPS")
        ax.set(xlabel=f"{a} (coded log2)", ylabel=f"{b} (coded log2)", title=f"{a} × {b}")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    _save(fig, path)


def sample_efficiency(rs, doe_traj, sel_val, B, best, path):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.fill_between(rs.n, rs.p10, rs.p90, color=S1, alpha=0.15, lw=0)
    ax.plot(rs.n, rs["median"], color=S1, label="Random search (median, 10–90th pct band)")
    ax.plot(np.arange(1, len(doe_traj) + 1), doe_traj, color=S2, label="DOE–RSM evaluations (best so far)")
    ax.scatter([B], [sel_val], color=S2, s=60, zorder=5, edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"DOE–RSM selection\n{sel_val:.1f} TFLOPS, {B} evaluations", (B, sel_val),
                xytext=(12, -34), textcoords="offset points", color=INK, fontsize=8.5)
    ax.axhline(best, color=MUTED, lw=1)
    ax.text(rs.n.max(), best, f"exhaustive optimum {best:.1f}", ha="right", va="bottom", color=INK2, fontsize=8.5)
    ax.set_xscale("log")
    ax.set(xlabel="unique configurations evaluated (log scale)", ylabel="best TFLOPS found",
           title="Sample efficiency at 4096³")
    ax.legend(loc="lower right")
    _save(fig, path)


def budget_histogram(best_at_B, sel_val, B, path):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.hist(best_at_B, bins=60, color=S1, edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(sel_val, color=S2, lw=2)
    ax.text(sel_val, ax.get_ylim()[1] * 0.95, f"  DOE–RSM {sel_val:.1f}", color=INK, va="top", fontsize=8.5)
    ax.set(xlabel=f"best TFLOPS found by random search with the same budget (B = {B})",
           ylabel="trials (of 10,000)", title="Equal-budget random search")
    _save(fig, path)


def predicted_vs_measured(ap, path):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    out, ins = ap[~ap.in_permitted_region], ap[ap.in_permitted_region]
    ax.scatter(out.measured_tflops, out.predicted_tflops, s=12, color=S2, alpha=0.7,
               edgecolor=SURFACE, linewidth=0.4, label=f"outside fitted region (n={len(out)})")
    ax.scatter(ins.measured_tflops, ins.predicted_tflops, s=16, color=S1,
               edgecolor=SURFACE, linewidth=0.6, label=f"inside fitted region (n={len(ins)})")
    lo, hi = ap[["measured_tflops", "predicted_tflops"]].min().min(), ap[["measured_tflops", "predicted_tflops"]].max().max()
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
    ax.set(xlabel="measured TFLOPS (exhaustive sweep)", ylabel="RSM-predicted TFLOPS",
           title="Model predictions vs ground truth")
    ax.legend(loc="upper left")
    _save(fig, path)


def final_comparison(cmp, path):
    c = cmp[cmp.cublas_mode == "pedantic"].set_index("config")
    order = ["default", "rsm_selected", "exhaustive_best"]
    labels = ["Untuned default", "DOE–RSM selected", "Exhaustive best", "cuBLAS (pedantic FP32)"]
    vals = [c.loc[k, "custom_tflops"] for k in order] + [c.cublas_tflops.median()]
    colors = [MUTED, S1, MUTED, MUTED]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", color=INK, fontsize=9)
    ax.set(ylabel="TFLOPS (median, 4096³ SGEMM)", title="Throughput: custom kernel configurations vs cuBLAS")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def shape_importance(df, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    f = df.index.tolist()
    y = np.arange(len(f))
    for (M, col), dy in zip(df.items(), (-0.22, 0, 0.22)):
        color = {4096: S1, 2048: S2, 1024: S3}[M]
        ax.scatter(col.values, y + dy, s=46, color=color, edgecolor=SURFACE, linewidth=1.5, label=f"{M}³", zorder=3)
    ax.set_yticks(y, f)
    ax.invert_yaxis()
    ax.set(xlabel="% of total variance (main effect + its 2-factor interactions)",
           title="Factor importance by matrix shape")
    ax.legend(loc="lower right", title="shape", title_fontsize=8.5)
    _save(fig, path)


def registers(ex, path):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    sp = ex.local_bytes_per_thread > 0
    ax.scatter(ex.regs_per_thread[~sp], ex.tflops[~sp], s=10, color=S1, alpha=0.55, lw=0,
               label=f"no spill (n={int((~sp).sum())})")
    if sp.any():
        ax.scatter(ex.regs_per_thread[sp], ex.tflops[sp], s=16, color=S2, lw=0, label=f"spills (n={int(sp.sum())})")
    ax.set(xlabel="registers per thread", ylabel="TFLOPS", title="Register use vs throughput (exhaustive sweep)")
    ax.legend(loc="upper left")
    _save(fig, path)
