"""Datasheet-style vector figures for the final deck and report.

Every chart is drawn from the saved results of run final_20261028 (no model
is refit except the deterministic response-surface refit for its contour
plot). Output is SVG with text kept as text; the font names are rewritten to
'Geist Mono' so the charts take the deck's and report's web fonts when the
SVG is inlined into the page.

Two variants per chart: 'deck' (1 SVG unit = 1 px on a 1920x1080 slide) and
'report' (sized for an A4 text column).

  python figures_v2.py <run_id>
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK = "#141414"
ACCENT = "#E4472B"
BLUE = "#2F4B66"
MUTED = "#8A857A"
FAINT = "#BDB6A6"
RULE = "#D9D3C4"
PAPER = "#F2EFE6"
RAMP = LinearSegmentedColormap.from_list("ink", ["#ECE7DA", "#B8C2CB", "#7A8FA3", "#2F4B66", "#141414"])

VARIANTS = {
    # name: (font size, line width, marker scale, px per inch)
    "deck": (22, 2.4, 1.6),
    "report": (11.5, 1.2, 0.9),
}


def setup(variant):
    fs, lw, _ = VARIANTS[variant]
    plt.rcdefaults()
    plt.rcParams.update({
        "svg.fonttype": "none",
        "font.family": "monospace",
        "font.monospace": ["DejaVu Sans Mono"],
        "font.size": fs,
        "axes.titlesize": fs,
        "axes.labelsize": fs * 0.9,
        "xtick.labelsize": fs * 0.85,
        "ytick.labelsize": fs * 0.85,
        "legend.fontsize": fs * 0.85,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": INK,
        "axes.linewidth": lw * 0.5,
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
        "axes.grid": True,
        "grid.color": RULE,
        "grid.linewidth": lw * 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": fs * 0.5,
        "ytick.major.pad": fs * 0.5,
        "legend.frameon": False,
        "lines.linewidth": lw,
        "lines.solid_capstyle": "butt",
    })


def fig(variant, w_px, h_px):
    """Figure whose SVG coordinate unit is one CSS px at its intended size."""
    return plt.figure(figsize=(w_px / 72.0, h_px / 72.0))


def save(f, outdir, name, variant):
    plt.rcParams["svg.hashsalt"] = f"{name}-{variant}"
    path = os.path.join(outdir, variant, f"{name}.svg")
    f.savefig(path, format="svg", transparent=True)
    plt.close(f)
    s = open(path, encoding="utf-8").read()
    s = s.replace("DejaVu Sans Mono", "Geist Mono").replace("DejaVu Sans", "Geist Mono")
    s = re.sub(r"<\?xml[^>]*\?>\s*", "", s)
    s = re.sub(r"<!DOCTYPE[^>]*>\s*", "", s)
    s = re.sub(r"<metadata>.*?</metadata>\s*", "", s, flags=re.S)
    open(path, "w", encoding="utf-8").write(s)


def ms(variant, base):
    return base * VARIANTS[variant][2]


# --------------------------------------------------------------- charts ---

def effects(A, variant, out, w, h):
    e = pd.read_csv(os.path.join(A, "screening_4096_effects.csv"))
    e["mag"] = e.effect_tflops.abs()
    main = e[~e.term.str.contains(":")]
    inter = e[e.term.str.contains(":")].sort_values("mag", ascending=False).head(6)
    d = pd.concat([main, inter]).sort_values("mag")
    f = fig(variant, w, h)
    ax = f.add_axes([0.25, 0.12, 0.70, 0.84])
    y = np.arange(len(d))
    colors = [ACCENT if t == "UNROLL" else (INK if ":" not in t else BLUE) for t in d.term]
    ax.barh(y, d.effect_tflops, color=colors, height=0.56)
    ax.errorbar(d.effect_tflops, y, xerr=[d.effect_tflops - d.ci95_low, d.ci95_high - d.effect_tflops],
                fmt="none", ecolor=MUTED, elinewidth=VARIANTS[variant][1] * 0.6, capsize=0)
    ax.axvline(0, color=INK, lw=VARIANTS[variant][1] * 0.5)
    ax.set_yticks(y, [t.replace(":", " × ") for t in d.term])
    ax.tick_params(axis="y", colors=INK)
    for yi, v, lo, hi in zip(y, d.effect_tflops, d.ci95_low, d.ci95_high):
        ax.text(hi + 0.25 if v >= 0 else lo - 0.25, yi, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", color=INK, fontsize=plt.rcParams["font.size"] * 0.8)
    ax.set_xlim(-5.6, 9.8)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("effect on TFLOPS, high − low level (95% CI)")
    save(f, out, "effects", variant)


def shapes(A, variant, out, w, h):
    s = json.load(open(os.path.join(A, "final_summary.json")))["shapes"]["total_contribution_pct"]
    df = pd.DataFrame({int(k): v for k, v in s.items()})
    order = ["UNROLL", "TM", "BN", "BK", "TN", "BM"]
    df = df.loc[order]
    f = fig(variant, w, h)
    ax = f.add_axes([0.14, 0.16, 0.83, 0.74])
    y = np.arange(len(order))
    for M, c, off in ((4096, INK, -0.18), (2048, MUTED, 0.0), (1024, ACCENT, 0.18)):
        ax.scatter(df[M], y + off, s=ms(variant, 70), color=c, zorder=3, label=f"{M}³", linewidths=0)
    for i, fac in enumerate(order):
        lo, hi = df.loc[fac].min(), df.loc[fac].max()
        ax.plot([lo, hi], [i, i], color=RULE, lw=VARIANTS[variant][1] * 0.6, zorder=1)
    ax.set_yticks(y, order)
    ax.tick_params(axis="y", colors=INK)
    ax.invert_yaxis()
    ax.set_xlim(-1, 60)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("% of variance: main effect + its 2-factor interactions")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.3, columnspacing=1.6)
    save(f, out, "shapes", variant)


def pred_vs_measured(A, run_dir, variant, out, w, h):
    ap = pd.read_csv(os.path.join(A, "model_predictions_all_feasible.csv"))
    ex = pd.read_csv(os.path.join(run_dir, "measurements.csv"))
    ex = ex[(ex.phase == "exhaustive") & (ex.validity == "valid")]
    key = lambda r: "_".join(f"{k}{int(r[k])}" for k in ["BM", "BN", "BK", "TM", "TN", "UNROLL"])
    ap["cfg_key"] = ap.apply(key, axis=1)
    ap = ap.merge(ex[["cfg_key", "tflops"]], on="cfg_key")
    f = fig(variant, w, h)
    axes = [f.add_axes([0.08, 0.15, 0.40, 0.72]), f.add_axes([0.57, 0.15, 0.40, 0.72])]
    for ax, col, title in ((axes[0], "predicted_tflops_quadratic", "QUADRATIC (PRE-REGISTERED)"),
                           (axes[1], "predicted_tflops", "QUADRATIC + 3-WAY TERMS")):
        out_, in_ = ap[~ap.in_permitted_region], ap[ap.in_permitted_region]
        ax.scatter(out_.tflops, out_[col], s=ms(variant, 9), color=FAINT, linewidths=0,
                   label=f"outside fitted region · {len(out_)}")
        ax.scatter(in_.tflops, in_[col], s=ms(variant, 14), color=BLUE, linewidths=0,
                   label=f"inside fitted region · {len(in_)}")
        ax.plot([0, 36], [0, 36], color=INK, lw=VARIANTS[variant][1] * 0.5)
        ax.axhline(0, color=MUTED, lw=VARIANTS[variant][1] * 0.4)
        ax.set_xlim(0, 36)
        ax.set_ylim(-32, 40)
        ax.set_title(title, loc="left", color=INK, fontsize=plt.rcParams["font.size"] * 0.8, pad=plt.rcParams["font.size"])
        ax.set_xlabel("measured TFLOPS")
    axes[0].set_ylabel("predicted TFLOPS")
    axes[0].legend(loc="lower right", markerscale=1.6, handletextpad=0.2)
    save(f, out, "pred_vs_measured", variant)


def sample_efficiency(A, variant, out, w, h):
    rs = pd.read_csv(os.path.join(A, "random_search_curve.csv"))
    tr = pd.read_csv(os.path.join(A, "doe_rsm_trajectory.csv"))
    s = json.load(open(os.path.join(A, "final_summary.json")))["sample_efficiency"]
    B, sel, best = s["doe_rsm_unique_evaluations_B"], s["doe_rsm_selected_tflops_exhaustive_measurement"], s["exhaustive"]["best_tflops"]
    fs = plt.rcParams["font.size"]
    f = fig(variant, w, h)
    ax = f.add_axes([0.09, 0.15, 0.88, 0.80])
    ax.fill_between(rs.n, rs.p10, rs.p90, color=BLUE, alpha=0.13, lw=0, label="random search, 10th–90th pct")
    ax.plot(rs.n, rs["median"], color=BLUE, label="random search, median of 2,000 runs")
    ax.step(tr.n, tr.best_so_far, where="post", color=ACCENT, label=f"DOE–RSM, best so far ({B} evaluations)")
    ax.scatter([B], [sel], s=ms(variant, 90), color=ACCENT, zorder=5, linewidths=0)
    ax.axhline(best, color=INK, lw=VARIANTS[variant][1] * 0.45)
    ax.text(1.05, best + 0.35, f"exhaustive optimum {best:.1f} · 1,396 evaluations", color=INK, fontsize=fs * 0.8)
    ax.legend(loc="lower right", handlelength=1.6)
    ax.set_xscale("log")
    ax.set_ylim(9, 36.5)
    ax.set_xlim(0.9, 1600)
    ax.set_xlabel("unique configurations evaluated (log)")
    ax.set_ylabel("best TFLOPS found")
    save(f, out, "sample_efficiency", variant)


def equal_budget(A, run_dir, variant, out, w, h):
    s = json.load(open(os.path.join(A, "final_summary.json")))["sample_efficiency"]
    ex = pd.read_csv(os.path.join(run_dir, "measurements.csv"))
    pool = ex[(ex.phase == "exhaustive") & (ex.validity == "valid")].groupby("cfg_key").tflops.mean().values
    B, sel = s["doe_rsm_unique_evaluations_B"], s["doe_rsm_selected_tflops_exhaustive_measurement"]
    rng = np.random.default_rng(20261028)
    best = np.array([pool[rng.choice(len(pool), B, replace=False)].max() for _ in range(10000)])
    fs = plt.rcParams["font.size"]
    f = fig(variant, w, h)
    ax = f.add_axes([0.09, 0.18, 0.88, 0.76])
    counts, _, _ = ax.hist(best, bins=60, color=BLUE, alpha=0.85, lw=0)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.axvline(sel, color=ACCENT, lw=VARIANTS[variant][1])
    ax.text(sel - 0.1, counts.max() * 1.2, f"DOE–RSM {sel:.1f}", color=ACCENT, ha="right", va="top", fontsize=fs * 0.85)
    ax.set_xlabel(f"best TFLOPS from random search, budget B = {B} (10,000 trials)")
    ax.set_ylabel("trials")
    save(f, out, "equal_budget", variant)


def final_comparison(A, variant, out, w, h):
    c = pd.read_csv(os.path.join(A, "final_cublas_comparison.csv"))
    c = c[c.cublas_mode == "pedantic"].set_index("config")
    rows = [("cuBLAS SGEMM, FP32", c.cublas_tflops.median(), INK),
            ("Exhaustive best", c.loc["exhaustive_best", "custom_tflops"], MUTED),
            ("DOE–RSM selection", c.loc["rsm_selected", "custom_tflops"], ACCENT),
            ("Untuned default", c.loc["default", "custom_tflops"], FAINT)]
    fs = plt.rcParams["font.size"]
    f = fig(variant, w, h)
    ax = f.add_axes([0.30, 0.14, 0.62, 0.82])
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.56)
    for yi, (lab, v, _) in zip(y, rows):
        ax.text(v + 0.6, yi, f"{v:.1f}", va="center", color=INK, fontsize=fs * 0.9)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", colors=INK)
    ax.set_xlim(0, 54)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("TFLOPS · 4096³ SGEMM · median of 900 interleaved launches")
    save(f, out, "final_comparison", variant)


def registers(run_dir, variant, out, w, h):
    ex = pd.read_csv(os.path.join(run_dir, "measurements.csv"))
    ex = ex[(ex.phase == "exhaustive") & (ex.validity == "valid")]
    sp = ex.local_bytes_per_thread > 0
    f = fig(variant, w, h)
    ax = f.add_axes([0.09, 0.15, 0.88, 0.80])
    ax.scatter(ex.regs_per_thread[~sp], ex.tflops[~sp], s=ms(variant, 12), color=BLUE, alpha=0.45, linewidths=0,
               label=f"no spill · {int((~sp).sum())}")
    ax.scatter(ex.regs_per_thread[sp], ex.tflops[sp], s=ms(variant, 26), color=ACCENT, linewidths=0,
               label=f"spills · {int(sp.sum())}")
    ax.set_xlabel("registers per thread")
    ax.set_ylabel("TFLOPS")
    ax.legend(loc="upper left", markerscale=1.5, handletextpad=0.2)
    save(f, out, "registers", variant)


def unroll(run_dir, variant, out, w, h):
    ex = pd.read_csv(os.path.join(run_dir, "measurements.csv"))
    ex = ex[(ex.phase == "exhaustive") & (ex.validity == "valid") & ex.cfg_key.str.startswith("BM64_BN64_BK32_TM8_TN4_")]
    ex = ex.sort_values("UNROLL")
    fs = plt.rcParams["font.size"]
    f = fig(variant, w, h)
    ax = f.add_axes([0.14, 0.17, 0.82, 0.78])
    x = np.arange(len(ex))
    colors = [ACCENT if u == 4 else INK for u in ex.UNROLL]
    ax.bar(x, ex.tflops, color=colors, width=0.56)
    for xi, v in zip(x, ex.tflops):
        ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", color=INK, fontsize=fs * 0.9)
    ax.set_xticks(x, [f"×{u}" for u in ex.UNROLL])
    ax.tick_params(axis="x", colors=INK)
    ax.set_ylim(0, 38)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("unroll factor · config (64, 64, 32, 8, 4)")
    ax.set_ylabel("TFLOPS")
    save(f, out, "unroll", variant)


def residuals(run_id, variant, out, w, h):
    import analysis
    from scipy import stats
    ctx = analysis.Ctx(run_id)
    R = ["BM", "BN", "BK", "TM", "TN", "UNROLL"]
    d = analysis.add_coded(ctx.phase("ccd"), R)
    fit, _ = analysis.fit_ccd_model(d, R, "tflops")
    r = pd.DataFrame({"order": d.order.values, "resid": fit.resid.values, "fitted": fit.fittedvalues.values}).sort_values("order")
    cooks = fit.get_influence().cooks_distance[0]
    fs = plt.rcParams["font.size"]
    f = fig(variant, w, h)
    gs = f.add_gridspec(2, 2, left=0.08, right=0.98, bottom=0.09, top=0.93, hspace=0.55, wspace=0.22)
    ax = [f.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    s = ms(variant, 14)
    ax[0].scatter(r.fitted, r.resid, s=s, color=BLUE, linewidths=0)
    ax[0].axhline(0, color=INK, lw=0.6)
    ax[0].set(xlabel="fitted TFLOPS", ylabel="residual")
    ax[0].set_title("RESIDUAL VS FITTED", loc="left", color=INK, fontsize=fs * 0.85)
    (osm, osr), (slope, icpt, _) = stats.probplot(r.resid, dist="norm")
    ax[1].scatter(osm, osr, s=s, color=BLUE, linewidths=0)
    ax[1].plot(osm, slope * osm + icpt, color=INK, lw=0.6)
    ax[1].set(xlabel="theoretical quantile", ylabel="residual")
    ax[1].set_title("NORMAL Q–Q", loc="left", color=INK, fontsize=fs * 0.85)
    ax[2].plot(np.arange(len(r)), r.resid, color=BLUE, lw=0.6, marker="o", ms=1.6)
    ax[2].axhline(0, color=INK, lw=0.6)
    ax[2].set(xlabel="run order", ylabel="residual")
    ax[2].set_title("RESIDUAL VS RUN ORDER", loc="left", color=INK, fontsize=fs * 0.85)
    ax[3].vlines(np.arange(len(cooks)), 0, cooks, color=BLUE, lw=0.7)
    ax[3].axhline(4 / len(cooks), color=ACCENT, lw=0.6)
    ax[3].set(xlabel="observation", ylabel="Cook's D")
    ax[3].set_title("COOK'S DISTANCE  (line: 4/n)", loc="left", color=INK, fontsize=fs * 0.85)
    save(f, out, "residuals", variant)


def surfaces(run_id, variant, out, w, h):
    import itertools
    import analysis
    from designs import C, H
    ctx = analysis.Ctx(run_id)
    surface, s = analysis.load_surface(ctx)
    best = {k: v for k, v in s["optimisation"]["augmented_enumeration_best"].items() if k != "predicted_tflops"}
    p = surface.fit.params
    R = surface.retained
    inter = sorted(((abs(p[f"x_{a}:x_{b}"]), a, b) for a, b in itertools.combinations(R, 2)), reverse=True)[:3]
    fs = plt.rcParams["font.size"]
    f = fig(variant, w, h)
    axes = [f.add_axes([0.075 + i * 0.335, 0.19, 0.205, 0.68]) for i in range(3)]
    for ax, (_, a, b) in zip(axes, inter):
        xs = np.linspace(-1, 1, 41)
        Z = np.zeros((41, 41))
        for i, ya in enumerate(xs):
            for j, xb in enumerate(xs):
                cfg = dict(best)
                cfg[a] = 2 ** (C[a] + xb * H[a])
                cfg[b] = 2 ** (C[b] + ya * H[b])
                Z[i, j] = surface.predict_tflops([cfg])[0]
        cs = ax.contourf(xs, xs, Z, levels=12, cmap=RAMP)
        ax.contour(xs, xs, Z, levels=12, colors=PAPER, linewidths=0.4)
        ax.grid(False)
        cb = f.colorbar(cs, ax=ax, fraction=0.06, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=fs * 0.75, colors=MUTED)
        ax.set(xlabel=f"{a} (coded)", ylabel=f"{b} (coded)")
        ax.set_title(f"{a} × {b}", loc="left", color=INK, fontsize=fs * 0.9)
    save(f, out, "surfaces", variant)


if __name__ == "__main__":
    run_id = sys.argv[1]
    run_dir = os.path.join(ROOT, "results", run_id)
    A = os.path.join(run_dir, "analysis")
    out = os.path.join(run_dir, "figs_v2")
    for v in VARIANTS:
        os.makedirs(os.path.join(out, v), exist_ok=True)
    sizes = {  # name: (deck w, h px, report w, h px)
        "effects": ((980, 620), (620, 400)),
        "shapes": ((1000, 600), (620, 360)),
        "pred_vs_measured": ((1664, 560), (640, 300)),
        "sample_efficiency": ((1100, 620), (640, 360)),
        "equal_budget": ((1000, 480), (640, 280)),
        "final_comparison": ((1050, 520), (620, 260)),
        "registers": ((1000, 600), (620, 340)),
        "unroll": ((760, 560), (420, 300)),
        "residuals": ((1600, 900), (640, 440)),
        "surfaces": ((1664, 520), (660, 260)),
    }
    for v in VARIANTS:
        setup(v)
        k = 0 if v == "deck" else 1
        effects(A, v, out, *sizes["effects"][k])
        shapes(A, v, out, *sizes["shapes"][k])
        pred_vs_measured(A, run_dir, v, out, *sizes["pred_vs_measured"][k])
        sample_efficiency(A, v, out, *sizes["sample_efficiency"][k])
        equal_budget(A, run_dir, v, out, *sizes["equal_budget"][k])
        final_comparison(A, v, out, *sizes["final_comparison"][k])
        registers(run_dir, v, out, *sizes["registers"][k])
        unroll(run_dir, v, out, *sizes["unroll"][k])
        residuals(run_id, v, out, *sizes["residuals"][k])
        surfaces(run_id, v, out, *sizes["surfaces"][k])
        print("done", v)
