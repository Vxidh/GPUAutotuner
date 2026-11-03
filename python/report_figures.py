"""Report-ready figures redrawn from the saved analysis outputs (no model is
refit and no decision is recomputed; every number comes from the CSV/JSON files
the analysis already wrote).

  python report_figures.py <run_id>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures as F
from figures import plt, S1, S2, S3, MUTED, INK, INK2, SURFACE, AXIS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def screening_effects(A, M, path):
    e = pd.read_csv(os.path.join(A, f"screening_{M}_effects.csv"))
    e["mag"] = e.effect_tflops.abs()
    main = e[~e.term.str.contains(":")]
    inter = e[e.term.str.contains(":")].sort_values("mag", ascending=False).head(6)
    d = pd.concat([main, inter]).sort_values("mag")
    sig = d.p_value < 0.05
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    y = np.arange(len(d))
    colors = [S1 if s else MUTED for s in sig]
    ax.barh(y, d.effect_tflops, color=colors, height=0.62)
    ax.errorbar(d.effect_tflops, y, xerr=[d.effect_tflops - d.ci95_low, d.ci95_high - d.effect_tflops],
                fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, [t.replace(":", " × ") for t in d.term])
    ax.set(xlabel="effect on TFLOPS (high level − low level), with 95% CI",
           title=f"Screening effects at {M}³: 6 main effects + 6 largest interactions")
    ax.grid(axis="y", visible=False)
    ax.text(0.99, 0.02, "blue = significant (p < 0.05)", transform=ax.transAxes,
            ha="right", va="bottom", color=INK2, fontsize=8.5)
    F._save(fig, path)


def sample_efficiency(A, path):
    rs = pd.read_csv(os.path.join(A, "random_search_curve.csv"))
    tr = pd.read_csv(os.path.join(A, "doe_rsm_trajectory.csv"))
    s = json.load(open(os.path.join(A, "final_summary.json")))["sample_efficiency"]
    B, sel, best = s["doe_rsm_unique_evaluations_B"], s["doe_rsm_selected_tflops_exhaustive_measurement"], s["exhaustive"]["best_tflops"]
    first_hit = int(tr[tr.best_so_far >= sel - 1e-9].n.iloc[0])
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    ax.fill_between(rs.n, rs.p10, rs.p90, color=S1, alpha=0.15, lw=0)
    ax.plot(rs.n, rs["median"], color=S1, label="Random search: median of 2,000 runs (band: 10th–90th pct)")
    ax.plot(tr.n, tr.best_so_far, color=S2, label="DOE–RSM: best configuration measured so far")
    for x, lab in ((32, "screening ends"), (45, "CCD ends")):
        ax.axvline(x, color="#e1e0d9", lw=1, zorder=0)
        ax.text(x, 12.0, lab, rotation=90, ha="right", va="bottom", color=MUTED, fontsize=7.5)
    ax.scatter([B], [sel], color=S2, s=60, zorder=5, edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"validated selection: {sel:.1f} TFLOPS\nafter {B} evaluations (first measured\n"
                f"at evaluation {first_hit}, during screening)",
                (B, sel), xytext=(70, 18.0), textcoords="data", color=INK, fontsize=8.3,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.axhline(best, color=MUTED, lw=1)
    ax.text(1.05, best + 0.25, f"exhaustive optimum {best:.1f} TFLOPS (1,396 evaluations)", color=INK2, fontsize=8.3)
    ax.set_xscale("log")
    ax.set_ylim(8, 36.5)
    ax.set(xlabel="unique configurations evaluated (log scale)", ylabel="best TFLOPS found",
           title="Sample efficiency at 4096³")
    ax.legend(loc="lower right")
    F._save(fig, path)


def predicted_vs_measured(A, path):
    ap = pd.read_csv(os.path.join(A, "model_predictions_all_feasible.csv"))
    ex = pd.read_csv(os.path.join(os.path.dirname(A), "measurements.csv"))
    ex = ex[(ex.phase == "exhaustive") & (ex.validity == "valid")]
    key = lambda r: "_".join(f"{k}{int(r[k])}" for k in ["BM", "BN", "BK", "TM", "TN", "UNROLL"])
    ap["cfg_key"] = ap.apply(key, axis=1)
    ap = ap.merge(ex[["cfg_key", "tflops"]], on="cfg_key")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), sharey=True)
    for ax, col, title in ((axes[0], "predicted_tflops_quadratic", "Second-order model (pre-registered)"),
                           (axes[1], "predicted_tflops", "Second-order + 3-factor interactions")):
        out, ins = ap[~ap.in_permitted_region], ap[ap.in_permitted_region]
        ax.scatter(out.tflops, out[col], s=10, color=S2, alpha=0.6, lw=0, label=f"outside fitted region (n={len(out)})")
        ax.scatter(ins.tflops, ins[col], s=14, color=S1, edgecolor=SURFACE, linewidth=0.5,
                   label=f"inside fitted region (n={len(ins)})")
        ax.plot([0, 36], [0, 36], color=MUTED, lw=1)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xlim(0, 36)
        ax.set(xlabel="measured TFLOPS (exhaustive sweep)", title=title)
    axes[0].set_ylabel("predicted TFLOPS")
    axes[0].legend(loc="upper left")
    fig.suptitle("Model predictions vs ground truth for all 1,396 feasible configurations",
                 x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    F._save(fig, path)


def shape_importance(A, path):
    s = json.load(open(os.path.join(A, "final_summary.json")))["shapes"]["total_contribution_pct"]
    df = pd.DataFrame({int(k): v for k, v in s.items()})
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    y = np.arange(len(df.index))
    for (M, dy, c) in ((4096, -0.22, S1), (2048, 0, S2), (1024, 0.22, S3)):
        ax.scatter(df[M].values, y + dy, s=46, color=c, edgecolor=SURFACE, linewidth=1.5, label=f"{M}³", zorder=3)
    ax.set_yticks(y, df.index)
    ax.invert_yaxis()
    ax.set_xlim(-1, 60)
    ax.set(xlabel="% of total variance (main effect + its 2-factor interactions)",
           title="Factor importance by matrix shape")
    ax.legend(loc="upper right", title="shape", title_fontsize=8.5)
    F._save(fig, path)


def response_surfaces(run_id, path):
    """Deterministic refit of the CCD models from the saved data (identical
    fit to the analysis run); both models drawn on the same pair grid."""
    import analysis
    ctx = analysis.Ctx(run_id)
    surface, s = analysis.load_surface(ctx)
    best = {k: v for k, v in s["optimisation"]["augmented_enumeration_best"].items() if k != "predicted_tflops"}
    F.response_surfaces(surface, best, path,
                        title="Response surface (second-order + 3-factor-interaction model; fails the adequacy gate, "
                              "shown for interpretation only)")


if __name__ == "__main__":
    A = os.path.join(ROOT, "results", sys.argv[1], "analysis")
    out = os.path.join(ROOT, "results", sys.argv[1], "report_figures")
    os.makedirs(out, exist_ok=True)
    for M in (4096, 2048, 1024):
        screening_effects(A, M, os.path.join(out, f"fig_screening_effects_{M}.png"))
    sample_efficiency(A, os.path.join(out, "fig_sample_efficiency.png"))
    predicted_vs_measured(A, os.path.join(out, "fig_predicted_vs_measured.png"))
    shape_importance(A, os.path.join(out, "fig_factor_importance_by_shape.png"))
    response_surfaces(sys.argv[1], os.path.join(out, "fig_ccd_response_surfaces.png"))
    print("wrote", sorted(os.listdir(out)))
