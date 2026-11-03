"""Statistical analysis for the final protocol. Every decision rule is the one
frozen in contract.json; decisions are written to results/<run>/decisions/ for
pipeline.py, results and figures to results/<run>/analysis/.

  python analysis.py <run_id> screening <M>
  python analysis.py <run_id> ccd            (fit, diagnostics, adequacy gate, optimisation)
  python analysis.py <run_id> validation
  python analysis.py <run_id> exhaustive
  python analysis.py <run_id> recheck
  python analysis.py <run_id> final          (sample efficiency, cuBLAS table, shapes, costs)
"""
import itertools
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.optimize import minimize
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from design_space import FACTORS, FULL_LEVELS, enumerate_valid, is_valid
import designs
from designs import C as LOG2C, H as LOG2H, code
import figures

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = json.load(open(os.path.join(ROOT, "contract.json")))
ALPHA_SIG = 0.05
PRIMARY = 4096


class Ctx:
    def __init__(self, run_id):
        self.run_id = run_id
        self.dir = os.path.join(ROOT, "results", run_id)
        self.out = os.path.join(self.dir, "analysis")
        self.dec = os.path.join(self.dir, "decisions")
        os.makedirs(self.out, exist_ok=True)
        os.makedirs(self.dec, exist_ok=True)
        self.m = pd.read_csv(os.path.join(self.dir, "measurements.csv"))

    def phase(self, phase, M=PRIMARY, valid_only=True):
        d = self.m[(self.m.phase == phase) & (self.m.M == M)].copy()
        return d[d.validity == "valid"].copy() if valid_only else d

    def save_json(self, name, obj, decision=False):
        p = os.path.join(self.dec if decision else self.out, name)
        json.dump(obj, open(p, "w"), indent=2, default=_jsonable)
        return p

    def load_decision(self, name):
        return json.load(open(os.path.join(self.dec, name)))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def residual_diagnostics(fit, d):
    """Diagnostics with residuals ordered by actual run order."""
    r = pd.DataFrame({"order": d["order"].values, "resid": fit.resid.values,
                      "fitted": fit.fittedvalues.values})
    r = r.sort_values("order")
    infl = fit.get_influence()
    cooks = infl.cooks_distance[0]
    bp = het_breuschpagan(fit.resid, fit.model.exog)
    return {
        "durbin_watson_run_order": float(durbin_watson(r["resid"])),
        "shapiro_wilk_p": float(stats.shapiro(fit.resid)[1]),
        "breusch_pagan_p": float(bp[1]),
        "max_cooks_distance": float(np.max(cooks)),
        "n_cooks_gt_4_over_n": int(np.sum(cooks > 4 / len(cooks))),
        "residual_sd": float(np.sqrt(fit.scale)),
    }, r, cooks


def contamination_summary(d):
    return {"measurements": int(len(d)),
            "retried": int((d["attempts"] > 1).sum()),
            "still_contended": int(d["contended"].astype(bool).sum()),
            "sm_clocks_seen_mhz": sorted(set(d["sm_clock_end"].dropna().astype(int)))}


# ================================================================ screening

def screening(ctx, M):
    dall = ctx.phase("screening", M, valid_only=False)
    d = dall[dall.validity == "valid"].copy()
    for f in FACTORS:
        d[f"c_{f}"] = d[f].map(lambda v, f=f: round(code(f, v)))
    terms = " + ".join(f"c_{f}" for f in FACTORS)
    fit = smf.ols(f"tflops ~ C(block) + ({terms})**2", data=d).fit()
    an = sm.stats.anova_lm(fit, typ=2)
    sst = float(((d.tflops - d.tflops.mean()) ** 2).sum())
    an["pct_of_total_ss"] = 100 * an["sum_sq"] / sst

    eff = []
    ci = fit.conf_int()
    for t in fit.params.index:
        if t.startswith("c_"):
            eff.append({"term": t.replace("c_", ""), "effect_tflops": 2 * fit.params[t],
                        "ci95_low": 2 * ci.loc[t, 0], "ci95_high": 2 * ci.loc[t, 1],
                        "p_value": fit.pvalues[t]})
    eff = pd.DataFrame(eff).sort_values("p_value")

    contrib = {}
    for f in FACTORS:
        rows = [i for i in an.index if f"c_{f}" in i.split(":") or i == f"c_{f}"]
        contrib[f] = {"main_effect_pct": float(an.loc[f"c_{f}", "pct_of_total_ss"]),
                      "total_incl_interactions_pct": float(an.loc[rows, "pct_of_total_ss"].sum()),
                      "main_p": float(an.loc[f"c_{f}", "PR(>F)"]),
                      "min_p_any_term": float(an.loc[rows, "PR(>F)"].min())}

    diag, rdf, cooks = residual_diagnostics(fit, d)
    rep = d.groupby("point_id")["tflops"].agg(["mean", "std"])
    diag["replicate_cv_pct_median"] = float((100 * rep["std"] / rep["mean"]).median())
    diag["replicate_cv_pct_max"] = float((100 * rep["std"] / rep["mean"]).max())
    diag["pure_error_sd_tflops"] = float(np.sqrt((rep["std"] ** 2).mean()))

    # sensitivity: same model on log(latency)
    fit_log = smf.ols(f"np.log(median_ms) ~ C(block) + ({terms})**2", data=d).fit()
    an_log = sm.stats.anova_lm(fit_log, typ=2)
    sig_main = sorted(f for f in FACTORS if an.loc[f"c_{f}", "PR(>F)"] < ALPHA_SIG)
    sig_main_log = sorted(f for f in FACTORS if an_log.loc[f"c_{f}", "PR(>F)"] < ALPHA_SIG)

    summary = {
        "shape": M, "design": "2^(6-1) res VI, I=ABCDEF", "blocks": int(d.block.nunique()),
        "r_squared": fit.rsquared, "adj_r_squared": fit.rsquared_adj,
        "block_effect_p": float(an.loc["C(block)", "PR(>F)"]),
        "factor_contributions": contrib, "diagnostics": diag,
        "contamination": contamination_summary(dall),
        "invalid_measurements": dall[dall.validity != "valid"].status.value_counts().to_dict(),
        "significant_main_effects_tflops": sig_main,
        "significant_main_effects_log_latency": sig_main_log,
        "response_transform_changes_conclusion": sig_main != sig_main_log,
    }
    tag = f"screening_{M}"
    an.to_csv(os.path.join(ctx.out, f"{tag}_anova.csv"))
    eff.to_csv(os.path.join(ctx.out, f"{tag}_effects.csv"), index=False)
    pd.DataFrame(designs.alias_table()).to_csv(os.path.join(ctx.out, "screening_alias_structure.csv"), index=False)
    ctx.save_json(f"{tag}_summary.json", summary)
    figures.residual_panel(rdf, fit, cooks, f"Screening {M}³ residual diagnostics",
                           os.path.join(ctx.out, f"fig_{tag}_residuals.png"))

    if M == PRIMARY:
        retained = [f for f in FACTORS if contrib[f]["min_p_any_term"] < ALPHA_SIG]
        fixed = {}
        for f in FACTORS:
            if f not in retained:
                means = d.groupby(f)["tflops"].mean()
                fixed[f] = int(means.idxmax())
        ctx.save_json("retention.json", {
            "rule": CONTRACT["screening"]["retention_rule"],
            "retained": retained, "fixed": fixed,
            "evidence": {f: contrib[f] for f in FACTORS}}, decision=True)
        summary["retained"], summary["fixed"] = retained, fixed
    return summary


# ===================================================================== CCD

def ccd_terms(retained):
    lin = [f"x_{f}" for f in retained]
    quad = [f"I(x_{f}**2)" for f in retained]
    inter = [f"x_{a}:x_{b}" for a, b in itertools.combinations(retained, 2)]
    return lin, quad, inter


def third_order_terms(retained):
    """Estimable 3-factor interactions. With the 2^(6-1) (I = ABCDEF) core each
    3FI is aliased with its complement (e.g. BM:BN:BK = TM:TN:UNROLL), so one
    representative per alias pair is used: the 10 triples containing the first
    factor. Returns (terms, alias labels)."""
    if len(retained) == 6:
        triples = [t for t in itertools.combinations(retained, 3) if retained[0] in t]
    else:
        triples = list(itertools.combinations(retained, 3))
    terms = [":".join(f"x_{f}" for f in t) for t in triples]
    labels = {}
    for t, term in zip(triples, terms):
        comp = [f for f in retained if f not in t]
        labels[term] = "·".join(t) + (" = " + "·".join(comp) if len(retained) == 6 else "")
    return terms, labels


# response name -> (patsy lhs, include 3FI terms, response is log latency)
MODEL_SPECS = {
    "tflops": ("tflops", False, False),
    "log_latency": ("np.log(median_ms)", False, True),
    "tflops_3fi": ("tflops", True, False),
}


def model_formula(retained, response):
    y, third, _ = MODEL_SPECS[response]
    lin, quad, inter = ccd_terms(retained)
    extra = third_order_terms(retained)[0] if third else []
    return f"{y} ~ C(block) + " + " + ".join(lin + quad + inter + extra)


def add_coded(df, retained):
    for f in retained:
        df[f"x_{f}"] = df[f].map(lambda v, f=f: code(f, v))
    return df


class Surface:
    """Fitted second-order model; predictions average over block levels."""

    def __init__(self, fit, retained, fixed, response, blocks):
        self.fit, self.retained, self.fixed = fit, retained, fixed
        self.response, self.blocks = response, blocks

    def predict_tflops(self, cfgs, M=PRIMARY):
        base = pd.DataFrame(cfgs)
        base = add_coded(base, self.retained)
        preds = []
        for b in self.blocks:
            x = base.copy()
            x["block"] = b
            preds.append(np.asarray(self.fit.predict(x)))
        p = np.mean(preds, axis=0)
        if MODEL_SPECS[self.response][2]:
            return 2.0 * M ** 3 / (np.exp(p) / 1e3) / 1e12
        return p

    def predict_coded(self, xvec):
        cfg = {f: 2 ** (LOG2C[f] + xvec[i] * LOG2H[f]) for i, f in enumerate(self.retained)}
        cfg.update(self.fixed)
        return float(self.predict_tflops([cfg])[0])


def fit_ccd_model(d, retained, response):
    y = MODEL_SPECS[response][0]
    fq = smf.ols(model_formula(retained, response), data=d).fit()
    fs = smf.ols(f"{y} ~ C(block) + C(cfg_key)", data=d).fit()
    lof = sm.stats.anova_lm(fq, fs)
    lof_F, lof_p = float(lof["F"].iloc[1]), float(lof["Pr(>F)"].iloc[1])
    lof_ss, lof_df = float(lof["ss_diff"].iloc[1]), float(lof["df_diff"].iloc[1])
    yv = fq.model.endog
    sst = float(((yv - yv.mean()) ** 2).sum())
    model_ms = (sst - fq.ssr) / fq.df_model
    h = fq.get_influence().hat_matrix_diag
    press = float(np.sum((fq.resid / (1 - h)) ** 2))
    adequate = (lof_p >= 0.01) or (lof_ss / lof_df < 0.05 * model_ms)
    X = fq.model.exog
    return fq, {
        "response": response, "n_parameters": int(X.shape[1]),
        "design_rank": int(np.linalg.matrix_rank(X)),
        "condition_number": float(np.linalg.cond(X)),
        "r_squared": fq.rsquared, "adj_r_squared": fq.rsquared_adj,
        "press": press, "predicted_r_squared": 1 - press / sst,
        "lack_of_fit_F": lof_F, "lack_of_fit_p": lof_p,
        "lack_of_fit_ms": lof_ss / lof_df, "model_ms": model_ms,
        "lof_ms_pct_of_model_ms": 100 * (lof_ss / lof_df) / model_ms,
        "pure_error_df": float(fs.df_resid), "lack_of_fit_df": lof_df,
        "adequacy_gate_passed": bool(adequate),
    }


def lopo_cv(d, retained, response):
    """Leave-one-design-configuration-out CV (all replicates of a configuration
    held out together). Folds whose removal makes the model rank-deficient
    are reported separately, not silently dropped."""
    formula = model_formula(retained, response)
    errs, deficient = [], []
    for key, g in d.groupby("cfg_key"):
        tr = d[d.cfg_key != key]
        fit = smf.ols(formula, data=tr).fit()
        if np.linalg.matrix_rank(fit.model.exog) < fit.model.exog.shape[1]:
            deficient.append(key)
            continue
        pred = np.asarray(fit.predict(g))
        if MODEL_SPECS[response][2]:
            pred = 2.0 * PRIMARY ** 3 / (np.exp(pred) / 1e3) / 1e12
        errs.append({"cfg_key": key, "measured": g.tflops.mean(), "predicted": float(pred.mean())})
    e = pd.DataFrame(errs)
    e["abs_pct_err"] = 100 * (e.predicted - e.measured).abs() / e.measured
    return e, deficient


def canonical(surface):
    """Stationary point and eigenvalues of the quadratic part (coded units)."""
    p, R = surface.fit.params, surface.retained
    k = len(R)
    b = np.array([p[f"x_{f}"] for f in R])
    B = np.zeros((k, k))
    for i, f in enumerate(R):
        B[i, i] = p[f"I(x_{f} ** 2)"] if f"I(x_{f} ** 2)" in p else p[f"I(x_{f}**2)"]
    for (i, a), (j, c) in itertools.combinations(enumerate(R), 2):
        B[i, j] = B[j, i] = p[f"x_{a}:x_{c}"] / 2
    ev = np.linalg.eigvalsh(B)
    try:
        xs = -0.5 * np.linalg.solve(B, b)
    except np.linalg.LinAlgError:
        xs = np.full(k, np.nan)
    nature = "maximum" if np.all(ev < 0) else ("minimum" if np.all(ev > 0) else "saddle")
    return {"stationary_point_coded": dict(zip(R, xs)), "eigenvalues": ev.tolist(),
            "nature_for_tflops": nature,
            "stationary_point_inside_factorial_box": bool(np.all(np.abs(xs) <= 1))}


def lattice_region(retained, fixed, permitted_only=True):
    valid, _ = enumerate_valid(FULL_LEVELS)
    out = []
    for c in valid:
        if any(c[f] != v for f, v in fixed.items()):
            continue
        if permitted_only and any(abs(code(f, c[f])) > 1 + 1e-9 for f in retained):
            continue
        out.append(c)
    return out


def sqp(surface):
    R, fixed = surface.retained, surface.fixed

    def nat(x):
        n = {f: 2 ** (LOG2C[f] + x[i] * LOG2H[f]) for i, f in enumerate(R)}
        n.update({k: float(v) for k, v in fixed.items()})
        return n

    def threads(n):
        return n["BM"] * n["BN"] / (n["TM"] * n["TN"])

    cons = [
        {"type": "ineq", "fun": lambda x: 1024 - threads(nat(x))},
        {"type": "ineq", "fun": lambda x: 49152 - 4 * (nat(x)["BM"] * nat(x)["BK"] + nat(x)["BK"] * nat(x)["BN"])},
        {"type": "ineq", "fun": lambda x: nat(x)["BM"] * nat(x)["BK"] - threads(nat(x))},
        {"type": "ineq", "fun": lambda x: nat(x)["BK"] * nat(x)["BN"] - threads(nat(x))},
    ]
    bounds = [(-1.0, 1.0)] * len(R)
    rng = np.random.default_rng(CONTRACT["randomisation"]["seed"])
    best, n_success = None, 0
    for _ in range(30):
        x0 = rng.uniform(-1, 1, len(R))
        res = minimize(lambda x: -surface.predict_coded(x), x0, method="SLSQP",
                       bounds=bounds, constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
        feas = all(c["fun"](res.x) >= -1e-6 for c in cons)
        if res.success and feas:
            n_success += 1
            if best is None or res.fun < best.fun:
                best = res
    x = dict(zip(R, best.x))
    projected = {f: designs.snap(f, LOG2C[f] + x[f] * LOG2H[f]) for f in R}
    projected.update(fixed)
    neigh = []
    choices = []
    for f in FACTORS:
        if f in fixed:
            choices.append([fixed[f]])
            continue
        lv = FULL_LEVELS[f]
        i = lv.index(projected[f])
        choices.append([lv[j] for j in (i - 1, i, i + 1)
                        if 0 <= j < len(lv) and abs(code(f, lv[j])) <= 1 + 1e-9])
    for combo in itertools.product(*choices):
        c = dict(zip(FACTORS, combo))
        if is_valid(c)[0]:
            neigh.append(c)
    return {"continuous_optimum_coded": x, "predicted_tflops_continuous": -best.fun,
            "n_successful_starts": n_success, "projected": {f: projected[f] for f in FACTORS},
            "projected_feasible": is_valid({f: projected[f] for f in FACTORS})[0]}, neigh


def ccd(ctx):
    ret = ctx.load_decision("retention.json")
    R, fixed = ret["retained"], ret["fixed"]
    dall = ctx.phase("ccd", valid_only=False)
    d = add_coded(dall[dall.validity == "valid"].copy(), R)
    blocks = sorted(d.block.unique())

    fits = {resp: fit_ccd_model(d, R, resp) for resp in MODEL_SPECS}
    surfaces = {resp: Surface(f, R, fixed, resp, blocks) for resp, (f, _) in fits.items()}

    model_reports = {}
    for resp, (fit, st) in fits.items():
        diag, rdf, cooks = residual_diagnostics(fit, d)
        cv, deficient = lopo_cv(d, R, resp)
        cv.to_csv(os.path.join(ctx.out, f"ccd_lopo_cv_{resp}.csv"), index=False)
        coef = pd.DataFrame({"coef": fit.params, "se": fit.bse, "p": fit.pvalues,
                             "ci95_low": fit.conf_int()[0], "ci95_high": fit.conf_int()[1]})
        coef.to_csv(os.path.join(ctx.out, f"ccd_coefficients_{resp}.csv"))
        if resp != "log_latency":
            title = {"tflops": "Second-order (quadratic) model",
                     "tflops_3fi": "Quadratic + 3-factor-interaction model"}[resp]
            figures.residual_panel(rdf, fit, cooks, f"CCD residual diagnostics: {title}",
                                   os.path.join(ctx.out, f"fig_ccd_residuals_{resp}.png"))
        model_reports[resp] = {**st, "diagnostics": diag,
                               "lopo_cv": {"folds_evaluated": int(len(cv)), "rank_deficient_folds": deficient,
                                           "rmse_tflops": float(np.sqrt(((cv.predicted - cv.measured) ** 2).mean())),
                                           "median_abs_pct_err": float(cv.abs_pct_err.median()),
                                           "max_abs_pct_err": float(cv.abs_pct_err.max())}}

    tfi = fits["tflops_3fi"][0]
    terms3, labels3 = third_order_terms(R)
    model_reports["tflops_3fi"]["third_order_terms"] = [
        {"term": labels3[t], "coef": float(tfi.params[t]), "p": float(tfi.pvalues[t])} for t in terms3]
    sse_q, sse_3 = fits["tflops"][0].ssr, tfi.ssr
    model_reports["tflops_3fi"]["share_of_quadratic_residual_ss_explained_by_3fi_pct"] = 100 * (sse_q - sse_3) / sse_q

    preregistered_passed = (model_reports["tflops"]["adequacy_gate_passed"]
                            or model_reports["log_latency"]["adequacy_gate_passed"])
    chosen = "tflops_3fi"
    surface = surfaces[chosen]
    quad = surfaces["tflops"]

    summary = {"retained": R, "fixed": fixed, "n_measurements": len(d),
               "distinct_configs": int(d.cfg_key.nunique()),
               "models": model_reports,
               "preregistered_gate": {
                   "second_order_tflops_passed": model_reports["tflops"]["adequacy_gate_passed"],
                   "second_order_log_latency_passed": model_reports["log_latency"]["adequacy_gate_passed"],
                   "outcome": "passed" if preregistered_passed else
                   "FAILED: the pre-registered second-order model is statistically inadequate (contract: do not optimise it)"},
               "deviation": {
                   "date": "2026-10-28", "decided_by": "project owner, after seeing the failed gate",
                   "action": "report the failed gate as the answer to RQ2; then, as a further investigation, "
                             "augment the second-order model with the 10 estimable 3-factor-interaction alias "
                             "pairs (same CCD data, no new measurements), re-run the same gate and CV, and use "
                             "the augmented model only as a proposal generator; the final configuration is "
                             "chosen by measured validation. The second-order model's own top-3 are added to "
                             "the validation set so its recommendation is tested too.",
                   "augmented_model_passes_gate": model_reports["tflops_3fi"]["adequacy_gate_passed"]},
               "model_used": chosen,
               "contamination": contamination_summary(dall)}
    summary["canonical_analysis_second_order"] = canonical(quad)

    # --- optimisation on the augmented model, validated by measurement ---
    region = lattice_region(R, fixed, permitted_only=True)
    pr = surface.predict_tflops(region)
    ranked = sorted(zip(pr, map(json.dumps, region)), key=lambda t: -t[0])
    top5 = [json.loads(c) for _, c in ranked[:5]]
    pq = quad.predict_tflops(region)
    quad_top3 = [json.loads(c) for _, c in sorted(zip(pq, map(json.dumps, region)), key=lambda t: -t[0])[:3]]
    sqp_info, neigh = sqp(surface)
    sqp_quad, _ = sqp(quad)
    neigh_pred = surface.predict_tflops(neigh) if neigh else []
    neigh_ranked = [json.loads(c) for _, c in sorted(zip(neigh_pred, map(json.dumps, neigh)), key=lambda t: -t[0])]

    cands, index = [], {}
    def add(c, src, cap):
        k = json.dumps({f: c[f] for f in FACTORS}, sort_keys=True)
        if k in index:
            if src not in index[k]["source"]:
                index[k]["source"] += "+" + src
            return
        if len(cands) >= cap:
            return
        entry = {**{f: c[f] for f in FACTORS}, "source": src}
        cands.append(entry)
        index[k] = entry
    for c in top5:
        add(c, "augmented_top5", 15)
    for c in quad_top3:
        add(c, "quadratic_top3", 15)
    for c in neigh_ranked:
        add(c, "augmented_sqp_neighbourhood", 15)
    pc = surface.predict_tflops([{f: c[f] for f in FACTORS} for c in cands])
    pcq = quad.predict_tflops([{f: c[f] for f in FACTORS} for c in cands])
    for i, c in enumerate(cands):
        c["candidate_id"] = f"V{i + 1:02d}"
        c["predicted_tflops"] = float(pc[i])
        c["predicted_tflops_quadratic"] = float(pcq[i])

    everywhere = lattice_region(R, fixed, permitted_only=False)
    allpred = pd.DataFrame(everywhere)
    allpred["predicted_tflops"] = surface.predict_tflops(everywhere)
    allpred["predicted_tflops_quadratic"] = quad.predict_tflops(everywhere)
    allpred["in_permitted_region"] = [all(abs(code(f, c[f])) <= 1 + 1e-9 for f in R) for c in everywhere]
    allpred.to_csv(os.path.join(ctx.out, "model_predictions_all_feasible.csv"), index=False)
    ext = allpred.sort_values("predicted_tflops", ascending=False).iloc[0]

    summary["optimisation"] = {
        "permitted_region_size": len(region),
        "augmented_enumeration_best": {**top5[0], "predicted_tflops": float(ranked[0][0])},
        "quadratic_enumeration_best": {**quad_top3[0], "predicted_tflops": float(max(pq))},
        "sqp_augmented": sqp_info, "sqp_quadratic": sqp_quad,
        "sqp_neighbourhood_size": len(neigh),
        "validation_candidates": len(cands),
        "extrapolated_optimum_outside_region": {**{f: int(ext[f]) for f in FACTORS},
                                                "predicted_tflops": float(ext.predicted_tflops)},
    }
    ctx.save_json("validation_candidates.json", {"candidates": cands}, decision=True)
    ctx.save_json("ccd_summary.json", summary)
    figures.response_surfaces(surface, top5[0], os.path.join(ctx.out, "fig_ccd_response_surfaces.png"))
    return summary


def load_surface(ctx):
    """Refit the chosen CCD model (deterministic) for later phases."""
    s = json.load(open(os.path.join(ctx.out, "ccd_summary.json")))
    R, fixed = s["retained"], s["fixed"]
    d = add_coded(ctx.phase("ccd"), R)
    fit, _ = fit_ccd_model(d, R, s["model_used"])
    return Surface(fit, R, fixed, s["model_used"], sorted(d.block.unique())), s


# ============================================================== validation

def validation(ctx):
    cands = pd.DataFrame(ctx.load_decision("validation_candidates.json")["candidates"])
    dall = ctx.phase("validation", valid_only=False)
    d = dall[dall.validity == "valid"]
    g = d.groupby("point_id").agg(measured_tflops=("tflops", "mean"), sd=("tflops", "std"),
                                  blocks=("tflops", "size"), median_ms=("median_ms", "mean"))
    v = cands.set_index("candidate_id").join(g)
    v["pred_err_pct"] = 100 * (v.predicted_tflops - v.measured_tflops) / v.measured_tflops
    v["pred_err_pct_quadratic"] = 100 * (v.predicted_tflops_quadratic - v.measured_tflops) / v.measured_tflops
    v["measured_rank"] = v.measured_tflops.rank(ascending=False).astype(int)
    v["augmented_rank"] = v.predicted_tflops.rank(ascending=False).astype(int)
    v["quadratic_rank"] = v.predicted_tflops_quadratic.rank(ascending=False).astype(int)
    v = v.sort_values("measured_tflops", ascending=False)
    v.to_csv(os.path.join(ctx.out, "validation_results.csv"))
    best = v.iloc[0]
    sel = {f: int(best[f]) for f in FACTORS}
    out = {"selected_candidate": best.name, "rsm_selected": sel,
           "measured_tflops": float(best.measured_tflops), "predicted_tflops": float(best.predicted_tflops),
           "prediction_error_pct_selected": float(best.pred_err_pct),
           "selected_source": best.source,
           "mean_abs_prediction_error_pct_candidates": float(v.pred_err_pct.abs().mean()),
           "mean_abs_prediction_error_pct_candidates_quadratic": float(v.pred_err_pct_quadratic.abs().mean()),
           "spearman_rank_measured_vs_augmented": float(stats.spearmanr(v.measured_tflops, v.predicted_tflops).statistic),
           "spearman_rank_measured_vs_quadratic": float(stats.spearmanr(v.measured_tflops, v.predicted_tflops_quadratic).statistic),
           "selected_was_augmented_rank_1": bool(best.augmented_rank == 1),
           "quadratic_top1_measured_tflops": float(v[v.source.str.contains("quadratic_top3")]
                                                   .sort_values("predicted_tflops_quadratic", ascending=False)
                                                   .measured_tflops.iloc[0]),
           "contamination": contamination_summary(dall)}
    ctx.save_json("rsm_selection.json", out, decision=True)
    ctx.save_json("validation_summary.json", out)
    return out


# ============================================================== exhaustive

def exhaustive(ctx):
    dall = ctx.phase("exhaustive", valid_only=False)
    d = dall[dall.validity == "valid"]
    top = d.sort_values("tflops", ascending=False).head(20)
    out = {"attempted": len(dall), "valid": len(d),
           "by_validity": dall.validity.value_counts().to_dict(),
           "numerically_invalid": dall[dall.validity == "numerically_invalid"].cfg_key.tolist(),
           "spilling_configs": int((d.local_bytes_per_thread > 0).sum()),
           "max_rel_fro_err_valid": float(d.rel_fro_err.max()),
           "contamination": contamination_summary(dall)}
    ctx.save_json("exhaustive_top20.json",
                  {"top": top[["cfg_key"] + FACTORS + ["tflops"]].to_dict("records")}, decision=True)
    ctx.save_json("exhaustive_summary.json", out)
    return out


def recheck(ctx):
    top = pd.DataFrame(ctx.load_decision("exhaustive_top20.json")["top"])
    m = ctx.m[(ctx.m.validity == "valid") & (ctx.m.M == PRIMARY)
              & ctx.m.phase.isin(["exhaustive", "exhaustive_recheck"]) & ctx.m.cfg_key.isin(top.cfg_key)]
    g = m.groupby("cfg_key").agg(tflops_mean=("tflops", "mean"), tflops_sd=("tflops", "std"),
                                 n=("tflops", "size"), median_ms=("median_ms", "mean")).sort_values(
        "tflops_mean", ascending=False)
    g.to_csv(os.path.join(ctx.out, "exhaustive_top20_recheck.csv"))
    best_key = g.index[0]
    best = m[m.cfg_key == best_key].iloc[0]
    sel = ctx.load_decision("rsm_selection.json")["rsm_selected"]
    t1, t2 = m[m.cfg_key == g.index[0]].tflops, m[m.cfg_key == g.index[1]].tflops
    out = {"exhaustive_best": {f: int(best[f]) for f in FACTORS}, "rsm_selected": sel,
           "t_min_ms": float(g.iloc[0].median_ms), "best_tflops": float(g.iloc[0].tflops_mean),
           "best_vs_second_welch_p": float(stats.ttest_ind(t1, t2, equal_var=False).pvalue),
           "top_rank_changed_by_recheck": bool(best_key != top.cfg_key.iloc[0])}
    ctx.save_json("final_selection.json", out, decision=True)
    ctx.save_json("recheck_summary.json", out)
    return out


# =================================================================== final

def final(ctx):
    res = {}
    key = lambda c: "_".join(f"{k}{c[k]}" for k in FACTORS)
    ex = ctx.phase("exhaustive")
    pool = ex.groupby("cfg_key").tflops.mean()
    lat = ex.groupby("cfg_key").median_ms.mean()
    rc = pd.read_csv(os.path.join(ctx.out, "exhaustive_top20_recheck.csv"), index_col=0)
    fs = ctx.load_decision("final_selection.json")
    sel_key, best_key = key(fs["rsm_selected"]), key(fs["exhaustive_best"])
    t_min = fs["t_min_ms"]

    # --- equal-budget random search (contract: baselines.random_search) ---
    doe = ctx.m[(ctx.m.M == PRIMARY) & ctx.m.phase.isin(["screening", "ccd", "validation"])]
    doe_order = doe.sort_values("order").drop_duplicates("cfg_key").cfg_key.tolist()
    B = len(doe_order)
    vals = pool.values
    rng = np.random.default_rng(CONTRACT["randomisation"]["seed"])
    best_at_B = np.array([vals[rng.choice(len(vals), B, replace=False)].max() for _ in range(10000)])
    sel_val = float(pool[sel_key])
    sel_lat = float(rc.loc[sel_key, "median_ms"]) if sel_key in rc.index else float(lat[sel_key])
    gap = lambda t: 100 * (t - t_min) / t_min
    perms = np.array([rng.permutation(vals) for _ in range(2000)])
    curve = np.maximum.accumulate(perms, axis=1)
    rs_curve = pd.DataFrame({"n": np.arange(1, len(vals) + 1), "median": np.median(curve, 0),
                             "p10": np.percentile(curve, 10, 0), "p90": np.percentile(curve, 90, 0)})
    doe_traj = np.maximum.accumulate([pool[k] for k in doe_order])
    rs_curve.to_csv(os.path.join(ctx.out, "random_search_curve.csv"), index=False)
    pd.DataFrame({"n": np.arange(1, B + 1), "cfg_key": doe_order, "best_so_far": doe_traj}).to_csv(
        os.path.join(ctx.out, "doe_rsm_trajectory.csv"), index=False)
    # evaluations random search needs to match DOE-RSM's selected config (median trial)
    hit = (curve >= sel_val).argmax(axis=1) + 1
    hit[~(curve >= sel_val).any(axis=1)] = len(vals)
    res["sample_efficiency"] = {
        "doe_rsm_unique_evaluations_B": B, "feasible_space": int(len(vals)),
        "doe_rsm_selected": sel_key, "doe_rsm_selected_tflops_exhaustive_measurement": sel_val,
        "doe_rsm_best_evaluated_tflops": float(doe_traj[-1]),
        "random_search_at_B": {"median_best": float(np.median(best_at_B)),
                               "p10": float(np.percentile(best_at_B, 10)),
                               "p90": float(np.percentile(best_at_B, 90)),
                               "prob_random_matches_or_beats_doe_rsm": float((best_at_B >= sel_val).mean())},
        "random_search_evals_to_match_doe_rsm": {"median": float(np.median(hit)),
                                                 "p10": float(np.percentile(hit, 10)),
                                                 "p90": float(np.percentile(hit, 90))},
        "exhaustive": {"best": best_key, "t_min_ms": t_min, "best_tflops": fs["best_tflops"]},
        "optimality_gap_pct": {"doe_rsm_selected": gap(sel_lat),
                               "random_search_median_at_B": gap(2.0 * PRIMARY ** 3 / np.median(best_at_B) / 1e9),
                               "untuned_default": gap(float(lat["BM64_BN64_BK8_TM4_TN4_UNROLL1"]))},
        "doe_rsm_selected_rank_in_exhaustive": int((pool > sel_val).sum() + 1),
    }
    figures.sample_efficiency(rs_curve, doe_traj, sel_val, B, fs["best_tflops"],
                              os.path.join(ctx.out, "fig_sample_efficiency.png"))
    figures.budget_histogram(best_at_B, sel_val, B, os.path.join(ctx.out, "fig_equal_budget.png"))

    # --- extrapolation check, measured against the exhaustive sweep ---
    surface, ccd_s = load_surface(ctx)
    ap = pd.read_csv(os.path.join(ctx.out, "model_predictions_all_feasible.csv"))
    ap["cfg_key"] = ap.apply(lambda r: key(r), axis=1)
    ap["measured_tflops"] = ap.cfg_key.map(pool)
    ap = ap.dropna(subset=["measured_tflops"])
    ap["abs_pct_err"] = 100 * (ap.predicted_tflops - ap.measured_tflops).abs() / ap.measured_tflops
    ap["abs_pct_err_q"] = 100 * (ap.predicted_tflops_quadratic - ap.measured_tflops).abs() / ap.measured_tflops
    eo = ccd_s["optimisation"]["extrapolated_optimum_outside_region"]
    res["model_vs_ground_truth"] = {
        "inside_region_median_abs_pct_err": float(ap[ap.in_permitted_region].abs_pct_err.median()),
        "outside_region_median_abs_pct_err": float(ap[~ap.in_permitted_region].abs_pct_err.median()),
        "quadratic_inside_region_median_abs_pct_err": float(ap[ap.in_permitted_region].abs_pct_err_q.median()),
        "quadratic_outside_region_median_abs_pct_err": float(ap[~ap.in_permitted_region].abs_pct_err_q.median()),
        "spearman_all_feasible_augmented": float(stats.spearmanr(ap.predicted_tflops, ap.measured_tflops).statistic),
        "spearman_all_feasible_quadratic": float(stats.spearmanr(ap.predicted_tflops_quadratic, ap.measured_tflops).statistic),
        "n_inside": int(ap.in_permitted_region.sum()), "n_outside": int((~ap.in_permitted_region).sum()),
        "extrapolated_optimum": eo["predicted_tflops"],
        "extrapolated_optimum_measured": float(pool.get(key(eo), np.nan)),
    }
    figures.predicted_vs_measured(ap, os.path.join(ctx.out, "fig_predicted_vs_measured.png"))

    # --- cuBLAS comparison (contract: baselines.cublas) ---
    raw = pd.read_csv(os.path.join(ctx.dir, "raw_launches.csv"))
    rows = []
    for mode in ("default", "pedantic"):
        cm = ctx.m[ctx.m.phase == f"compare_{mode}"]
        for pid, g in cm.groupby("point_id"):
            r = raw[raw.order.isin(g.order)]
            tc, tb = r[r.impl == "custom"].latency_ms, r[r.impl == "cublas"].latency_ms
            fl = 2.0 * PRIMARY ** 3
            rows.append({"cublas_mode": mode, "config": pid, "cfg_key": g.cfg_key.iloc[0],
                         "custom_median_ms": tc.median(), "custom_iqr_ms": tc.quantile(.75) - tc.quantile(.25),
                         "cublas_median_ms": tb.median(), "cublas_iqr_ms": tb.quantile(.75) - tb.quantile(.25),
                         "custom_tflops": fl / (tc.median() / 1e3) / 1e12,
                         "cublas_tflops": fl / (tb.median() / 1e3) / 1e12,
                         "speed_ratio_S": tb.median() / tc.median(),
                         "attained_pct": 100 * tb.median() / tc.median(),
                         "n_launches_each": int(len(tc)),
                         "rel_fro_err_vs_cublas": float(g.rel_fro_err.iloc[0]),
                         "contended_blocks": int(g.contended.astype(bool).sum())})
    cmp = pd.DataFrame(rows)
    cmp.to_csv(os.path.join(ctx.out, "final_cublas_comparison.csv"), index=False)
    res["cublas_comparison"] = cmp.to_dict("records")
    figures.final_comparison(cmp, os.path.join(ctx.out, "fig_final_comparison.png"))

    # --- factor importance across shapes ---
    shp = {}
    for M in CONTRACT["screening"]["shapes"]:
        p = os.path.join(ctx.out, f"screening_{M[0]}_summary.json")
        if os.path.exists(p):
            s = json.load(open(p))
            shp[M[0]] = {f: s["factor_contributions"][f]["total_incl_interactions_pct"] for f in FACTORS}
    if len(shp) > 1:
        df = pd.DataFrame(shp)
        taus = {f"{PRIMARY}_vs_{M}": float(stats.kendalltau(df[PRIMARY], df[M]).statistic)
                for M in df.columns if M != PRIMARY}
        res["shapes"] = {"total_contribution_pct": shp, "kendall_tau_rank_agreement": taus}
        figures.shape_importance(df, os.path.join(ctx.out, "fig_factor_importance_by_shape.png"))

    # --- registers / spilling (synopsis problem statement, bullet 2) ---
    res["registers"] = {"configs_with_spills": int((ex.local_bytes_per_thread > 0).sum()),
                        "regs_range": [int(ex.regs_per_thread.min()), int(ex.regs_per_thread.max())],
                        "spearman_tflops_vs_regs": float(stats.spearmanr(ex.tflops, ex.regs_per_thread).statistic),
                        "spearman_tflops_vs_occupancy": float(stats.spearmanr(ex.tflops, ex.occupancy).statistic)}
    figures.registers(ex, os.path.join(ctx.out, "fig_registers.png"))

    # --- costs ---
    cl = pd.read_csv(os.path.join(ctx.dir, "compile_log.csv"))
    res["offline_cost"] = {"unique_compiles": int(len(cl)), "compile_s_total": float(cl.compile_s.sum()),
                           "compile_s_median": float(cl.compile_s.median()),
                           "measurements_total": int(len(ctx.m)),
                           "retried_measurements": int((ctx.m.attempts > 1).sum()),
                           "still_contended_measurements": int(ctx.m.contended.astype(bool).sum())}
    ctx.save_json("final_summary.json", res)
    return res


if __name__ == "__main__":
    run_id, what = sys.argv[1], sys.argv[2]
    ctx = Ctx(run_id)
    fn = {"screening": lambda: screening(ctx, int(sys.argv[3])), "ccd": lambda: ccd(ctx),
          "validation": lambda: validation(ctx), "exhaustive": lambda: exhaustive(ctx),
          "recheck": lambda: recheck(ctx), "final": lambda: final(ctx)}[what]
    out = fn()
    print(json.dumps(out, indent=2, default=_jsonable))
