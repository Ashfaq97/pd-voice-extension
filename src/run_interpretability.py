"""Phase 5 -- what the honest model actually uses.

    python src/run_interpretability.py

SHAP is computed on LOGISTIC REGRESSION, not XGBoost. LR is the model that
wins under subject-level validation on the UCI set, and a linear model with
real coefficients is easier to defend clinically. XGBoost appears here only
in the leaky-versus-honest comparison, never as a headline importance
ranking.

Three outputs.

1. Global SHAP importance for LR under subject-level training. SHAP values
   are computed per outer fold, on that fold's held-out rows, using that
   fold's own fitted scaler -- so no scaling information crosses a fold
   boundary, exactly as in the CV itself.

2. Leaky versus subject-level feature reliance. The same procedure is run
   under regime A and regime B and the resulting importance rankings are
   compared (Spearman). If a model leans on different features when it can
   exploit speaker identity, that is direct evidence it was learning the
   speaker rather than the disease. Reported for LR and, as the contrast,
   for XGBoost.

3. Standardised LR coefficients with 95% confidence intervals, fitted with
   CLUSTER-ROBUST standard errors grouped by subject. Ordinary SEs would
   treat 6 recordings from one speaker as 6 independent observations, which
   is the same mistake at the inference level that the paper is about at the
   validation level.

   Collinearity. Both feature sets contain definitional duplicates: Jitter:DDP
   is exactly 3 x MDVP:RAP and Shimmer:DDA is exactly 3 x Shimmer:APQ3
   (r = 1.000000 to six decimals in both datasets), and in Sakar
   Number_of_periods tracks Number_of_pulses at r = 0.999. Fitting an
   unpenalised logistic regression on the raw set produces coefficients with
   confidence intervals spanning thousands of units -- not a result, an
   artefact. The inference model therefore drops the derived duplicate of
   each near-perfectly-collinear pair and reports VIFs for what remains.

   Removing the exact duplicates is not enough: the remaining jitter measures
   are near-substitutes for one another, as are the shimmer measures, leaving
   VIFs in the hundreds. A coefficient with VIF 741 is not interpretable
   whatever its confidence interval says. The coefficient table is therefore
   reported twice -- on the duplicate-pruned set, and on a VIF-pruned set
   built by dropping the highest-VIF feature until every VIF is below 10.
   Only the second is interpretable feature by feature.

   This affects the coefficient table ONLY. The prediction models in phases
   2-4 keep the full feature set, because that is what the published
   pipelines use and what the paper is measuring. Worth noting in its own
   right: the most-used PD voice dataset ships 22 "features" of which two are
   exact multiples of two others, so its effective dimensionality is 20.

Writes results/tables/shap_importance.csv, lr_coefficients.csv,
collinearity_report.csv, and four figures to results/figures/.
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from models import get_fixed_models
from cv_core import outer_splits

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                     "font.size": 9, "axes.grid": False})

# Sakar-2018 is deliberately excluded here. SHAP over 752 features on 252
# subjects produces a ranking dominated by collinear near-duplicates, and the
# coefficient model would prune almost all of them -- neither is informative.
# The interpretability claim in the paper rests on the two corpora with
# interpretable feature sets.
DATASETS = {
    "UCI/MDVP": (C.UCI_OUT, C.UCI_FEATURES, "uci"),
    "Sakar-2013": (C.SAKAR_TRAIN_OUT, C.SAKAR_FEATURES, "sakar"),
}


def fold_shap(X, y, g, features, model_name, regime, seed=C.SEED):
    """Per-fold SHAP values on held-out rows. Returns (values, rows_used)."""
    vals, rows = [], []
    for tr, te in outer_splits(regime, X, y, g, 5, seed):
        pipe = get_fixed_models()[model_name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X[tr], y[tr])
        scaler, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
        Xtr_s, Xte_s = scaler.transform(X[tr]), scaler.transform(X[te])

        if model_name == "Logistic Regression":
            expl = shap.LinearExplainer(clf, Xtr_s)
        else:
            expl = shap.TreeExplainer(clf)
        sv = expl.shap_values(Xte_s)
        if isinstance(sv, list):          # older binary-classifier convention
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                  # (n, features, classes)
            sv = sv[:, :, 1]
        vals.append(sv)
        rows.append(Xte_s)
    return np.vstack(vals), np.vstack(rows)


def prune_collinear(X, features, thresh=0.999):
    """Drop the later member of each near-perfectly-correlated pair.

    Returns (kept_indices, kept_names, report_rows). Feature order in
    config.py places the derived quantity after its parent (RAP before DDP,
    APQ3 before DDA), so keeping the earlier member keeps the primitive.
    """
    corr = np.corrcoef(np.asarray(X, float), rowvar=False)
    drop, report = set(), []
    for i in range(len(features)):
        if i in drop:
            continue
        for j in range(i + 1, len(features)):
            if j in drop:
                continue
            r = abs(corr[i, j])
            if r > thresh:
                drop.add(j)
                report.append({"kept": features[i], "dropped": features[j],
                               "abs_corr": round(float(r), 6)})
    keep = [i for i in range(len(features)) if i not in drop]
    return keep, [features[i] for i in keep], report


def lr_coefficients(X, y, g, features):
    """Standardised LR coefficients with subject-clustered 95% CIs."""
    Xs = StandardScaler().fit_transform(X)
    design = sm.add_constant(Xs, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.Logit(y, design).fit(
            disp=0, method="bfgs", maxiter=2000,
            cov_type="cluster", cov_kwds={"groups": pd.factorize(g)[0]},
        )
    ci = fit.conf_int()
    out = pd.DataFrame({
        "feature": ["(intercept)"] + list(features),
        "coef_standardised": fit.params,
        "std_err_clustered": fit.bse,
        "ci_lo": ci[:, 0], "ci_hi": ci[:, 1],
        "p_value": fit.pvalues,
    }).iloc[1:].reset_index(drop=True)
    # Benjamini-Hochberg across the features in this model. Without it,
    # finding 1-2 "significant" features among 20 is the null expectation.
    out["p_value_bh"] = multipletests(out["p_value"], method="fdr_bh")[1]
    return out


def prune_by_vif(X, features, max_vif=10.0):
    """Drop the highest-VIF feature until every remaining VIF is below max_vif."""
    keep = list(range(len(features)))
    dropped = []
    while len(keep) > 1:
        v = vifs(X[:, keep], [features[i] for i in keep])
        worst = v.vif.idxmax()
        if v.vif[worst] <= max_vif:
            break
        dropped.append((features[keep[worst]], float(v.vif[worst])))
        keep.pop(worst)
    return keep, [features[i] for i in keep], dropped


def vifs(X, features):
    Xs = StandardScaler().fit_transform(X)
    Xs = sm.add_constant(Xs, has_constant="add")
    return pd.DataFrame({
        "feature": features,
        "vif": [variance_inflation_factor(Xs, i + 1) for i in range(len(features))],
    })


def main() -> None:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    imp_rows, coef_rows, collin_rows = [], [], []

    for ds_name, (path, features, slug) in DATASETS.items():
        df = pd.read_csv(path)
        X = df[features].values
        y = df["label"].values
        g = df["subject"].values
        print(f"\n=== {ds_name} ===")

        shap_by = {}
        for model_name in ["Logistic Regression", "XGBoost"]:
            for regime in ["A_leaky_recording", "B_group_kfold"]:
                sv, rows_s = fold_shap(X, y, g, features, model_name, regime)
                mean_abs = np.abs(sv).mean(axis=0)
                shap_by[(model_name, regime)] = (sv, rows_s, mean_abs)
                for f, v in zip(features, mean_abs):
                    imp_rows.append({"dataset": ds_name, "model": model_name,
                                     "regime": regime, "feature": f,
                                     "mean_abs_shap": v})

            a = shap_by[(model_name, "A_leaky_recording")][2]
            b = shap_by[(model_name, "B_group_kfold")][2]
            rho, p = spearmanr(a, b)
            top_a = [features[i] for i in np.argsort(-a)[:5]]
            top_b = [features[i] for i in np.argsort(-b)[:5]]
            print(f"  {model_name:20s} leaky-vs-honest importance "
                  f"Spearman rho={rho:+.3f} (p={p:.3g})")
            print(f"      top-5 leaky : {top_a}")
            print(f"      top-5 honest: {top_b}")
            print(f"      overlap: {len(set(top_a) & set(top_b))}/5")

        # --- figure: SHAP summary, LR under subject-level training --------
        sv, rows_s, _ = shap_by[("Logistic Regression", "B_group_kfold")]
        plt.figure()
        shap.summary_plot(sv, rows_s, feature_names=features, show=False,
                          max_display=15)
        plt.title(f"{ds_name} - SHAP, logistic regression, subject-level folds",
                  fontsize=10)
        plt.savefig(C.FIGURES / f"fig4_shap_summary_{slug}.png")
        plt.close()

        # --- figure: leaky vs honest reliance, both models -----------------
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=False)
        for ax, model_name in zip(axes, ["Logistic Regression", "XGBoost"]):
            a = shap_by[(model_name, "A_leaky_recording")][2]
            b = shap_by[(model_name, "B_group_kfold")][2]
            order = np.argsort(-(a + b))[:12][::-1]
            ypos = np.arange(len(order))
            ax.barh(ypos - 0.2, a[order] / a.sum(), height=0.4,
                    label="recording-level (leaky)", color="#c44e52")
            ax.barh(ypos + 0.2, b[order] / b.sum(), height=0.4,
                    label="subject-level", color="#4c72b0")
            ax.set_yticks(ypos)
            ax.set_yticklabels([features[i] for i in order], fontsize=7)
            rho = spearmanr(a, b).statistic
            ax.set_title(f"{model_name}  (rho = {rho:+.2f})", fontsize=9)
            ax.set_xlabel("share of total mean |SHAP|")
        handles = [Patch(facecolor="#c44e52", label="recording-level (leaky)"),
                   Patch(facecolor="#4c72b0", label="subject-level")]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                   fontsize=8, bbox_to_anchor=(0.5, -0.03))
        fig.suptitle(f"{ds_name} - feature reliance, leaky vs subject-level",
                     fontsize=10)
        fig.savefig(C.FIGURES / f"fig4b_shap_leaky_vs_honest_{slug}.png")
        plt.close(fig)

        # --- standardised coefficients ------------------------------------
        keep, kept_names, report = prune_collinear(X, features)
        for row in report:
            row["dataset"] = ds_name
            collin_rows.append(row)
        if report:
            print("  dropped from the inference model as definitionally "
                  "collinear:")
            for row in report:
                print(f"      {row['dropped']:24s} (r={row['abs_corr']:.6f} "
                      f"with {row['kept']})")

        Xk = X[:, keep]
        coefs = lr_coefficients(Xk, y, g, kept_names)
        coefs = coefs.merge(vifs(Xk, kept_names), on="feature", how="left")
        coefs.insert(0, "dataset", ds_name)
        coef_rows.append(coefs)

        top = coefs.reindex(coefs.coef_standardised.abs().sort_values(
            ascending=False).index).head(8)
        print(f"  top standardised LR coefficients "
              f"(subject-clustered 95% CI, max VIF {coefs.vif.max():.1f}):")
        for r in top.itertuples():
            star = "*" if (r.ci_lo > 0) or (r.ci_hi < 0) else " "
            print(f"      {star} {r.feature:22s} {r.coef_standardised:+.3f} "
                  f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  p={r.p_value:.3g}"
                  f"  vif={r.vif:.1f}")
        n_sig = int(((coefs.ci_lo > 0) | (coefs.ci_hi < 0)).sum())
        print(f"      {n_sig} of {len(coefs)} coefficients exclude zero "
              f"(not interpretable individually: max VIF {coefs.vif.max():.0f})")

        # --- VIF-pruned, interpretable model ------------------------------
        vkeep, vnames, vdropped = prune_by_vif(Xk, kept_names, max_vif=10.0)
        vcoefs = lr_coefficients(Xk[:, vkeep], y, g, vnames)
        vcoefs = vcoefs.merge(vifs(Xk[:, vkeep], vnames), on="feature", how="left")
        vcoefs.insert(0, "dataset", ds_name)
        vcoefs.insert(1, "model_set", "vif_pruned")
        coefs.insert(1, "model_set", "duplicate_pruned")
        coef_rows.append(vcoefs)

        print(f"  VIF-pruned model: {len(vnames)} of {len(kept_names)} features "
              f"retained (all VIF < 10), {len(vdropped)} dropped")
        vsig = vcoefs[(vcoefs.ci_lo > 0) | (vcoefs.ci_hi < 0)]
        for r in vcoefs.reindex(vcoefs.coef_standardised.abs()
                                .sort_values(ascending=False).index).head(8).itertuples():
            star = "*" if (r.ci_lo > 0) or (r.ci_hi < 0) else " "
            print(f"      {star} {r.feature:22s} {r.coef_standardised:+.3f} "
                  f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  p={r.p_value:.3g}"
                  f"  vif={r.vif:.1f}")
        n_bh = int((vcoefs.p_value_bh < 0.05).sum())
        print(f"      {len(vsig)} of {len(vcoefs)} coefficients exclude zero "
              f"with subject-clustered SEs; {n_bh} survive "
              f"Benjamini-Hochberg at FDR 0.05")

    pd.DataFrame(imp_rows).to_csv(C.TABLES / "shap_importance.csv", index=False)
    pd.concat(coef_rows, ignore_index=True).to_csv(
        C.TABLES / "lr_coefficients.csv", index=False)
    pd.DataFrame(collin_rows).to_csv(
        C.TABLES / "collinearity_report.csv", index=False)
    print(f"\nwrote shap_importance.csv, lr_coefficients.csv, "
          f"collinearity_report.csv and 4 figures to "
          f"{C.FIGURES.relative_to(C.ROOT)}/")


if __name__ == "__main__":
    main()
