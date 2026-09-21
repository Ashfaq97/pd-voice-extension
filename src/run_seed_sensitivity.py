"""How much does the regime contrast move with the fold split alone?

    python src/run_seed_sensitivity.py

Diagnostic for Phase 2, and the basis for the paper's headline numbers.

StratifiedKFold and StratifiedGroupKFold have to place 195 recordings (32
subjects, 8 of them controls) or 1040 recordings (40 subjects) into 5 folds.
With so few subjects, which subject lands in which fold is itself a source of
variance, and a single seed gives a falsely precise-looking number.

Regimes A and B are re-run across 10 fold seeds, varying ONLY the CV seed --
model random_state stays at 27 -- so the spread is attributable to fold
construction and nothing else. Because both regimes see the same seed, the
inflation is measured as a PAIRED difference within each seed, which removes
the shared fold-difficulty component and estimates the quantity the paper
actually claims.

LOSO is excluded: its outer loop is deterministic, which is exactly what
makes it the stable reference reported alongside these.

Writes results/tables/seed_sensitivity.csv and seed_sensitivity_summary.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from models import get_models, MODEL_ORDER
from cv_core import nested_cv

SEEDS = list(range(10))
REGIMES = ["A_leaky_recording", "B_group_kfold"]


def main() -> None:
    models = get_models()
    datasets = {
        "UCI/MDVP": (pd.read_csv(C.UCI_OUT), C.UCI_FEATURES),
        "Sakar": (pd.read_csv(C.SAKAR_TRAIN_OUT), C.SAKAR_FEATURES),
    }

    rows = []
    t0 = time.time()
    for ds_name, (df, features) in datasets.items():
        X, y, g = df[features].values, df["label"].values, df["subject"].values
        for model_name in MODEL_ORDER:
            pipe, grid = models[model_name]
            for regime in REGIMES:
                for seed in SEEDS:
                    res = nested_cv(X, y, g, pipe, grid, regime,
                                    n_splits=5, inner_splits=5, seed=seed)
                    p = res["pooled"]
                    rows.append({
                        "dataset": ds_name, "model": model_name,
                        "regime": regime, "cv_seed": seed,
                        "accuracy": p["accuracy"], "roc_auc": p["roc_auc"],
                        "mcc": p["mcc"], "recall_pd": p["recall_pd"],
                        "recall_hc": p["recall_hc"],
                    })
            print(f"  done {ds_name:9s} {model_name:20s} ({time.time() - t0:.0f}s)")

    raw = pd.DataFrame(rows)
    raw.to_csv(C.TABLES / "seed_sensitivity.csv", index=False)

    # --- paired summary --------------------------------------------------
    wide = raw.pivot_table(
        index=["dataset", "model", "cv_seed"], columns="regime",
        values=["accuracy", "recall_hc"],
    )
    summary = []
    for (ds, model), grp in wide.groupby(level=["dataset", "model"]):
        a = grp[("accuracy", "A_leaky_recording")].values
        b = grp[("accuracy", "B_group_kfold")].values
        d = a - b                      # inflation attributable to the leak
        ha = grp[("recall_hc", "A_leaky_recording")].values
        hb = grp[("recall_hc", "B_group_kfold")].values
        t, p = stats.ttest_rel(a, b)
        summary.append({
            "dataset": ds, "model": model, "n_seeds": len(d),
            "A_mean": a.mean(), "A_sd": a.std(ddof=1),
            "B_mean": b.mean(), "B_sd": b.std(ddof=1),
            "inflation_mean": d.mean(), "inflation_sd": d.std(ddof=1),
            "inflation_ci_lo": d.mean() - 2.262 * d.std(ddof=1) / np.sqrt(len(d)),
            "inflation_ci_hi": d.mean() + 2.262 * d.std(ddof=1) / np.sqrt(len(d)),
            "paired_t": t, "p_value": p,
            "recall_hc_A_mean": ha.mean(), "recall_hc_B_mean": hb.mean(),
        })

    out = pd.DataFrame(summary).set_index(["dataset", "model"]).reindex(
        [(d, m) for d in ["UCI/MDVP", "Sakar"] for m in MODEL_ORDER]
    ).reset_index()
    out.to_csv(C.TABLES / "seed_sensitivity_summary.csv", index=False)

    show = out[["dataset", "model", "A_mean", "A_sd", "B_mean", "B_sd",
                "inflation_mean", "inflation_sd", "inflation_ci_lo",
                "inflation_ci_hi", "p_value"]]
    print("\n=== accuracy over 10 fold seeds; inflation is paired A-B (95% CI) ===")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== healthy-control recall, leaky vs subject-level (seed means) ===")
    print(out[["dataset", "model", "recall_hc_A_mean", "recall_hc_B_mean"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
