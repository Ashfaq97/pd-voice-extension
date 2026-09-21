"""Phase 2 -- the leakage analysis. Five models x three regimes x two datasets.

    python src/run_cv_comparison.py

Writes
    results/tables/cv_comparison.csv        one row per dataset/regime/model
    results/tables/cv_comparison_folds.csv  per-fold metrics
    results/tables/cv_best_params.csv       parameters chosen in each outer fold
    results/tables/oof_predictions.csv      pooled out-of-fold predictions

Every constraint in the handoff's section 3 is honoured: grouping by subject,
nested tuning with a grouped inner loop, StandardScaler inside the pipeline,
and a fixed seed (27). See src/cv_core.py for the regime definitions and for
why regime A's inner loop is deliberately ungrouped.

Grid size. No reduction was needed. The full grids in src/models.py (43 points
across the five models) run under 5-fold nested CV in seconds, and LOSO on
Sakar -- the worst case, 40 outer folds x 5 inner folds -- costs about four
minutes for all five models. Inner CV is 5-fold in every regime, so the
regimes differ only in their splitting discipline.

Headline metrics are computed on pooled out-of-fold predictions: each
recording is predicted exactly once by a model that never saw it. Per-fold
mean and standard deviation accompany them for regimes A and B.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from models import get_models, MODEL_ORDER
from cv_core import nested_cv, majority_baseline, REGIMES

N_SPLITS = 5
INNER_SPLITS = 5


def load_datasets():
    uci = pd.read_csv(C.UCI_OUT)
    sakar = pd.read_csv(C.SAKAR_TRAIN_OUT)
    return {
        "UCI/MDVP": (uci, C.UCI_FEATURES),
        "Sakar": (sakar, C.SAKAR_FEATURES),
    }


def main() -> None:
    C.TABLES.mkdir(parents=True, exist_ok=True)
    models = get_models()

    rows, fold_rows, param_rows, oof_rows = [], [], [], []
    t_start = time.time()

    for ds_name, (df, features) in load_datasets().items():
        X = df[features].values
        y = df["label"].values
        groups = df["subject"].values
        baseline = majority_baseline(y)
        print(
            f"\n=== {ds_name}: {len(df)} recordings, {df['subject'].nunique()} "
            f"subjects, baseline {baseline:.4f} ==="
        )

        for regime in REGIMES:
            for model_name in MODEL_ORDER:
                pipe, grid = models[model_name]
                t0 = time.time()
                res = nested_cv(
                    X, y, groups, pipe, grid, regime,
                    n_splits=N_SPLITS, inner_splits=INNER_SPLITS, seed=C.SEED,
                )
                dt = time.time() - t0
                p = res["pooled"]
                folds = res["folds"]
                per_fold_defined = regime != "C_loso"

                rows.append({
                    "dataset": ds_name,
                    "regime": regime,
                    "model": model_name,
                    "accuracy": p["accuracy"],
                    "accuracy_fold_mean": folds["accuracy"].mean() if per_fold_defined else np.nan,
                    "accuracy_fold_std": folds["accuracy"].std(ddof=1) if per_fold_defined else np.nan,
                    "roc_auc": p["roc_auc"],
                    "mcc": p["mcc"],
                    "recall_pd": p["recall_pd"],
                    "recall_hc": p["recall_hc"],
                    "majority_baseline": baseline,
                    "n_outer_folds": len(folds),
                    "n_recordings": len(df),
                    "n_subjects": int(df["subject"].nunique()),
                    "runtime_s": round(dt, 1),
                })

                f = folds.copy()
                f.insert(0, "model", model_name)
                f.insert(0, "regime", regime)
                f.insert(0, "dataset", ds_name)
                fold_rows.append(f)

                for i, params in enumerate(res["best_params"]):
                    param_rows.append({
                        "dataset": ds_name, "regime": regime, "model": model_name,
                        "fold": i,
                        **{k.replace("clf__", ""): v for k, v in params.items()},
                    })

                oof_rows.append(pd.DataFrame({
                    "dataset": ds_name, "regime": regime, "model": model_name,
                    "subject": groups, "y_true": y,
                    "y_pred": res["oof_pred"], "y_prob": res["oof_prob"],
                }))

                print(
                    f"  {regime:18s} {model_name:20s} "
                    f"acc {p['accuracy']:.3f}  auc {p['roc_auc']:.3f}  "
                    f"mcc {p['mcc']:+.3f}  PD-rec {p['recall_pd']:.3f}  "
                    f"HC-rec {p['recall_hc']:.3f}  ({dt:.0f}s)"
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(C.TABLES / "cv_comparison.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(
        C.TABLES / "cv_comparison_folds.csv", index=False)
    pd.DataFrame(param_rows).to_csv(C.TABLES / "cv_best_params.csv", index=False)
    pd.concat(oof_rows, ignore_index=True).to_csv(
        C.TABLES / "oof_predictions.csv", index=False)

    print(f"\ntotal runtime {time.time() - t_start:.0f}s")
    print(f"wrote 4 tables to {C.TABLES.relative_to(C.ROOT)}/")

    # --- the paper's headline contrast -----------------------------------
    print("\n=== accuracy by regime (pooled out-of-fold) ===")
    piv = summary.pivot_table(
        index=["dataset", "model"], columns="regime", values="accuracy"
    )
    piv["drop_A_minus_B"] = piv["B_group_kfold"] - piv["A_leaky_recording"]
    piv = piv.reindex(
        [(d, m) for d in ["UCI/MDVP", "Sakar"] for m in MODEL_ORDER]
    )
    print(piv.round(3).to_string())

    print("\n=== healthy-control recall, leaky vs subject-level ===")
    hc = summary.pivot_table(
        index=["dataset", "model"], columns="regime", values="recall_hc"
    ).reindex([(d, m) for d in ["UCI/MDVP", "Sakar"] for m in MODEL_ORDER])
    print(hc.round(3).to_string())


if __name__ == "__main__":
    main()
