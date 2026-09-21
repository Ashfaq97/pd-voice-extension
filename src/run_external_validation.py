"""Phase 4 -- independent-cohort sensitivity check on the Sakar held-out set.

    python src/run_external_validation.py

NOT an external validation, and the paper does not call it one.

data/raw/sakar/test_data.txt contains 28 PD speakers and NO healthy controls,
six sustained-vowel recordings each (/a/ and /o/, three times). That is the
published Sakar et al. 2013 protocol, not a defect. Consequences:

  * specificity is undefined -- there are no true negatives to find;
  * MCC is undefined -- y_true has zero variance, so its denominator is 0;
  * the confusion matrix collapses to a single row;
  * accuracy is identical to sensitivity; and
  * a model that predicts "PD" unconditionally scores 1.000.

The number this file produces is therefore one-sided and weak on its own. It
is reported next to the same model's honest specificity on the TRAINING
cohort under subject-level CV, which is defined, so a reader can see whether
a high test sensitivity reflects detection or merely a PD-biased operating
point.

Two training configurations are reported, because the test set is
vowels-only:

  vowel_only  train on the 120 sustained-vowel rows (rec_index 1-3), matching
              the test set's recording type
  full_train  train on all 1040 rows

Model selection uses subject-level CV on the TRAINING data alone. The test
set is then predicted exactly once per configuration. Nothing about the test
set informs selection; the code is arranged so that it cannot.

Writes results/tables/external_validation.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from models import get_models, MODEL_ORDER
from cv_core import nested_cv


def select_model(X, y, g):
    """Pick a model by subject-level nested CV on the training data only.

    Returns (name, pooled_metrics_by_model). The test set is not loaded at
    the point this runs.
    """
    models = get_models()
    scores = {}
    for name in MODEL_ORDER:
        pipe, grid = models[name]
        res = nested_cv(X, y, g, pipe, grid, "B_group_kfold",
                        n_splits=5, inner_splits=5, seed=C.SEED)
        scores[name] = res["pooled"]
        print(f"    {name:20s} CV acc {res['pooled']['accuracy']:.3f}  "
              f"HC-recall {res['pooled']['recall_hc']:.3f}  "
              f"MCC {res['pooled']['mcc']:+.3f}")
    best = max(scores, key=lambda n: scores[n]["accuracy"])
    print(f"    -> selected: {best}")
    return best, scores


def fit_final(X, y, g, name):
    """Refit the selected model on all training data, tuning by grouped CV."""
    pipe, grid = get_models()[name]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    search = GridSearchCV(pipe, grid, cv=inner, scoring="accuracy", n_jobs=-1)
    search.fit(X, y, groups=g)
    return search.best_estimator_, search.best_params_


def main() -> None:
    train = pd.read_csv(C.SAKAR_TRAIN_OUT)
    feats = C.SAKAR_FEATURES

    configs = {
        "vowel_only": train[train["rec_type"] == "vowel"],
        "full_train": train,
    }

    # --- selection phase: training data only -----------------------------
    selected, cv_scores, fitted = {}, {}, {}
    for cfg, tr in configs.items():
        print(f"\n=== selection on {cfg} "
              f"({len(tr)} recordings, {tr['subject'].nunique()} subjects) ===")
        X, y, g = tr[feats].values, tr["label"].values, tr["subject"].values
        name, scores = select_model(X, y, g)
        selected[cfg], cv_scores[cfg] = name, scores
        est, params = fit_final(X, y, g, name)
        fitted[cfg] = (est, params)
        print(f"    final hyper-parameters: "
              f"{ {k.replace('clf__',''): v for k, v in params.items()} }")

    # --- the single pass over the held-out set ---------------------------
    test = pd.read_csv(C.SAKAR_TEST_OUT)
    assert set(test["label"]) == {1}, "test set is expected to be PD-only"
    Xte, yte = test[feats].values, test["label"].values
    print(f"\n=== predicting the held-out cohort once "
          f"({len(test)} recordings, {test['subject'].nunique()} PD speakers, "
          f"0 controls) ===")

    rows = []
    for cfg, tr in configs.items():
        name = selected[cfg]
        est, params = fitted[cfg]
        pred = est.predict(Xte)
        prob = est.predict_proba(Xte)[:, 1]

        sens = recall_score(yte, pred, pos_label=1)
        # Per-speaker majority vote: does the cohort look PD at speaker level?
        per_subj = (
            pd.DataFrame({"subject": test["subject"], "pred": pred})
            .groupby("subject")["pred"].mean()
        )
        subj_sens = float((per_subj > 0.5).mean())

        internal = cv_scores[cfg][name]
        rows.append({
            "config": cfg,
            "train_recordings": len(tr),
            "train_subjects": int(tr["subject"].nunique()),
            "selected_model": name,
            "hyperparameters": str(
                {k.replace("clf__", ""): v for k, v in params.items()}),
            "train_cv_accuracy_subject_level": round(internal["accuracy"], 4),
            "train_cv_specificity_subject_level": round(internal["recall_hc"], 4),
            "train_cv_mcc_subject_level": round(internal["mcc"], 4),
            "test_recordings": len(test),
            "test_subjects": int(test["subject"].nunique()),
            "test_sensitivity_recording": round(sens, 4),
            "test_sensitivity_speaker": round(subj_sens, 4),
            "test_mean_pd_probability": round(float(prob.mean()), 4),
            "test_accuracy": round(sens, 4),          # identical by construction
            "test_specificity": np.nan,               # undefined: no controls
            "test_mcc": np.nan,                       # undefined: zero variance
            "constant_pd_baseline": 1.0,
        })
        print(f"  {cfg:11s} {name:20s} "
              f"test sensitivity {sens:.3f} (recording) / {subj_sens:.3f} (speaker)  "
              f"mean P(PD) {prob.mean():.3f}  |  "
              f"train-cohort specificity {internal['recall_hc']:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES / "external_validation.csv", index=False)

    # --- secondary: all five models, for completeness --------------------
    # Selection above was made on training CV alone, BEFORE any of these were
    # computed. These are reported so the reader can see the spread; they are
    # not used to choose anything.
    sec = []
    for cfg, tr in configs.items():
        X, y, g = tr[feats].values, tr["label"].values, tr["subject"].values
        for name in MODEL_ORDER:
            est, params = fit_final(X, y, g, name)
            pred = est.predict(Xte)
            sec.append({
                "config": cfg, "model": name,
                "test_sensitivity": round(recall_score(yte, pred, pos_label=1), 4),
                "train_cv_specificity_subject_level":
                    round(cv_scores[cfg][name]["recall_hc"], 4),
                "selected": name == selected[cfg],
            })
    sec_df = pd.DataFrame(sec)
    sec_df.to_csv(C.TABLES / "external_validation_all_models.csv", index=False)

    print("\n=== all models, reported for completeness (not used for selection) ===")
    print(sec_df.to_string(index=False))
    print("\nReminder: specificity and MCC are undefined on this cohort and a")
    print("constant-PD predictor scores 1.000. Read test sensitivity only")
    print("alongside the training-cohort specificity in the same row.")
    print(f"\nwrote {(C.TABLES / 'external_validation.csv').relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
