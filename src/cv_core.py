"""Nested cross-validation machinery shared by the analysis phases.

The three regimes differ ONLY in how rows are assigned to folds. Models,
grids, preprocessing, scoring and seed are identical across them, so the gap
between regimes is attributable to the split and nothing else.

    A_leaky_recording  StratifiedKFold over recordings, outer AND inner.
                       Reproduces the 2023 thesis and acts as the control.
    B_group_kfold      StratifiedGroupKFold by subject, outer AND inner.
    C_loso             LeaveOneGroupOut outer, StratifiedGroupKFold inner.

Note on regime A's inner loop. It is ungrouped, matching its outer loop.
Grouping only the inner loop would produce a hybrid that neither reproduces
the thesis nor measures anything of interest; the regime is the splitting
strategy, applied consistently. Regimes B and C group the inner loop, which
is essential -- a grouped outer loop wrapped around an ungrouped GridSearchCV
reintroduces the very leak being measured.

Metrics are computed on pooled out-of-fold predictions (every row predicted
exactly once, by a model that never saw it). Per-fold mean and standard
deviation are also reported for A and B. They are not reported for LOSO,
where a fold holds one subject and therefore one class, leaving ROC-AUC, MCC
and one of the two recalls undefined per fold.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    StratifiedGroupKFold,
    LeaveOneGroupOut,
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    recall_score,
    confusion_matrix,
)

REGIMES = ["A_leaky_recording", "B_group_kfold", "C_loso"]


def outer_splits(regime, X, y, groups, n_splits, seed):
    """Yield (train_idx, test_idx) for the given regime."""
    if regime == "A_leaky_recording":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from cv.split(X, y)
    elif regime == "B_group_kfold":
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from cv.split(X, y, groups)
    elif regime == "C_loso":
        yield from LeaveOneGroupOut().split(X, y, groups)
    else:
        raise ValueError(regime)


def inner_cv(regime, n_splits, seed):
    """Inner splitter, matching the regime's grouping discipline."""
    if regime == "A_leaky_recording":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), False
    return (
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        True,
    )


def score_block(y_true, y_pred, y_prob):
    """Metrics for a set of predictions. NaN where undefined."""
    y_true = np.asarray(y_true)
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    both_classes = len(np.unique(y_true)) == 2
    out["roc_auc"] = roc_auc_score(y_true, y_prob) if both_classes else np.nan
    out["mcc"] = matthews_corrcoef(y_true, y_pred) if both_classes else np.nan
    out["recall_pd"] = (
        recall_score(y_true, y_pred, pos_label=1, zero_division=np.nan)
        if (y_true == 1).any() else np.nan
    )
    out["recall_hc"] = (
        recall_score(y_true, y_pred, pos_label=0, zero_division=np.nan)
        if (y_true == 0).any() else np.nan
    )
    return out


def nested_cv(X, y, groups, pipeline, grid, regime, n_splits, inner_splits, seed,
              n_jobs=-1):
    """Run nested CV for one model under one regime.

    GridSearchCV sits inside each outer fold, so no information from the
    outer test fold reaches tuning. Returns pooled out-of-fold predictions,
    per-fold metrics, and the parameters chosen in each fold.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    inner, inner_needs_groups = inner_cv(regime, inner_splits, seed)

    oof_pred = np.full(len(y), -1, dtype=int)
    oof_prob = np.full(len(y), np.nan, dtype=float)
    fold_rows, chosen = [], []

    for fold, (tr, te) in enumerate(
        outer_splits(regime, X, y, groups, n_splits, seed)
    ):
        search = GridSearchCV(
            pipeline, grid, cv=inner, scoring="accuracy", n_jobs=n_jobs, refit=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if inner_needs_groups:
                search.fit(X[tr], y[tr], groups=groups[tr])
            else:
                search.fit(X[tr], y[tr])

        pred = search.predict(X[te])
        prob = search.predict_proba(X[te])[:, 1]
        oof_pred[te] = pred
        oof_prob[te] = prob
        chosen.append(search.best_params_)

        row = {"fold": fold, "n_test": len(te)}
        row.update(score_block(y[te], pred, prob))
        fold_rows.append(row)

    assert (oof_pred >= 0).all(), "some rows were never in a test fold"

    pooled = score_block(y, oof_pred, oof_prob)
    folds = pd.DataFrame(fold_rows)
    return {
        "pooled": pooled,
        "folds": folds,
        "oof_pred": oof_pred,
        "oof_prob": oof_prob,
        "best_params": chosen,
        "confusion": confusion_matrix(y, oof_pred, labels=[0, 1]),
    }


def majority_baseline(y):
    return float(pd.Series(y).value_counts(normalize=True).max())
