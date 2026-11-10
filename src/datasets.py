"""The corpora, in one place, so every phase agrees on what they are.

Added when the third corpus arrived: run_cv_comparison, run_seed_sensitivity
and run_interpretability had each been constructing their own dataset dict,
which is exactly how two phases end up quietly disagreeing about a feature
list.

REGIMES_FOR records which validation regimes are run on which corpus, and
why. Leave-one-subject-out is the deterministic reference on the two small
corpora, where 5-fold grouped CV has substantial fold-construction variance
(SD up to 0.048 with 32 subjects). On Sakar-2018 that argument does not
apply -- 252 subjects put ~50 per fold, so the variance LOSO exists to avoid
is already small -- and 252 outer folds with nested tuning over 752 features
is not affordable. The per-corpus choice is recorded here rather than buried
in a script, and run_seed_sensitivity.py reports the variance that justifies
it.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

REGIMES_FOR = {
    "UCI/MDVP": ["A_leaky_recording", "B_group_kfold", "C_loso"],
    "Sakar-2013": ["A_leaky_recording", "B_group_kfold", "C_loso"],
    "Sakar-2018": ["A_leaky_recording", "B_group_kfold"],
}

DISPLAY_ORDER = ["UCI/MDVP", "Sakar-2013", "Sakar-2018"]


def load_all(include_2018=True):
    """Return {name: (dataframe, feature_list)} for the analysis corpora."""
    out = {
        "UCI/MDVP": (pd.read_csv(C.UCI_OUT), C.UCI_FEATURES),
        "Sakar-2013": (pd.read_csv(C.SAKAR_TRAIN_OUT), C.SAKAR_FEATURES),
    }
    if include_2018 and C.SAKAR2018_OUT.exists():
        out["Sakar-2018"] = (pd.read_csv(C.SAKAR2018_OUT), C.sakar2018_features())
    return out


def describe(name, df, features):
    subj = df.groupby("subject")["label"].first()
    return {
        "dataset": name,
        "recordings": len(df),
        "subjects": int(df["subject"].nunique()),
        "pd_subjects": int((subj == 1).sum()),
        "hc_subjects": int((subj == 0).sum()),
        "features": len(features),
        "majority_baseline": round(
            float(df["label"].value_counts(normalize=True).max()), 4),
    }
