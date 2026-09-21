"""Compare the Phase 2 run against the checkpoint figures in the handoff.

    python src/verify_checkpoint.py

The handoff's section 1 records an analysis done elsewhere. It is a
checkpoint, not a target: this script reports deviations, and nothing in the
pipeline is adjusted to close them. A deviation larger than 0.02 is flagged.

Writes results/tables/checkpoint_comparison.csv and
results/tables/checkpoint_comparison_rectype.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from models import MODEL_ORDER
from cv_core import outer_splits

TOL = 0.02

# Handoff section 1. "subject_level" is reported there as a single column;
# the handoff states that LOSO agrees with it, so both B and C are compared.
CHECKPOINT = {
    ("UCI/MDVP", "Gradient Boosting"): (0.908, 0.795),
    ("UCI/MDVP", "XGBoost"): (0.903, 0.780),
    ("UCI/MDVP", "KNN"): (0.892, 0.769),
    ("UCI/MDVP", "Decision Tree"): (0.851, 0.697),
    ("UCI/MDVP", "Logistic Regression"): (0.846, 0.836),
    ("Sakar", "XGBoost"): (0.701, 0.543),
    ("Sakar", "Decision Tree"): (0.615, 0.486),
    ("Sakar", "Gradient Boosting"): (0.688, 0.578),
    ("Sakar", "KNN"): (0.664, 0.575),
    ("Sakar", "Logistic Regression"): (0.637, 0.570),
}

# Handoff section 1, recording-type breakdown (Sakar, subject-level).
CHECKPOINT_RECTYPE = {
    "Logistic Regression": {"vowel": 0.558, "number": 0.535, "sentence": 0.588, "word": 0.489},
    "Gradient Boosting":   {"vowel": 0.558, "number": 0.565, "sentence": 0.556, "word": 0.528},
    "XGBoost":             {"vowel": 0.567, "number": 0.560, "sentence": 0.562, "word": 0.550},
    "Decision Tree":       {"vowel": 0.533, "number": 0.532, "sentence": 0.569, "word": 0.506},
    "KNN":                 {"vowel": 0.400, "number": 0.585, "sentence": 0.525, "word": 0.553},
}
TYPE_ORDER = ["vowel", "number", "sentence", "word"]


def check_recording_type() -> None:
    """Compare Phase 3 against the handoff's recording-type table.

    Phase 3 reports each cell as a mean over 10 fold seeds with its own SD,
    so a deviation is judged against that SD as well as against the flat
    0.02 tolerance -- a single-seed checkpoint value that sits within one SD
    of a 10-seed mean is not evidence of a discrepancy.
    """
    path = C.TABLES / "recording_type.csv"
    if not path.exists():
        print("\n(recording_type.csv not found -- run src/run_recording_type.py)")
        return
    rt = pd.read_csv(path)
    b = rt[rt.regime == "B_group_kfold"]
    mean = b.pivot_table(index="model", columns="rec_type", values="accuracy_mean")
    sd = b.pivot_table(index="model", columns="rec_type", values="accuracy_sd")

    rows = []
    for model, exp in CHECKPOINT_RECTYPE.items():
        for t in TYPE_ORDER:
            obs, s = mean.loc[model, t], sd.loc[model, t]
            d = obs - exp[t]
            rows.append({
                "model": model, "rec_type": t,
                "expected": exp[t], "observed": round(obs, 3),
                "delta": round(d, 3), "observed_sd": round(s, 3),
                "delta_in_sd": round(d / s, 2),
                "within_1sd": abs(d) <= s,
                "within_tol": abs(d) <= TOL,
            })
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES / "checkpoint_comparison_rectype.csv", index=False)

    print("\n=== Phase 3: recording-type breakdown vs checkpoint ===")
    print(out.to_string(index=False))
    n_out = int((~out.within_1sd).sum())
    print(f"\n{int((~out.within_tol).sum())} of {len(out)} cells exceed the "
          f"flat {TOL} tolerance; {n_out} of {len(out)} lie beyond 1 SD of the "
          "10-seed mean")
    print(f"wrote {(C.TABLES / 'checkpoint_comparison_rectype.csv').relative_to(C.ROOT)}")


def verify_leak_magnitude() -> None:
    """Confirm that regime A leaks and regime B does not.

    This is the mechanism the whole paper rests on, so it is measured rather
    than asserted: the fraction of test recordings whose speaker also appears
    in the same fold's training set.
    """
    print("=== leak magnitude by regime ===")
    for name, path, feats in [
        ("UCI/MDVP", C.UCI_OUT, C.UCI_FEATURES),
        ("Sakar", C.SAKAR_TRAIN_OUT, C.SAKAR_FEATURES),
    ]:
        df = pd.read_csv(path)
        X, y, g = df[feats].values, df["label"].values, df["subject"].values
        for regime in ["A_leaky_recording", "B_group_kfold"]:
            frac = [
                np.mean([s in set(g[tr]) for s in g[te]])
                for tr, te in outer_splits(regime, X, y, g, 5, C.SEED)
            ]
            pct = float(np.mean(frac)) * 100
            print(f"  {name:9s} {regime:18s} "
                  f"test recordings whose speaker is also in training: {pct:5.1f}%")
            if regime == "A_leaky_recording":
                assert pct > 99, "regime A is supposed to leak; it does not"
            else:
                assert pct == 0, "regime B leaked -- grouping is broken"
    print()


def main() -> None:
    verify_leak_magnitude()
    cv = pd.read_csv(C.TABLES / "cv_comparison.csv")
    acc = cv.pivot_table(index=["dataset", "model"], columns="regime", values="accuracy")

    rows = []
    for (ds, model), (exp_a, exp_sub) in CHECKPOINT.items():
        got_a = acc.loc[(ds, model), "A_leaky_recording"]
        got_b = acc.loc[(ds, model), "B_group_kfold"]
        got_c = acc.loc[(ds, model), "C_loso"]
        rows.append({
            "dataset": ds, "model": model,
            "A_expected": exp_a, "A_observed": round(got_a, 3),
            "A_delta": round(got_a - exp_a, 3),
            "subject_expected": exp_sub,
            "B_observed": round(got_b, 3), "B_delta": round(got_b - exp_sub, 3),
            "C_observed": round(got_c, 3), "C_delta": round(got_c - exp_sub, 3),
        })

    out = pd.DataFrame(rows)
    out["flag"] = [
        " ".join(
            tag for tag, d in (("A", r.A_delta), ("B", r.B_delta), ("C", r.C_delta))
            if abs(d) > TOL
        ) or "-"
        for r in out.itertuples()
    ]
    out = out.set_index(["dataset", "model"]).reindex(
        [(d, m) for d in ["UCI/MDVP", "Sakar"] for m in MODEL_ORDER]
    ).reset_index()
    out.to_csv(C.TABLES / "checkpoint_comparison.csv", index=False)

    print(out.to_string(index=False))
    flagged = out[out["flag"] != "-"]
    print(f"\ntolerance +/-{TOL}")
    print(f"{len(flagged)} of {len(out)} model/dataset pairs deviate on at least one regime")
    for r in flagged.itertuples():
        for tag in r.flag.split():
            d = {"A": r.A_delta, "B": r.B_delta, "C": r.C_delta}[tag]
            exp = r.A_expected if tag == "A" else r.subject_expected
            print(f"  {r.dataset:9s} {r.model:20s} regime {tag}: "
                  f"expected {exp:.3f}, observed {exp + d:.3f}, delta {d:+.3f}")
    print(f"\nwrote {(C.TABLES / 'checkpoint_comparison.csv').relative_to(C.ROOT)}")

    check_recording_type()


if __name__ == "__main__":
    main()
