"""Phase 3 -- does any recording type rescue subject-level performance?

    python src/run_recording_type.py

Sakar only; the UCI set has no recording-type structure. Answers the obvious
reviewer objection that sustained vowels -- the standard PD voice task --
would do better than connected speech if only they were analysed separately.

Step 1 re-verifies the recording-type ordering empirically rather than
assuming it. Sakar ships 26 recordings per subject in a fixed order with no
type column; the blocks are recovered from acoustic structure, since
sustained vowels are long and almost fully voiced while numbers and words are
short and heavily unvoiced. The assertions below fail if that structure is
not present.

Step 2 runs regimes A and B within each type.

Hyper-parameters are FIXED here (src/models.py: get_fixed_models) rather than
tuned per fold. Three reasons: the question is whether any type carries more
signal than another, which requires the models to be identical across types,
not individually optimised; the per-type subsets are small (120-400
recordings) so nested tuning inside them would be noisier than the effect
being measured; and it keeps a 4 types x 2 regimes x 5 models x 10 seeds grid
affordable. Regime A and B remain identical to each other in every respect
but the split.

Each cell is run across 10 fold seeds and reported as mean +/- SD. Per-type
subsets are small enough that a single seed is not informative -- see
run_seed_sensitivity.py for the same effect at full sample size.

Writes results/tables/recording_type.csv and recording_type_profile.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from models import get_fixed_models, MODEL_ORDER
from cv_core import outer_splits, score_block

SEEDS = list(range(10))
REGIMES = ["A_leaky_recording", "B_group_kfold"]
TYPE_ORDER = ["vowel", "number", "sentence", "word"]


def verify_type_ordering(df: pd.DataFrame) -> pd.DataFrame:
    """Recover the recording-type blocks from acoustic structure."""
    prof = df.groupby("rec_index").agg(
        mean_pulses=("Number_of_pulses", "mean"),
        mean_pct_unvoiced=("Fraction_of_locally_unvoiced_frames", "mean"),
        mean_periods=("Number_of_periods", "mean"),
    )
    prof["assigned_type"] = [C.sakar_rec_type(k) for k in prof.index]

    vowel = prof[prof.assigned_type == "vowel"]
    sentence = prof[prof.assigned_type == "sentence"]
    short = prof[prof.assigned_type.isin(["number", "word"])]

    # Sustained vowels: long and almost fully voiced.
    assert vowel.mean_pulses.min() > 250, (
        f"vowel block pulses too low: {vowel.mean_pulses.min():.0f}")
    assert vowel.mean_pct_unvoiced.max() < 6, (
        f"vowel block unvoiced too high: {vowel.mean_pct_unvoiced.max():.1f}%")
    # Sentences: intermediate length.
    assert 100 < sentence.mean_pulses.min() and sentence.mean_pulses.max() < 250, (
        f"sentence block pulses outside 100-250: "
        f"{sentence.mean_pulses.min():.0f}-{sentence.mean_pulses.max():.0f}")
    # Numbers and words: short.
    assert short.mean_pulses.max() < 100, (
        f"number/word block pulses too high: {short.mean_pulses.max():.0f}")
    # The three blocks must not overlap on length.
    assert vowel.mean_pulses.min() > sentence.mean_pulses.max()
    assert sentence.mean_pulses.min() > short.mean_pulses.max()

    print("recording-type ordering verified from acoustic structure:")
    for t in TYPE_ORDER:
        sub = prof[prof.assigned_type == t]
        print(f"  {t:9s} idx {sub.index.min():2d}-{sub.index.max():2d}  "
              f"pulses {sub.mean_pulses.min():6.1f}-{sub.mean_pulses.max():6.1f}  "
              f"unvoiced {sub.mean_pct_unvoiced.min():4.1f}-"
              f"{sub.mean_pct_unvoiced.max():4.1f}%")
    return prof


def run_cell(X, y, g, pipe, regime, seed):
    """Plain (non-nested) CV -- hyper-parameters are fixed, so nothing to tune."""
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_prob = np.full(len(y), np.nan, dtype=float)
    for tr, te in outer_splits(regime, X, y, g, 5, seed):
        est = pipe.fit(X[tr], y[tr])
        oof_pred[te] = est.predict(X[te])
        oof_prob[te] = est.predict_proba(X[te])[:, 1]
    assert (oof_pred >= 0).all()
    return score_block(y, oof_pred, oof_prob)


def main() -> None:
    df = pd.read_csv(C.SAKAR_TRAIN_OUT)
    prof = verify_type_ordering(df)
    prof.to_csv(C.TABLES / "recording_type_profile.csv")

    counts = df.groupby("rec_type").size().to_dict()
    print("\nrecordings per type:",
          {t: counts[t] for t in TYPE_ORDER}, "\n")

    rows = []
    t0 = time.time()
    for rec_type in TYPE_ORDER:
        sub = df[df["rec_type"] == rec_type]
        X = sub[C.SAKAR_FEATURES].values
        y = sub["label"].values
        g = sub["subject"].values
        assert sub["subject"].nunique() == 40
        for model_name in MODEL_ORDER:
            for regime in REGIMES:
                # A fresh estimator per cell; sklearn clones on fit anyway,
                # but this keeps state unambiguous.
                accs, hcs, aucs = [], [], []
                for seed in SEEDS:
                    pipe = get_fixed_models()[model_name]
                    s = run_cell(X, y, g, pipe, regime, seed)
                    accs.append(s["accuracy"])
                    hcs.append(s["recall_hc"])
                    aucs.append(s["roc_auc"])
                a = np.array(accs)
                rows.append({
                    "dataset": "Sakar-2013", "rec_type": rec_type, "model": model_name,
                    "regime": regime, "n_recordings": len(sub),
                    "n_seeds": len(SEEDS),
                    "accuracy_mean": a.mean(), "accuracy_sd": a.std(ddof=1),
                    "accuracy_min": a.min(), "accuracy_max": a.max(),
                    "roc_auc_mean": float(np.mean(aucs)),
                    "recall_hc_mean": float(np.mean(hcs)),
                    "majority_baseline": 0.5,
                })
        print(f"  {rec_type:9s} done ({time.time() - t0:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES / "recording_type.csv", index=False)

    print("\n=== subject-level accuracy by recording type "
          "(regime B, mean of 10 fold seeds; chance = 0.500) ===")
    b = out[out.regime == "B_group_kfold"]
    piv = b.pivot_table(index="model", columns="rec_type", values="accuracy_mean")
    sd = b.pivot_table(index="model", columns="rec_type", values="accuracy_sd")
    piv = piv[TYPE_ORDER].reindex(MODEL_ORDER)
    sd = sd[TYPE_ORDER].reindex(MODEL_ORDER)
    shown = piv.round(3).astype(str) + " +/-" + sd.round(3).astype(str)
    print(shown.to_string())

    print("\n=== same, regime A (recording-level) for contrast ===")
    a = out[out.regime == "A_leaky_recording"]
    print(a.pivot_table(index="model", columns="rec_type", values="accuracy_mean")
          [TYPE_ORDER].reindex(MODEL_ORDER).round(3).to_string())

    best = piv.max().max()
    print(f"\nbest subject-level cell across all types and models: {best:.3f} "
          f"(chance 0.500)")
    print(f"wrote {(C.TABLES / 'recording_type.csv').relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
