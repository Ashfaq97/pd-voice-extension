"""Build data/processed/ from data/raw/. Run this before any analysis script.

    python src/build_datasets.py

Produces three files, each with a uniform schema: a `subject` column (the
grouping variable for every CV regime), a `label` column (1 = PD, 0 = HC),
and the dataset's acoustic features.

    uci_mdvp.csv     195 recordings,  32 subjects,  24 PD /  8 HC,  22 features
    sakar_train.csv  1040 recordings,  40 subjects,  20 PD / 20 HC,  26 features
    sakar_test.csv    168 recordings,  28 subjects,  28 PD /  0 HC,  26 features
    sakar2018.csv     756 recordings, 252 subjects, 188 PD / 64 HC, 752 features

Provenance note. data/raw/sakar/{train,test}_data.txt are the untouched files
from the UCI archive Parkinson_Multiple_Sound_Recording.rar (files dated
2014-06-11), not the corrupted 2023 working copy. The 2023 copy is kept at
data/raw/excluded/train_data_corrupted_2023.txt purely as evidence; see
README "Excluded data". The trap-1 and trap-2 defects documented in the
project handoff were introduced locally in 2023, not by UCI, so the checks
below are *assertions on pristine input* rather than repairs. They are kept
because a silent regression here would poison every downstream result.

Every count is asserted. This script fails loudly or not at all.
"""
import sys
import hashlib
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_uci() -> pd.DataFrame:
    df = pd.read_csv(C.UCI_RAW)
    assert df.shape == (195, 24), f"uci raw shape {df.shape}"
    assert df.isnull().sum().sum() == 0, "uci has nulls"
    assert not df.duplicated().any(), "uci has duplicate rows"
    assert df["name"].nunique() == 195, "uci recording names not unique"

    # TRAP 7: subject ID is `name` with the trailing _N stripped. This yields
    # 32 subjects / 24 PD, not the 31 / 23 stated in the 2023 thesis.
    df["subject"] = df["name"].str.rsplit("_", n=1).str[0]
    df["label"] = df["status"].astype(int)

    assert df["subject"].nunique() == 32, f"uci subjects {df['subject'].nunique()}"
    per_subject = df.groupby("subject")["label"].nunique()
    assert (per_subject == 1).all(), "a uci subject has mixed labels"
    subj_label = df.groupby("subject")["label"].first()
    assert int((subj_label == 1).sum()) == 24, "uci PD subjects != 24"
    assert int((subj_label == 0).sum()) == 8, "uci HC subjects != 8"
    # 29 subjects contribute 6 recordings and 3 contribute 7.
    counts = df.groupby("subject").size().value_counts().to_dict()
    assert counts == {6: 29, 7: 3}, f"uci recordings per subject {counts}"
    assert df["label"].value_counts().to_dict() == {1: 147, 0: 48}

    out = df[["name", "subject", "label"] + C.UCI_FEATURES].copy()
    assert out.shape == (195, 25), f"uci out shape {out.shape}"
    return out


def _read_sakar(path: Path, cols: list) -> pd.DataFrame:
    """Read a raw Sakar file. The UCI originals have no header row."""
    df = pd.read_csv(path, header=None, names=cols)
    assert df.isnull().sum().sum() == 0, f"{path.name} has nulls"
    return df


def _clean_sakar(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Apply the trap-1 and trap-2 checks.

    TRAP 1 -- 178 rows from youtube_data.txt were appended to the 2023 working
    copy with only 28 of 29 fields, shifting every column by one so the class
    label landed in UPDRS and class_information read NaN. Pristine input has
    no such rows; assert it.

    TRAP 2 -- the 2023 copy also carried 4 fabricated rows under id 41 whose
    jitter/shimmer are fractions where every real row is a percentage
    (Jitter_local 0.012-0.069 against a true range of 0.190-14.376), i.e. off
    by ~100x. Pristine input has no id 41; assert it.
    """
    n_before = len(df)
    df = df[df["class_information"].notna()].copy()
    assert len(df) == n_before, (
        f"{source}: trap 1 present -- {n_before - len(df)} rows have a NaN "
        "class_information, meaning youtube rows were appended"
    )

    jitter_min = df["Jitter_local"].min()
    assert jitter_min >= 0.1, (
        f"{source}: trap 2 present -- Jitter_local minimum {jitter_min} is "
        "below the 0.19 floor of correctly-scaled rows"
    )
    assert 41 not in set(df["id"]), f"{source}: trap 2 present -- id 41 found"

    df["label"] = df["class_information"].astype(int)
    df["subject"] = source + "_" + df["id"].astype(int).astype(str).str.zfill(2)

    per_subject = df.groupby("subject")["label"].nunique()
    assert (per_subject == 1).all(), f"{source}: a subject has mixed labels"

    # TRAP 3: drop UPDRS here so it can never reach data/processed/.
    return df.drop(columns=[c for c in C.SAKAR_FORBIDDEN if c in df.columns])


def build_sakar_train() -> pd.DataFrame:
    df = _read_sakar(C.SAKAR_TRAIN_RAW, C.SAKAR_TRAIN_COLS)
    assert df.shape == (1040, 29), f"sakar train raw shape {df.shape}"

    # Confirm the UPDRS leak before discarding the column, so the claim in the
    # paper is checked by the pipeline rather than asserted from memory.
    hc_updrs = set(df.loc[df["class_information"] == 0, "UPDRS"])
    pd_updrs = set(df.loc[df["class_information"] == 1, "UPDRS"])
    assert hc_updrs == {1}, f"expected UPDRS==1 for all HC, got {sorted(hc_updrs)}"
    assert not (hc_updrs & pd_updrs), "UPDRS overlaps between classes"

    df = _clean_sakar(df, "sakar")
    assert len(df) == 1040, f"sakar train rows {len(df)}"
    assert df["subject"].nunique() == 40, f"sakar train subjects {df['subject'].nunique()}"
    assert set(df.groupby("subject").size()) == {26}, "not every subject has 26 recordings"
    subj_label = df.groupby("subject")["label"].first()
    assert int((subj_label == 1).sum()) == 20, "sakar train PD subjects != 20"
    assert int((subj_label == 0).sum()) == 20, "sakar train HC subjects != 20"
    assert df["label"].value_counts().to_dict() == {1: 520, 0: 520}

    # 1-based recording index within subject; the basis for recording type.
    df["rec_index"] = df.groupby("subject").cumcount() + 1
    assert set(df["rec_index"]) == set(range(1, 27))
    df["rec_type"] = df["rec_index"].map(C.sakar_rec_type)

    out = df[["subject", "rec_index", "rec_type", "label"] + C.SAKAR_FEATURES].copy()
    assert "UPDRS" not in out.columns
    assert out.shape == (1040, 30), f"sakar train out shape {out.shape}"
    return out


def build_sakar_test() -> pd.DataFrame:
    """The independent Sakar cohort.

    NOTE -- this set contains 28 PD speakers and NO healthy controls, and only
    sustained vowels (6 per speaker: /a/ and /o/, three times each). That is
    the published Sakar et al. 2013 protocol, not a defect. It means
    specificity and MCC are undefined on this set and accuracy is identical to
    sensitivity; see run_external_validation.py.
    """
    df = _read_sakar(C.SAKAR_TEST_RAW, C.SAKAR_TEST_COLS)
    assert df.shape == (168, 28), f"sakar test raw shape {df.shape}"

    df = _clean_sakar(df, "sakartest")
    assert len(df) == 168, f"sakar test rows {len(df)}"
    assert df["subject"].nunique() == 28, f"sakar test subjects {df['subject'].nunique()}"
    assert set(df.groupby("subject").size()) == {6}, "not every test subject has 6 recordings"
    assert set(df["label"]) == {1}, "sakar test is expected to be PD-only"

    df["rec_index"] = df.groupby("subject").cumcount() + 1
    df["rec_type"] = "vowel"

    out = df[["subject", "rec_index", "rec_type", "label"] + C.SAKAR_FEATURES].copy()
    assert out.shape == (168, 30), f"sakar test out shape {out.shape}"
    return out


def build_sakar2018() -> pd.DataFrame:
    """The 2018 Sakar corpus (UCI #470).

    Features arrive already extracted, so there is no audio processing and no
    cleaning to do -- but the counts are asserted exactly as for the other two
    corpora, because a silent regression here would be just as damaging.

    Two things about the raw file are easy to get wrong and are handled here:
    the first row is a feature-group banner rather than column names, and
    `gender` is a subject-constant demographic that is dropped by default
    (see config.SAKAR2018_USE_GENDER).
    """
    df = pd.read_csv(C.SAKAR2018_RAW, header=C.SAKAR2018_HEADER_ROW)
    assert df.shape == (756, 755), f"sakar2018 raw shape {df.shape}"
    assert df.isnull().sum().sum() == 0, "sakar2018 has nulls"
    assert C.SAKAR2018_ID in df.columns, "sakar2018 lost its subject id column"

    # Guard against the banner row being read as data, which would make every
    # column object-typed and every count meaningless.
    assert df[C.SAKAR2018_ID].dtype.kind in "iu", (
        "sakar2018 id column is not integer -- header row likely misread")

    df["label"] = df[C.SAKAR2018_TARGET].astype(int)
    df["subject"] = "sakar2018_" + df[C.SAKAR2018_ID].astype(int).astype(str).str.zfill(3)

    assert df["subject"].nunique() == 252, f"sakar2018 subjects {df['subject'].nunique()}"
    assert set(df.groupby("subject").size()) == {3}, "not every subject has 3 recordings"
    per_subject = df.groupby("subject")["label"].nunique()
    assert (per_subject == 1).all(), "a sakar2018 subject has mixed labels"
    subj_label = df.groupby("subject")["label"].first()
    assert int((subj_label == 1).sum()) == 188, "sakar2018 PD subjects != 188"
    assert int((subj_label == 0).sum()) == 64, "sakar2018 HC subjects != 64"
    assert df["label"].value_counts().to_dict() == {1: 564, 0: 192}

    features = [c for c in df.columns
                if c not in C.SAKAR2018_NON_FEATURES + ["label", "subject"]]
    assert len(features) == 752, f"sakar2018 feature count {len(features)}"
    assert "gender" not in features or C.SAKAR2018_USE_GENDER
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in features), \
        "a sakar2018 feature is non-numeric"

    out = df[["subject", "label"] + features].copy()
    assert out.shape == (756, 754), f"sakar2018 out shape {out.shape}"
    return out


def main() -> None:
    C.PROCESSED.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("uci_mdvp", build_uci, C.UCI_OUT, [C.UCI_RAW]),
        ("sakar_train", build_sakar_train, C.SAKAR_TRAIN_OUT, [C.SAKAR_TRAIN_RAW]),
        ("sakar_test", build_sakar_test, C.SAKAR_TEST_OUT, [C.SAKAR_TEST_RAW]),
        ("sakar2018", build_sakar2018, C.SAKAR2018_OUT, [C.SAKAR2018_RAW]),
    ]

    manifest = {}
    for key, fn, dest, sources in jobs:
        df = fn()
        df.to_csv(dest, index=False)
        subj = df.groupby("subject")["label"].first()
        manifest[key] = {
            "output": str(dest.relative_to(C.ROOT)),
            "rows": int(len(df)),
            "features": int(len(df.columns) - (4 if "rec_index" in df else 3)),
            "subjects": int(df["subject"].nunique()),
            "pd_subjects": int((subj == 1).sum()),
            "hc_subjects": int((subj == 0).sum()),
            "majority_baseline": round(float(df["label"].value_counts(normalize=True).max()), 4),
            "sources": {str(s.relative_to(C.ROOT)): sha256(s)[:16] for s in sources},
            "output_sha256_16": sha256(dest)[:16],
        }
        m = manifest[key]
        print(
            f"{key:12s} -> {m['output']:30s} "
            f"{m['rows']:5d} rows  {m['subjects']:3d} subjects  "
            f"{m['pd_subjects']:2d} PD / {m['hc_subjects']:2d} HC  "
            f"baseline {m['majority_baseline']:.4f}"
        )

    (C.PROCESSED / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {C.PROCESSED / 'manifest.json'}")
    print("all assertions passed")


if __name__ == "__main__":
    main()
