"""Verify that the guards in build_datasets.py actually fire.

    python src/test_traps.py

The pristine UCI archive has none of the defects documented in the handoff,
so the assertions in _clean_sakar() never trip during a normal build. That
makes them untested by construction. This script feeds them the corrupted
2023 working copy -- which does have the defects -- and checks that each
guard raises. If a guard is ever weakened, this fails.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from build_datasets import _clean_sakar

CORRUPT = C.RAW / "excluded" / "train_data_corrupted_2023.txt"


def expect_assertion(fn, must_mention: str, label: str) -> None:
    try:
        fn()
    except AssertionError as exc:
        assert must_mention in str(exc), (
            f"{label}: raised, but message does not mention {must_mention!r}: {exc}"
        )
        print(f"PASS  {label}\n      -> {exc}")
        return
    raise SystemExit(f"FAIL  {label}: no AssertionError raised")


def main() -> None:
    # The 2023 copy has a hand-added quoted header row, so header=0.
    corrupt = pd.read_csv(CORRUPT, header=0, names=C.SAKAR_TRAIN_COLS)
    assert corrupt.shape == (1222, 29), f"corrupt fixture shape {corrupt.shape}"
    assert corrupt["class_information"].isna().sum() == 178, "fixture lost its trap-1 rows"
    print(f"fixture: {CORRUPT.relative_to(C.ROOT)}  {corrupt.shape[0]} rows\n")

    # Trap 1: the 178 column-shifted youtube rows.
    expect_assertion(
        lambda: _clean_sakar(corrupt, "sakar"),
        "trap 1 present",
        "trap 1 (column-shifted youtube rows) is detected",
    )

    # Trap 2: with the youtube rows removed, the 4 mis-scaled id-41 rows remain.
    trap2_only = corrupt[corrupt["class_information"].notna()].copy()
    assert len(trap2_only) == 1044 and (trap2_only["id"] == 41).sum() == 4
    expect_assertion(
        lambda: _clean_sakar(trap2_only, "sakar"),
        "trap 2 present",
        "trap 2 (mis-scaled subject 41) is detected",
    )

    # Trap 3: UPDRS must never survive into a cleaned frame.
    clean = _clean_sakar(
        pd.read_csv(C.SAKAR_TRAIN_RAW, header=None, names=C.SAKAR_TRAIN_COLS), "sakar"
    )
    assert "UPDRS" not in clean.columns, "trap 3: UPDRS survived cleaning"
    print("PASS  trap 3 (UPDRS) is dropped by _clean_sakar")

    for out in (C.SAKAR_TRAIN_OUT, C.SAKAR_TEST_OUT):
        if out.exists():
            assert "UPDRS" not in pd.read_csv(out, nrows=1).columns, f"UPDRS in {out.name}"
    print("PASS  trap 3 (UPDRS) is absent from data/processed/")

    print("\nall guards fire correctly")


if __name__ == "__main__":
    main()
