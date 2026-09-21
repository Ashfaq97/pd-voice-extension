"""Check every number in the manuscript's tables against results/tables/.

    python src/verify_manuscript.py

The manuscript is written by hand; the tables it quotes are generated. This
script parses the markdown tables back out and compares them to the CSVs, so
a number that drifts after a re-run is caught rather than published. Exits
non-zero on any mismatch.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

MS = C.ROOT / "manuscript" / "manuscript.md"
TOL = 0.0006          # quoted to 3 decimals
failures, checks = [], 0


def check(label, expected, got, tol=TOL):
    global checks
    checks += 1
    if expected is None or pd.isna(got) or abs(float(expected) - float(got)) > tol:
        failures.append(f"{label}: manuscript {expected}, tables {got}")


def rows_of(table_marker, text):
    """Return the markdown rows of the table following a marker line."""
    start = text.index(table_marker)
    block = text[start:].split("\n\n")
    for chunk in block:
        if chunk.lstrip().startswith("|"):
            return [
                [c.strip() for c in ln.strip().strip("|").split("|")]
                for ln in chunk.strip().splitlines()
                if ln.strip().startswith("|") and not set(ln) <= set("|- :")
            ][1:]
    raise AssertionError(f"no table after {table_marker!r}")


def num(cell):
    m = re.search(r"-?\d+\.\d+", cell.replace("**", ""))
    return float(m.group()) if m else None


def main() -> None:
    text = MS.read_text()
    summ = pd.read_csv(C.TABLES / "seed_sensitivity_summary.csv").set_index(
        ["dataset", "model"])
    cv = pd.read_csv(C.TABLES / "cv_comparison.csv")
    loso = cv[cv.regime == "C_loso"].set_index(["dataset", "model"])
    regb = cv[cv.regime == "B_group_kfold"].set_index(["dataset", "model"])
    rt = pd.read_csv(C.TABLES / "recording_type.csv")
    rtb = rt[rt.regime == "B_group_kfold"].set_index(["model", "rec_type"])
    ext = pd.read_csv(C.TABLES / "external_validation.csv").set_index("config")

    # --- Table 3: accuracy by regime ------------------------------------
    for r in rows_of("**Table 3.**", text):
        ds, model = r[0], r[1]
        key = (ds, model)
        check(f"T3 {ds} {model} A", num(r[2]), summ.loc[key, "A_mean"])
        check(f"T3 {ds} {model} A sd",
              float(re.search(r"\(([\d.]+)\)", r[2]).group(1)),
              summ.loc[key, "A_sd"])
        check(f"T3 {ds} {model} B", num(r[3]), summ.loc[key, "B_mean"])
        check(f"T3 {ds} {model} C", num(r[4]), loso.loc[key, "accuracy"])
        infl = re.findall(r"-?\d+\.\d+", r[5])
        check(f"T3 {ds} {model} inflation", float(infl[0]),
              summ.loc[key, "inflation_mean"])
        check(f"T3 {ds} {model} ci_lo", float(infl[1]),
              summ.loc[key, "inflation_ci_lo"])
        check(f"T3 {ds} {model} ci_hi", float(infl[2]),
              summ.loc[key, "inflation_ci_hi"])

    # --- Table 4: secondary metrics, regime B ---------------------------
    for r in rows_of("**Table 4.**", text):
        key = (r[0], r[1])
        for col, name in zip(r[2:6], ["roc_auc", "mcc", "recall_pd", "recall_hc"]):
            check(f"T4 {key} {name}", num(col), regb.loc[key, name])

    # --- Table 5: recording type ----------------------------------------
    for r in rows_of("**Table 5.**", text):
        model = r[0]
        for cell, t in zip(r[1:5], ["vowel", "number", "sentence", "word"]):
            check(f"T5 {model} {t}", num(cell), rtb.loc[(model, t), "accuracy_mean"])

    # --- Table 6: independent cohort ------------------------------------
    for r, cfg in zip(rows_of("**Table 6.**", text), ["vowel_only", "full_train"]):
        check(f"T6 {cfg} sens_rec", num(r[2]),
              ext.loc[cfg, "test_sensitivity_recording"])
        check(f"T6 {cfg} sens_subj", num(r[3]),
              ext.loc[cfg, "test_sensitivity_speaker"])
        check(f"T6 {cfg} train spec", num(r[4]),
              ext.loc[cfg, "train_cv_specificity_subject_level"])
        assert r[1] == ext.loc[cfg, "selected_model"], (
            f"T6 {cfg}: manuscript says {r[1]}, tables say "
            f"{ext.loc[cfg, 'selected_model']}")

    # --- prose claims ----------------------------------------------------
    infl_all = summ["inflation_mean"]
    check("prose: min inflation", 0.058, infl_all.min())
    check("prose: max inflation", 0.155, infl_all.max())
    check("prose: best rec-type cell", 0.596, rtb["accuracy_mean"].max())
    hc_b = summ.loc["UCI/MDVP", "recall_hc_B_mean"]
    hc_a = summ.loc["UCI/MDVP", "recall_hc_A_mean"]
    check("prose: UCI HC recall B min", 0.285, hc_b.min())
    check("prose: UCI HC recall B max", 0.392, hc_b.max())
    check("prose: UCI HC recall A min", 0.583, hc_a.min())
    check("prose: UCI HC recall A max", 0.804, hc_a.max())
    check("prose: UCI baseline", 0.754,
          cv[cv.dataset == "UCI/MDVP"].majority_baseline.iloc[0], tol=0.0005)

    # speaker counts quoted in 4.5
    for cfg, quoted in [("vowel_only", 16), ("full_train", 10)]:
        got = round(ext.loc[cfg, "test_sensitivity_speaker"] * 28)
        checks_ok = quoted == got
        if not checks_ok:
            failures.append(f"4.5 {cfg}: manuscript {quoted} speakers, tables {got}")

    print(f"{checks} numeric checks against results/tables/")
    if failures:
        print(f"\n{len(failures)} MISMATCH(ES):")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("all manuscript numbers match the generated tables")


if __name__ == "__main__":
    main()
