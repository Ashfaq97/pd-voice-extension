# Subject-level leakage in Parkinson's disease voice classification

A methodological correction. Two widely-used public voice datasets are re-analysed
under recording-level and subject-level cross-validation to quantify how much
published accuracy is attributable to the same speaker appearing in both training
and test sets.

**Author:** Ashfaq Hussain Mohammad · **Senior author:** Prof. Li Liu

This repository supersedes the analysis in the 2023 MS thesis *"Key Feature Analysis
for Predicting Speech of People with Parkinson's Disease in Machine Learning"*
(CSUN). The thesis split at the recording level, tuned on the full dataset, and
scaled before splitting; its headline figures (XGBoost ~96.6% train / 93.2% test)
are inflated and appear here only as an explicitly labelled "as originally
reported" comparison.

---

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
brew install libomp          # macOS only; xgboost links against it
pip install -r requirements.txt

python src/build_datasets.py   # data/raw/ -> data/processed/
python src/test_traps.py       # verify the data guards still fire
```

Everything in `data/processed/`, `results/tables/` and `results/figures/` is
generated. Nothing there is hand-edited. `data/raw/` is never written to.

---

## Data provenance

### Included

| File | Source | Obtained | Shape | Notes |
|---|---|---|---|---|
| `data/raw/parkinsons.csv` | UCI ML Repository #174, *Parkinsons* (Little et al. 2007). <https://archive.ics.uci.edu/dataset/174/parkinsons> | 2023-09-27 | 195 × 24 | Byte-identical to the UCI `parkinsons.data` distribution. sha256 `a36116c2deac0789…` |
| `data/raw/sakar/train_data.txt` | UCI ML Repository #301, *Parkinson Speech Dataset with Multiple Types of Sound Recordings* (Sakar et al. 2013). <https://archive.ics.uci.edu/dataset/301/> | archive downloaded 2022-09-25; files bear UCI's own 2014-06-11 timestamps | 1040 × 29 | No header row. sha256 `eb1cec4315f2b35b…` |
| `data/raw/sakar/test_data.txt` | same archive | same | 168 × 28 | No header row, no `UPDRS` column. sha256 `bd358e986f4fdef4…` |
| `data/raw/sakar/Parkinson_Multiple_Sound_Recording.rar` | same archive, kept intact | 2022-09-25 | 94,343 B | The distribution as downloaded, retained so the two files above can be re-verified. sha256 `24113bc9c7bcb76d…` |

The Sakar files were **not** re-downloaded. They were recovered from the original
2022 archive, which was still on the author's machine; the 2023 working copy in the
project directory had been corrupted (see below). Three identical copies of the
archive existed (`md5 5ea5aab434600aef2f72449ff05f43cb`).

### Dataset composition

|  | UCI / MDVP | Sakar train | Sakar test |
|---|---|---|---|
| Recordings | 195 | 1,040 | 168 |
| Subjects | 32 | 40 | 28 |
| PD / HC subjects | 24 / 8 | 20 / 20 | 28 / 0 |
| Recordings per subject | 6 (29 subjects), 7 (S21, S27, S35) | 26 | 6 |
| Features | 22 (MDVP) | 26 (Praat) | 26 (Praat) |
| Majority-class baseline | 0.7538 | 0.5000 | 1.0000 |

Subject IDs: UCI = `name` with the trailing `_N` stripped (`phon_R01_S01_1` →
`phon_R01_S01`); Sakar = the `id` column. Train and test subject IDs are namespaced
(`sakar_01`, `sakartest_01`) because the two files both number from 1 and the
cohorts are disjoint.

**The UCI subject count is 32 / 24 PD, not the 31 / 23 reported in the 2023
thesis.** The thesis figure was an undercount; the corrected value is used
throughout and flagged in the manuscript's Methods.

Sakar recording types, by within-subject recording index (verified empirically in
`src/run_recording_type.py`, not assumed): **1–3** sustained vowels, **4–13**
numbers 1–10, **14–17** short sentences, **18–26** words.

### Excluded

Everything in `data/raw/excluded/` is retained for the record and used by nothing
except `src/test_traps.py`.

| File | Obtained | Why excluded |
|---|---|---|
| `train_data_corrupted_2023.txt` | 2023-04-04 | The 2023 working copy of Sakar `train_data.txt`, 1 header + 1,222 rows. Two defects. **(a)** 178 rows from `youtube_data.txt` were appended with 28 of 29 fields, shifting every column by one so the class label landed in `UPDRS` and `class_information` read NaN. **(b)** 4 fabricated rows under `id 41` whose jitter/shimmer are fractions where every real row is a percentage (`Jitter_local` 0.012–0.069 against a true range of 0.190–14.376), i.e. ~100× off. Superseded by the pristine archive. Kept as the fixture that proves the guards in `build_datasets.py` fire. |
| `data.csv` | 2023-03-20 | Duplicate of `parkinsons.csv` with the header and `name` column stripped. Confirmed numerically identical (features, then `status`). |
| `actual_test_data.txt` | 2023-01-05 | 5 rows, the first duplicated. The 4 unique rows are exactly the mis-scaled `id 41` rows already present in the corrupted training file (exact float match). Not a test set. |
| `real_test_data_2.rtf` | 2023-01-05 | RTF-wrapped copy of the same 4 rows. |
| `x_test.csv` | 2023-10-20 | A saved recording-level test split from `less_features_parkinsons_prediction.ipynb`: 59 recordings spanning **31 of the 32 subjects**. Retained as direct evidence of the leak this paper measures — almost every subject in that test set also appeared in its training set. |
| `mdvp_voice_dataset.csv` | 2023-10-20 | figshare, Hlavnička et al. 2019, *Synthetic vowels of speakers with Parkinson's disease and Parkinsonism*. **Synthesised** signals built to benchmark pitch-detection algorithms, not patient recordings. Also has no `status` column, mixed units (jitter 0.00168–4.438, a 2,642× spread), and is not a PD/HC set at all: 10 MSA, 10 HC, 10 PD, 8 PSP — 18 of 38 rows are atypical parkinsonian syndromes. Not usable for external validation and not rescalable. |
| `youtube_data.txt` | 2023-01-26 | 178 rows, 30 IDs, **all labelled PD, no controls**. Extraction method unknown; subjects never consented; labels taken from video descriptions rather than clinical diagnosis. Byte-identical to the 178 rows appended to the corrupted training file. Fails provenance, ethics and label quality. |
| `yt_pd10_npd10.csv` / `.txt` | 2023-09-13 | 86 rows, 15 subjects, same YouTube pipeline. Excluded for the same reasons. |

`Test Recordings/` (YouTube-sourced audio) is git-ignored and must not be
redistributed, for the same consent reasons.

### A note on the Sakar test set

`test_data.txt` contains **28 PD speakers and no healthy controls**, and only
sustained vowels (6 per speaker: /a/ and /o/, three times each). This is the
published Sakar et al. 2013 protocol, not a defect in the file.

It follows that on this set **specificity and MCC are undefined** (there are no
true negatives to find, and `y_true` has zero variance), the confusion matrix
collapses to one row, and accuracy is identical to sensitivity — a model that
predicts "PD" unconditionally scores 1.000. The corresponding section of the paper
is therefore an *independent-cohort sensitivity check*, not an external validation,
and is reported as such.

---

## Layout

```
data/raw/          never modified, exactly as downloaded
data/raw/excluded/ retained for the record; used only by src/test_traps.py
data/processed/    written only by src/build_datasets.py
src/               all analysis code
results/tables/    generated
results/figures/   generated, never hand-edited
notebooks/archive/ 2023 notebooks, reference only -- do not run
manuscript/        paper draft
```

## Reproducing

Run in order:

| # | Command | Writes | Runtime |
|---|---|---|---|
| 1 | `python src/build_datasets.py` | `data/processed/*.csv`, `manifest.json` | <1 s |
| 2 | `python src/test_traps.py` | — (checks the data guards fire) | <1 s |
| 3 | `python src/run_cv_comparison.py` | `cv_comparison.csv`, `_folds.csv`, `cv_best_params.csv`, `oof_predictions.csv` | ~8 min |
| 4 | `python src/run_seed_sensitivity.py` | `seed_sensitivity.csv`, `seed_sensitivity_summary.csv` | ~18 min |
| 5 | `python src/run_recording_type.py` | `recording_type.csv`, `recording_type_profile.csv` | ~3 min |
| 6 | `python src/run_external_validation.py` | `external_validation.csv`, `_all_models.csv` | ~2 min |
| 7 | `python src/run_interpretability.py` | `shap_importance.csv`, `lr_coefficients.csv`, `collinearity_report.csv`, figures 4/4b | ~2 min |
| 8 | `python src/make_figures.py` | figures 1, 2, 3, 5 | <1 min |
| 9 | `python src/verify_checkpoint.py` | `checkpoint_comparison.csv`, `_rectype.csv` | <1 min |
| 10 | `python src/verify_manuscript.py` | — (144 checks of the manuscript against the tables) | <1 s |

### Walkthrough notebook

`notebooks/leakage_walkthrough.ipynb` is a guided tour for presenting the result. It reads
the precomputed tables rather than recomputing them, so the whole notebook runs in about
five seconds, and it demonstrates the leak live (100% vs 0% of test recordings sharing a
speaker with training) plus one short live model comparison.

```bash
jupyter lab notebooks/leakage_walkthrough.ipynb
```

The notebooks in `notebooks/archive/` are the 2023 analysis, kept for reference only. They
contain the errors this work corrects and should not be presented or run.

Steps 3-7 are independent of each other and depend only on step 1. Step 8 needs 3, 4 and
5; step 9 needs 3 and 5; step 10 needs 3-7.

### Verification

Two scripts exist so that nothing in the write-up drifts away from the data.

`verify_checkpoint.py` compares the run against the figures recorded in the project
handoff, and against the handoff's recording-type table, flagging any deviation beyond
0.02. It also asserts the leakage mechanism directly: under regime A, 100% of test
recordings come from a speaker already in the training fold; under regime B, 0%.

`verify_manuscript.py` parses the manuscript's tables back out of the markdown and checks
all 144 quoted numbers against the generated CSVs. It exits non-zero on any mismatch, so
re-running the analysis and forgetting to update the paper is a hard error rather than a
silent inconsistency.

`data/processed/manifest.json` records the sha256 of every input and output, so a
rebuild can be checked against the run that produced the published tables.

## Findings

Accuracy falls by 5.8 to 15.5 percentage points when folds are formed by speaker rather
than by recording — measured as a paired difference across 10 fold seeds, all p < 0.001,
on both corpora. The inflation is concentrated in flexible models: logistic regression
loses 0.065 (UCI) and 0.058 (Sakar) while the tree ensembles lose 0.129 to 0.143. Under
leave-one-subject-out on UCI, logistic regression is the most accurate model (0.836)
having been last of five under recording-level validation.

Healthy-control recall on UCI falls from 0.583-0.804 to 0.285-0.392, against a
majority-class baseline of 0.754. No recording type rescues performance: the best of 20
model-by-type cells on Sakar is 0.596 against chance 0.500, and sustained vowels are the
weakest task for four of five models.

Two results differ from what the project handoff anticipated, and are reported as such:

- **Leaky models do not use different features.** SHAP importance rankings under
  recording-level and subject-level training correlate at rho +0.87 to +0.96 with 4-5 of
  the top 5 features shared. The leak is memorisation within a fixed feature geometry,
  not feature substitution.
- **No feature survives correction.** After pruning definitional duplicates
  (`Jitter:DDP` is exactly 3x `MDVP:RAP`; `Shimmer:DDA` exactly 3x `Shimmer:APQ3`) and
  then by VIF, with standard errors clustered by speaker, zero coefficients survive
  Benjamini-Hochberg at FDR 0.05 on either corpus.

## Method constraints

These are load-bearing. Deviating from any of them invalidates the central claim.

1. **Group by subject.** `StratifiedGroupKFold` for k-fold, `LeaveOneGroupOut` for LOSO.
2. **Nested tuning with a grouped inner loop.** `GridSearchCV` sits inside each outer
   fold and its own CV is grouped. An ungrouped inner loop reintroduces the leak
   being measured.
3. **Preprocessing inside the pipeline.** `StandardScaler` is a pipeline step, fit
   per fold, never on full data.
4. **Three regimes, identical in every other respect**, so the contrast is
   like-for-like: `A_leaky_recording` (StratifiedKFold over recordings — reproduces
   the thesis and acts as the control), `B_group_kfold`, `C_loso`.
5. **Report beyond accuracy**: ROC-AUC, MCC, per-class recall — healthy-control
   recall especially — the majority-class baseline, and per-fold standard deviation.
   With 8 HC subjects in UCI a fold holds ~1.6 controls and metrics swing.
6. **The held-out set is touched exactly once.** No tuning, no model selection.
7. **Seed everything** (`random_state=27`) and pin library versions.

`UPDRS` is dropped in `build_datasets.py` and never written to `data/processed/`:
in Sakar it is 1 for every HC row and 5–55 for every PD row with zero overlap, so
including it as a predictor yields 100% accuracy for free.
