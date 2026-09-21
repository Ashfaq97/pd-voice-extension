# Project handoff: Parkinson's voice detection — methodological correction paper

You are taking over a research project start to finish. Read this whole document before
running anything. It contains findings, data traps, and correctness constraints that are
not recoverable from the files alone.

**Repo:** `pd-voice-extension` (GitHub, already connected)
**Local data:** the project directory you have access to
**Author:** Ashfaq Hussain Mohammad (first author). Prof. Li Liu is senior/last author,
nominal — he is not doing analysis. Do not wait on him for anything.
**Target journal:** *Computers in Biology and Medicine* (primary) or *Journal of
Biomedical Informatics* (secondary). IEEE Access if speed is preferred.

---

## 1. Background

In 2023 the author completed an MS thesis at CSUN: *"Key Feature Analysis for Predicting
Speech of People with Parkinson's Disease in Machine Learning."* It trained five
classifiers (Logistic Regression, KNN, Decision Tree, Gradient Boosting, XGBoost) on the
UCI/Kaggle MDVP voice dataset and reported XGBoost as best at ~96.6% training / 93.2%
testing accuracy.

Those numbers are inflated by a validation error. The corrected analysis is the paper.

### The error

The UCI dataset has ~6 recordings per subject. The thesis split at the **recording**
level, so the same speaker appeared in both training and test sets. The models could
recognise voices rather than learn disease markers — subject-level data leakage.

Two compounding errors in the same code:

1. **Tuning contamination.** `GridSearchCV` was fit on the entire dataset, then
   performance reported on rows included in that search. One cell even scores with
   `best_clf.score(X, Y)` on all data.
2. **Preprocessing leakage.** `StandardScaler` fit before splitting.

### The corrected result (already established — reproduce, do not re-derive)

**UCI/MDVP** — 195 recordings, 32 subjects (24 PD / 8 HC), 22 features:

| Model | Recording-level split | Subject-level split | Drop |
|---|---|---|---|
| Gradient Boosting | 0.908 | 0.795 | −0.113 |
| XGBoost | 0.903 | 0.780 | −0.123 |
| KNN | 0.892 | 0.769 | −0.123 |
| Decision Tree | 0.851 | 0.697 | −0.154 |
| Logistic Regression | 0.846 | 0.836 | −0.010 |

Leave-one-subject-out agrees with the subject-level column, so this is not a
fold-construction artifact.

**Sakar** — 1,040 recordings, 40 subjects (20 PD / 20 HC), 26 features:

| Model | Recording-level | Subject-level | Drop |
|---|---|---|---|
| XGBoost | 0.701 | 0.543 | −0.158 |
| Decision Tree | 0.615 | 0.486 | −0.130 |
| Gradient Boosting | 0.688 | 0.578 | −0.110 |
| KNN | 0.664 | 0.575 | −0.089 |
| Logistic Regression | 0.637 | 0.570 | −0.066 |

**Recording-type breakdown (Sakar, subject-level, chance = 0.50):**

| Model | Vowels | Numbers | Sentences | Words |
|---|---|---|---|---|
| Logistic Regression | 0.558 | 0.535 | 0.588 | 0.489 |
| Gradient Boosting | 0.558 | 0.565 | 0.556 | 0.528 |
| XGBoost | 0.567 | 0.560 | 0.562 | 0.550 |
| Decision Tree | 0.533 | 0.532 | 0.569 | 0.506 |
| KNN | 0.400 | 0.585 | 0.525 | 0.553 |

### The three claims the paper makes

1. Subject-level leakage inflates accuracy by 7–16 points on two independent public
   datasets.
2. The inflation is **concentrated in flexible models**. Tree ensembles lose 11–16
   points; logistic regression loses 1–7. This **reverses the published model ranking** —
   under honest validation LR is the best model, not XGBoost.
3. No recording type rescues performance. Sustained vowels do not beat connected speech;
   all types sit just above chance. This forecloses the obvious reviewer objection.

Also report: healthy-control recall collapses (UCI ~0.75 → ~0.35–0.40), and the UCI
majority-class baseline is 0.754, so XGBoost's honest 0.780 is barely above trivial.

**This is not a "we built a classifier" paper.** That space is saturated. It is a
methodological correction. Frame every section accordingly.

---

## 2. Data traps — read before touching any file

These are confirmed, not hypothetical. Each one silently corrupts results.

| # | Trap | Detail | Required action |
|---|---|---|---|
| 1 | **`train_data.txt` is corrupted** | 1,222 rows. The first 1,044 are Sakar (29 cols, ends `UPDRS,class_information`). 178 rows from `youtube_data.txt` were appended with only 28 cols, so columns shift by one: the class label lands in `UPDRS` and `class_information` reads NaN. | Drop rows where `class_information` is NaN. |
| 2 | **Mis-scaled subject 41** | After trap 1, subject 41 has 4 rows where every other subject has 26. Those 4 are ~100× off scale (jitter 0.012–0.069 vs 0.19–14.38) — fractions among percentages. | Drop `id == 41`. Result: exactly 1,040 rows, 40 subjects, 26 each, 20/20 balanced. |
| 3 | **UPDRS is a target leak** | In Sakar, UPDRS is 1.0 for every HC and 5–55 for every PD subject. Including it gives 100% accuracy for free. | **Never** use as a predictor. Drop from feature list. |
| 4 | **`actual_test_data.txt` is not a test set** | 5 rows, first duplicated, = the same 4 mis-scaled subject-41 rows that are in the training file. | Do not use. |
| 5 | **figshare set is synthetic** | `mdvp_voice_dataset.csv` (38 rows) comes from Hlavnička et al. 2019, *"Synthetic vowels of speakers with Parkinson's disease and Parkinsonism"* — synthesized signals built to benchmark pitch detectors, not patient recordings. It also has no `status` column and mixed units (jitter 0.00168–4.438, a 2,600× spread, with 7/10 HC rows in percent). | **Excluded from the paper.** Do not use for external validation. Do not attempt to rescale. |
| 6 | **YouTube data is retired** | `youtube_data.txt` (178 rows, all labelled PD, no controls) and `yt_pd10_npd10.*` (86 rows, 15 subjects). Extraction method unknown; subjects never consented; labels from video descriptions, not clinical diagnosis. | **Excluded.** Fails provenance, ethics, and label quality. Do not revive. |
| 7 | **Subject count** | Thesis says 31 subjects / 23 PD. Parsing `name` in `parkinsons.csv` (`phon_R01_S01_1` → `phon_R01_S01`) gives **32 / 24**. | Use 32/24 and note the correction. |
| 8 | **`data.csv` is a duplicate** | Same 195 rows as `parkinsons.csv` with header and `name` column stripped. | Ignore. Use `parkinsons.csv`. |

### Sakar recording-type order

Each subject's 26 rows are in fixed order: **1–3** sustained vowels, **4–13** numbers
1–10, **14–17** short sentences, **18–26** words. This was confirmed empirically, not
assumed — indices 1–3 average ~360 glottal pulses at 1–4% unvoiced frames; 4–13 and 18–26
average ~55–62 pulses at 15–48% unvoiced; 14–17 sit at ~165. Re-verify this before using
it, and assign type by within-subject `cumcount()`.

---

## 3. Methodology constraints — non-negotiable

Any deviation invalidates the paper's central claim.

1. **Group by subject.** Use `StratifiedGroupKFold` (and `LeaveOneGroupOut` for LOSO)
   with subject as the group. UCI subject ID = `name` with the trailing `_N` stripped.
   Sakar subject ID = `id` column.
2. **Nested tuning, grouped inner loop.** `GridSearchCV` goes *inside* each outer fold,
   and its inner CV must also be grouped. Tuning on ungrouped inner folds reintroduces
   the leak you are measuring.
3. **Preprocessing inside the pipeline.** `StandardScaler` as a pipeline step so it is fit
   per fold, never on full data.
4. **Run all three regimes** on both datasets so the contrast is like-for-like:
   - `A_leaky_recording` — `StratifiedKFold` over recordings (reproduces the thesis)
   - `B_group_kfold` — `StratifiedGroupKFold` by subject
   - `C_loso` — leave-one-subject-out
   Regime A must be identical to B/C in every other respect. It is the control.
5. **Report beyond accuracy.** Accuracy, ROC-AUC, MCC, per-class recall (especially
   healthy-control recall), and the majority-class baseline. Accuracy alone hides the
   specificity collapse. Include per-fold standard deviation — with 8 HC subjects in UCI,
   each fold holds ~1.6 controls and metrics swing.
6. **Touch the held-out test set exactly once.** No tuning, no model selection, no
   "let me just try one more thing" against it.
7. **Seed everything** (`random_state=27` used previously) and pin library versions in
   `requirements.txt`. The paper must be reproducible from a clean clone.

---

## 4. Work plan

Commit after each phase with a clear message. Push to `pd-voice-extension`.

### Phase 0 — Inventory and verify

Before writing analysis code, confirm what is actually on disk. Do not assume the file
list below is complete or correctly named.

```
data/raw/
  parkinsons.csv                 195 rows, 24 cols  (UCI/MDVP — KEEP)
  train_data.txt                 1222 rows           (Sakar, corrupted — see trap 1)
  test_data.txt                  ~168 rows           (Sakar held-out — NEEDED, may be missing)
  data.csv                       195 rows            (duplicate — ignore)
  mdvp_voice_dataset.csv         38 rows             (figshare, synthetic — excluded)
  youtube_data.txt               178 rows            (retired)
  yt_pd10_npd10.csv / .txt       86 rows             (retired)
  actual_test_data.txt           5 rows              (not a test set)
  *.ipynb                        2023 notebooks      (archive, do not run)
```

**`test_data.txt` is the one blocker.** It is the Sakar held-out set (~168 recordings,
~28 subjects) and is referenced by `Parkinson_s_Speech_Analysis.ipynb`, so it existed in
2023. Search the local directory for it. If it is absent, download the Sakar dataset from
the UCI ML Repository — *"Parkinson Speech Dataset with Multiple Types of Sound
Recordings"* (Sakar et al. 2013) — and use its `test_data.txt`. Note in the README that it
was re-downloaded, with the date and URL.

If you cannot obtain it, **stop and report**. Do not substitute another file, do not
carve a test set out of the training data and call it external, and do not proceed to
Phase 4 without it.

Report the inventory before continuing.

### Phase 1 — Repo structure and clean data

File sprawl (`Copy1`, `-Results-Copy1`, an empty notebook, four near-identical `.txt`
files) is what produced traps 1, 2 and 4 in the first place. Fix the structure first.

```
pd-voice-extension/
├── data/
│   ├── raw/          # never modified, exactly as downloaded
│   └── processed/    # written only by scripts in src/
├── src/
├── results/
│   ├── tables/
│   └── figures/
├── notebooks/archive/   # 2023 notebooks, for reference only
├── requirements.txt
└── README.md
```

`README.md` records provenance for **every** raw file: source, URL, download date, row
and column counts, and any known defect. Anyone must be able to rebuild `data/processed/`
from `data/raw/` by running scripts in `src/`.

Write `src/build_datasets.py` producing:
- `data/processed/uci_mdvp.csv` — from `parkinsons.csv`, with a `subject` column
- `data/processed/sakar_train.csv` — traps 1 and 2 applied; assert 1,040 rows / 40
  subjects / 26 each / 20 PD / 20 HC and fail loudly otherwise
- `data/processed/sakar_test.csv` — same cleaning logic applied to `test_data.txt`

Assertions, not comments. A silent regression here poisons everything downstream.

### Phase 2 — Reproduce the leakage analysis

`src/run_cv_comparison.py`: all five models × three regimes × both datasets, honouring
every constraint in §3. Write `results/tables/cv_comparison.csv`.

**Verification:** compare against the tables in §1. Small differences from library-version
drift are expected. If any number differs by more than ~0.02, **stop and report the
discrepancy**. Do not adjust code to match the expected values — the §1 numbers are a
checkpoint, not a target. If your run is right and the checkpoint is wrong, say so.

LOSO on Sakar with nested tuning is slow. Reduce grid size before dropping the regime,
and record whatever you chose in the script docstring.

### Phase 3 — Recording-type breakdown

`src/run_recording_type.py`: re-verify the type ordering empirically (§2), then run
regimes A and B per type. Fixed hyperparameters are acceptable here for speed and
comparability — state that choice in the docstring. Write
`results/tables/recording_type.csv` and verify against §1.

### Phase 4 — External validation (needs `test_data.txt`)

Train on `sakar_train.csv` with subject-level CV for model selection only. Then predict
once on `sakar_test.csv`. Report accuracy, sensitivity, specificity, MCC, and the
confusion matrix. One pass. Write `results/tables/external_validation.csv`.

### Phase 5 — Interpretability

SHAP on **logistic regression**, not XGBoost — LR is the model that wins under honest
validation, and a linear model with real coefficients is easier to defend clinically. Use
`LinearExplainer` (or `KernelExplainer` if needed), on the correctly-scaled pipeline.

Produce: global summary plot, dependence plots for the top features, and a comparison of
which features matter under leaky vs. subject-level training — if the leaky model relies
on different features, that is direct evidence it was learning speaker identity, and it
belongs in the paper.

Also report standardised LR coefficients with confidence intervals. Do not report XGBoost
feature importances as a headline.

### Phase 6 — Figures and manuscript

Figures (`results/figures/`, generated by scripts, never hand-edited):
1. Schematic: recording-level vs subject-level splitting
2. Grouped bar chart: accuracy by model × regime, both datasets, with the majority-class
   baseline drawn in
3. Recording-type breakdown
4. SHAP summary
5. Healthy-control recall, leaky vs honest

Draft `manuscript/manuscript.md` in IMRaD:
1. **Introduction** — PD voice detection; the accuracies reported in the literature
2. **Related work** — how widely the UCI set is used and how it is typically split
3. **Methods** — subject-level CV, nested grouped tuning, per-fold preprocessing;
   datasets with exact subject/recording counts
4. **Results** — both datasets, all three regimes, side by side
5. **External validation** — Sakar held-out set
6. **Discussion** — implications for the published literature; why flexible models are
   hit hardest
7. **Limitations** — small subject counts (32 and 40); two datasets only; single feature
   extraction pipeline per dataset; no clinical severity modelling

The original thesis PDF is available in the project for the Related Work section and for
quoting the original reported figures. Note in Methods that the corrected subject count is
32/24, not the 31/23 stated in the thesis.

---

## 5. Things that will go wrong if you are not careful

- **Do not include UPDRS as a feature.** If you see accuracy near 1.0 on Sakar, this is
  why. Check first.
- **Do not train on `train_data.txt` directly.** It is corrupted. Always go through
  `build_datasets.py`.
- **Do not reintroduce leakage through the inner CV loop.** The most common way this
  analysis fails is a grouped outer loop wrapped around an ungrouped `GridSearchCV`.
- **Do not report only accuracy.** The specificity collapse is half the finding.
- **Do not treat the figshare or YouTube data as recoverable.** Both are excluded for
  documented reasons. If you think you have found a way to use them, raise it — do not
  just include them.
- **Do not tune against the held-out test set.**
- **Do not quietly adjust code until numbers match §1.** Report discrepancies.
- **Do not restore the thesis's headline figures anywhere** except as the explicitly
  labelled "as originally reported" comparison.

---

## 6. Definition of done

- Clean clone + `pip install -r requirements.txt` + documented command sequence
  reproduces every table and figure
- `results/tables/` holds cv_comparison, recording_type, external_validation
- `results/figures/` holds all five figures, script-generated
- `manuscript/manuscript.md` is a complete IMRaD draft with real numbers, no placeholders
- `README.md` documents provenance for every raw file and how to reproduce everything
- All work committed and pushed to `pd-voice-extension`

---

## 7. How to work

Go step by step — this is a research project, and correctness matters more than speed.

Report at the end of each phase: what you ran, what came out, what surprised you. Flag
anything that contradicts this document rather than working around it; this handoff was
written from an analysis done elsewhere and may itself contain errors.

Start with Phase 0 and report the inventory before writing any analysis code.