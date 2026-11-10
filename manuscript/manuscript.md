# Subject-level data leakage inflates reported accuracy in Parkinson's disease voice classification

**Ashfaq Hussain Mohammad**¹, **Li Liu**¹

¹ Department of Computer Science, California State University, Northridge

---

## Abstract

Machine-learning studies of Parkinson's disease (PD) detection from voice routinely
report accuracies above 90% on small public datasets in which each speaker contributes
several recordings. When such a dataset is partitioned at the level of the recording
rather than the speaker, recordings from the same person appear in both the training and
the test set, and a classifier can succeed by recognising the speaker instead of the
disease. We quantify the resulting inflation on two independent public corpora. Holding
models, hyper-parameter grids, preprocessing and random seed fixed, and varying only the
partitioning rule, accuracy falls by 5.8 to 15.5 percentage points when folds are formed
by speaker rather than by recording (all *p* < 0.001, paired across 10 fold seeds). The
inflation is not uniform across model families: logistic regression loses 6.5 points on
the UCI/MDVP corpus and 5.8 on the Sakar-2013 corpus, while gradient boosting, XGBoost and
k-nearest neighbours lose 11.0 to 15.5. Because the most flexible models are the most
inflated, the published ordering of models does not survive honest validation. Accuracy
alone conceals the severity: healthy-control recall on UCI/MDVP falls from 0.58–0.80 to
0.29–0.39, and the majority-class baseline on that corpus is 0.754, so most models are
at or below trivial performance once leakage is removed. No recording type rescues
performance — across 20 model-by-type combinations on Sakar-2013 the best subject-level
accuracy is 0.596 against a chance level of 0.500, and sustained vowels are the weakest
type for four of five models. On an independent cohort of 28 unseen PD speakers,
sensitivity is 0.41–0.59. Finally, with standard errors clustered by speaker, no
acoustic feature in either corpus retains a non-zero association with PD status after
correction for multiple comparisons. We conclude that a substantial part of the reported
performance in this literature is an artefact of partitioning, and we release code that
reproduces every number from the raw distributions.

**Keywords:** Parkinson's disease, dysphonia, data leakage, cross-validation, subject-level
partitioning, reproducibility

---

## 1. Introduction

Hypokinetic dysarthria appears early in Parkinson's disease and affects the majority of
patients over the course of the illness. Because voice can be captured with a microphone
and characterised by a small set of acoustic measures, automatic detection of PD from
speech has become an active area, with the promise of a cheap, non-invasive screening
aid.

The reported results are striking. On the UCI/MDVP corpus of Little et al., accuracies
in the 90–99% range are common, and individual studies have claimed near-perfect
discrimination. Taken at face value, these figures would place voice analysis among the
most accurate diagnostic tests in neurology, on a condition that has no definitive
ante-mortem biomarker.

They should not be taken at face value. The UCI/MDVP corpus contains 195 recordings from
32 speakers — roughly six recordings per person. If those 195 rows are shuffled and split
into training and test sets, then essentially every speaker in the test set has other
recordings of their own voice in the training set. We measured this directly: under a
stratified 5-fold split over recordings, **100%** of test recordings come from a speaker
already present in that fold's training data. A classifier can then reach a high score by
learning to recognise individual voices, a far easier task than learning a disease marker
and one with no clinical value whatsoever.

This is subject-level leakage. It is well documented in other biomedical domains, and it
is not subtle, but it is easy to introduce because the standard tools make it the default:
`train_test_split` and `KFold` partition rows, and nothing in their interface signals that
rows may be clustered within people.

This paper is a correction rather than a new classifier. The space of "we applied five
classifiers to the UCI voice dataset" papers is saturated; what is missing is a
quantitative account of how much of the reported performance survives honest validation.
We take as our starting point a study by the first author — an MS thesis that reported
XGBoost as the best model at 96.61% training and 93.22% testing accuracy — and re-analyse
its data under corrected methodology, extending the analysis to a second, independent
corpus.

Our contributions are:

1. A like-for-like quantification of accuracy inflation under recording-level versus
   subject-level partitioning, on two public corpora, with all other analysis choices
   held fixed and uncertainty estimated by repeating the whole comparison across ten fold
   constructions.
2. Evidence that the inflation is concentrated in high-capacity models, which reverses the
   published ordering of model families.
3. A recording-type analysis showing that no speech task — sustained vowels included —
   restores performance, foreclosing the most natural objection to the result.
4. An analysis of per-feature inference under speaker-clustered standard errors, showing
   that the same independence assumption that inflates validation also inflates the
   feature-level claims made in this literature.

---

## 2. Related work

### 2.1 The datasets and how they are partitioned

The UCI/MDVP corpus is the most heavily used resource in this area. It contains 195
sustained-vowel phonations from 32 speakers, 24 with PD and 8 healthy controls,
summarised by 22 acoustic measures derived from the Kay Pentax MDVP system. The
recording identifier encodes the speaker (`phon_R01_S01_1` through `_6`), so the
clustering is recoverable — but only if one looks for it. The dataset's own
documentation presents it as 195 instances, and a large fraction of published analyses
treat it that way.

The Sakar-2013 corpus provides an independent check with a different feature extraction
pipeline. Its training partition holds 1,040 recordings from 40 speakers, evenly balanced
at 20 PD and 20 controls, with 26 measures computed in Praat. Each speaker contributes 26
recordings in a fixed order spanning four speech tasks.

### 2.2 The specific analysis being corrected

The 2023 thesis underlying this work reported the results in Table 1. We reproduce them
here only as the "as originally reported" comparison.

**Table 1.** Accuracies as originally reported in the 2023 thesis.

| Model | Training accuracy (%) | "Testing" accuracy (%) |
|---|---|---|
| Logistic Regression | 88.13 | 86.44 |
| KNN | 84.74 | 79.66 |
| Decision Tree | 89.83 | 81.35 |
| XGBoost | **96.61** | **93.22** |

Inspection of the archived analysis code identifies three compounding errors.

**Recording-level partitioning.** The split is `train_test_split(X, Y, test_size=0.3,
random_state=7)` over the 195 rows, with no speaker grouping.

**Tuning on the full dataset.** Hyper-parameter search is fit to all rows —
`gsearch1.fit(dataset[predictors], dataset['status'])` — and performance is then reported
on rows included in that search. One cell reports `best_clf.score(X, Y)`, scoring the
tuned model on the entire dataset.

**Preprocessing before splitting.** Feature scaling is fit on the full data rather than
within each fold.

A fourth issue concerns the second column of Table 1. It is not a held-out partition of
the UCI corpus. It is accuracy on a separately assembled set of recordings harvested from
public video, which contained **only PD speakers and no controls**, and for which only 15
of the 22 features could be extracted. On a cohort with no negative cases, accuracy
equals sensitivity and a classifier that answers "PD" unconditionally scores 100%. The
thesis text acknowledges the imbalance but reports the figure as a testing accuracy
nonetheless. That collection is excluded from the present work on provenance, consent and
label-quality grounds.

We also correct a counting error: parsing the speaker identifier yields **32 speakers, 24
with PD**, not the 31 and 23 stated in the thesis.

---

## 3. Methods

### 3.1 Data

**Table 2.** Corpora after cleaning.

| | UCI/MDVP | Sakar-2013 (training) | Sakar-2013 (held-out) |
|---|---|---|---|
| Recordings | 195 | 1,040 | 168 |
| Speakers | 32 | 40 | 28 |
| PD / control speakers | 24 / 8 | 20 / 20 | 28 / 0 |
| Recordings per speaker | 6 (29 speakers), 7 (3) | 26 | 6 |
| Features | 22 | 26 | 26 |
| Majority-class baseline | 0.754 | 0.500 | 1.000 |

Speaker identity is the `name` field with its trailing index removed for UCI/MDVP, and the
`id` column for Sakar.

Three data defects required handling and are documented in full in the repository. The
Sakar-2013 training file used in the 2023 analysis had been corrupted locally: 178 rows from
the excluded video-harvested collection had been appended with one field too few, shifting
every column and placing the class label in the `UPDRS` field, and four further rows
carried jitter and shimmer values expressed as fractions where every genuine row uses
percentages. We recovered the pristine distribution from the original archive and build
from that; the corrupted file is retained solely as a regression fixture.

`UPDRS` is excluded as a predictor and is not written to the processed data at all. In the
Sakar-2013 corpus it takes the value 1 for every control row and 5–55 for every PD row, with no
overlap, and including it yields perfect accuracy for free.

### 3.2 Validation regimes

Three regimes differ **only** in how rows are assigned to folds. Models, grids,
preprocessing, scoring and seed are identical throughout, so any difference between them
is attributable to the partitioning rule.

- **A — recording-level.** `StratifiedKFold` over recordings, in both the outer and the
  inner loop. This reproduces the conventional analysis and serves as the control.
- **B — subject-level.** `StratifiedGroupKFold` with speaker as the group, outer and inner.
- **C — leave-one-subject-out.** `LeaveOneGroupOut` outer, grouped inner.

Regime A's inner loop is ungrouped by design, matching its outer loop. Grouping only the
inner loop would produce a hybrid that neither reproduces conventional practice nor
measures anything of interest: the regime *is* the partitioning discipline, applied
consistently.

We verified the mechanism directly. Under regime A, 100% of test recordings belong to a
speaker present in the same fold's training set; under regime B, 0%. This is asserted in
the analysis code.

### 3.3 Tuning and preprocessing

Hyper-parameter search runs **inside** each outer fold, and its own cross-validation is
grouped in regimes B and C. A grouped outer loop wrapped around an ungrouped
`GridSearchCV` reintroduces precisely the leakage under study; this is, in our experience,
the most common way the analysis fails. Feature standardisation is a pipeline step and is
therefore fit within each training fold, never on full data.

Five models are compared: logistic regression, k-nearest neighbours, a decision tree,
gradient boosting and XGBoost, with grids of 5 to 12 points each. All randomness is seeded
at 27 and library versions are pinned.

### 3.4 Metrics and uncertainty

Metrics are computed on pooled out-of-fold predictions: every recording is predicted
exactly once, by a model that never saw it. We report accuracy, ROC-AUC, Matthews
correlation coefficient, per-class recall, and the majority-class baseline. Accuracy alone
is misleading on UCI/MDVP, where 75.4% of recordings are from PD speakers.

With 32 and 40 speakers, the assignment of speakers to folds is itself a material source
of variance. A single seed yields a falsely precise number. We therefore repeat the entire
regime A versus regime B comparison across ten fold seeds, varying only the partitioning
and holding model seeds fixed, and report the inflation as a **paired** difference within
each seed. Pairing removes the shared fold-difficulty component. Leave-one-subject-out has
a deterministic outer loop and is reported as a single value.

---

## 4. Results

### 4.1 Accuracy inflation

**Table 3.** Accuracy by regime. A and B are means over ten fold seeds (SD in
parentheses); C is deterministic. Inflation is the paired A−B difference with a 95%
confidence interval.

| Dataset | Model | A recording-level | B subject-level | C LOSO | Inflation (95% CI) |
|---|---|---|---|---|---|
| UCI/MDVP | Logistic Regression | 0.849 (0.010) | 0.785 (0.035) | **0.836** | 0.065 [0.037, 0.092] |
| UCI/MDVP | KNN | 0.912 (0.007) | 0.756 (0.023) | 0.759 | 0.155 [0.139, 0.172] |
| UCI/MDVP | Decision Tree | 0.856 (0.030) | 0.743 (0.048) | 0.738 | 0.114 [0.067, 0.161] |
| UCI/MDVP | Gradient Boosting | 0.916 (0.011) | 0.787 (0.022) | 0.810 | 0.129 [0.109, 0.149] |
| UCI/MDVP | XGBoost | 0.917 (0.012) | 0.786 (0.028) | 0.795 | 0.131 [0.103, 0.158] |
| Sakar-2013 | Logistic Regression | 0.630 (0.007) | **0.572** (0.017) | 0.519 | 0.058 [0.043, 0.072] |
| Sakar-2013 | KNN | 0.670 (0.008) | 0.559 (0.023) | 0.526 | 0.110 [0.094, 0.127] |
| Sakar-2013 | Decision Tree | 0.621 (0.017) | 0.534 (0.037) | 0.482 | 0.087 [0.060, 0.115] |
| Sakar-2013 | Gradient Boosting | 0.688 (0.005) | 0.558 (0.017) | 0.512 | 0.130 [0.116, 0.144] |
| Sakar-2013 | XGBoost | 0.697 (0.009) | 0.555 (0.017) | 0.516 | 0.143 [0.128, 0.158] |

Majority-class baselines: 0.754 (UCI/MDVP), 0.500 (Sakar). All paired differences are
significant at *p* < 0.001.

Every model on both corpora loses accuracy when the partition respects the speaker. The
loss ranges from 5.8 to 15.5 percentage points. Figure 2 presents the comparison
graphically with the baselines drawn in.

Two features of Table 3 deserve emphasis. First, on UCI/MDVP the majority-class baseline
is 0.754, and under subject-level validation only logistic regression clears it by a
meaningful margin; the decision tree at 0.743 falls **below** it, and KNN at 0.756 matches
it. A reader given only the recording-level column would conclude that four of five models
exceed 85% accuracy. Second, on Sakar-2013 the leave-one-subject-out column sits between 0.482
and 0.526 against a chance level of 0.500 — that is, at chance. Whatever the models are
learning on that corpus does not generalise to an unseen speaker at all.

### 4.2 The inflation is concentrated in flexible models

Logistic regression is the least inflated model on both corpora, at 0.065 and 0.058, while
gradient boosting and XGBoost lose 0.129 to 0.143 — roughly double. KNN, which memorises
training instances by construction, is the most inflated on UCI/MDVP at 0.155.

The consequence is a change in the ordering of models. Under recording-level validation on
UCI/MDVP, logistic regression is **last** of five (0.849, against 0.917 for XGBoost).
Under leave-one-subject-out it is **first** (0.836, against 0.810 for the next best), and
it is the only model whose MCC exceeds 0.5 (0.515, against 0.422 for gradient boosting).
On Sakar it is likewise the best model under subject-level 5-fold validation (0.572).

We state one qualification precisely rather than round it away. Under subject-level 5-fold
validation on UCI/MDVP the top three models are separated by less than their fold-seed
standard deviation — gradient boosting 0.787, XGBoost 0.786, logistic regression 0.785 —
and should be read as tied. The clean reversal appears under leave-one-subject-out, which
is the regime with no fold-construction variance and therefore the one we treat as the
reference. The defensible claim is that honest validation removes the apparent advantage
of tree ensembles, and that under the most stringent regime the linear model is ahead.

### 4.3 Accuracy conceals a collapse in specificity

**Table 4.** Secondary metrics under subject-level validation (regime B, seed 27).

| Dataset | Model | ROC-AUC | MCC | PD recall | Control recall |
|---|---|---|---|---|---|
| UCI/MDVP | Logistic Regression | 0.785 | 0.386 | 0.946 | 0.354 |
| UCI/MDVP | KNN | 0.779 | 0.279 | 0.925 | 0.292 |
| UCI/MDVP | Decision Tree | 0.690 | 0.213 | 0.830 | 0.375 |
| UCI/MDVP | Gradient Boosting | 0.668 | 0.331 | 0.884 | 0.417 |
| UCI/MDVP | XGBoost | 0.724 | 0.320 | 0.878 | 0.417 |
| Sakar-2013 | Logistic Regression | 0.617 | 0.148 | 0.546 | 0.602 |
| Sakar-2013 | KNN | 0.619 | 0.163 | 0.590 | 0.573 |
| Sakar-2013 | Decision Tree | 0.553 | 0.051 | 0.454 | 0.596 |
| Sakar-2013 | Gradient Boosting | 0.613 | 0.177 | 0.562 | 0.615 |
| Sakar-2013 | XGBoost | 0.632 | 0.187 | 0.560 | 0.627 |

Averaged over ten fold seeds, healthy-control recall on UCI/MDVP falls from 0.583–0.804
under recording-level validation to 0.285–0.392 under subject-level validation (Figure 5).
The models retain high PD recall — 0.83 to 0.95 — and achieve their accuracy largely by
predicting the majority class. Under honest validation, between 61% and 72% of healthy
speakers are labelled as having Parkinson's disease. A screening instrument with that
false-positive rate is not usable, and the accuracy column does not reveal it.

Per-fold standard deviations are large on UCI/MDVP under subject-level validation — up to
0.200 for logistic regression — because each of five folds holds approximately 1.6 control
speakers. This is an unavoidable consequence of the corpus having only eight controls and
is itself a reason for caution about any result derived from it.

### 4.4 No recording type rescues performance

The most natural objection is that the Sakar-2013 corpus mixes speech tasks, and that sustained
vowels — the standard clinical elicitation, and the only task in UCI/MDVP — would perform
better in isolation.

We first recovered the task structure empirically rather than assuming it. Sakar ships 26
recordings per speaker in a fixed order with no task label. Sustained vowels are long and
almost fully voiced, whereas isolated numbers and words are short and heavily unvoiced. The
per-index acoustic profile separates cleanly and without overlap: indices 1–3 average
334–383 glottal pulses at 1.4–4.1% unvoiced frames, indices 14–17 average 152–190 pulses,
and indices 4–13 and 18–26 average 35–72 pulses at 15–48% unvoiced. This confirms the
documented ordering of three vowels, ten numbers, four sentences and nine words.

**Table 5.** Subject-level accuracy by recording type (regime B, fixed hyper-parameters,
mean of ten fold seeds; chance = 0.500).

| Model | Vowels | Numbers | Sentences | Words |
|---|---|---|---|---|
| Logistic Regression | 0.577 | **0.595** | 0.585 | 0.541 |
| KNN | 0.446 | **0.596** | 0.514 | 0.519 |
| Decision Tree | 0.480 | 0.516 | **0.548** | 0.507 |
| Gradient Boosting | 0.527 | 0.546 | **0.566** | 0.517 |
| XGBoost | 0.509 | 0.546 | **0.571** | 0.510 |

No cell reaches 0.60. The best of the twenty combinations is 0.596, less than ten points
above chance. Sustained vowels are the **weakest** task for four of the five models, and
for KNN they fall below chance at 0.446. Under recording-level validation the same cells
range from 0.578 to 0.705, reproducing the appearance of a usable signal.

### 4.5 Independent cohort

The Sakar-2013 distribution includes a held-out partition of 168 recordings from 28 speakers
disjoint from the training set. It is **not** usable as an external validation set in the
usual sense: it contains 28 PD speakers and no controls, and only sustained vowels. On such
a cohort specificity is undefined, MCC is undefined because the labels have no variance,
the confusion matrix collapses to a single row, accuracy is identical to sensitivity, and a
classifier that answers "PD" unconditionally scores 1.000.

We therefore report it as a one-sided sensitivity check, selecting a model by subject-level
cross-validation on the training data alone and predicting the cohort exactly once.

**Table 6.** Independent-cohort sensitivity. Specificity and MCC are undefined on this
cohort; the training-cohort specificity is shown alongside so the operating point is visible.

| Training set | Selected model | Sensitivity (recording) | Sensitivity (speaker) | Training-cohort specificity |
|---|---|---|---|---|
| Sustained vowels only (n = 120) | KNN | 0.589 | 0.571 | 0.500 |
| Full training set (n = 1,040) | XGBoost | 0.411 | 0.357 | 0.627 |

Neither model detects PD in unseen speakers at a rate that would be useful. The
vowel-trained model identifies 59% of genuine PD recordings and 16 of 28 PD speakers; the
model trained on the full corpus identifies 41% of recordings and 10 of 28 speakers — worse
than answering "PD" every time, and worse than a coin flip. Across all five models the
sensitivity ranges from 0.149 to 0.649, a spread wide enough to be consistent with noise.

### 4.6 Which features matter, and how confidently

SHAP values were computed for logistic regression per outer fold, on that fold's held-out
rows, using that fold's own scaler (Figure 4).

We tested the hypothesis that leaky models rely on different features — which would be
direct evidence that they were learning speaker identity. **The hypothesis is not
supported.** Importance rankings under recording-level and subject-level training correlate
at Spearman ρ = +0.894 (logistic regression) and +0.964 (XGBoost) on UCI/MDVP, and +0.869
and +0.932 on Sakar-2013, with four or five of the top five features shared in every case
(Figure 4b). The leak does not change *which* acoustic quantities a model uses; it changes
how well those quantities appear to separate the classes, because the test speaker's own
other recordings sit nearby in the same feature space. The mechanism is memorisation within
a fixed feature geometry, not feature substitution.

Per-feature inference requires care that this literature has generally not taken. Both
feature sets contain definitional duplicates: `Jitter:DDP` is exactly three times
`MDVP:RAP`, and `Shimmer:DDA` is exactly three times `Shimmer:APQ3`, to six decimal places
in both corpora. The UCI/MDVP corpus therefore has an effective dimensionality of 20, not
the 22 usually reported. Fitting an unpenalised logistic regression to the raw feature set
produces confidence intervals spanning thousands of units. After removing duplicates the
remaining jitter measures, and separately the shimmer measures, are near-substitutes, with
variance inflation factors up to 741 on UCI/MDVP and 145 on Sakar-2013.

We therefore fit the inference model on a VIF-pruned subset (12 of 20 features on
UCI/MDVP, 19 of 23 on Sakar-2013, all VIF < 10), with standard errors clustered by speaker.
Clustering matters: treating six recordings from one person as six independent
observations is the same error at the inference level that recording-level partitioning is
at the validation level.

The result is unambiguous. On UCI/MDVP, **no** coefficient has a 95% confidence interval
excluding zero. On Sakar, two of nineteen do — `Jitter_local_absolute` (*p* = 0.033) and
`Shimmer_apq11` (*p* = 0.048) — which is exactly the number expected under the null across
nineteen tests. **After Benjamini–Hochberg correction at a false discovery rate of 0.05,
no feature survives on either corpus.** We accordingly make no claim about individual
acoustic markers and report SHAP as descriptive of model behaviour rather than as evidence
about disease biology.

---

## 5. Discussion

### 5.1 Implications for the published literature

The size of the effect — 5.8 to 15.5 percentage points, replicated on two corpora with
different speakers, languages, recording protocols and feature-extraction pipelines —
implies that a substantial share of reported performance in this area is attributable to
partitioning rather than to acoustic signal. Any study using the UCI/MDVP corpus without
speaker-aware partitioning should be read as reporting an upper bound inflated by roughly
this amount, with the inflation larger for higher-capacity models.

The practical consequence is not only that reported numbers are too high. It is that the
ordering of methods is unreliable. A field that compares models under leaky validation
will systematically prefer the models that memorise best. Our results show this concretely:
logistic regression ranks last of five under recording-level validation on UCI/MDVP and
first under leave-one-subject-out. Ten years of incremental "we improved on the previous
best" results, if evaluated this way, may have been selecting for capacity to exploit
leakage.

### 5.2 Why flexible models are hit hardest

Under recording-level partitioning, a test recording's nearest neighbours in feature space
are very often other recordings of the same speaker. A model that can carve fine-grained,
locally adaptive decision boundaries — a deep tree, a boosted ensemble, a small-*k* nearest
neighbour rule — can exploit this directly, effectively performing speaker identification.
A linear model cannot: it is constrained to a single global hyperplane and cannot
individuate speakers within it. Its recording-level score is consequently closer to its
honest score.

Our SHAP analysis sharpens this. Because leaky and honest models rely on the *same*
features, the advantage is not gained by switching to speaker-identifying features. It is
gained by fitting the local geometry around each speaker's cluster of recordings. Model
flexibility is precisely the capacity to do that, which is why inflation tracks capacity.

### 5.3 What honest performance looks like

Once leakage is removed, the picture is sober. On UCI/MDVP the best subject-level accuracy
is 0.836 against a majority-class baseline of 0.754 — an improvement of about eight points
over always answering "PD" — achieved with healthy-control recall below 0.50. On Sakar,
leave-one-subject-out accuracy is at chance for every model. On an independent cohort,
sensitivity is 0.41–0.59.

This does not establish that voice carries no information about Parkinson's disease; the
speaker-level signal in UCI/MDVP is above baseline, and larger studies with hundreds of
speakers report more encouraging results. It does establish that these two widely used
corpora, at 32 and 40 speakers, cannot support the claims that have been built on them,
and that the apparent strength of those claims is largely a methodological artefact.

### 5.4 Recommendations

We suggest four practices, none novel, all cheap:

1. Partition by speaker. Use `StratifiedGroupKFold` or `LeaveOneGroupOut` with the speaker
   as the group, and state the grouping variable explicitly in the methods section.
2. Place hyper-parameter search inside the outer fold, with a grouped inner loop.
3. Report the majority-class baseline, per-class recall and MCC alongside accuracy.
4. Report the variability induced by fold construction. With fewer than about fifty
   speakers, a single seed is not a measurement; our fold-seed standard deviations reach
   0.048, so differences smaller than roughly 0.10 between two models are not
   interpretable.

---

## 6. Limitations

**Few speakers.** Thirty-two and forty speakers are small. This is a limitation of the
available public corpora and is part of our argument, but it also limits our own estimates:
subject-level accuracy on UCI/MDVP varies by up to 0.149 across fold constructions. We
mitigate this by averaging over ten seeds and by reporting leave-one-subject-out, but the
underlying corpora remain small.

**Two corpora.** Both are widely used and mutually independent, but two is not many. We
cannot exclude the possibility that the effect size differs in larger or differently
constructed datasets.

**One feature pipeline per corpus.** UCI/MDVP is analysed with its distributed MDVP
measures and Sakar with its distributed Praat measures. We did not re-extract features from
audio, so we cannot separate the contribution of the feature set from that of the cohort.

**No severity modelling.** We treat PD detection as binary. The Sakar-2013 corpus carries UPDRS
scores, but they are unusable as a predictor — they take the value 1 for every control and
5–55 for every PD speaker — and we did not model severity as a separate regression task.

**The independent cohort is one-sided.** With no controls and only sustained vowels, it
supports a sensitivity estimate and nothing more. It cannot speak to specificity or to
overall discrimination, and its vowel-only protocol introduces a domain shift relative to
the full training corpus.

**Excluded data.** A figshare collection of synthesised parkinsonian vowels and a
video-harvested collection were both excluded — the first because its signals are
synthetic, carry inconsistent units, and include multiple-system atrophy and progressive
supranuclear palsy cases alongside PD; the second because its provenance, consent status
and label quality cannot be established. Neither exclusion affects the results reported
here, but both narrow the evidence base.

---

## 7. Reproducibility

All code, the raw distributions, and the exact commands required to regenerate every table
and figure are available at `https://github.com/Ashfaq97/pd-voice-extension`. Library
versions are pinned; all randomness is seeded at 27. The processed datasets are rebuilt
from the raw files by a single script under assertions on every count, and a companion
test feeds the historical corrupted file to those assertions to confirm they fire.

---

## Figures

**Figure 1.** Recording-level versus subject-level partitioning. Rows are speakers, cells
are that speaker's recordings.

**Figure 2.** Accuracy by model and regime on both corpora, with majority-class baselines.
Bars for regimes A and B are means over ten fold seeds with 1 SD error bars.

**Figure 3.** Subject-level accuracy by recording type on the Sakar-2013 corpus, against chance.

**Figure 4.** SHAP summary for logistic regression under subject-level training.
**Figure 4b.** Feature reliance under leaky versus subject-level training.

**Figure 5.** Healthy-control recall, recording-level versus subject-level.
