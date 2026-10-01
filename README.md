# Tumour Subtype Classification

Code for predicting tumour subtype from proteomic data and from histopathology whole-slide image features. There are three scripts. The first two work together (feature selection, then evaluation of the selected panels). The third is independent and uses image features.

`random_state=42` is used for all splits, models and resampling.

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn scipy h5py openpyxl venn
```

## Repo structure

```
.
├── bootstrap_ci.py                           # bootstrap CI for multiclass ROC-AUC
├── feature_selection.py                      # 1. protein selection by fold voting
├── protein_panel_classification.py           # 2. evaluate the selected panels
└── histopathology_slide_classification.py    # 3. slide/patient-level classification
```

## 1. Feature selection (`feature_selection.py`)

Finds a small panel of proteins that separate the subtypes.

**Input:** `tumour_zscore_clusters.csv` (z-scored protein abundances plus `cluster`), and `idmapping.xlsx` (UniProt ID mapping of literature biomarkers, reviewed entries only).

**Steps:**
1. Keep only proteins with no missing values.
2. 5-fold stratified CV. In each training fold:
   - standardise the proteins
   - keep the top 50 by ANOVA p-value
   - run forward Sequential Feature Selection (SFS; logistic regression, balanced accuracy, 3-fold inner CV) to pick 3 proteins
   - run Recursive Feature Elimination (RFE; logistic regression) to pick 3 proteins
3. Count how many folds picked each protein. The three proteins with the most votes for each method are the SFS and RFE panels.
4. Compare the top 3 and the full set of selected proteins against the literature biomarker list, and draw a Venn diagram of SFS vs RFE selections (`venn_sfs_vs_rfe.png`).

**Outputs:** `feature_votes_sfs.csv`, `feature_votes_rfe.csv`, `venn_sfs_vs_rfe.png`.

## 2. Protein panel classification (`protein_panel_classification.py`)

Evaluates the two panels from step 1.

| Panel | Proteins |
|-------|----------|
| SFS | F8VZS0, Q13885, O95479 |
| RFE | Q9NR12, Q04206, Q8NCW5 |

**Classifiers:** Random Forest, SVM (RBF), XGBoost, Logistic Regression, MLP (balanced class weights where supported).

**Cross-validation:** 5-fold stratified, predictions pooled across folds.

**Metrics:** balanced accuracy per fold; pooled accuracy, weighted precision/recall/F1; weighted one-vs-rest ROC-AUC with a 95% bootstrap CI (1,000 resamples). Also saves a classification report, confusion matrix and per-subtype ROC curves for every model.

## 3. Histopathology classification (`histopathology_slide_classification.py`)

Classifies subtype from slide-level TITAN features (`.h5` files), with slides linked to patients by `case_submitter_id`. A patient can have several slides.

**Class imbalance:**
- inverse-frequency class weights, with an extra multiplier on one minority cluster of interest (`FOCUS_MULTIPLIER`)
- SMOTE on the training fold only
- XGBoost uses per-sample weights instead of `class_weight`

**Per-fold preprocessing (training fold only):** `SelectKBest` (ANOVA F, top 64) then SMOTE then `StandardScaler`.

**Cross-validation:** 5-fold `StratifiedGroupKFold` grouped by patient, so a patient's slides are never split between train and test.

**Classifiers:** Logistic Regression, SVM, Random Forest, XGBoost, MLP. The MLP has no class weight option, so it only benefits from SMOTE.

**Evaluation levels:**
- slide level: each slide's own prediction
- patient level: slide probabilities averaged per patient, then argmax

**Metrics (both levels):** accuracy, weighted precision/recall/F1, weighted one-vs-rest ROC-AUC with a 95% bootstrap CI, plus precision/recall/F1 for the cluster of interest. The bootstrap resamples patients, not slides.

## Bootstrap CI (`bootstrap_ci.py`)

`bootstrap_multiclass_auc_ci` resamples whole groups (e.g. patients) with replacement 1,000 times and returns the 2.5th and 97.5th percentiles of the weighted multiclass ROC-AUC. If there is no natural grouping, pass `groups=np.arange(len(y_true))`.

## Notes on interpretation

- **Step 1 does not give a performance estimate.** Selection is run on training folds, but the votes are summed over all five folds, so the final panels have been influenced by every sample. Step 2 then evaluates those panels with cross-validation on the same samples, so its scores are likely optimistic. Treat step 2 as a comparison of classifiers and panels, not as an unbiased estimate of how the panels would perform on new data. An independent validation cohort, or selection repeated inside every training fold of the evaluation, would be needed for that.
- Only proteins with complete data are used in step 1, which may exclude informative proteins that are sometimes missing.
- When several proteins have the same vote count, the top 3 is decided by the order they were first selected, so ties are arbitrary.
- In the histopathology pipeline, SMOTE and class weights both correct for imbalance, so minority classes are effectively up-weighted twice.
