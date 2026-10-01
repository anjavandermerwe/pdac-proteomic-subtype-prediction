# Tumour Subtype Classification

Code for predicting tumour subtype from proteomic data and from histopathology whole-slide image features. There are three scripts. The first two work together (feature selection of protein biomarkers, then evaluation of the selected panels). The third is independent and uses image features obtained from the TITAN foundation model, but any WSI-level image features can be used.

`random_state=42` is used for all splits, models and resampling.

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn scipy h5py openpyxl venn
```



## 1. Feature selection (`feature_selection.py`)

Finds a small panel of proteins that separate the subtypes.

**Input:** `tumour_zscore_clusters.csv` 

**Steps:**
1. Keep only proteins with no missing values.
2. 5-fold stratified CV. In each training fold:
   - standardise the proteins
   - keep the top 50 by ANOVA p-value
   - run forward Sequential Feature Selection
   - run Recursive Feature Elimination
3. Count how many folds picked each protein. The three proteins with the most votes for each method are the SFS and RFE panels.

## 2. Protein panel classification (`protein_panel_classification.py`)

Evaluates the two panels from the output of step 1.

**Classifiers:** Random Forest, SVM (RBF), XGBoost, Logistic Regression, MLP (balanced class weights where supported).

**Cross-validation:** 5-fold stratified, predictions pooled across folds.


## 3. Histopathology classification (`histopathology_slide_classification.py`)

Classifies subtype from slide-level TITAN features (`.h5` files), with slides linked to patients by `case_submitter_id`. A patient can have several slides.

**Class imbalance:**
- inverse-frequency class weights, with an extra multiplier on one minority cluster of interest (`FOCUS_MULTIPLIER`)
- SMOTE on the training fold only
- XGBoost uses per-sample weights instead of `class_weight`

**Per-fold preprocessing (training fold only):** `SelectKBest` (ANOVA F, top 64) then SMOTE then `StandardScaler`.

**Cross-validation:** 5-fold `StratifiedGroupKFold` grouped by patient, so a patient's slides are never split between train and test.

**Classifiers:** Logistic Regression, SVM, Random Forest, XGBoost, MLP. The MLP has no class weight option, so it only benefits from SMOTE.


## Bootstrap CI (`bootstrap_ci.py`)

`bootstrap_multiclass_auc_ci` resamples whole groups (e.g. patients) with replacement 1,000 times and returns the 2.5th and 97.5th percentiles of the weighted multiclass ROC-AUC. If there is no natural grouping, pass `groups=np.arange(len(y_true))`.


