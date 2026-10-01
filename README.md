# Tumour Subtype Classification — Modeling Pipelines

This repository contains three complementary classification pipelines developed for tumour subtype prediction, using proteomic and histopathology (whole-slide image) features. All pipelines fix `random_state=42` for reproducibility and fit any preprocessing/feature-selection step only on training folds to avoid leakage.

The proteomic pipelines are run in sequence: the nested CV pipeline (Section 1) discovers which proteins are worth using, and those proteins are then locked in as the fixed panels evaluated in Section 2.

## Contents
1. [Nested CV Feature Selection Pipeline](#1-nested-cv-feature-selection-pipeline)
2. [Fixed Protein Panel Classification](#2-fixed-protein-panel-classification)
3. [Histopathology Slide/Patient-Level Classification](#3-histopathology-slidepatient-level-classification)
4. [Shared Utility: Bootstrap ROC-AUC Confidence Intervals](#shared-utility-bootstrap-roc-auc-confidence-intervals)

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn statsmodels scipy h5py
```

## Suggested Repo Structure

```
.
├── utils/
│   └── bootstrap_ci.py                       # shared bootstrap CI helpers (see below)
├── nested_cv_feature_selection.py            # Section 1
├── protein_panel_classification.py           # Section 2
└── histopathology_slide_classification.py    # Section 3
```

---

## 1. Nested CV Feature Selection Pipeline

Selects proteins **from scratch inside each outer fold**, so the selection process itself is never exposed to the test data — this is the discovery step that produced the SFS and RFE protein panels used in Section 2.

**Structure:** Outer 5-fold CV (unbiased performance estimate) → per training fold: filter → impute → scale → select → fit. Feature selection itself uses an inner 5-fold CV.

**Per-fold selection steps (train data only):**
1. Global missingness filter (proteins missing in >20% of *all* samples dropped once, upfront, as a structural step — not data-driven per fold)
2. Upregulation filter — a protein must be elevated in at least one class relative to its global median (log2 scale)
3. ANOVA F-test per protein with Benjamini–Hochberg FDR correction (q < 0.05), falling back to the top raw p-values if too few survive
4. **SFS** — Sequential Feature Selector (forward, logistic regression base, scored on balanced accuracy via inner CV)
5. **RFE** — Recursive Feature Elimination (same logistic regression base)

SFS and RFE each select a target number of features independently, producing two competing feature sets per fold.

**Classifiers evaluated per feature set:** Random Forest, SVM, XGBoost.

**Feature stability:** counts how often each protein is selected across the 5 outer folds, to identify consistently informative proteins (e.g. selected in ≥3/5 folds) versus fold-specific noise. **The most stable proteins from this step are what get carried forward as the fixed SFS/RFE panels in Section 2.**

<details>
<summary>Show code — <code>nested_cv_feature_selection.py</code></summary>

```python
"""
Nested cross-validation with in-fold feature selection.

All feature selection steps (upregulation filter, ANOVA, SFS, RFE) are
performed exclusively on training folds — test folds are never seen
during selection or preprocessing — giving honest, unbiased performance
estimates.

Structure:
    Outer CV (5-fold) → unbiased performance estimation
      └── Inner CV (5-fold) → used by SFS/RFE for selection
            └── Per-fold: filter → impute → scale → select → fit
"""

from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

N_SELECT = 20  # number of features SFS/RFE each select per fold

# ── Load data ────────────────────────────────────────────────────
df = pd.read_csv("tumour_zscore_clusters.csv")

meta_cols = ["cluster", "case_submitter_id", "sample_name"]
protein_cols = [c for c in df.columns if c not in meta_cols]

X_full = df[protein_cols].copy()
le = LabelEncoder()
y = le.fit_transform(df["cluster"])
target_names = [str(cls) for cls in le.classes_]
n_classes = len(target_names)

print(f"Samples: {len(df)} | Proteins (raw): {len(protein_cols)} | Classes: {target_names}")

# Global missingness filter: proteins missing in >20% of ALL samples are
# structurally unobserved, so it's safe to filter these once, globally.
missing_frac = X_full.isna().mean(axis=0)
keep_global = missing_frac[missing_frac < 0.20].index.tolist()
X_full = X_full[keep_global]
print(f"Proteins after global missingness filter (<20%): {len(keep_global)}")

classifiers = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
    ),
    "SVM": SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
    ),
}


def select_features_on_train(X_train_raw, y_train, n_select=N_SELECT):
    """
    Runs the full filter → ANOVA → SFS/RFE selection pipeline using only
    the training fold. Returns the SFS and RFE column selections.
    """
    X = pd.DataFrame(X_train_raw, columns=keep_global)

    # (a) Upregulation filter
    log2_X = np.log2(X + 1)
    upreg = []
    for protein in log2_X.columns:
        global_med = log2_X[protein].median()
        if any(log2_X.loc[y_train == c, protein].mean() > global_med + 0.5 for c in np.unique(y_train)):
            upreg.append(protein)
    X = X[upreg] if upreg else X

    # (b) ANOVA with FDR correction
    pvals, tested_proteins = [], []
    for protein in X.columns:
        groups = [X.loc[y_train == c, protein].dropna().values for c in np.unique(y_train)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2:
            _, p = f_oneway(*groups)
            pvals.append(p)
            tested_proteins.append(protein)

    if pvals:
        _, fdr_pvals, _, _ = multipletests(pvals, method="fdr_bh")
        sig_proteins = [p for p, q in zip(tested_proteins, fdr_pvals) if q < 0.05]
    else:
        sig_proteins = tested_proteins

    if len(sig_proteins) < n_select:  # fall back to top-p-value proteins
        ranked = sorted(zip(pvals, tested_proteins))
        sig_proteins = [p for _, p in ranked[:max(n_select, 50)]]

    X = X[sig_proteins]

    # (c) Impute + scale
    X_imp = KNNImputer(n_neighbors=5).fit_transform(X)
    X_sc = StandardScaler().fit_transform(X_imp)

    # (d) SFS — uses its own inner CV on the training fold
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = LogisticRegression(max_iter=2000, multi_class="auto")
    n_sel = min(n_select, X_sc.shape[1])

    sfs = SequentialFeatureSelector(
        estimator=base_model, n_features_to_select=n_sel, direction="forward",
        scoring="balanced_accuracy", cv=inner_cv, n_jobs=-1,
    )
    sfs.fit(X_sc, y_train)
    sfs_cols = np.array(sig_proteins)[sfs.get_support()].tolist()

    # (e) RFE
    rfe = RFE(estimator=base_model, n_features_to_select=n_sel)
    rfe.fit(X_sc, y_train)
    rfe_cols = np.array(sig_proteins)[rfe.support_].tolist()

    return sfs_cols, rfe_cols, sig_proteins


# ── Outer CV loop ──────────────────────────────────────────────────
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {
    fs: {clf: {"fold_accs": [], "all_true": [], "all_pred": []} for clf in classifiers}
    for fs in ["SFS", "RFE"]
}
feature_counts = {"SFS": defaultdict(int), "RFE": defaultdict(int)}

X_arr = X_full.values
col_to_idx = {c: i for i, c in enumerate(keep_global)}

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_arr, y), 1):
    print(f"\n── Outer fold {fold_idx}/5 ──")

    X_train_raw, X_test_raw = X_arr[train_idx], X_arr[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    sfs_cols, rfe_cols, anova_proteins = select_features_on_train(X_train_raw, y_train)
    print(f"  ANOVA-significant proteins (FDR<0.05): {len(anova_proteins)}")
    print(f"  SFS selected ({len(sfs_cols)}): {sfs_cols}")
    print(f"  RFE selected ({len(rfe_cols)}): {rfe_cols}")

    for col in sfs_cols:
        feature_counts["SFS"][col] += 1
    for col in rfe_cols:
        feature_counts["RFE"][col] += 1

    for fs_name, selected_cols in [("SFS", sfs_cols), ("RFE", rfe_cols)]:
        sel_idx = [col_to_idx[c] for c in selected_cols if c in col_to_idx]

        X_train_sub, X_test_sub = X_train_raw[:, sel_idx], X_test_raw[:, sel_idx]

        # Leakage-free impute + scale, refit for this feature subset
        fold_imputer = KNNImputer(n_neighbors=5)
        fold_scaler = StandardScaler()
        X_train_sc = fold_scaler.fit_transform(fold_imputer.fit_transform(X_train_sub))
        X_test_sc = fold_scaler.transform(fold_imputer.transform(X_test_sub))

        for clf_name, clf in classifiers.items():
            clf.fit(X_train_sc, y_train)
            y_pred = clf.predict(X_test_sc)

            results[fs_name][clf_name]["fold_accs"].append(balanced_accuracy_score(y_test, y_pred))
            results[fs_name][clf_name]["all_true"].extend(y_test)
            results[fs_name][clf_name]["all_pred"].extend(y_pred)

# ── Results summary ──────────────────────────────────────────────
summary_rows = []
for fs_name in ["SFS", "RFE"]:
    print(f"\n{'#' * 60}\n  Feature Set: {fs_name}\n{'#' * 60}")
    for clf_name in classifiers:
        accs = results[fs_name][clf_name]["fold_accs"]
        all_true = results[fs_name][clf_name]["all_true"]
        all_pred = results[fs_name][clf_name]["all_pred"]

        print(f"\n  {clf_name}")
        print(f"    Balanced Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(classification_report(all_true, all_pred, target_names=target_names, zero_division=0))

        summary_rows.append({
            "feature_set": fs_name,
            "classifier": clf_name,
            "mean_balanced_acc": round(np.mean(accs), 4),
            "std_balanced_acc": round(np.std(accs), 4),
        })

pd.DataFrame(summary_rows).to_csv("nested_cv_performance.csv", index=False)

# ── Feature stability across outer folds ──────────────────────────
for fs_name in ["SFS", "RFE"]:
    counts = feature_counts[fs_name]
    stable = sorted(counts.items(), key=lambda x: -x[1])

    print(f"\n{fs_name} — proteins selected in ≥3/5 folds:")
    for protein, cnt in stable:
        if cnt >= 3:
            print(f"  {protein}: {cnt}/5 folds")

    pd.DataFrame(stable, columns=["protein", "n_folds_selected"]).to_csv(
        f"feature_stability_{fs_name.lower()}.csv", index=False
    )

print("\nPipeline complete. All selection was performed inside CV folds — performance estimates are unbiased.")
```

</details>

---

## 2. Fixed Protein Panel Classification

Takes the SFS and RFE protein panels identified as most stable in Section 1 and locks them in, evaluating them against five classifiers with flat (non-nested) cross-validation. This is the "does the panel we found actually generalize well across model types" step.

**Data:** `tumour_zscore_clusters.csv`; target is `Subtype` or `cluster`, label-encoded.

**Feature panels (carried forward from Section 1's feature stability results):**

| Panel | Proteins |
|-------|----------|
| SFS | F8VZS0, Q13885, O95479 |
| RFE | Q9NR12, Q04206, Q8NCW5 |

**Classifiers:** Random Forest, SVM (RBF), XGBoost, Logistic Regression, MLP — all with balanced class weights where supported.

**Cross-validation:** 5-fold `StratifiedKFold` (`shuffle=True`). Each classifier is trained/evaluated per fold; predictions are aggregated across folds.

**Metrics:** balanced accuracy (per fold), aggregated accuracy, weighted precision/recall/F1, aggregated ROC-AUC (weighted OvR) with a 95% bootstrap CI (1,000 resamples).

<details>
<summary>Show code — <code>protein_panel_classification.py</code></summary>

```python
"""
Fixed protein panel classification.

Compares the two protein panels selected in the nested CV feature
selection step (SFS, RFE) across five classifiers using flat 5-fold
stratified cross-validation.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from xgboost import XGBClassifier

from utils.bootstrap_ci import  bootstrap_multiclass_auc_ci

warnings.filterwarnings("ignore")

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Panels carried forward from the nested CV feature selection step
FEATURE_GROUPS = {
    "SFS": ["F8VZS0", "Q13885", "O95479"],
    "RFE": ["Q9NR12", "Q04206", "Q8NCW5"],
}

# ── Load data ────────────────────────────────────────────────────
df = pd.read_csv("tumour_zscore_clusters.csv")
target_col = "Subtype" if "Subtype" in df.columns else "cluster"

le = LabelEncoder()
y = le.fit_transform(df[target_col])

target_names = []
for cls in le.classes_:
    cls_str = str(cls)
    if cls_str.isdigit():
        target_names.append(f"Subtype {cls_str}")
    else:
        target_names.append(cls_str.replace("Cluster", "Subtype").replace("cluster", "Subtype"))

n_classes = len(target_names)
summary_data = []

# ── Evaluate each feature group ───────────────────────────────────
for group_name, proteins in FEATURE_GROUPS.items():
    print(f"Running pipeline for feature group: {group_name} ({proteins})")

    X = df[proteins].copy()

    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0, class_weight="balanced", random_state=42, max_iter=1000, n_jobs=-1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100, 50), activation="relu", solver="adam",
            max_iter=500, random_state=42, early_stopping=True,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {
        name: {"fold_accs": [], "fold_rocs": [], "all_true": [], "all_pred": [], "all_probs": []}
        for name in classifiers
    }

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)

            fold_roc = (
                roc_auc_score(y_test, y_prob[:, 1]) if n_classes == 2
                else roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
            )

            cv_results[name]["fold_accs"].append(balanced_accuracy_score(y_test, y_pred))
            cv_results[name]["fold_rocs"].append(fold_roc)
            cv_results[name]["all_true"].extend(y_test)
            cv_results[name]["all_pred"].extend(y_pred)
            cv_results[name]["all_probs"].extend(y_prob)

    # ── Aggregate, report, and plot per classifier ──────────────────
    for name, res in cv_results.items():
        all_true = np.array(res["all_true"])
        all_pred = np.array(res["all_pred"])
        all_probs = np.array(res["all_probs"])

        accuracy = accuracy_score(all_true, all_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true, all_pred, average="weighted", zero_division=0
        )
        global_roc = (
            roc_auc_score(all_true, all_probs[:, 1]) if n_classes == 2
            else roc_auc_score(all_true, all_probs, multi_class="ovr", average="weighted")
        )

        # No independent grouping column exists, so each sample is its own bootstrap unit.
        sample_groups = np.arange(len(all_true))
       
        roc_lo, roc_hi, _ = bootstrap_multiclass_auc_ci(
                all_true, all_probs, classes=range(n_classes), groups=sample_groups
            )

        summary_data.append({
            "Feature Group": group_name,
            "Model": name,
            "Weighted Precision": precision,
            "Weighted Recall": recall,
            "Weighted F1-Score": f1,
            "Accuracy": accuracy,
            "Mean Fold ROC-AUC": np.mean(res["fold_rocs"]),
            "Global ROC-AUC": global_roc,
            "Global ROC-AUC 95% CI": f"[{roc_lo:.2f}-{roc_hi:.2f}]",
            "CV Balanced Accuracy (Mean)": np.mean(res["fold_accs"]),
            "CV Balanced Accuracy (Std)": np.std(res["fold_accs"]),
        })

        safe_name = name.replace(" ", "_")

        pd.DataFrame(
            classification_report(all_true, all_pred, target_names=target_names,
                                   zero_division=0, output_dict=True)
        ).T.to_csv(f"{OUTPUT_DIR}/{group_name}_{safe_name}_overall_classification_report.csv")

        # Confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            confusion_matrix(all_true, all_pred, labels=range(n_classes)),
            annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names,
        )
        plt.title(f"Confusion Matrix: {name} ({group_name} Features)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{group_name}_{safe_name}_confusion_matrix.png", dpi=300)
        plt.close()

        # ROC curve(s)
        plt.figure(figsize=(7, 6))
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(all_true, all_probs[:, 1])
            plt.plot(fpr, tpr, color="darkorange", lw=2,
                     label=f"ROC curve (AUC = {auc(fpr, tpr):.2f}) [{roc_lo:.2f}-{roc_hi:.2f}]")
        else:
            y_true_bin = label_binarize(all_true, classes=range(n_classes))
            for i, class_name in enumerate(target_names):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
                plt.plot(fpr, tpr, lw=2,
                         label=f"{class_name.replace('Subtype ', 'S-')} (AUC = {auc(fpr, tpr):.2f})")
            plt.plot([], [], " ", label=f"Overall Weighted AUC = {global_roc:.2f} [{roc_lo:.2f}-{roc_hi:.2f}]")

        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves: {name} ({group_name} Features)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{group_name}_{safe_name}_roc_curve.png", dpi=300)
        plt.close()

        print(f"{name} ({group_name}) — Balanced Acc: {np.mean(res['fold_accs']):.2f} ± "
              f"{np.std(res['fold_accs']):.2f} | ROC-AUC: {global_roc:.2f} [{roc_lo:.2f}-{roc_hi:.2f}]")

# ── Export master comparison table ─────────────────────────────────
column_order = [
    "Feature Group", "Model", "Weighted Precision", "Weighted Recall",
    "Weighted F1-Score", "Accuracy", "Mean Fold ROC-AUC", "Global ROC-AUC",
    "Global ROC-AUC 95% CI", "CV Balanced Accuracy (Mean)", "CV Balanced Accuracy (Std)",
]
pd.DataFrame(summary_data)[column_order].to_csv(
    f"{OUTPUT_DIR}/comprehensive_model_performance_summary.csv", index=False
)
```

</details>

---

## 3. Histopathology Slide/Patient-Level Classification

Classifies tumour subtype from **deep slide-level feature vectors** (extracted via TITAN, stored as `.h5` files), aggregated to the patient level. This pipeline is independent of the proteomic panels above — it evaluates a separate, image-derived feature source against the same subtype labels.

**Data:** slide feature files matched to patients via `case_submitter_id`, merged with a cluster/subtype mapping. Multiple slides can belong to one patient.

**Class imbalance handling:**
- Inverse-frequency class weights, with a tunable extra multiplier applied to one minority cluster of particular interest (`CLUSTER2_MULTIPLIER`)
- **SMOTE** oversampling applied to each training fold only (after feature selection, before scaling)
- XGBoost uses per-sample weights (balanced + the same cluster multiplier) instead of `class_weight`, since it has no native support for the latter

**Per-fold preprocessing:** `SelectKBest` (ANOVA F-value, top 64 features) → SMOTE → `StandardScaler`, all fit on the training fold only.

**Cross-validation:** 5-fold `StratifiedGroupKFold`, grouped by patient — ensures all slides from one patient stay entirely within either the train or test split.

**Classifiers:** Logistic Regression, SVM, Random Forest, XGBoost, MLP — each initialized with the custom class weights described above.

**Two levels of evaluation:**
- **Slide-level:** raw per-slide predictions and probabilities
- **Patient-level:** slide probabilities averaged per patient, then argmax to get a single patient-level prediction

**Metrics (both levels):** accuracy, balanced accuracy, weighted precision/recall/F1, weighted ROC-AUC (OvR), plus precision/recall/F1/ROC-AUC specifically for the minority cluster of interest — all with 95% bootstrap CIs (block bootstrap by patient to avoid pseudo-replication from multiple slides per patient).

<details>
<summary>Show code — <code>histopathology_slide_classification.py</code></summary>

```python
"""
Slide- and patient-level tumour subtype classification from
TITAN whole-slide-image features.
"""

from pathlib import Path
import warnings

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from utils.bootstrap_ci import bootstrap_multiclass_auc_ci

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────
FEATURES_DIR = Path("path/to/trident_processed/20x_512px_0px_overlap/slide_features_titan")
CLUSTER_MAP_PATH = "path/to/cluster_mapping.csv"

N_FEATURES_SELECT = 64
CLUSTER_OF_INTEREST = 2         # e.g. a rare/clinically important subtype
CLUSTER2_MULTIPLIER = 2.0       # extra weight multiplier applied to it
ROC_PALETTE = ["#E8735A", "#4C9ED9", "#6DBF7E"]

# ── Load and merge slide-level features with subtype labels ────────
cluster_map = pd.read_csv(CLUSTER_MAP_PATH, usecols=["case_submitter_id", "cluster"])

slide_records = []
for h5_file in sorted(FEATURES_DIR.glob("*.h5")):
    patient_id = "-".join(h5_file.stem.split("-")[:2])
    with h5py.File(h5_file, "r") as f:
        features = f["features"][:]
    slide_records.append({
        "slide_id": h5_file.stem,
        "case_submitter_id": patient_id,
        **{i: v for i, v in enumerate(features)},
    })

slide_df = pd.DataFrame(slide_records)
slide_merged = slide_df.merge(cluster_map, on="case_submitter_id", how="inner")

print(f"Slides after merge: {len(slide_merged)}")
print(f"Patients after merge: {slide_merged['case_submitter_id'].nunique()}")
print(f"Subtype distribution (slides):\n{slide_merged['cluster'].value_counts().sort_index()}")

non_feat = {"slide_id", "case_submitter_id", "cluster"}
feat_cols = [c for c in slide_merged.columns if c not in non_feat]

X = slide_merged[feat_cols].values.astype(np.float32)
y = slide_merged["cluster"].values
groups = slide_merged["case_submitter_id"].values

# XGBoost requires sequential 0-indexed class labels
unique_clusters = sorted(np.unique(y))
cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
idx_to_cluster = {i: c for c, i in cluster_to_idx.items()}
y_encoded = np.array([cluster_to_idx[v] for v in y])
classes = sorted(np.unique(y_encoded))
c2_encoded_idx = cluster_to_idx[CLUSTER_OF_INTEREST]

# ── Class weights (inverse frequency + manual multiplier on the cluster of interest) ─
counts = pd.Series(y_encoded).value_counts().sort_index()
total = counts.sum()
base_weights = {c: total / (len(classes) * counts[c]) for c in classes}
class_weights = {
    c: base_weights[c] * (CLUSTER2_MULTIPLIER if c == c2_encoded_idx else 1.0)
    for c in classes
}
print(f"Class weights: { {idx_to_cluster[c]: round(w, 2) for c, w in class_weights.items()} }")

# ── Define models ──────────────────────────────────────────────────
models = {
    "LogisticRegression": LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", multi_class="multinomial",
        class_weight=class_weights, max_iter=5000, tol=1e-3, random_state=42, n_jobs=-1,
    ),
    "SVM": SVC(
        kernel="rbf", C=1.0, gamma="scale", class_weight=class_weights,
        probability=True, random_state=42,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        class_weight=class_weights, random_state=42, n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        objective="multi:softprob", eval_metric="mlogloss", random_state=42, n_jobs=-1,
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-3,
        learning_rate_init=1e-3, max_iter=2000, early_stopping=True,
        n_iter_no_change=20, validation_fraction=0.15, random_state=42,
    ),
}

slide_comparison_metrics, patient_comparison_metrics = [], []
cv = StratifiedGroupKFold(n_splits=5)

for model_name, clf in models.items():
    print(f"\n{'=' * 70}\nModel: {model_name}\n{'=' * 70}")

    fold_accs, fold_bal_accs, fold_c2_f1s = [], [], []
    all_y_true, all_y_pred, all_y_prob = [], [], []
    all_test_slides, all_test_patients = [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded, groups=groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        print(f"  Fold {fold}: {len(np.unique(groups[train_idx]))} train patients "
              f"({len(train_idx)} slides) | {len(np.unique(groups[test_idx]))} test patients "
              f"({len(test_idx)} slides)")

        # Feature selection — fit on the training fold only
        selector = SelectKBest(score_func=f_classif, k=N_FEATURES_SELECT)
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)

        # SMOTE oversampling — training fold only
        X_train_sel, y_train = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_train_sel, y_train)

        # Scaling — fit on the (resampled) training fold only
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_sel)
        X_test_s = scaler.transform(X_test_sel)

        if model_name == "XGBoost":
            # XGBoost has no native class_weight — use per-sample weights instead
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
            sample_weights[y_train == c2_encoded_idx] *= CLUSTER2_MULTIPLIER
            clf.fit(X_train_s, y_train, sample_weight=sample_weights)
        else:
            clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)
        y_prob = clf.predict_proba(X_test_s)

        fold_accs.append((y_pred == y_test).mean())
        fold_bal_accs.append(balanced_accuracy_score(y_test, y_pred))
        fold_c2_f1s.append(f1_score(y_test, y_pred, labels=[c2_encoded_idx], average=None)[0])

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        all_test_slides.extend(slide_merged["slide_id"].iloc[test_idx].values)
        all_test_patients.extend(groups[test_idx])

    print(f"\n  Mean slide accuracy    : {np.mean(fold_accs):.2f} ± {np.std(fold_accs):.2f}")
    print(f"  Mean balanced accuracy : {np.mean(fold_bal_accs):.2f} ± {np.std(fold_bal_accs):.2f}")
    print(f"  Mean Cluster {CLUSTER_OF_INTEREST} F1      : {np.mean(fold_c2_f1s):.2f} ± {np.std(fold_c2_f1s):.2f}")

    all_y_true, all_y_pred, all_y_prob = np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob)
    all_y_true_decoded = np.array([idx_to_cluster[v] for v in all_y_true])
    all_y_pred_decoded = np.array([idx_to_cluster[v] for v in all_y_pred])
    class_labels = [f"S-{c}" for c in unique_clusters]

    # ── Slide-level metrics ─────────────────────────────────────────
    y_true_bin_c2 = (all_y_true == c2_encoded_idx).astype(int)
    y_prob_c2 = all_y_prob[:, c2_encoded_idx]

    slide_roc_auc_weighted = roc_auc_score(
        label_binarize(all_y_true, classes=classes), all_y_prob, average="weighted", multi_class="ovr"
    )
    slide_auc_ci_low, slide_auc_ci_high, _ = bootstrap_multiclass_auc_ci(
        all_y_true, all_y_prob, classes, groups=np.array(all_test_patients)
    )
    c2_roc_auc = roc_auc_score(y_true_bin_c2, y_prob_c2)
    

    slide_comparison_metrics.append({
        "Model": model_name,
        "Weighted Precision": round(precision_score(all_y_true, all_y_pred, average="weighted"), 2),
        "Weighted Recall": round(recall_score(all_y_true, all_y_pred, average="weighted"), 2),
        "Weighted F1-score": round(f1_score(all_y_true, all_y_pred, average="weighted"), 2),
        "Accuracy": round((all_y_true == all_y_pred).mean(), 2),
        "ROC-AUC (Weighted)": round(slide_roc_auc_weighted, 2),
        "ROC-AUC (Weighted) 95% CI": f"[{slide_auc_ci_low:.2f}-{slide_auc_ci_high:.2f}]",
        f"Cluster {CLUSTER_OF_INTEREST} Precision": round(
            precision_score(all_y_true_decoded, all_y_pred_decoded, labels=[CLUSTER_OF_INTEREST], average=None)[0], 2
        ),
        f"Cluster {CLUSTER_OF_INTEREST} Recall": round(
            recall_score(all_y_true_decoded, all_y_pred_decoded, labels=[CLUSTER_OF_INTEREST], average=None)[0], 2
        ),
        f"Cluster {CLUSTER_OF_INTEREST} F1-score": round(
            f1_score(all_y_true_decoded, all_y_pred_decoded, labels=[CLUSTER_OF_INTEREST], average=None)[0], 2
        ),
        f"Cluster {CLUSTER_OF_INTEREST} ROC-AUC": round(c2_roc_auc, 2),
        f"Cluster {CLUSTER_OF_INTEREST} ROC-AUC 95% CI": f"[{slide_c2_auc_ci_low:.2f}-{slide_c2_auc_ci_high:.2f}]",
    })

    print(f"\n  Slide-level Weighted ROC-AUC  : {slide_roc_auc_weighted:.2f} "
          f"[95% CI {slide_auc_ci_low:.2f}-{slide_auc_ci_high:.2f}]")
    print(classification_report(all_y_true_decoded, all_y_pred_decoded, target_names=class_labels, digits=2))

    # ── Slide-level plots ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(all_y_true_decoded, all_y_pred_decoded, labels=unique_clusters),
        display_labels=class_labels,
    ).plot(ax=ax, colorbar=False, cmap="PuRd")
    ax.set_title(model_name, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{model_name}_slide_confusion_matrix.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    y_bin = label_binarize(all_y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_y_prob[:, i])
        label_name = f"Cluster {idx_to_cluster[c]}" if idx_to_cluster[c] == CLUSTER_OF_INTEREST else f"Subtype {idx_to_cluster[c]}"
        ax.plot(fpr, tpr, color=ROC_PALETTE[i], lw=2, label=f"{label_name} (AUC = {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot([], [], " ", label=f"Overall Weighted AUC = {slide_roc_auc_weighted:.2f} "
                               f"[{slide_auc_ci_low:.2f}-{slide_auc_ci_high:.2f}]")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(model_name, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{model_name}_slide_roc_curves.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ── Patient-level aggregation (mean slide probability → argmax) ──
    pred_df = pd.DataFrame({
        "slide_id": all_test_slides,
        "case_submitter_id": all_test_patients,
        "true_subtype": all_y_true_decoded,
        "pred_subtype": all_y_pred_decoded,
        **{f"prob_subtype_{idx_to_cluster[c]}": all_y_prob[:, i].round(2) for i, c in enumerate(classes)},
    })

    prob_cols = [f"prob_subtype_{c}" for c in unique_clusters]
    patient_prob_df = pred_df.groupby("case_submitter_id")[prob_cols].mean()
    patient_true = pred_df.groupby("case_submitter_id")["true_subtype"].first()
    patient_pred = patient_prob_df.idxmax(axis=1).str.replace("prob_subtype_", "").astype(int)

    patient_results = pd.DataFrame({"true_subtype": patient_true, "pred_subtype": patient_pred})

    patient_prob_full = patient_prob_df.copy()
    patient_prob_full.columns = unique_clusters
    y_pat_bin = label_binarize(patient_results["true_subtype"], classes=unique_clusters)
    y_pat_prob = patient_prob_full.loc[patient_results.index].values

    pat_roc_auc_weighted = roc_auc_score(y_pat_bin, y_pat_prob, average="weighted", multi_class="ovr")
    pat_auc_ci_low, pat_auc_ci_high, _ = bootstrap_multiclass_auc_ci(
        patient_results["true_subtype"].values, y_pat_prob, unique_clusters, groups=patient_results.index.values
    )

    pat_true_bin_c2 = (patient_results["true_subtype"] == CLUSTER_OF_INTEREST).astype(int)
    pat_prob_c2 = y_pat_prob[:, unique_clusters.index(CLUSTER_OF_INTEREST)]
    pat_c2_roc_auc = roc_auc_score(pat_true_bin_c2, pat_prob_c2)
    

    patient_comparison_metrics.append({
        "Model": model_name,
        "Weighted Precision": round(precision_score(patient_results["true_subtype"], patient_results["pred_subtype"], average="weighted"), 2),
        "Weighted Recall": round(recall_score(patient_results["true_subtype"], patient_results["pred_subtype"], average="weighted"), 2),
        "Weighted F1-score": round(f1_score(patient_results["true_subtype"], patient_results["pred_subtype"], average="weighted"), 2),
        "Accuracy": round((patient_results["true_subtype"] == patient_results["pred_subtype"]).mean(), 2),
        "ROC-AUC (Weighted)": round(pat_roc_auc_weighted, 2),
        "ROC-AUC (Weighted) 95% CI": f"[{pat_auc_ci_low:.2f}-{pat_auc_ci_high:.2f}]",
        f"Cluster {CLUSTER_OF_INTEREST} Precision": round(
            precision_score(patient_results["true_subtype"], patient_results["pred_subtype"], labels=[CLUSTER_OF_INTEREST], average=None)[0], 2
        ),
        f"Cluster {CLUSTER_OF_INTEREST} Recall": round(
            recall_score(patient_results["true_subtype"], patient_results["pred_subtype"], labels=[CLUSTER_OF_INTEREST], average=None)[0], 2
        ),
        f"Cluster {CLUSTER_OF_INTEREST} F1-score": round(
            f1_score(patient_results["true_subtype"], patient_results["pred_subtype"], labels=[CLUSTER_OF_INTEREST], average=None)[0], 2
        ),
        f"Cluster {CLUSTER_OF_INTEREST} ROC-AUC": round(pat_c2_roc_auc, 2),
        f"Cluster {CLUSTER_OF_INTEREST} ROC-AUC 95% CI": f"[{pat_c2_auc_ci_low:.2f}-{pat_c2_auc_ci_high:.2f}]",
    })

    print(f"\n  Patient-level Weighted ROC-AUC  : {pat_roc_auc_weighted:.2f} "
          f"[95% CI {pat_auc_ci_low:.2f}-{pat_auc_ci_high:.2f}]")
    print(classification_report(
        patient_results["true_subtype"], patient_results["pred_subtype"], target_names=class_labels, digits=2
    ))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(patient_results["true_subtype"], patient_results["pred_subtype"], labels=unique_clusters),
        display_labels=class_labels,
    ).plot(ax=ax, colorbar=False, cmap="PuRd")
    ax.set_title(model_name, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{model_name}_patient_confusion_matrix.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

# ── Final comparison tables ────────────────────────────────────────
slide_comparison_df = pd.DataFrame(slide_comparison_metrics).set_index("Model")
patient_comparison_df = pd.DataFrame(patient_comparison_metrics).set_index("Model")

print(f"\n{'#' * 70}\n FINAL MODEL COMPARISON SUMMARY\n{'#' * 70}")
print("\n--- SLIDE-LEVEL EVALUATION METRICS ---")
print(slide_comparison_df.to_string())
print("\n--- PATIENT-LEVEL EVALUATION METRICS ---")
print(patient_comparison_df.to_string())

slide_comparison_df.to_csv("model_comparison_slide_level.csv")
patient_comparison_df.to_csv("model_comparison_patient_level.csv")
```

</details>

---

## Shared Utility: Bootstrap ROC-AUC Confidence Intervals

Used by all three pipelines to attach 95% confidence intervals to ROC-AUC scores via block bootstrap (resampling by group — e.g. patient ID — to avoid pseudo-replication when multiple rows belong to the same group; pass `groups=np.arange(len(y_true))` when there's no natural grouping).

<details>
<summary>Show code — <code>utils/bootstrap_ci.py</code></summary>

```python
"""Block-bootstrap 95% confidence intervals for ROC-AUC scores."""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

N_BOOTSTRAPS = 1000
CI_ALPHA = 0.95


def percentile_ci(values, alpha=CI_ALPHA):
    """Two-sided percentile confidence interval."""
    lo = (1 - alpha) / 2 * 100
    hi = (1 + alpha) / 2 * 100
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))



def bootstrap_multiclass_auc_ci(y_true, y_score, classes, groups, average="weighted",
                                 n_bootstraps=N_BOOTSTRAPS, alpha=CI_ALPHA, random_state=42):
    """Block bootstrap 95% CI for the one-vs-rest multiclass weighted ROC-AUC."""
    rng = np.random.RandomState(random_state)
    y_true, y_score, groups = np.asarray(y_true), np.asarray(y_score), np.asarray(groups)

    unique_groups = np.unique(groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}

    aucs = []
    for _ in range(n_bootstraps):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            y_bin = label_binarize(yt, classes=classes)
            if y_bin.shape[1] < 2:
                continue
            aucs.append(roc_auc_score(y_bin, y_score[idx], average=average, multi_class="ovr"))
        except ValueError:
            continue

    aucs = np.array(aucs)
    lo, hi = percentile_ci(aucs, alpha)
    return lo, hi, aucs
```

</details>

---

## Shared Methodology Notes
- **Reproducibility:** `random_state=42` fixed across every model, split, and resampling step.
- **No leakage:** all imputation, scaling, feature selection, and oversampling is fit on training folds/splits only and applied (not refit) to test data.
- **Uncertainty quantification:** ROC-AUC values throughout are reported with 95% confidence intervals from a block bootstrap (1,000 resamples), grouped by patient where relevant to avoid treating correlated samples (e.g. multiple slides per patient) as independent.
