"""
Slide- and patient-level subtype classification from TITAN slide features.
"""

import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, auc, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from utils.bootstrap_ci import bootstrap_multiclass_auc_ci

warnings.filterwarnings("ignore")

FEATURES_DIR = Path("path/to/trident_processed/20x_512px_0px_overlap/slide_features_titan")
CLUSTER_MAP_PATH = "path/to/cluster_mapping.csv"

N_FEATURES = 64
FOCUS_CLUSTER = 2            # minority cluster we care most about
FOCUS_MULTIPLIER = 2.0       # extra weight on that cluster
ROC_COLOURS = ["#E8735A", "#4C9ED9", "#6DBF7E"]

# load slide features and attach the subtype for each patient
cluster_map = pd.read_csv(CLUSTER_MAP_PATH, usecols=["case_submitter_id", "cluster"])

records = []
for h5_path in sorted(FEATURES_DIR.glob("*.h5")):
    with h5py.File(h5_path, "r") as f:
        feats = f["features"][:]
    records.append({
        "slide_id": h5_path.stem,
        "case_submitter_id": "-".join(h5_path.stem.split("-")[:2]),
        **dict(enumerate(feats)),
    })

data = pd.DataFrame(records).merge(cluster_map, on="case_submitter_id", how="inner")
print(f"{len(data)} slides, {data['case_submitter_id'].nunique()} patients")
print(data["cluster"].value_counts().sort_index())

feat_cols = [c for c in data.columns if c not in ("slide_id", "case_submitter_id", "cluster")]
X = data[feat_cols].values.astype(np.float32)
groups = data["case_submitter_id"].values

# xgboost needs labels 0..k-1, so encode and keep a mapping back
clusters = sorted(data["cluster"].unique())
to_idx = {c: i for i, c in enumerate(clusters)}
y = data["cluster"].map(to_idx).values
focus_idx = to_idx[FOCUS_CLUSTER]

# inverse-frequency weights, with an extra boost for the focus cluster
counts = pd.Series(y).value_counts().sort_index()
class_weights = {c: len(y) / (len(counts) * counts[c]) for c in counts.index}
class_weights[focus_idx] *= FOCUS_MULTIPLIER
print("class weights:", {clusters[c]: round(w, 2) for c, w in class_weights.items()})

models = {
    "LogisticRegression": LogisticRegression(C=1.0, class_weight=class_weights, max_iter=5000,
                                             tol=1e-3, random_state=42),
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=class_weights,
               probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                           class_weight=class_weights, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             objective="multi:softprob", eval_metric="mlogloss",
                             random_state=42, n_jobs=-1),
    "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, learning_rate_init=1e-3,
                         max_iter=2000, early_stopping=True, n_iter_no_change=20,
                         validation_fraction=0.15, random_state=42),
}


def summarise(model_name, y_true, y_pred, y_prob, group_ids):
    """Metrics for one model. y_true/y_pred use the original cluster labels,
    y_prob columns follow the order of `clusters`."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    fp, fr, ff, _ = precision_recall_fscore_support(y_true, y_pred, labels=[FOCUS_CLUSTER],
                                                    zero_division=0)
    y_bin = label_binarize(y_true, classes=clusters)
    weighted_auc = roc_auc_score(y_bin, y_prob, average="weighted")
    lo, hi = bootstrap_multiclass_auc_ci(y_true, y_prob, clusters, group_ids)

    row = {
        "Model": model_name,
        "Weighted Precision": round(p, 2),
        "Weighted Recall": round(r, 2),
        "Weighted F1-score": round(f, 2),
        "Accuracy": round((y_true == y_pred).mean(), 2),
        "ROC-AUC (Weighted)": round(weighted_auc, 2),
        "ROC-AUC (Weighted) 95% CI": f"[{lo:.2f}-{hi:.2f}]",
        f"Cluster {FOCUS_CLUSTER} Precision": round(fp[0], 2),
        f"Cluster {FOCUS_CLUSTER} Recall": round(fr[0], 2),
        f"Cluster {FOCUS_CLUSTER} F1-score": round(ff[0], 2),
    }
    return row, weighted_auc, lo, hi


def save_confusion_matrix(y_true, y_pred, labels, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=clusters),
                           display_labels=labels).plot(ax=ax, colorbar=False, cmap="PuRd")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


slide_rows, patient_rows = [], []
cv = StratifiedGroupKFold(n_splits=5)
labels = [f"S-{c}" for c in clusters]

for model_name, clf in models.items():
    print(f"\n=== {model_name} ===")

    fold_acc, fold_bal_acc, fold_focus_f1 = [], [], []
    true_all, pred_all, prob_all, patients_all, slides_all = [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"  fold {fold}: {len(set(groups[train_idx]))} train patients ({len(train_idx)} slides), "
              f"{len(set(groups[test_idx]))} test patients ({len(test_idx)} slides)")

        # everything below is fit on the training fold only
        selector = SelectKBest(f_classif, k=N_FEATURES)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        X_train, y_train = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if model_name == "XGBoost":
            # no class_weight option, so use sample weights
            w = compute_sample_weight("balanced", y_train)
            w[y_train == focus_idx] *= FOCUS_MULTIPLIER
            clf.fit(X_train, y_train, sample_weight=w)
        else:
            clf.fit(X_train, y_train)

        pred = clf.predict(X_test)
        prob = clf.predict_proba(X_test)

        fold_acc.append((pred == y_test).mean())
        fold_bal_acc.append(balanced_accuracy_score(y_test, pred))
        fold_focus_f1.append(f1_score(y_test, pred, labels=[focus_idx], average=None)[0])

        true_all.extend(y_test)
        pred_all.extend(pred)
        prob_all.extend(prob)
        patients_all.extend(groups[test_idx])
        slides_all.extend(data["slide_id"].iloc[test_idx])

    print(f"  accuracy {np.mean(fold_acc):.2f} +/- {np.std(fold_acc):.2f}, "
          f"balanced accuracy {np.mean(fold_bal_acc):.2f} +/- {np.std(fold_bal_acc):.2f}, "
          f"cluster {FOCUS_CLUSTER} F1 {np.mean(fold_focus_f1):.2f} +/- {np.std(fold_focus_f1):.2f}")

    true_all = np.array([clusters[i] for i in true_all])
    pred_all = np.array([clusters[i] for i in pred_all])
    prob_all = np.array(prob_all)
    patients_all = np.array(patients_all)

    # slide level
    row, slide_auc, lo, hi = summarise(model_name, true_all, pred_all, prob_all, patients_all)
    slide_rows.append(row)
    print(f"\n  slide-level weighted AUC {slide_auc:.2f} [{lo:.2f}-{hi:.2f}]")
    print(classification_report(true_all, pred_all, target_names=labels, digits=2))

    save_confusion_matrix(true_all, pred_all, labels, model_name,
                          f"{model_name}_slide_confusion_matrix.png")

    y_bin = label_binarize(true_all, classes=clusters)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, c in enumerate(clusters):
        fpr, tpr, _ = roc_curve(y_bin[:, i], prob_all[:, i])
        name = f"Cluster {c}" if c == FOCUS_CLUSTER else f"Subtype {c}"
        ax.plot(fpr, tpr, color=ROC_COLOURS[i], lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot([], [], " ", label=f"Weighted AUC = {slide_auc:.2f} [{lo:.2f}-{hi:.2f}]")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(model_name, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{model_name}_slide_roc_curves.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # patient level: average slide probabilities per patient, then take the argmax
    prob_df = pd.DataFrame(prob_all, columns=clusters)
    prob_df["patient"] = patients_all
    patient_prob = prob_df.groupby("patient")[clusters].mean()

    patient_true = pd.Series(true_all, index=patients_all).groupby(level=0).first()
    patient_true = patient_true.loc[patient_prob.index]
    patient_pred = patient_prob.idxmax(axis=1)

    row, pat_auc, lo, hi = summarise(model_name, patient_true.values, patient_pred.values,
                                     patient_prob.values, patient_prob.index.values)
    patient_rows.append(row)
    print(f"\n  patient-level weighted AUC {pat_auc:.2f} [{lo:.2f}-{hi:.2f}]")
    print(classification_report(patient_true, patient_pred, target_names=labels, digits=2))

    save_confusion_matrix(patient_true, patient_pred, labels, model_name,
                          f"{model_name}_patient_confusion_matrix.png")

slide_df = pd.DataFrame(slide_rows).set_index("Model")
patient_df = pd.DataFrame(patient_rows).set_index("Model")

print("\n--- slide level ---")
print(slide_df.to_string())
print("\n--- patient level ---")
print(patient_df.to_string())

slide_df.to_csv("model_comparison_slide_level.csv")
patient_df.to_csv("model_comparison_patient_level.csv")
