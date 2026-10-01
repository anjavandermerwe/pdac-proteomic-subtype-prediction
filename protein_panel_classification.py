"""
Compare the SFS and RFE protein panels from feature_selection.py across five
classifiers using 5-fold stratified CV.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier

from utils.bootstrap_ci import bootstrap_multiclass_auc_ci

warnings.filterwarnings("ignore")

out_dir = Path("results")
out_dir.mkdir(exist_ok=True)

panels = {
    "SFS": ["F8VZS0", "Q13885", "O95479"],
    "RFE": ["Q9NR12", "Q04206", "Q8NCW5"],
}

df = pd.read_csv("tumour_zscore_clusters.csv")
target_col = "Subtype" if "Subtype" in df.columns else "cluster"

le = LabelEncoder()
y = le.fit_transform(df[target_col])
n_classes = len(le.classes_)


def subtype_label(c):
    c = str(c)
    if c.isdigit():
        return f"Subtype {c}"
    return c.replace("Cluster", "Subtype").replace("cluster", "Subtype")


target_names = [subtype_label(c) for c in le.classes_]


def make_classifiers():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                random_state=42, n_jobs=-1),
        "SVM": SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                 eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0),
        "Logistic Regression": LogisticRegression(C=1.0, class_weight="balanced",
                                                  max_iter=1000, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500,
                             early_stopping=True, random_state=42),
    }


summary = []

for panel_name, proteins in panels.items():
    print(f"\n{panel_name}: {proteins}")
    X = df[proteins]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    res = {name: {"bal_acc": [], "roc": [], "true": [], "pred": [], "prob": []}
           for name in make_classifiers()}
    classifiers = make_classifiers()

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            prob = clf.predict_proba(X_test)

            r = res[name]
            r["bal_acc"].append(balanced_accuracy_score(y_test, pred))
            r["roc"].append(roc_auc_score(y_test, prob, multi_class="ovr", average="weighted"))
            r["true"].extend(y_test)
            r["pred"].extend(pred)
            r["prob"].extend(prob)

    for name, r in res.items():
        true, pred, prob = np.array(r["true"]), np.array(r["pred"]), np.array(r["prob"])

        precision, recall, f1, _ = precision_recall_fscore_support(
            true, pred, average="weighted", zero_division=0)
        pooled_auc = roc_auc_score(true, prob, multi_class="ovr", average="weighted")

        # each sample is its own bootstrap unit here
        lo, hi = bootstrap_multiclass_auc_ci(true, prob, range(n_classes),
                                             groups=np.arange(len(true)))

        summary.append({
            "Feature Group": panel_name,
            "Model": name,
            "Weighted Precision": precision,
            "Weighted Recall": recall,
            "Weighted F1-Score": f1,
            "Accuracy": accuracy_score(true, pred),
            "Mean Fold ROC-AUC": np.mean(r["roc"]),
            "Global ROC-AUC": pooled_auc,
            "Global ROC-AUC 95% CI": f"[{lo:.2f}-{hi:.2f}]",
            "CV Balanced Accuracy (Mean)": np.mean(r["bal_acc"]),
            "CV Balanced Accuracy (Std)": np.std(r["bal_acc"]),
        })

        prefix = f"{out_dir}/{panel_name}_{name.replace(' ', '_')}"
        title = f"{name} ({panel_name})"

        report = classification_report(true, pred, target_names=target_names,
                                       zero_division=0, output_dict=True)
        pd.DataFrame(report).T.to_csv(f"{prefix}_overall_classification_report.csv")

        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(true, pred, labels=range(n_classes)), annot=True, fmt="d",
                    cmap="Blues", xticklabels=target_names, yticklabels=target_names)
        plt.title(f"Confusion matrix: {title}")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(f"{prefix}_confusion_matrix.png", dpi=300)
        plt.close()

        # one-vs-rest ROC curve for each subtype
        true_bin = label_binarize(true, classes=range(n_classes))
        plt.figure(figsize=(7, 6))
        for i, label in enumerate(target_names):
            fpr, tpr, _ = roc_curve(true_bin[:, i], prob[:, i])
            plt.plot(fpr, tpr, lw=2, label=f"{label.replace('Subtype ', 'S-')} (AUC = {auc(fpr, tpr):.2f})")
        plt.plot([], [], " ", label=f"Weighted AUC = {pooled_auc:.2f} [{lo:.2f}-{hi:.2f}]")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"ROC curves: {title}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{prefix}_roc_curve.png", dpi=300)
        plt.close()

        print(f"  {name}: bal acc {np.mean(r['bal_acc']):.2f} +/- {np.std(r['bal_acc']):.2f}, "
              f"AUC {pooled_auc:.2f} [{lo:.2f}-{hi:.2f}]")

pd.DataFrame(summary).to_csv(f"{out_dir}/comprehensive_model_performance_summary.csv", index=False)
