"""
Protein feature selection by voting across CV folds.

In each training fold: scale, keep the top 50 proteins by ANOVA p-value, then
run SFS and RFE to pick 3 proteins each. Proteins are ranked by how many folds
picked them, and the top 3 per method become the panels used in
protein_panel_classification.py. Selected proteins are also compared against
a list of literature biomarkers.
"""

import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from venn import venn

warnings.filterwarnings("ignore")

N_SELECT = 3     # proteins picked per fold by each method
N_PREFILTER = 50 # proteins kept after the ANOVA pre-filter

# literature biomarkers (reviewed UniProt entries only)
try:
    marker_df = pd.read_excel("idmapping.xlsx")
    marker_df = marker_df[marker_df["Reviewed"] == "reviewed"]
    markers = set(marker_df["Entry"].dropna())
    print(f"{len(markers)} literature biomarkers loaded")
except Exception as e:
    print(f"couldn't load idmapping.xlsx ({e}), continuing without literature overlap")
    markers = set()

df = pd.read_csv("tumour_zscore_clusters.csv")
meta_cols = ["cluster", "case_submitter_id", "sample_name"]
protein_cols = [c for c in df.columns if c not in meta_cols]
y = LabelEncoder().fit_transform(df["cluster"])

# only keep proteins with no missing values at all
missing = df[protein_cols].isna().mean()
complete = missing[missing == 0].index.tolist()
X = df[complete]
print(f"{len(complete)} of {len(protein_cols)} proteins have no missing values")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sfs_counts, rfe_counts = Counter(), Counter()

for train_idx, _ in cv.split(X, y):
    y_train = y[train_idx]
    X_train = pd.DataFrame(StandardScaler().fit_transform(X.iloc[train_idx]), columns=complete)

    # keep the 50 proteins with the smallest ANOVA p-values
    pvals = {p: f_oneway(*[X_train.loc[y_train == c, p] for c in np.unique(y_train)])[1]
             for p in complete}
    top50 = pd.Series(pvals).nsmallest(N_PREFILTER).index.tolist()
    X_sub = X_train[top50]

    lr = LogisticRegression(max_iter=2000)

    sfs = SequentialFeatureSelector(lr, n_features_to_select=N_SELECT, direction="forward",
                                    scoring="balanced_accuracy", cv=3, n_jobs=-1)
    sfs.fit(X_sub, y_train)
    sfs_counts.update(np.array(top50)[sfs.get_support()])

    rfe = RFE(lr, n_features_to_select=N_SELECT).fit(X_sub, y_train)
    rfe_counts.update(np.array(top50)[rfe.support_])

pd.DataFrame(sfs_counts.most_common(), columns=["protein", "n_folds_selected"]).to_csv(
    "feature_votes_sfs.csv", index=False)
pd.DataFrame(rfe_counts.most_common(), columns=["protein", "n_folds_selected"]).to_csv(
    "feature_votes_rfe.csv", index=False)

top_sfs = [p for p, _ in sfs_counts.most_common(3)]
top_rfe = [p for p, _ in rfe_counts.most_common(3)]
sfs_all, rfe_all = set(sfs_counts), set(rfe_counts)
pipeline_union = sfs_all | rfe_all

print("\nTop 3 by votes")
print("SFS:", top_sfs)
print("RFE:", top_rfe)

print("\nOverlap with literature biomarkers")
print("SFS top 3:", sorted(markers & set(top_sfs)))
print("RFE top 3:", sorted(markers & set(top_rfe)))

overlap = markers & pipeline_union
print(f"\n{len(pipeline_union)} proteins selected in any fold by either method")
print(f"{len(overlap)} of them are literature biomarkers: {sorted(overlap)}")


def plot_venn(sfs_set, rfe_set, filename):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    venn({"SFS features": sfs_set, "RFE features": rfe_set},
         cmap=["#377EB8", "#E41A1C"], alpha=0.35, fontsize=11,
         legend_loc="upper right", fmt="", ax=ax)

    # protein names go inside each region instead of counts
    regions = [
        (0.26, sorted(sfs_set - rfe_set), "#1F4E79", "medium"),
        (0.74, sorted(rfe_set - sfs_set), "#8B0000", "medium"),
        (0.50, sorted(sfs_set & rfe_set), "#2A2A2A", "bold"),
    ]
    for x, names, colour, weight in regions:
        ax.text(x, 0.58, "\n".join(names), transform=ax.transAxes, ha="center", va="center",
                fontsize=7.5, color=colour, fontweight=weight)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


plot_venn(sfs_all, rfe_all, "venn_sfs_vs_rfe.png")
