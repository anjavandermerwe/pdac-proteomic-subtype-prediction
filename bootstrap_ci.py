import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def bootstrap_multiclass_auc_ci(y_true, y_score, classes, groups,
                                n_boot=1000, ci=0.95, seed=42):
    """
    Bootstrap CI for the weighted one-vs-rest multiclass ROC-AUC.
    Whole groups (e.g. patients) are resampled together so that rows from
    the same patient aren't treated as independent. If there is no grouping,
    pass groups=np.arange(len(y_true)).
    """
    rng = np.random.RandomState(seed)
    y_true, y_score, groups = np.asarray(y_true), np.asarray(y_score), np.asarray(groups)

    unique_groups = np.unique(groups)
    rows_by_group = {g: np.where(groups == g)[0] for g in unique_groups}

    aucs = []
    for _ in range(n_boot):
        picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([rows_by_group[g] for g in picked])
        y_bin = label_binarize(y_true[idx], classes=classes)
        try:
            aucs.append(roc_auc_score(y_bin, y_score[idx], average="weighted"))
        except ValueError:
            # resample was missing a class, can't compute AUC
            continue

    tail = (1 - ci) / 2 * 100
    lo, hi = np.percentile(aucs, [tail, 100 - tail])
    return lo, hi
