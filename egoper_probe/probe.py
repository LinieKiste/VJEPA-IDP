"""Train + evaluate a light probe on cached V-JEPA 2 window features for EgoPER error detection.

Loads the per-video ``.npz`` caches from ``extract.py``, splits train/test **by video**
(GroupShuffleSplit — no window leakage across the split), trains a logistic-regression
probe on the frozen features, and reports window-level and video-level error-detection
metrics (ROC-AUC, average precision). Everything is logged to mlflow.

Two feature views (``--view``): ``feats`` (mean-pool, 1024-d) or ``feats_sf`` (SlowFast
detail, 24576-d: slow 4x4 spatial + fast 8-step temporal). Two settings per run:
  - **supervised**: logreg, error videos in the train split.
  - **one-class** (EgoPER-faithful): fit on NORMAL-video windows only, score test windows
    by distance to the normal distribution. Mahalanobis is well-conditioned only in low
    dim, so for high-dim views we use **kNN distance to the normal-window bank** (mean
    distance to the k nearest normal training windows) which is dimension-robust.

Usage:
    .venv/bin/python egoper_probe/probe.py --task coffee --view feats_sf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_DIR = Path(__file__).parent / "features"


def load_task(task: str, view: str = "feats"):
    """Return (feats [N,D], labels [N], groups [N] video-id, video_has_error dict).

    ``view`` selects the cached pooling: ``feats`` (mean-pool, 1024-d) or ``feats_sf``
    (SlowFast detail, 24576-d). Older caches only have ``feats``."""
    files = sorted((FEATURES_DIR / task).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no cached features in {FEATURES_DIR / task} — run extract.py first")
    feats, labels, groups, vid_has_err = [], [], [], {}
    for f in files:
        d = np.load(f)
        vid = f.stem
        if view not in d:
            raise KeyError(f"{f} has no '{view}' array (keys={list(d.keys())}) — re-run extract.py")
        feats.append(d[view])
        labels.append(d["labels"])
        groups.extend([vid] * len(d["labels"]))
        vid_has_err[vid] = "error" in vid  # filename convention; matches GT has_error
    return np.concatenate(feats), np.concatenate(labels), np.asarray(groups), vid_has_err


def oneclass_scores(X_tr_normal, X_te, k=10):
    """One-class anomaly scores for test windows, dimension-robust.

    Standardize on the normal-train windows, then score each test window by the **mean
    distance to its k nearest normal-train windows** (higher = more anomalous). kNN avoids
    the D x D covariance that makes Mahalanobis unusable for the 24k-d SlowFast view."""
    sc = StandardScaler().fit(X_tr_normal)
    nn = NearestNeighbors(n_neighbors=min(k, len(X_tr_normal))).fit(sc.transform(X_tr_normal))
    dist, _ = nn.kneighbors(sc.transform(X_te))
    return dist.mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="coffee")
    ap.add_argument("--view", default="feats", choices=["feats", "feats_sf"],
                    help="feats=mean-pool 1024d, feats_sf=SlowFast detail 24576d")
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 reg strength")
    ap.add_argument("--knn_k", type=int, default=10, help="k for one-class kNN scoring")
    args = ap.parse_args()

    X, y, groups, vid_has_err = load_task(args.task, args.view)
    print(f"{args.task} [{args.view}, dim={X.shape[1]}]: {len(X)} windows from "
          f"{len(set(groups))} videos | error windows: {y.sum()} ({100 * y.mean():.1f}%)")

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    tr, te = next(gss.split(X, y, groups))

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=args.C, class_weight="balanced"),
    )
    clf.fit(X[tr], y[tr])
    prob = clf.predict_proba(X[te])[:, 1]

    # window-level
    win_auc = roc_auc_score(y[te], prob)
    win_ap = average_precision_score(y[te], prob)

    # video-level: a video is "predicted error" by its max window prob; GT = has_error
    test_vids = sorted(set(groups[te]))
    vid_score = {v: prob[(groups[te] == v)].max() for v in test_vids}
    vid_y = np.array([int(vid_has_err[v]) for v in test_vids])
    vid_p = np.array([vid_score[v] for v in test_vids])
    vid_auc = roc_auc_score(vid_y, vid_p) if len(set(vid_y)) > 1 else float("nan")

    # one-class (EgoPER-faithful): fit only on NORMAL-video windows from the train split,
    # score test windows by kNN distance to that normal bank (dimension-robust).
    tr_normal = X[tr][np.array([not vid_has_err[v] for v in groups[tr]])]
    oc_score = oneclass_scores(tr_normal, X[te], k=args.knn_k)
    oc_win_auc = roc_auc_score(y[te], oc_score)
    oc_win_ap = average_precision_score(y[te], oc_score)

    print(f"[supervised] window ROC-AUC={win_auc:.3f}  AP={win_ap:.3f}")
    print(f"[supervised] video  ROC-AUC={vid_auc:.3f}  ({len(test_vids)} test videos)")
    print(f"[one-class ] window ROC-AUC={oc_win_auc:.3f}  AP={oc_win_ap:.3f}  "
          f"(kNN k={args.knn_k} on {len(tr_normal)} normal-video windows)")

    mlflow.set_experiment(f"egoper_probe_{args.task}")
    with mlflow.start_run():
        mlflow.log_params({
            "task": args.task, "view": args.view, "test_frac": args.test_frac,
            "seed": args.seed, "C": args.C, "knn_k": args.knn_k,
            "n_windows": len(X), "n_videos": len(set(groups)), "dim": X.shape[1],
            "probe": "logreg_balanced", "oneclass": "knn",
        })
        mlflow.log_metrics({
            "window_roc_auc": win_auc, "window_ap": win_ap,
            "video_roc_auc": vid_auc, "error_window_frac": float(y.mean()),
            "oneclass_window_roc_auc": oc_win_auc, "oneclass_window_ap": oc_win_ap,
        })
    print("logged to mlflow.")


if __name__ == "__main__":
    main()
