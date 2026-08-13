"""Figure for the EgoPER phase-2 probe: ROC curves + per-window probability timelines.

Reads the cached features (egoper_probe/features/coffee/*.npz), reproduces the
probe.py protocol (GroupShuffleSplit by video, logreg, one-class kNN), and plots:
  left:  window-level ROC curves (supervised mean-pool, supervised SlowFast, one-class)
  right: predicted error probability over time for a few held-out videos
         (GT error segments shaded)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "egoper_probe"))
from probe import load_task, oneclass_scores  # noqa: E402

TASK = "coffee"
SEED = 0
TEST_FRAC = 0.3
OUT = ROOT / "egoper_probe" / "qc_probe_roc.png"

X, y, groups, vid_has_err = load_task(TASK, "feats")
X_sf, _, _, _ = load_task(TASK, "feats_sf")
print(f"{TASK}: {len(X)} windows, {len(set(groups))} videos, "
      f"error windows {y.sum()} ({100*y.mean():.1f}%)")

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
tr, te = next(gss.split(X, y, groups))
test_vids = sorted(set(groups[te]))
print(f"train {len(set(groups[tr]))} vids / test {len(test_vids)} vids")

def fit_predict(Xtr, Xte, C):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=C, class_weight="balanced"))
    clf.fit(Xtr, y[tr])
    return clf.predict_proba(Xte)[:, 1]

prob_mean = fit_predict(X[tr], X[te], 1.0)
prob_sf = fit_predict(X_sf[tr], X_sf[te], 0.001)
tr_normal = X[tr][np.array([not vid_has_err[v] for v in groups[tr]])]
oc = oneclass_scores(tr_normal, X[te], k=10)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

ax = axes[0]
for name, p, c in [("mean-pool logreg (C=1)", prob_mean, "#005293"),
                   ("SlowFast logreg (C=0.001)", prob_sf, "#E37222"),
                   ("one-class kNN (normal-only)", oc, "#A2AD00")]:
    fpr, tpr, _ = roc_curve(y[te], p)
    auc = roc_auc_score(y[te], p)
    ax.plot(fpr, tpr, c=c, lw=2, label=f"{name}  AUC {auc:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_title("Window-level error detection (held-out videos)")
ax.legend(loc="lower right", fontsize=9)
ax.set_aspect("equal")

ax = axes[1]
times = {}
for f in sorted((ROOT / "egoper_probe" / "features" / TASK).glob("*.npz")):
    d = np.load(f)
    times[f.stem] = d["times"]
show = [v for v in test_vids if vid_has_err[v]][:3] + [v for v in test_vids if not vid_has_err[v]][:1]
for v in show:
    m = groups[te] == v
    t = times[v][:, 0]
    ax.plot(t, prob_mean[m], c="#005293", lw=1.6, label=f"{v} (GT: {'error' if vid_has_err[v] else 'normal'})")
    if vid_has_err[v]:
        err_t = times[v][y[te][m] == 1]
        if len(err_t):
            for s, e in err_t:
                ax.axvspan(s, e, color="#E37222", alpha=0.25)
ax.set_xlabel("time in video (s)")
ax.set_ylabel("predicted error probability")
ax.set_title("Predicted error probability over time (held-out videos)")
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(-0.05, 1.05)

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
