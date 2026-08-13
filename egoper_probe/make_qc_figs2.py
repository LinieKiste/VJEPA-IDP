"""QC figures for the EgoPER phase-2 probe (German labels, English GT terms).

Fig 1: 6 example videos (3 successful detections, 3 missed errors) — representative
       frame from the error segment + label below.
Fig 2: per error-category detection rate over all held-out videos.

Reproduces the probe.py protocol (GroupShuffleSplit by video, logreg C=1, mean-pool).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from decord import VideoReader, cpu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "egoper_probe"))
from probe import load_task  # noqa: E402

TASK = "coffee"
SEED = 0
TEST_FRAC = 0.3
ANN = json.loads((ROOT / "datasets" / "egoper" / "annotation.json").read_text())
SEGS = {s["video_id"]: s["labels"] for s in ANN[TASK]["segments"]}
I2T = {v: k for k, v in ANN[TASK]["actiontype2idx"].items()}
VIDEO_DIR = ROOT / "datasets" / "egoper" / "Coffee" / "trim_videos"

X, y, groups, vid_has_err = load_task(TASK, "feats")
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
tr, te = next(gss.split(X, y, groups))
clf = make_pipeline(StandardScaler(),
                    LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
clf.fit(X[tr], y[tr])
prob = clf.predict_proba(X[te])[:, 1]
test_vids = sorted(set(groups[te]))


def err_segments(vid: str) -> list[tuple[float, float, str]]:
    lab = SEGS[vid]
    return [(ts[0], ts[1], I2T[at]) for at, ts in zip(lab["action_type"], lab["time_stamp"])
            if at != 0]


def err_descs(vid: str) -> list[tuple[float, float, str, str]]:
    lab = SEGS[vid]
    return [(ts[0], ts[1], I2T[at], desc)
            for at, ts, desc in zip(lab["action_type"], lab["time_stamp"], lab["error_description"])
            if at != 0]


def window_probs(vid: str) -> tuple[np.ndarray, np.ndarray]:
    m = groups[te] == vid
    times = np.load(ROOT / "egoper_probe" / "features" / TASK / f"{vid}.npz")["times"]
    return prob[m], times


def peak_inside(vid: str, s: float, e: float) -> float:
    p, times = window_probs(vid)
    ctr = (times[:, 0] + times[:, 1]) / 2
    sel = (ctr >= s) & (ctr <= e)
    return float(p[sel].max()) if sel.any() else 0.0


def frame_at(vid: str, t: float) -> np.ndarray:
    vr = VideoReader(str(VIDEO_DIR / f"{vid}.mp4"), ctx=cpu(0))
    fps = float(vr.get_avg_fps()) or 15.0
    idx = min(int(round(t * fps)), len(vr) - 1)
    return vr[idx].asnumpy()


# ---- pick examples -----------------------------------------------------------
success = ["coffee_u1_a7_error_007", "coffee_u1_a5_error_024", "coffee_u1_a2_error_023"]
fail = ["coffee_u1_a7_error_002", "coffee_u1_a6_error_034", "coffee_u1_a6_error_031"]

examples = []
for vid in success + fail:
    descs = err_descs(vid)
    s, e, cat, desc = max(descs, key=lambda d: peak_inside(vid, d[0], d[1]))
    p_in = peak_inside(vid, s, e)
    t_show = (s + e) / 2
    examples.append({"vid": vid, "cat": cat, "desc": desc, "p_in": p_in,
                     "t_show": t_show, "ok": vid in success})

# ---- Fig 1: 6 examples ------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
for ax, ex in zip(axes.ravel(), examples):
    frame = frame_at(ex["vid"], ex["t_show"])
    ax.imshow(frame)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(ex["vid"].replace("coffee_", ""), fontsize=10, pad=4)
    if ex["ok"]:
        head = f"Erfolg — Fehler erkannt (p = {ex['p_in']:.2f})"
        color = "#005293"
    else:
        head = f"Fehler übersehen (p = {ex['p_in']:.2f})"
        color = "#E37222"
    ax.set_xlabel(f"{head}\n{ex['cat']}: {ex['desc']}", fontsize=8.5, color=color,
                  labelpad=6, linespacing=1.25)
fig.suptitle("Beispiele: Fehlererkennung auf gehaltenen Videos (EgoPER, Kaffee)",
             fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(ROOT / "egoper_probe" / "qc_probe_examples.png", dpi=150)
print("wrote qc_probe_examples.png")

# ---- Fig 2: detection rate per error category --------------------------------
cat_tot, cat_det = {}, {}
for vid in test_vids:
    for s, e, cat in err_segments(vid):
        cat_tot[cat] = cat_tot.get(cat, 0) + 1
        if peak_inside(vid, s, e) > 0.5:
            cat_det[cat] = cat_det.get(cat, 0) + 1

cats = ["Error_Modification", "Error_Slip", "Error_Correction", "Error_Addition"]
tot = [cat_tot.get(c, 0) for c in cats]
det = [cat_det.get(c, 0) for c in cats]
rate = [d / t if t else 0 for d, t in zip(det, tot)]

fig, ax = plt.subplots(figsize=(8.5, 4.6))
x = np.arange(len(cats))
bars = ax.bar(x, rate, color="#005293", width=0.55)
for xi, r, d, t in zip(x, rate, det, tot):
    ax.text(xi, r + 0.02, f"{d}/{t}", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=10)
ax.set_ylabel("Erkennungsrate (Anteil erkannter Fehler-Segmente)", fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_title("Erkennungsrate je Fehlerkategorie (gehaltene Videos, Schwellwert p = 0.5)",
             fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(ROOT / "egoper_probe" / "qc_probe_categories.png", dpi=150)
print("wrote qc_probe_categories.png")
print("per-category:", {c: f"{det[i]}/{tot[i]}" for i, c in enumerate(cats)})
