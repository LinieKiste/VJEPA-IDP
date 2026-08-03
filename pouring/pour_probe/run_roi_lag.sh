#!/usr/bin/env bash
# ROI-vs-center-crop cross-view comparison, rerun WITH the 0.7 s water-transit lag.
#
# The published ROI comparison (CAM2-native 0.899 vs 0.738, CAM2->CAM3 0.524 vs 0.706)
# was run at lag 0 on both arms. The lag correction was worth +0.09 R2 elsewhere, so both
# arms are rerun here at lag 0.7; comparing a lagged ROI run against the old un-lagged
# center-crop bar would confound crop with lag.
#
# Two 80-min trainings + two cross-view evals, sequential (one GPU).
set -euo pipefail

cd "$(dirname "$0")"
PY=../../.venv/bin/python
CENTER=/home/casimir/.cache/pour_probe/clips_frames288
ROI=/home/casimir/.cache/pour_probe/clips_frames288_roi
LOG=/home/casimir/.cache/pour_probe/roi_lag_logs
mkdir -p "$LOG"

echo "### [1/4] center-crop CAM2 @ lag 0.7  ($(date +%H:%M))"
POUR_FRAMES288_DIR=$CENTER $PY clips_train_attn.py \
  --target flow --cam CAM2 --lag_s 0.7 --minutes 80 2>&1 | tee "$LOG/train_center.log"

echo "### [2/4] detector-ROI CAM2 @ lag 0.7  ($(date +%H:%M))"
POUR_FRAMES288_DIR=$ROI $PY clips_train_attn.py \
  --target flow --cam CAM2 --lag_s 0.7 --minutes 80 --tag_extra roi 2>&1 | tee "$LOG/train_roi.log"

echo "### [3/4] cross-view, center crop  ($(date +%H:%M))"
POUR_FRAMES288_DIR=$CENTER $PY clips_eval_crossview.py \
  --target flow --tag _lag0.7 --lag_s 0.7 2>&1 | tee "$LOG/eval_center.log"

echo "### [4/4] cross-view, detector ROI  ($(date +%H:%M))"
POUR_FRAMES288_DIR=$ROI $PY clips_eval_crossview.py \
  --target flow --tag _lag0.7_roi --lag_s 0.7 2>&1 | tee "$LOG/eval_roi.log"

echo "### done  ($(date +%H:%M))"
