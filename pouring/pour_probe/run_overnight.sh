#!/usr/bin/env bash
# Overnight batch, two independent studies, sequential on the one GPU. ~11 h.
#
# STUDY 1 (~5.7 h): error bars on the crop-vs-cross-view result.
#   The published ROI comparison rests on ONE split ({8,13,21,24} = fold A) and its
#   headline conclusion already reversed once, when the lag bug in baselines_on_split
#   was fixed. Folds B and C repeat both arms (center crop, detector ROI) at lag 0.7 so
#   the +0.23 cross-view gain gets a mean and a spread instead of a single point.
#
# STUDY 2 (~5.3 h): 4-fold attentive VOLUME on both cams.
#   The audio-vs-video volume comparison is currently protocol-mismatched: the audio row
#   is frozen+ridge 4-fold, the V-JEPA volume 0.667 is the attentive probe on one split.
#   This produces the matched 4-fold attentive number. Volume is lag-insensitive
#   (the sweep spans 0.08 R2 and peaks negative), so it runs at lag 0, matching
#   clips_cnn_baseline's convention.
#
# Folds are clips_cnn_baseline.FOLDS, the same disjoint trial groups every other
# 4-fold number in the project uses. Fold A of study 1 already exists and is skipped.
set -uo pipefail

cd "$(dirname "$0")"
PY=../../.venv/bin/python
CENTER=/home/casimir/.cache/pour_probe/clips_frames288
ROI=/home/casimir/.cache/pour_probe/clips_frames288_roi
LOG=/home/casimir/.cache/pour_probe/overnight_logs
mkdir -p "$LOG"

FOLD_A="13,21,24,8"
FOLD_B="11,12,7,9"
FOLD_C="15,16,25,26,5"
FOLD_D="17,18,20,22,27"

step() { echo ""; echo "### $1  ($(date '+%F %H:%M'))"; }

# ---------------------------------------------------------------- study 1
for F in B C; do
  case $F in
    B) TRIALS=$FOLD_B ;;
    C) TRIALS=$FOLD_C ;;
  esac

  step "[study1] center-crop CAM2 fold$F @ lag 0.7"
  POUR_FRAMES288_DIR=$CENTER $PY clips_train_attn.py \
    --target flow --cam CAM2 --lag_s 0.7 --minutes 80 \
    --val_trials "$TRIALS" --fold "fold$F" 2>&1 | tee "$LOG/s1_center_fold$F.log"

  step "[study1] detector-ROI CAM2 fold$F @ lag 0.7"
  POUR_FRAMES288_DIR=$ROI $PY clips_train_attn.py \
    --target flow --cam CAM2 --lag_s 0.7 --minutes 80 \
    --val_trials "$TRIALS" --fold "fold$F" --tag_extra roi 2>&1 | tee "$LOG/s1_roi_fold$F.log"

  step "[study1] cross-view eval, center crop, fold$F"
  POUR_FRAMES288_DIR=$CENTER $PY clips_eval_crossview.py \
    --target flow --tag "_lag0.7_fold$F" --lag_s 0.7 \
    --val_trials "$TRIALS" 2>&1 | tee "$LOG/s1_eval_center_fold$F.log"

  step "[study1] cross-view eval, detector ROI, fold$F"
  POUR_FRAMES288_DIR=$ROI $PY clips_eval_crossview.py \
    --target flow --tag "_lag0.7_fold${F}_roi" --lag_s 0.7 \
    --val_trials "$TRIALS" 2>&1 | tee "$LOG/s1_eval_roi_fold$F.log"
done

# ---------------------------------------------------------------- study 2
for F in A B C D; do
  case $F in
    A) TRIALS=$FOLD_A ;;
    B) TRIALS=$FOLD_B ;;
    C) TRIALS=$FOLD_C ;;
    D) TRIALS=$FOLD_D ;;
  esac

  step "[study2] attentive volume both-cam fold$F"
  POUR_FRAMES288_DIR=$CENTER $PY clips_train_attn.py \
    --target volume --cam both --lag_s 0 --minutes 80 \
    --val_trials "$TRIALS" --fold "vol$F" 2>&1 | tee "$LOG/s2_volume_fold$F.log"
done

step "ALL DONE"
