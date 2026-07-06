# pour_probe — pouring flow/volume regression from frozen V-JEPA 2 (UWLPD)

Frozen V-JEPA 2 ViT-L feature probe for **pouring volume / flow estimation**, on the
**UW Liquid Pouring Dataset (UWLPD)**. Mirrors `exprt_probe/` / `egoper_probe/`: cache
frozen sliding-window features → train a light head → track in mlflow (`pour_probe`).

## Data
`datasets/UWLPD/` = UWLPD **Real Robot Dataset**, `large_bowls` subset: 5 zips
(source→target combos `bowl←{bottle,cup,mug}`, `fruitBowl←{bottle,cup}`), 36 conditions
each = **180 sequences** (~61 GB, kept zipped — frames read straight from the zips).
Condition grid: fill `{empty,30,60,90%}` × profile `{dump,hold,partial}` × motion
`{minimal,moderate,high}` (from each seq's `render_v3/sim_args.txt`). Per frame:
`data*.jpg` (RGB 640×480), `ground_truth*.png` (**binary liquid mask**, liquid = blue
`(0,0,255,255)` on transparent-black).

**No mL ground truth here** — that lives in the separate *Simulated* dataset's
`bowl_volume.csv` (requested from Schenck; not yet in hand). So the target is a
mask-derived **proxy**, used to smoke-test the pipeline; the real mL trace drops in later.

## Targets (proxy, per window)
- `flow`   = mean liquid-pixel area over the window's frames (visible-liquid signal).
- `volume` = running-max liquid area up to the window's last frame (monotone accumulation).

## Pipeline
```
python pour_probe/dataset.py                 # build+verify mapping.json (180 seqs)
python pour_probe/extract.py                 # frozen ViT-L token grids + proxy targets -> features/
python pour_probe/pool.py --pool ek100       # frozen attentive pooler -> pooled/ (1 vec/clip)
python pour_probe/pool.py --pool mean        #   ablation: plain mean over tokens -> pooled_mean/
python pour_probe/train.py --target flow     # SmoothL1 linear probe, GroupKFold by seq, mlflow
python pour_probe/train.py --target volume
python pour_probe/attn_map.py                # attention-map interpretability (Stage-1 gate)
```
Feature/pooled caches live on the **Storage HDD** (`/mnt/storage/pour_probe/`) to keep the
SSD free; override with `$POUR_FEATURES_DIR` / `$POUR_POOLED_ROOT`.

## Design notes
- **Reuse:** `_encoder.py` → `video_qa/model.py::build_encoder` (`checkpoints/vitl.pt`,
  `target_encoder`); `head.py` = `AttentiveClassifier(num_outputs=1)` from
  `vjepa2/src/models/attentive_pooler.py`, warm-started from the EK100 probe
  (`checkpoints/ek100-vitl-256.pt`, action query) — same as `exprt_probe/head.py`.
- **Windows in FRAME units** (`window_frames`/`stride_frames`) — the real-robot recordings
  ship no reliable fps; default 32-frame span sampled to 16, stride 16 (~28 windows/seq).
- **Eval = clip-level, grouped by sequence** (`GroupKFold`): whole sequences held out so
  appearance can't leak. Headline = **test MAE vs predict-train-mean baseline** + R².
- **`--train_pooler` stretch:** fine-tune the pooler in-loop on token grids (the faithful
  V-JEPA 2 Appendix-C probe) — deferred until the real mL target arrives.

## Status
Pipeline built + verified end to end (mask-proxy smoke test beats the mean baseline). The
reported volume number waits for the Simulated-dataset mL; then: pretrain on sim →
fine-tune on own-lab data (RGB + cumulative-mL, matched to this loader).
