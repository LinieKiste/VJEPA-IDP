# egoper_probe — frozen V-JEPA 2 feature probe for EgoPER error detection

**This is the project's centerpiece** (V-JEPA evaluation), not a baseline. It asks the core
question directly: *do frozen V-JEPA 2 latents separate correct vs. erroneous procedure
steps?* The `egoper_vqa/` Qwen2.5-VL work is the generic-VLM **baseline/contrast**.

## Pipeline
1. **`extract.py`** — frozen V-JEPA 2 ViT-L over sliding windows of each EgoPER video.
   Each window's 2048-token grid is pooled into **two cached views** (one forward pass):
   `feats` (global mean-pool, 1024-d) and `feats_sf` (SlowFast detail, 24,576-d: slow =
   temporal-mean + 4×4 spatial pool = 16 tokens preserving *where*; fast = spatial-mean
   over all 8 temporal slices = 8 tokens preserving *when*). Plus `[start,end]` time +
   error label (1 if the window overlaps a GT error segment, i.e. any
   `action_type != Normal`). Cached per video to `features/<task>/<video_id>.npz`
   (~1.5 GB/task). bf16 autocast + decode-once: ~45 s/video, peak ~5.3 GB GPU.
2. **`probe.py`** — loads the cache, splits train/test **by video** (no window leakage),
   trains a logistic-regression probe on the frozen features (`--view feats|feats_sf`),
   plus a one-class kNN scorer (fit on normal-video windows only), reports window- and
   video-level error-detection ROC-AUC / AP, logs to **mlflow**.

## Run
Uses the project `.venv` (torch, decord, sklearn, mlflow all present). Reuses
`video_qa/model.py::build_encoder` (local `vjepa2` pkg + `checkpoints/vitl.pt`).

```bash
# 1) extract features (use the GPU — CPU is ~100x slower). ~50 min for coffee at stride 2s.
.venv/bin/python egoper_probe/extract.py --task coffee --window_s 4 --stride_s 2

# 2) train + evaluate the probe (best config)
.venv/bin/python egoper_probe/probe.py --task coffee --view feats_sf --C 0.001

# 3) inspect runs
.venv/bin/mlflow ui   # then open http://localhost:5000
```

## Results (coffee, 68 vids → 14,744 windows, 3.9% error, 5–10 seed GroupShuffleSplit)
| setting | view | window ROC-AUC | window AP |
|---|---|---|---|
| supervised logreg C=1 | mean-pool 1024d | 0.749 ± 0.045 | 0.132 ± 0.039 |
| **supervised logreg C=0.001** | **SlowFast 24,576d** | **0.779 ± 0.049** | **0.163 ± 0.026** |
| one-class kNN (k=10) | mean-pool | 0.606 ± 0.036 | 0.073 |
| one-class kNN / max-over-tokens | SlowFast | ~0.60 | ~0.07 |

- **The V-JEPA thesis holds:** frozen latents separate correct vs. erroneous windows at
  ROC-AUC ≈ 0.78 with only a linear head. Detail-preserving (SlowFast) pooling adds
  +0.03 AUC/AP over mean-pool **but only with strong L2** (C=1 overfits 24k-d → 0.659).
- **One-class is the open problem** (~0.60 for every view/scorer tried; kNN >> Mahalanobis
  0.554). Subtle errors aren't outliers in frozen-feature space. Ideas: per-position token
  banks, temporal context across windows, task-step conditioning.

## Notes
- **Supervised vs one-class.** The supervised probe uses error videos in training. The
  one-class variant (fit on normal only) is closer to **EgoPER's own setting** (they assume
  only normal videos at train).
- **Comparability caveat.** EgoPER's EgoPED assumes no task-graph access and only normal
  training videos. The supervised probe uses error videos, so its numbers aren't a
  like-for-like comparison to their benchmark — it's a representation-quality probe.
- **Window labeling** is overlap-based; brief errors need a tight `--stride_s` (≤ window
  length) or they fall between windows.
- sklearn logreg on the 24k-d view is CPU-only (~1 min/fit); switch to a torch linear head
  on GPU if doing larger sweeps.
