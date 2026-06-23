# exprt_probe — anomaly classification on eXprt with a frozen V-JEPA 2 + EK100-transferred head

Classify short clips of the **eXprt tea-making** recordings as **anomaly vs normal** (primary)
and **8-way error type** (secondary), using the frozen V-JEPA 2 ViT-L encoder and a small
attentive head **warm-started from the Epic-Kitchens-100 (EK100) attentive probe**. Loss and
metrics are tracked in mlflow. This mirrors the `egoper_probe/` pipeline, adapted to a new
dataset (PNG frame sequences, video-level labels) and a transferred head.

## Dataset (eXprt / "CAM1 Aufnahmen Patrick")
- Egocentric tea-making, one subject, **PNG frame sequences** (`frame_NNNN.png`, 1886×1056 @ 20 fps),
  ~80–280 s each. **40 trials = 8 classes × 5 iterations**: `Normal` (only non-anomaly) +
  `2tb 2stir`, `Spüli`, `glass and fork`, `no tea bag`, `not enough water`, `perplexity`, `sequence`.
- **Labels are video-level only** — one `start_time` per trial, no within-video localization of the
  anomaly. A clip inherits its whole video's label (weak supervision); positives are noisy for
  *transient* anomalies (`perplexity`, `sequence`), so we also report **video-aggregated** metrics.
- **Limitations:** tiny N (40 videos, only **5 Normal**) → splits grouped by video are high-variance;
  clip-level metrics (hundreds of clips) are the stabler signal.

### Trial → video mapping
The log has no video-id column and there are more recording dirs than trials (aborted re-takes).
PNGs carry no capture date (file mtime = rclone copy time); the only recording-time signal is the
**directory-name timestamp**. `dataset.py` maps each trial by: *bucket each recording dir to the
most-recent trial `Start Time` that precedes it; keep the longest recording per trial* (drops
aborts). Verified clean **40↔40, 5/class**. Persisted to `mapping.json` and as a `video_id` column
added to `trial_execution_log_*.csv` (frame folders are left untouched).

## Model
- **Encoder:** frozen V-JEPA 2 ViT-L (`checkpoints/vitl.pt`, `target_encoder`), forward
  `(B,C,T,H,W) → (B,N,1024)`, tubelet 2 / patch 16. **Not V-JEPA 2-AC** (AC is an action-conditioned
  robot world model needing an action stream we don't have, and has no classification head).
- **Head:** `AttentiveClassifier` (attentive pooler + linear, `vjepa2/src/models/attentive_pooler.py`)
  **warm-started from `checkpoints/ek100-vitl-256.pt`** — EK100's pooler (depth 4, 16 heads, 1024-d,
  trained for kitchen action anticipation) is loaded (49/49 params; its "action" query kept), its
  verb/noun/action linears are discarded, and a fresh 2-way / 8-way linear is attached. By default the
  pooler is frozen and only the linear trains (`--train_pooler` to fine-tune the pooler at a low LR).

## Files
- `dataset.py` — trial→video mapping + labels + frame access. Run once: `python exprt_probe/dataset.py`.
- `_encoder.py` — shared frozen encoder loader (reuses `video_qa/model.py::build_encoder`).
- `extract.py` — PNG loader + sliding clips → cache **full token grids** `(n_clips, 2048, 1024)` fp16
  + per-clip times/labels to `features/<video_id>.npz` (~9.8 GB).
- `head.py` — build + EK100 warm-start of the attentive head.
- `pool.py` — run the frozen pooler **once** → one 1024-d vector/clip. Running the pooler inside
  training reloads 9.8 GB/epoch (I/O-bound, GPU idle); precomputing makes `train.py` seconds.
  `--pool {ek100,mean,rand}` → `pooled/` / `pooled_mean/` / `pooled_rand/`.
- `train.py` — linear probe (torch loop, mlflow loss + metrics). `--level video` (primary) or `clip`.

## Run
```bash
.venv/bin/python exprt_probe/dataset.py                          # build & persist mapping (once)
.venv/bin/python exprt_probe/extract.py                          # token grids, all 40 videos (~1h, ~9.8 GB)
.venv/bin/python exprt_probe/pool.py --pool ek100                # pooled feats (also: --pool mean / rand)
.venv/bin/python exprt_probe/pool.py --pool mean                 # mean-pool feats (best for binary)
.venv/bin/python exprt_probe/train.py --level video --target binary
.venv/bin/python exprt_probe/train.py --level video --target multiclass --features_dir pooled
.venv/bin/mlflow ui                                              # experiment "exprt_anomaly"
```
Clip config defaults: 16 frames / 4 s window / 2 s stride / 256 px → 2048 tokens/clip.
Flags: `--level {video,clip}`, `--features_dir {pooled,pooled_mean,pooled_rand}`, `--seeds`, `--epochs`, `--lr`.

## Results (torch linear probe; see CLAUDE.md `exprt_probe/` for detail)
| level | task | mean-pool | EK100-pooled | random-pooler |
|---|---|---|---|---|
| **video** | binary AUC | **0.60** (~0.70 w/ sklearn LogReg) | 0.34 | 0.52 |
| **video** | 8-way acc (chance 0.125) | 0.39 | **0.49** | 0.43 |
| clip | binary AUC | 0.53 | 0.51 | 0.53 |
| clip | 8-way acc | 0.17 | 0.18 | 0.16 |

**Takeaways:** clip-level classification is **at chance** (whole-video labels + only 5 videos/class →
the probe overfits recording appearance, in-sample AUC ~1.0; most "anomaly" clips look normal). At the
**video level** frozen V-JEPA separates **8-way error type at ~4× chance**, and **EK100 warm-start helps**
(0.49 > 0.43 random) — the action-recognition pretraining transfers. **Binary anomaly detection is only
weakly above chance** and high-variance (5 normal videos, no within-video localization, unlike EgoPER's
localized labels → 0.745); plain mean-pool beats the action-tuned attentive pooler for it.
