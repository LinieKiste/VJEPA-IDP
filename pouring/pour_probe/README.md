# pour_probe/ — probing frozen V-JEPA 2 for pouring flow and volume

**Recipe used throughout:** keep the video backbone frozen, cache its features over sliding
1.0 s windows (16 frames, stride 0.5 s), train a light head, and evaluate with
**GroupKFold by trial**: clips from one trial share scene and container, so whole trials are
held out. Two targets per window, both taken from the scale curve:
- `flow`: Δweight across the window, in g/s. This is the main target. It is sampled 0.7 s
  later (`--lag_s 0.7`) because the scale registers the water that long after it is visible.
- `volume`: cumulative poured mass at the window centre, in g.

All runs log to mlflow (`mlflow_util.setup()`, experiments `pour_probe_*`). Caches go to
`$POUR_CACHE` (default `~/.cache/pour_probe`, see `paths.py`). Run the scripts from the repo
root. Every script's docstring gives its purpose, usage and the result it produced.

## Script map

**Shared building blocks**
| | |
|---|---|
| `_encoder.py` | Frozen V-JEPA 2 ViT-L loader (wraps `video_qa/model.py::build_encoder`) |
| `_dino_encoder.py` | Frozen DINOv3 ViT-L/16, a drop-in alternative backbone (with a per-frame time embedding) |
| `head.py` | Attentive regression head (V-JEPA 2 `AttentiveClassifier`, optionally warm-started from the EK100 probe) |
| `paths.py`, `mlflow_util.py` | Cache root; pinned mlflow store |

**1. Caching** (GPU for features, CPU for frames)
| | |
|---|---|
| `clips_extract.py` | Mean-pooled V-JEPA features and targets per window → used by the ridge probes |
| `clips_grid_cache.py` | Decoded 288-px frames and the GT curve per clip → used by the attentive trainer (`--size 416` for the 384-px run) |
| `clips_roi_cache.py` | The same, cropped to the containers (zero-shot GroundingDINO); for the cross-view study |

**2. Training**
| | |
|---|---|
| `clips_train.py` | Ridge on mean-pooled features, 4-fold OOF, plus the simple baselines. The fast (CPU) probe. |
| `clips_train_attn.py` | **The main model**: attentive probe trained end-to-end over the frozen encoder with augmentation. Options: backbone (`--backbone dinov3`), partial fine-tuning (`--unfreeze_blocks`), resolution, and running on the SoW dataset (`--dataset sow`). |
| `run_overnight.sh`, `run_roi_lag.sh` | Batch drivers for the multi-fold studies |

**3. Baselines** (same windows, same folds)
| | |
|---|---|
| `clips_cnn_baseline.py` | ImageNet ResNet-50 per frame. It also defines `FOLDS` and `LAG_FLOW`, which the rest of the code imports. |
| `clips_cnn3d_baseline.py` | Kinetics video CNNs (r2plus1d / s3d): the fair video baseline |
| `clips_sow_baseline.py` | *Sound of Water* audio model (frozen wav2vec2) + ridge |
| `sow_model.py`, `sow_physics.py`, `sow_targets.py`, `sow_grid_cache.py` | SoW model transcription, its wavelength→volume physics, and targets and frames for running our probe on *their* data |
| `dino_pilot.py` | CPU ridge pilot: which temporal representation of DINOv3 features carries flow |

**4. Evaluation and analysis**
| | |
|---|---|
| `clips_eval_protocol.py` | The strict protocol: within-pour R², skill score against the best non-visual baseline, per-pour totals (CPU, ~1 min) |
| `clips_headline_metrics.py` | 4-fold attentive metrics in physical units (g/s, g, tolerance bands, Bland-Altman); caches the predictions |
| `clips_eval_crossview.py` | Zero-shot CAM2→CAM3 transfer, center crop vs ROI |
| `clips_lag_sweep.py` | Measures the 0.7 s water-transit lag |
| `clips_stability.py` (+`_figs`) | Dense 1-frame-stride inference: how stable the predictions are |
| `clips_attn_slope.py`, `clips_bias_diag.py`, `clips_calibrate.py` | Diagnose and try to calibrate the compression bias (short pours over-, long pours under-predicted) |
| `clips_oracle_container.py` | Go/no-go for a container-size model (answer: no) |

**5. Figures, external test set, demos**
| | |
|---|---|
| `clips_viz.py`, `clips_viz_attn.py`, `clips_viz_weight.py`, `clips_attn_map.py` | QC figures: predicted vs true curves, reconstructed weight, attention maps |
| `eval_external.py` (+`_figs`) | Runs the probe on 9 out-of-domain iPhone pours (`datasets/eval/`) |
| `eval_videos.py` → `eval_videos_render.py` | Side-by-side demo videos for the talk (on-screen text in German on purpose) |

**mlflow utilities:** `mlflow_export.py` (db → `mlflow_export/*.csv`), `mlflow_relocate.py`
(repoint artifact paths after moving the db).

`RELATED_WORK.md` has literature notes. The numbers and their caveats are all in the root
`CLAUDE.md`.

*History:* the folder began as a UWLPD smoke test with a mask-derived proxy target. Those
scripts were removed during cleanup and remain in git history.
