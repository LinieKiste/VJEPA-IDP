# IDP SS26: Exploration von JEPA-Architekturen zur Anomalidetektion in Alltagsaktivitäten

Sammlung verschiedener Experimente mit V-JEPA.
Das Projekt besteht aus 2 Hauptphasen: Anomalidetektion mit V-JEPA und Schüttvolumenschätzung.

Zur Anomaliedetektion gehören:
`egoper_probe/`
`egoper_vqa/`
`exprt_probe/`
`video_qa/`

Code zur Schüttvolumenschätzung ist in `pouring/`.

`mlflow.db` enthält [mlflow](https://mlflow.org/)-kompatible Daten zu Training-runs und anderen experimenten.

The project started on video **anomaly detection** (EgoPER, eXprt tea dataset) and pivoted to
pouring in July 2026. That earlier work is kept in the repo as background.

## Where things are

| Path | Status | What it is |
|---|---|---|
| **`pouring/`** | **current** | The deliverable. `clip_split/` turns raw lab recordings into labelled pour clips; `pour_probe/` trains and evaluates the probes. See [`pouring/README.md`](pouring/README.md). |
| **`presentation_final/`** | **current** | The final talk (Slidev, TUM theme, German). See its README. |
| `mlflow_export/` | results | Every logged run (177) as CSV: `runs.csv`, `params.csv`, `metrics.csv`. |
| `mlruns/` | results | Artifacts logged to mlflow (figures, small files). |
| `presentation/` | background | Earlier, longer analysis deck (English, ~47 slides) with generated figures. It builds on its own from `presentation/data/`. |
| `egoper_probe/` | background | Frozen V-JEPA 2 probe for procedural-error detection on EgoPER (window ROC-AUC ≈ 0.75). |
| `exprt_probe/` | background | Anomaly and action probes on the eXprt tea-making dataset. |
| `egoper_vqa/` | background | Zero-shot video-QA baseline (Qwen2.5-VL) and a zero-shot EK100 action-head check. |
| `video_qa/` | background / **shared** | Replication of V-JEPA 2 Appendix E (encoder plus LLM). **`video_qa/model.py::build_encoder` is the encoder loader every other folder uses.** |
| `vjepa2/` | submodule | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) @ `204698b`: model code only. |
| `OCR_Scale_REader/` | submodule (**private**) | The supervisor's scale-OCR repo. `clip_split/run_ocr.py` imports its segment geometry; our own `lcd_ocr.py` replaced its OCR backends. You need access to `Paetriq/OCR_Scale_REader` to clone it. |
| `CLAUDE.md` | notes | The detailed lab notebook: every result, number, caveat and gotcha. Read this for the *why* behind any number. |

Not in git: `datasets/`, `checkpoints/`, `mlflow.db`, and all feature caches (see Setup).

## Setup

```bash
git clone --recurse-submodules <this-repo>     # OCR_Scale_REader needs repo access (SSH)
uv sync                                        # Python 3.10, torch cu132 (Blackwell GPU)
```

- **Checkpoints** → `checkpoints/vitl.pt` (V-JEPA 2 ViT-L, `target_encoder` key) and,
  optionally, `checkpoints/ek100-vitl-256.pt` (EK100 attentive probe, used as a warm start).
  Both are from the V-JEPA 2 release.
- **Data.** The curated pour clips (121 clips, CAM2 and CAM3, plus a per-clip GT weight curve,
  410 MB) are on the TUM NAS under `Datenverarbeitung/pouring_clips/`. Place them at
  `datasets/pouring_processed/clips/`. Raw recordings are in `Dateneingang/` (read-only).
- **Caches.** Frame and feature caches (tens of GB) go to `$POUR_CACHE`, which defaults to
  `~/.cache/pour_probe`. Put it on an SSD.
- **mlflow.** The run database is not in git: GitHub's secret scanner rejects the sqlite file
  (it flags a false-positive "Twilio SID"). Read results from `mlflow_export/*.csv`. If you
  have a copy of `mlflow.db`, run `pouring/pour_probe/mlflow_relocate.py` and then
  `mlflow ui --backend-store-uri sqlite:///$PWD/mlflow.db`.

Tested on a single RTX 5060 Ti 16 GB. Run all scripts from the repo root.

## Reproducing the headline numbers

```bash
P=.venv/bin/python; D=pouring/pour_probe
$P $D/clips_extract.py                         # mean-pool features, both cams (GPU)
$P $D/clips_eval_protocol.py --cam both        # ridge + controls, skill scores (CPU, ~1 min)
$P $D/clips_grid_cache.py --cam CAM2           # 288-px frame cache for the attentive probe
$P $D/clips_grid_cache.py --cam CAM3
# attentive flow probe, one run per trial fold (GPU, ~80 min each). The folds are
# defined in clips_cnn_baseline.FOLDS: A=8,13,21,24  B=7,9,11,12  C=5,15,16,25,26  D=17,18,20,22,27
$P $D/clips_train_attn.py --target flow --cam both --lag_s 0.7 \
    --val_trials 8,13,21,24 --fold foldA --minutes 60        # …repeat for B, C, D
$P $D/clips_headline_metrics.py                # 4-fold attentive metrics in g/s and g
```

`clips_headline_metrics.py` also expects the four volume checkpoints (`--target volume --cam
both --fold volA …`; see `run_overnight.sh`, study 2).

`pouring/pour_probe/README.md` has the full script map.
