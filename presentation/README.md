# Presentation

Slidev deck for the IDP. **Self-contained**: a clone of this repo is enough to build the
deck, regenerate every figure, and browse the experiment record. No dataset, no GPU, and
no `/mnt/storage` needed.

## Build

```bash
cd presentation
pnpm install          # or npm install
npx slidev            # dev server on :3030
npx slidev build      # static site into dist/
```

`slidev export` (PDF) needs playwright, which is not installed here. To screenshot, build
first, serve `dist/` through a static server with SPA fallback, and drive
`chromium --headless --screenshot`.

## Layout

`slides.md` is only headmatter, the title slide, and `src:` imports. All content lives in
`pages/`, one file per topic — edit those:

| file | topic |
|---|---|
| `pages/00-arc.md` | project arc |
| `pages/10-background.md` | V-JEPA 2, related work |
| `pages/20-data.md` | own-lab pouring dataset, OCR ground truth |
| `pages/30-method.md` | frozen features + attentive probe |
| `pages/40-results.md` | headline flow / volume numbers |
| `pages/50-ablations.md` | lag, views, ROI crop, head init |
| `pages/60-crossmodal.md` | Sound of Water |
| `pages/70-outlook.md` | next steps |

Images are referenced **relatively** (`../figs/…`, `../assets/…`). Do not move them into
`public/` and switch to absolute `/foo.png` paths — Vite's slide-import guard rejects
those inside imported page files (`resolves outside server.fs.allow`).

`colorSchema: light` is forced in the headmatter because the matplotlib figures have white
backgrounds and look broken on Slidev's default dark scheme.

## Figures

```bash
../.venv/bin/python make_figs.py     # writes all 19 figs/ + syncs assets/
```

- `figs/` — generated summary charts.
- `assets/` — QC images copied from the experiment dirs, plus the cropped 3-row ROI sheet.
  The originals are gitignored (`qc_*.png`), so the copies here are the tracked ones and
  `make_figs.py` leaves them alone when the sources are absent.

**Numbers in `make_figs.py` are transcribed by hand** from mlflow and from the analyses
never logged as runs (lag sweep, calibration, oracle-container). It is the single place to
fix a number — change it there, re-run, and the deck updates.

## `data/` — why it exists

The figures originally read from the 410 MB clip set and a prediction cache under
`~/.cache`, neither of which is in git. `data/` mirrors just the parts the figures touch
(4.4 MB), and `make_figs.py` prefers the originals when present and falls back to these:

| path | used by | substitutes for |
|---|---|---|
| `headline_preds.npz` | `fig_volume_curves` | `~/.cache/pour_probe/headline_preds.npz` |
| `clips_manifest.csv` | `fig_dataset` | `datasets/pouring_processed/clips/clips_manifest.csv` |
| `frames/CAM2_0001_t2s.png` | `fig_inputs` | frame from `clips/CAM2/0001.mp4` |
| `frames/CAM{2,3}_0001_mid.png` | `fig_views_example` | middle frames of clip 0001 |
| `clip_curves/*.csv` | — | the 121 per-clip ground-truth curves (`t_s,weight`), for new plots |

Verified: with `datasets/` and `~/.cache` both absent, all 19 figures regenerate
**byte-identical** to the committed ones.

## Experiment record

All 177 runs are tracked as CSV in `../mlflow_export/` — `runs.csv`, `metrics.csv` (final
value per run × metric), `params.csv`. Readable without mlflow installed, and diffable.
The experiments behind the deck are `pour_probe_clips_attn` (29, the attentive probes),
`pour_probe_baselines` (40), `pour_probe_clips` (7), and `pour_probe_sow_attn` (10, Sound
of Water). Logged artifacts are in `../mlruns/`.

```bash
# e.g. every held-out R² for the attentive probes
grep pour_probe_clips_attn ../mlflow_export/metrics.csv | grep best_val_r2
```

**The binary `mlflow.db` is deliberately NOT tracked.** GitHub's secret scanner rejects it:
a 32-hex mlflow `run_uuid` that happens to sit after the bytes `AC` in a sqlite page is
byte-identical to a Twilio Account SID. Two runs trip it (`attn_flow_CAM2_roi`,
`multiclass_seed1_fold4`) — a false positive, but it blocks the push. Copy the db across
by hand if you want the mlflow UI, then:

```bash
python ../pouring/pour_probe/mlflow_relocate.py    # fix absolute artifact paths
mlflow ui --backend-store-uri sqlite:///"$(cd .. && pwd)"/mlflow.db
python ../pouring/pour_probe/mlflow_export.py      # refresh the CSVs after new runs
```

`summary.md` is the prose companion — the full result tables and the reasoning behind
them. `../CLAUDE.md` is the project-wide summary, with the pre-2026-07-20 history in
`../CLAUDE.md.bak`.
