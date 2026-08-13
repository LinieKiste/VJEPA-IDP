# IDP — Interdisziplinäres Projekt (SS26)

## General instructions
Update this file whenever you learn something durable about the project. Keep it a
*coherent summary*, not a lab notebook: record decisions, results, and gotchas that stay
true — prune dead-end debugging narratives once they're resolved. Use the Misc section for
things without a home. The full pre-2026-07-20 blow-by-blow history is preserved in
`CLAUDE.md.bak`.

## Notion page
Canonical project page: https://app.notion.com/p/34830491519b800eb334c130d1478d73
(page id `34830491-519b-800e-b334-c130d1478d73`). Fetch with `mcp__notion__notion-fetch`
for meeting notes, tasks, datasets, the Gantt diagram (needed for submission).

## Project arc (one paragraph)
Systematic evaluation of **frozen V-JEPA 2 features** as a probe. The project explored
V-JEPA-latent *anomaly detection* (EgoPER procedural errors, then the eXprt tea dataset)
and, after the 05.07.2026 supervisor meeting, **pivoted to its LOCKED centerpiece:
pouring volume / flow-rate estimation from a frozen V-JEPA 2 backbone** — "the most
innovative field, could make a nice small publication." All current work lives under
`pouring/`. The anomaly experiments (`egoper_probe/`, `exprt_probe/`, `egoper_vqa/`,
`video_qa/`) are **background/history** — their findings are archived below.

Throughout, the recurring method is the same: **freeze V-JEPA 2 ViT-L, cache features,
train a light head, evaluate with group-held-out CV, track in mlflow.** SOTA is not the
bar — a working, honest result is.

## Hardware & infra essentials
- **GPU:** RTX 5060 Ti 16 GB (Blackwell sm_120). Needs torch ≥2.12/cu132 + transformers
  5.x. Old-torch stacks (LLaVA-NeXT pins 2.1.2) have **no Blackwell kernels → can't use the
  GPU at all** — this is why several off-the-shelf VLMs were ruled out.
- **One `.venv` per project** (user preference — no per-experiment conda envs).
- **Storage:** large datasets on the **Storage HDD** (1.8 TB NTFS, `/mnt/storage`, fstab
  `nofail`). `datasets/` holds symlinks into it. If symlinks look broken the drive isn't
  mounted: `sudo -A mount /mnt/storage` (always `sudo -A`; plain sudo hangs).
- **⚠ ntfs3 write hang:** `/mnt/storage` HANGS on sustained writes (uninterruptible
  D-state in `do_truncate`, corrupts the in-flight file). **Reads are fine.** Write all
  caches/datasets to the **SSD** (`~/.cache/...`); only read from the HDD.
- **TUM NAS** (`tum-nas`, SMB, needs TUM VPN): `Dateneingang/` = originals, kept
  READ-ONLY (DOS read-only attr set via `smbclient setmode +r`; rclone can't set SMB
  attrs). Processed outputs → `Datenverarbeitung/`. NEVER write into `Dateneingang`.
  Integrity manifest: `pouring/nas_manifest_eigene-Experimente.txt`.
- **mlflow:** single canonical store `sqlite:////home/casimir/UNI/SS_26/idp/mlflow.db`
  (absolute path — sqlite's default is CWD-relative and spawned a stray second store once).
  All scripts route through `mlflow_util.setup()`. Launch UI with
  `mlflow ui --backend-store-uri sqlite:////home/casimir/UNI/SS_26/idp/mlflow.db`.
  The stray `pouring/pour_probe/mlflow.db` still exists but is **strictly redundant**
  (verified 2026-08-03: same 40 baselines run names, canonical is a superset) — ignore it.
- **⚠ `mlflow.db` CANNOT BE PUSHED TO GITHUB.** Its secret scanner reads a 32-hex mlflow
  `run_uuid` that happens to follow the bytes `AC` in a sqlite page as a **Twilio Account
  SID** and rejects the push (trips on `attn_flow_CAM2_roi`, `multiclass_seed1_fold4`).
  False positive, but blocking. So the db is gitignored and the run record travels as CSV in
  **`mlflow_export/`** (`pour_probe/mlflow_export.py`, 177 runs / 240 KB); `mlruns/` (logged
  artifacts, 532 KB) IS tracked. Copy the db by hand if you need the UI elsewhere, then run
  **`pour_probe/mlflow_relocate.py`** — artifact paths are stored ABSOLUTE, so a clone at a
  different path shows metrics but opens no artifacts.
- **Workflow discipline:** pilot-first, build small QC utilities, get user sign-off at
  "gates" before any compute-heavy batch. Keep datasets pristine — extract writable working
  copies, never chmod/modify originals.
- **Env note:** `nvidia-smi` may error `NVML Driver/library version mismatch` after a
  background driver update — cosmetic, needs a reboot to resync; **torch/CUDA are
  unaffected**.

## The V-JEPA 2 encoder (shared building block)
Reuse `video_qa/model.py::build_encoder`: local `vjepa2/` pkg (pristine git submodule,
facebookresearch/vjepa2 @ 204698b) + `checkpoints/vitl.pt` (`target_encoder` key). Forward
`(B,C,T,H,W) → (B,N,1024)`, tubelet=2, patch=16; **bf16 autocast on the frozen encoder is
the key speed win**. Attentive head = `AttentiveClassifier`
(`vjepa2/src/models/attentive_pooler.py`, depth-4 pooler, 16 heads, 1024-d), optionally
warm-started from the EK100 action probe (`checkpoints/ek100-vitl-256.pt`).

---

# CENTERPIECE: Pouring flow / volume estimation (`pouring/`)

**Plan:** pretrain a probe on simulation volume data → fine-tune on own-lab data.
**Result so far:** frozen V-JEPA 2 reads instantaneous pouring **flow rate** from a wide
3rd-person video at **R² ≈ 0.85** (attentive probe, MAE ~7 g/s), well above every
non-V-JEPA baseline. This is the deliverable.

## Datasets
- **Own-lab pouring (THE fine-tune GT)** — recorded 2026-07-13, 3 GoPro views (CAM1 scale
  cam @119.88 fps; CAM2/CAM3 @29.97 fps), 18 trials, 12 source→target container combos
  (kettle/teapot/bottle → blue_mug/white_mug/glass/ikea_glass). **GT = OCR of a scale
  display in frame.** Finished, curated dataset: **121 pour clips, 8–362 g (median 140 g),
  2.6–8.7 s** at `datasets/pouring_processed/clips/{CAM2,CAM3,csv}/NNNN.*` +
  `clips_manifest.csv` (410 MB), uploaded to `tum-nas:.../Datenverarbeitung/pouring_clips/`.
  Final per-clip GT CSV = 2 cols `t_s,weight` (poured mass since clip start,
  baseline-subtracted, monotone by construction, rise == annotated final mass).
- **UWLPD** (`datasets/UWLPD/`, Schenck & Fox real-robot set) — **36 scene directories locally**
  (verified 2026-07-31: 4 fill levels × 3 motions × 3 speeds; an earlier note saying "180
  sequences" was wrong for what is on disk). Each is a REAL 640×480 pouring video (frames
  `data*.jpg`) plus a per-frame mask render (`data*.png`) under `render_v3/`.
  **No mL labels** (fill % fixed at 30/60/90); volume/flow must be derived
  from the per-frame binary **liquid mask** (liquid = BLUE `(0,0,255)`; `convert("L")`
  erases it — count `max(RGB)>127`). Used only for a **mask-derived proxy** smoke test. The
  clean mL trace lives in UWLPD's separate *Simulated* set (`bowl_volume.csv`) — requested
  from the author, not in hand.
- **SimLiquid** (`pouring/SimLiquid/`, BlenderProc renderer) — set up + validated. Renders
  photoreal liquid-in-cup images with **per-cup volume labels in mL** → the clean-volume
  sim-pretrain source. `blenderproc` in the project `.venv`; it auto-downloaded a **portable
  Blender 4.2.1** to `~/blender/` (pacman Blender 5.1.2 CANNOT drive BlenderProc — needs the
  portable 4.2/py3.11 bundled-python layout; don't retry). Assets: 977 Haven HDRIs +
  textures under `liquid_render/assets/hdri`. One code fix inlined a missing 3D-Print-Toolbox
  helper. Full 10k-image render (`./render.sh`) not yet run.
- **Sound of Water** (`pouring/pour_probe/third_party/SoundOfWater`, reference only, never on
  `sys.path`) — audio pouring dataset + model, used as a cross-modal baseline (see below).
  **Composition verified from `splits/*.csv` 2026-07-31:** 1010 videos / 48 containers across
  train+test_I/II/III; **1000 clean / 45 containers**; our S3 frame cache holds **780**. Shapes
  semiconical 487 / cylindrical 406 / bottleneck 107; materials ceramic, glass, paperboard,
  plastic, steel; liquid water_normal 695 / water_hot 305. `flow_rate_appx`: **constant 988,
  non-constant 1, TODO 21**.

## Own-lab data pipeline (`pouring/clip_split/`)
Trials → per-pour clips, a gated workflow (`trials.csv` → OCR → detect pours → annotate →
cut clips). The parts that stay true:
- **Scale physics (critical):** cup tared ≈0 g → pour ramps up → plateau = poured-mass GT →
  cup removed makes the scale go NEGATIVE but the OCR can't read the minus sign → bogus
  positive spike. A real pour rises out of a stable ~0 baseline; rises from plateau/chaos =
  removal artifacts, ignored.
- **OCR = own `lcd_ocr.py`** (`--backend lcd`, default). Supervisor's tesseract/segment/
  template backends FAIL on the wide-shot ~170×80 px 7-seg display. Ours: static camera →
  per-pixel background model (p90 over gain-normalized frames) → read the ratio image
  (bg−frame)/bg, per-cell correlation vs synthetic 7-seg templates over a ±8 px window
  (absorbs display drift). ~2 ms/frame, runs at full native rate. **Final batch: 95.2%
  valid overall, 16/19 trials ≥99.2%.**
- **Trace cleaning = monotonicity prior** (`--filter mono`): while pouring, true weight
  never decreases → two-scale rolling-median bounds reject both dropout (low) and
  digit-blend (high) misreads, then **isotonic regression** per pour interval makes each
  exported curve monotone by construction.
- **Pour detection = plateau-chain** (`--detector chain`): stable plateaus → pour = maximal
  ascending chain from a settled start to a settled end; weight = end − start. "CUP BARRIER"
  rule ends a pour before the removal step (any step landing at the trial's modal
  cup-constant level).
- **Interactive annotator** (`annotate.py` + `annotate_ui.html`, stdlib-only local web app):
  CAM1 video + zoomable trace, draggable event spans, per-clip weight override (rescales the
  GT curve monotone-preserving), per-sample OCR corrections (stored as time-ranges in
  `ocr_overrides.json`, filter re-runs live). User annotated all 121 events (weights
  human-verified). `cut_clips.py` cuts only completed, non-excluded events.

## The probe & results (`pouring/pour_probe/`)
`clips_extract.py`/`clips_grid_cache.py` cache frozen ViT-L features over sliding 1.0 s
windows (16 frames, short-side-256 + center-crop); targets from the per-clip curve
(`flow` = Δweight/window in g/s, `volume` = cumulative mass). Caches on the SSD
(`~/.cache/pour_probe/...`). CV = **GroupKFold by trial** (clips of a trial share
scene/container). mlflow experiments `pour_probe*`.

**Headline results (own-lab clips, held-out CV):**

| target | method | R² | notes |
|---|---|---|---|
| **flow** | **V-JEPA 2 attentive probe** | **0.81 ± 0.04** (4-fold) | MAE ~7 g/s; best single split 0.85–0.90 |
| flow | V-JEPA 2 ridge mean-pool | 0.69 | linear quick-check |
| flow | Kinetics video CNN (r2plus1d / s3d) | ~0.53 | the fair CNN baseline; gap to V-JEPA = **+0.28** |
| flow | Sound-of-Water audio model | 0.65 | cross-modal reference (below V-JEPA) |
| flow | temporal-profile prior (norm-time poly4) | 0.62 | needs pour boundaries; only predicts mean profile |
| flow | motion-energy | 0.18 | |
| flow | ResNet-50 per-frame (strawman) | ~0.00 | 2D CNN structurally can't see motion |
| flow | shuffle null | −0.32 | |
| **volume** | V-JEPA 2 attentive | **0.576 ± 0.087** (4-fold) | harder target; the old 0.67 was ONE split |
| volume | raw-time-linear (clock) | **0.78** | volume is clock-dominated |

**Volume is clock-dominated, but the two signals ARE complementary (measured 2026-07-26,
matched 4-fold-by-trial ridge, both cams, mean-pool cache):**
| target | clock only | V-JEPA only | V-JEPA + clock |
|---|---|---|---|
| volume | **0.778** | 0.398 | **0.818** |
| flow | −0.012 | 0.577 | 0.586 |
So V-JEPA adds a real **+0.04** on top of the clock for volume (it just can't supply the clock
term itself), and the clock adds **nothing** to flow. Why the clock is so strong on volume:
(a) **72% of volume variance is WITHIN a pour** ("how far in are we"), which elapsed time answers
directly; (b) the baseline is literally one global slope, `volume ≈ −37 + 50·t`, i.e. it assumes
every pour runs at the mean 50 g/s; (c) clip **duration alone correlates 0.81 with total poured
mass** (R² 0.65), so the clock also picks up totals. For flow the split is reversed: **83% of
variance is within-pour** and a straight line can't draw a bell curve.
- **The raw-time-linear FLOW baseline is LAG-DEPENDENT: 0.001 at lag 0, 0.237 at lag 0.7**
  (identical for CAM2/CAM3/both — it is a pure function of time, so the camera cannot matter).
  **The original 0.221 was essentially RIGHT; a 2026-07-26 note "correcting" it to ~0.00 was
  itself wrong** — it recomputed the baseline at lag 0 and compared it against lagged probes,
  the same un-lagged-baseline error as the `baselines_on_split` bug below. **Use 0.237 in any
  lag-0.7 flow table.** Why the lag creates clock-predictability at all: shifting the target
  0.7 s earlier in window-time slides the flow bell toward the clip start, so more of the clip
  sits in the declining phase and a negative-slope line fits it partially.
- **Per-clip TOTAL mass (6-fold OOF, CAM2):** clock R² 0.543 / MAE 51 g, V-JEPA ridge R² −0.18 /
  MAE 82 g. Both poor; the integrated-flow route (median 14 g) is far better than either.

**The story:** frozen V-JEPA 2 reads **instantaneous flow** from appearance and generalizes
across held-out trials/containers — decisively beating a per-frame CNN (~0), beating a
credible **video** CNN by +0.28, edging a boundary-cheating temporal prior, and above the
audio model. Predicted flow curves track the GT bell shape (`qc_attn_flow_*.png`).
**Volume (absolute fill level) is intrinsically clock-dominated** — the clock alone gets
0.78, V-JEPA 0.36 (ridge) / 0.67 (attentive); absolute fill from a wide 3rd-person shot is
near the aleatoric floor.

**Strict evaluation protocol (`clips_eval_protocol.py`, added 2026-07-27, CPU-only on the
mean-pool cache, ~1 min).** Answers the fair objection *"R² against the global mean just
rewards anything that rises with time, so of course a line wins"*. Borrows what we already
demanded of the SoW runs (`sow_baselines_on_split`) but never applied to our own data: score
the trivial controls under the SAME within-pour metric as the probe, and report a **skill
score** `1 − SSE_model/SSE_ref` against the best NON-VISUAL method. Both cams, 4-fold OOF:

| target | method | R² global | R² within-pour | skill vs best non-visual |
|---|---|---|---|---|
| flow | raw time (causal) | 0.237 | 0.439 | −0.85 |
| flow | time profile (**oracle** duration) | 0.588 | 0.698 | 0 (ref) |
| flow | **V-JEPA 2** | **0.718** | **0.771** | **+0.32** |
| volume | raw time (causal) | 0.777 | 0.864 | 0 (ref) |
| volume | V-JEPA 2 | 0.371 | 0.622 | −1.82 |
| volume | **V-JEPA 2 + clock** | **0.818** | **0.886** | **+0.19** |

- **Removing the between-pour offset changes NEITHER verdict** → the ranking is not a metric
  artefact. Flow stays a vision problem, volume stays a clock problem.
- **Report the skill score, not the raw R² gap.** "+0.04 R² on volume" is really **+0.19 skill**
  (19% of the clock's residual error removed); flow's "0.72 vs 0.59" is **+0.32 skill** over a
  baseline handed the oracle duration.
- `time_prof` needs each clip's DURATION to normalise time — an **oracle** at test time. Flagged
  as such everywhere now; `raw_time` is the causal one.
- Within-pour share of total variance: **flow 0.84, volume 0.72**.

**Overnight batch 2026-07-28 (`run_overnight.sh`, 11 h, both studies converged).**

**(1) 4-fold attentive VOLUME, both cams, lag 0 — CORRECTS a headline number.** Folds A/B/C/D
give 0.687 / 0.602 / 0.496 / 0.518 = **0.576 ± 0.087**. The previously reported **0.667 was a
single split, and it was the LUCKIEST of the four** (fold A). Same folds: ridge_meanpool
0.370 ± 0.080, time_prof 0.552 ± 0.050.
- **The attentive probe barely beats the normalized-time profile on volume: +0.024 mean, and it
  LOSES on fold D (−0.051).** It beats the mean-pool ridge by +0.206. So the attentive head buys
  a lot over a linear probe but almost nothing over a clock-shaped prior — the strongest
  statement yet of "volume is clock-dominated".
- Closes the protocol-mismatch caveat: vs audio 0.802 and clock 0.778, V-JEPA volume 0.576 is
  clearly below BOTH. "Audio beats video on volume" now holds by a wider margin, not a narrower one.

**(2) Crop/ROI cross-view, folds A+B+C at lag 0.7 — the claim SURVIVES and sharpens.**
| fold | ctr-in | roi-in | ctr→CAM3 | roi→CAM3 | CAM3 ridge |
|---|---|---|---|---|---|
| A (8,13,21,24) | 0.885 | 0.781 | 0.507 | 0.735 | 0.764 |
| B (7,9,11,12) | 0.863 | 0.863 | **0.043** | 0.622 | 0.714 |
| C (5,15,16,25,26) | 0.906 | 0.875 | 0.422 | 0.635 | 0.638 |
| **mean ± sd** | 0.885 ±0.022 | 0.840 ±0.051 | **0.324 ±0.247** | **0.664 ±0.062** | 0.705 ±0.063 |
- **Crop gain cross-view = +0.340 ± 0.207, positive on all three folds** (fold A's +0.228 was the
  LOW end, so the old single split understated it).
- **The real finding is VARIANCE, not mean.** Un-cropped transfer swings 0.043–0.507 (sd 0.247);
  cropped is 0.622–0.735 (sd 0.062), 4× tighter. Cropping buys *predictability* on an unseen view.
  Fold B is a near-total collapse of the un-cropped arm.
- **Crop cost within-view is only −0.045 ± 0.053** (was −0.104 on fold A alone); exactly 0.000 on fold B.
- **Yesterday's reversal is CONFIRMED on all 3 folds independently:** roi→CAM3 still loses to a
  linear ridge trained on the target view (−0.029, −0.092, −0.003; mean −0.041). Cropping remains
  the tool ONLY for an unseen angle with no labels at all.

**HEADLINE METRICS IN PHYSICAL UNITS (`clips_headline_metrics.py`, 2026-07-31, attentive
probe, 4-fold OOF, both cams, ~8 min GPU inference).** R² is a poor instrument here: its
denominator is whatever spread the test set happens to have, and it hides bias entirely.
Per-window predictions are cached to `~/.cache/pour_probe/headline_preds.npz` so the deck's
curve figure and the metric table come from identical numbers.

| quantity | method | MAE | medAE | P90 | bias | nMAE |
|---|---|---|---|---|---|---|
| flow | **V-JEPA attentive** | **8.46 g/s** | 4.02 | 22.0 | −0.49 | **25%** |
| flow | raw-time clock | 24.2 g/s | 21.6 | 44.7 | +0.03 | 72% |
| volume | V-JEPA + clock (ridge) | 28.5 g | 21.2 | 63.3 | −2.5 | 30% |
| volume | raw-time clock | 30.3 g | 17.5 | 71.2 | +0.25 | 32% |
| volume | V-JEPA attentive | 39.9 g | 24.6 | 101.1 | −1.1 | 42% |

**Per-clip TOTAL MASS (integrate the predicted flow curve, n=121, true mean 141 g):**
| method | MAE | medAE | ≤10 g | ≤25 g | ≤50 g | 95% limits of agreement |
|---|---|---|---|---|---|---|
| **V-JEPA attentive** | **23.2 g** | 14.5 | 32% | **74%** | **88%** | −69 to +63 g |
| V-JEPA ridge | 32.6 g | 25.8 | 25% | 49% | 79% | −90 to +82 g |
| predict-mean | 48.0 g | 42.1 | 12% | 29% | 60% | −127 to +105 g |
| raw-time clock | 63.2 g | 60.3 | 7% | 17% | 40% | −159 to +139 g |

- Confirms the older "median 14 g / mean 22 g" figure (now 14.5 / 23.2 on all 121 clips, 4-fold).
- **On per-clip total mass the clock is WORSE than predict-mean (63.2 vs 48.0 g)** despite
  topping the volume-R² table at 0.777 — the sharpest demonstration that volume-R² measures
  "early or late in the pour", not "how much was poured".
- **The volume skill score still oversold vision: "+0.19 skill" is 30.3 → 28.5 g MAE, ~2 g.**
- Report **tolerance bands** ("3 of 4 pours within 25 g") as the headline, MAE+nMAE next, and
  **Bland-Altman limits** (−69 to +63 g) as the honest worst case. Keep R² only for
  comparability with other papers, always paired with the skill score.

**SINGLE FINAL READING — one prediction per pour (2026-08-06, `final_reading` analysis,
CPU, 121 clips, 4-fold OOF).** Asks the deployment question directly: take each model's LAST
window prediction, compare against the scale's final reading. Distinct from every volume
number above, which averages over all windows and so mixes in "how far into the pour are we".

| single final reading | modality | MAE | medAE | bias | R² | ≤50 g |
|---|---|---|---|---|---|---|
| SoW wav2vec + V-JEPA + clock | A+V | **33.5 g** | 28.7 | +1.0 | 0.781 | 79% |
| **SoW wav2vec + clock** | audio | **34.0 g** | 28.9 | +4.8 | 0.772 | 78% |
| SoW wav2vec (no clock) | audio | 37.9 g | 32.2 | +3.4 | 0.698 | 71% |
| V-JEPA + clock (ridge) | video | 41.3 g | 35.1 | +1.4 | 0.668 | 64% |
| raw-time clock | clock | 52.2 g | 53.3 | +27.7 | 0.525 | 46% |
| V-JEPA attentive volume | video | 66.5 g | 60.3 | −0.7 | 0.134 | 47% |
| *predict-mean final mass* | *trivial* | *76.7 g* | *76.2* | +0.2 | −0.016 | 31% |
| time_prof (ORACLE duration) | clock | 77.1 g | 72.4 | +8.3 | −0.023 | 34% |
| V-JEPA ridge (vision only) | video | 79.6 g | 78.0 | −5.4 | −0.094 | 32% |

- **Audio beats video on the final reading, and the fusion adds ~nothing** (34.0 → 33.5 g).
  Strongest form yet of "audio wins on volume": SoW+clock beats V-JEPA+clock by 7 g.
- **Vision ALONE is worse than guessing the mean** (79.6 vs 76.7 g). Frozen V-JEPA carries
  essentially no readable absolute fill level at the end of a pour; so does the oracle-duration
  time profile (77.1). Yet V-JEPA+clock (41.3) beats clock alone (52.2) — **synergy, not
  addition**: vision cancels the clock's +27.7 g systematic overshoot (down to +1.4).
- **Reading the last window of a cumulative-volume model is the WRONG tool** (66.5 g, LoA
  ±162 g). Integrating the FLOW probe scores 26.2 g against the same truth — the trajectory
  carries far more than the final frame.
- **⚠ The 23.2 g headline total-mass figure is scored against the wrong truth.**
  `totals_from_flow` (`clips_eval_protocol.py:112`) trapezoids over window CENTRES, dropping the
  first/last 0.5 s, so its "GT" sits **12.1 g below** the scale's actual final reading. Against
  the true final mass the attentive probe is **26.2 g MAE / 17.8 medAE, bias −15.1 g**
  (63% ≤25 g, not 74%). Extending the integral to the clip bounds recentres bias to +6.7 g at
  the same MAE and tightens the low LoA (−65 vs −85 g) — the better convention. **Not yet fixed
  in the script or the deck.**
- `predict-mean` in `clips_eval_protocol.py` averages over WINDOWS, so as a final-reading control
  it is unfair (84.3 g, −57 g bias). The honest floor is the mean training-fold FINAL mass.

**EXTERNAL OUT-OF-DOMAIN SET (`datasets/eval/`, 9 iPhone 4K pours, 2026-08-06).**
`eval_external.py` (inference, ~5 min GPU) + `eval_external_figs.py` (4 figures →
`datasets/eval/figs/`). 3 scenes: kettle→stockpot (0866/0868/0872), jug→cup-on-tray
(0875/0877), and a very wide outdoor table (0879–0882) where the vessel is ~2% of frame.
GT = one final volume per video, no trace. Ensemble of the 4 lag-0.7 fold checkpoints;
inter-fold spread doubles as an uncertainty band.
- **Totals do NOT transfer: MAE 153 g / medAE 81 g, bias +74 g, r 0.62** (vs 23–26 g in-domain).
- **But the SHAPE does.** The predicted flow curves localise *when* pouring happens correctly in
  every video — single bells for the outdoor pours, a 12 s plateau for the long kettle pours,
  a 3-hump curve for the multi-pour 0875 — in scenes and at scales the probe never saw.
- **Root cause of the indoor over-prediction: the probe has no concept of "not pouring".** Every
  training clip is tightly cut to a pour, so it never saw a non-pour frame and has no zero. Its
  resting predicted flow is **20–35 g/s indoors** vs 3–4 g/s outdoors, and the integral
  accumulates that phantom flow over the whole untrimmed video. Subtracting each video's 10th
  percentile lifts **r 0.62 → 0.81** (MAE 153 → 130 g, bias flips to −91 g): the floor is an
  offset problem, not a shape problem. **Fix = train with non-pour segments, or gate on a
  pour/no-pour detector; do NOT feed untrimmed video to the current probe.**
- Outdoor set under-predicts (bias −32 g) — at ~2% of frame the stream is near-invisible.
- **`datasets/eval/gt.csv` is a bare column of 9 numbers, no filename column**; row order ==
  filename sort order (capture timestamps agree). **CONFIRMED CORRECT by the user 2026-08-06.**
- **IMG_0868 is a deliberate TRICKLE test and is the single most damning result in the set.**
  It is a kettle→stockpot pour visually near-identical to IMG_0866 (895 g) but poured slowly:
  **GT 128 g, predicted 721 g — 5.6× over.** The probe cannot tell a trickle from a gush in an
  unseen scene; it anchors on the *appearance* of "kettle tilted over pot" and reports the
  flow rate that pose implies in the training set. Directly reinforces the phantom-floor
  finding: on 0868 the predicted flow sits on a flat ~45 g/s plateau for 12 s with no
  amplitude modulation at all. **Rate discrimination, not just the zero point, fails
  out-of-domain.**
- **Naming: the set was `datasets/eval/Evan/` (an autocorrect of "eval"); renamed to
  `datasets/eval/videos/` 2026-08-08.** No `Evan` string survives in code or notes.

**ZERO-POUR STRESS TEST — YouTube street walk (2026-08-09).** User asked: do the methods
predict flow on *unrelated* videos? Downloaded the first 15 s of
`youtube.com/watch?v=1aedKShR1rA` ("Manhattan Evening Walk", 640×360) with
`uvx yt-dlp ... --downloader ffmpeg --external-downloader-args "ffmpeg_i:-ss 00:00:00 -t 00:00:15"`
→ `datasets/eval/videos/manhattan_15s.mp4`, wired into the demo pipeline as a new `"yt"`
source kind (`eval_videos.py` `load_yt`; renderer skips the GT reference line when
`gt_total` is NaN → row shows "k.A."). Demo: `datasets/eval/demo_videos/yt_manhattan_15s.mp4`.
- **Both probes hallucinate flow on a scene with ZERO pouring:** V-JEPA attentive
  mean **37 g/s** (peak 48.6), integral **560 g** in 15 s; DINOv3 mean **66 g/s**
  (peak 128.4), integral **994 g**. The V-JEPA curve is a near-flat 30–48 g/s plateau —
  the same phantom floor as the external pours, but with no kettle/pour pose in frame
  at all. The probe's resting prediction is a constant offset, not a response to any
  visual pouring cue. This is the strongest form yet of "the probe has no concept of
  not pouring": it was never shown a non-pour frame, so 0 g/s is off its manifold.
  GT row shows k.A. (no scale was involved).
- The `"yt"` kind stays in `SOURCES` like the external videos; predictions land in the
  usual `eval_videos_preds.npz` under `yt_manhattan_15s__*`.

**SIDE-BY-SIDE DEMO VIDEOS (`eval_videos.py` + `eval_videos_render.py`, 2026-08-08).**
Renders one mp4 per pour: the video on the right (with the 256 px centre crop the encoders
see outlined), and on the left a live readout of instantaneous flow + cumulative poured mass
for GT and three probes, over synchronised time plots with a sweeping cursor. Built for the
talk — it makes both the successes and the out-of-domain failures legible in one glance.
Outputs in `datasets/eval/demo_videos/`; predictions cached to
`~/.cache/pour_probe/eval_videos_preds.npz` (`--infer` = GPU ~4 min, `--render` = CPU ~3 min).
- **All three probes are the SAME fold-A split** (val trials 8/13/21/24) so the comparison is
  matched and every own-lab source shown is genuinely held out: V-JEPA 2 attentive, DINOv3
  attentive (+temporal embedding), and Sound-of-Water wav2vec→ridge (refit here on the
  non-fold-A windows, alpha chosen by inner CV over folds B/C/D; inner R² +0.552).
- **Totals use the CLIP-BOUND integral, not the window-centre one** — `cumulative()` holds
  the flow constant out to [0, dur], fixing the ~12 g low bias of `totals_from_flow`. The
  printed table and the plotted curves share that one function, so they cannot drift apart.
- GT flow is drawn exactly as `build_windows` defines the training target (1.0 s difference
  of the weight curve sampled at +0.7 s), so the black curve is the thing the probe was
  actually trained to predict. External pours have no trace → "k.A." instantaneous, a dotted
  final-mass line on the cumulative panel.
- **ALL ON-SCREEN TEXT IS GERMAN — keep it that way** (the talk is in German). Code, comments
  and docstrings stay English. That includes the `SOURCES` blurbs (they become the video
  titles) and the numbers: `eval_videos_render.de()` swaps in a German decimal comma, so
  readouts show `58,1` not `58.1`. Every model row carries an explicit label and the
  reference row is named **"Ground Truth (Waage)"** / **"Ground Truth: SoW-Physik,
  GEMESSENER Radius"** — never leave the GT row unlabelled.
- **Locked German terms (user's wording 2026-08-09 — do NOT paraphrase):** flow rate =
  **Flussrate**, cumulative poured mass = **Masse insgesamt**, poured (g) = **Schüttvolumen
  (g)**, trickle = **langsames Schütten**, a pour = **Schüttvorgang**. `Schüttvolumen` labels
  an axis in GRAMS (water: 1 g = 1 mL) — the user's choice, and it matches the deck.
- **The audio row is switched OFF for every source** (2026-08-09). The wiring stays in
  `eval_videos.py` — put `"sow"` back in a source's models tuple to re-enable — but it is not
  SoW's method and mixing it in invited exactly that misreading. `fit_sow_ridge()` is skipped
  entirely when no source asks for it.

**THREE DEMO SLIDES AT THE END OF `presentation_final/slides.md` (2026-08-09).** Videos copied
into `public/` as `demo_eigen_schnell.mp4` (clip 0016, in-domain, works), `demo_extern_langsam.mp4`
(IMG_0868, the trickle failure) and `demo_sow_vergleich.mp4` (the SoW comparison) — the arc is
*works → fails out of domain → fair fight against audio*.
- **Size the `<video>` by HEIGHT, not width.** `canvasWidth: 720` means 1 CSS px = 1 pt, so
  `width: 100%` makes a 16:9 clip ~354 pt tall and it runs off the slide (the plots and the
  caption vanish under the footer). `height: 234px; width: auto` fits with room for a
  one-line caption. The same one-line-HTML rule as for images applies.
- **Re-encode the deck copies** (`-crf 26 -preset slow -an`): 8.3 MB → 1.0 MB. The full-size
  file also made headless Chromium fail to screenshot that slide at all, which is how the
  overflow was nearly missed. Originals in `datasets/eval/demo_videos/` are untouched.

| source | GT g | V-JEPA | DINOv3 | SoW |
|---|---|---|---|---|
| 0047 own-lab, teapot→mug, 31 g / 3.3 s (9.3 g/s) | 31 | 87 | 76 | 39 |
| 0016 own-lab, kettle→glass, 335 g / 5.7 s (59 g/s) | 335 | 244 | 350 | 229 |
| 0045 own-lab CAM3, teapot→mug, 244 g / 7.0 s | 244 | 236 | 227 | 251 |
| IMG_0866 external, kettle→stockpot | 895 | 719 | 763 | 255 |
| **IMG_0868 external, TRICKLE** | **128** | **855** | **604** | **668** |

- **The 0866/0868 pair is the whole out-of-domain story in two videos.** Near-identical
  scenes; GT 895 vs 128 g. V-JEPA predicts **719 vs 855 g** — i.e. it predicts *more* for the
  pour that delivered **7× less**. The rate ordering is not merely compressed, it is
  **inverted**. DINOv3 (763 vs 604) and SoW (255 vs 668) get the ordering right but the
  magnitude hopelessly wrong. Restates the 5.6× figure above under the fold-A / clip-bound
  convention (that 721 g was the 4-fold ensemble with the window-centre integral).
- **In-domain the same probes are fine** (0016: 244/350/229 vs 335; 0045: 236/227/251 vs 244)
  and track the GT bell shape closely — so the demo shows competence and failure side by side
  rather than only one of them.
- **The slow in-domain clip 0047 (9.3 g/s) already over-predicts ~2.5×** (87 vs 31 g) even
  though its trial is only held out, not out of domain. So the trickle failure is not purely
  a domain-shift effect: **low flow rates are the probe's weak regime everywhere**, and the
  out-of-domain trickle is the extreme of a bias that is visible in-distribution. This is a
  new finding — the aggregate metrics never separated it out.
- **The audio row was REMOVED from the two external videos (2026-08-08) and must stay out.**
  It is not "Sound of Water" — it is `sow_feats_for` taking ONLY the 768-d wav2vec2 features
  (`_, feats = sm.predict_axial(...)`; the decoded λ is discarded and `sow_physics` is never
  imported) and OUR ridge supplying the g/s. On the external pours that ridge faces three
  stacked shifts: iPhone mic + kitchen vs GoPro + lab, stockpot vs mug/glass, and — the one
  specific to audio — **durations of 14.2 / 17.0 s against a ridge fit on 2.6–8.7 s clips.**
  SoW injects ABSOLUTE time into its features (`TimeEncodingDiscreteSinusoidal`, 49 fps), so
  past ~8.7 s the ridge is extrapolating, not transferring. Its numbers there said nothing
  about audio pouring estimation. The row is still shown on the own-lab clips, where the
  comparison is matched, and is labelled **"SoW wav2vec2 + our ridge"** everywhere.

**WHY SoW's PHYSICS ROUTE IS CLOSED ON OUR DATA — and it is NOT missing measurements
(settled 2026-08-08).** The natural objection is "SoW estimates the container radius from
audio, so run their real pipeline". They do, and it works: MAE 1.39 cm (Test I) / 1.88 cm
(Test II, unseen containers). But the estimator is **not** the `radial_head` (the paper:
"we only use axial resonance in training"). It is `shared/utils/physics.py:320`:

    def estimate_cylinder_radius(wavelengths, timestamps=None, beta=0.62):
        radius_pred = ((1. / beta) * (wavelengths[-1] / 4.)).item()

i.e. Eq. (6) `R = λ(T)/4β` from the **last** axial wavelength, with β a fixed global 0.62
(per SHAPE — 1.28 semi-conical — **not** per container; an earlier note in `sow_physics.py`
said per-container and was wrong).
- **It carries the boundary condition l(T) = 0: the vessel is FULL when the audio ends.**
  That is how Eq. (5)/(6) is derived, and it is the same fill-to-completion assumption behind
  the `t/T` degeneracy already documented above — biting in a second, independent place.
- **Our pours never fill the vessel**, so λ(T) encodes the leftover air column. Measured over
  our 121 clips: the SAME blue_mug across 48 pours gives **R = 1.7–11.4 cm**; within-container
  variance is **28×** the between-container variance; and **corr(R̂, poured mass) = −0.48**.
  It is reading fill level, not geometry. On the eval videos, the same stockpot yields an
  implied diameter of **34.5 cm (IMG_0866) vs 12.1 cm (IMG_0868)** — 2.9× apart.
- **So the frozen-features + our-ridge route was the only way to put SoW on our data at all.**
  Say it that way: the physics route is blocked by a design assumption of their task, not by
  anything missing on our side, and not by a flaw in their paper.

**SoW COMPARISON VIDEO — on THEIR data, where it IS a fair fight
(`eval_videos.py --sow`, 2026-08-08).** `datasets/eval/demo_videos/sow_VID_20240417_000535_2.2_8.0.mp4`.
Their pours fill the vessel, so Eq. (6) runs as published and all three estimates are
commensurable. Video `VID_20240417_000535_2.2_8.0`, container_7 (transparent cylinder,
**held out** from the S1 probe — the split at `--split_seed 0` holds out container_3 +
container_7, 81/212 videos). Panels: poured volume in mL, plus **λ(t)**, which shows where
the radius estimate comes from.

| final volume | mL |
|---|---|
| SoW physics, MEASURED radius (2.88 cm) — the target our probe trained against | 198 |
| SoW physics, radius ESTIMATED from λ(T) (2.49 cm, 0.87×) | 149 |
| V-JEPA 2 attentive (ours) | 247 |

- **Their radius estimator is accurate here** (2.49 vs 2.88 cm, 13% low) — exactly the regime
  the paper reports. Volume scales with R², so that 13% becomes a **−25% volume** offset; the
  green curve is the right SHAPE, scaled down.
- Our probe over-shoots by +25%; theirs under-shoots by −25%. Neither is clearly better on
  this one video, which is the honest read — and note their "GT" is itself a physics decode,
  not a scale, so this compares a video probe against an audio-physics estimate.
- The λ(t) curve is a clean monotone 37 → 6 cm. That is the signal our own recordings cannot
  supply an endpoint for.

**BACKBONE / CAPACITY / RESOLUTION SWEEP (2026-08-07, all fold A, flow, both cams,
lag 0.7, 31 epochs / 3193 steps each unless noted, all logged to `pour_probe_clips_attn`).**
**Compare runs on `ckpt_val_r2`** (the selected checkpoint, = the printed "attn ckpt
(combined)"). Mixing it with `val_r2` (final epoch) silently inflates deltas — that mistake
was made once here.

| fold A, flow, lag 0.7, both cams | ckpt_val_r2 | CAM2 | CAM3 | MAE |
|---|---|---|---|---|
| **V-JEPA 2, img 384** | **0.870** | 0.901 | **0.840** | 8.66 |
| V-JEPA 2, img 256 (the baseline) | 0.849 | 0.898 | 0.801 | 8.08 |
| V-JEPA 2, img 256, unfreeze last 4 blocks | 0.840 | 0.872 | 0.808 | 8.67 |
| V-JEPA 2, unfreeze 4 + **ROI crop** | 0.795 | 0.819 | 0.771 | 9.70 |
| **DINOv3 + temporal embedding** | 0.792 | 0.821 | 0.762 | 9.87 |
| V-JEPA 2 ridge mean-pool (LINEAR) | 0.766 | — | — | 13.13 |
| DINOv3 **without** temporal embedding | 0.679 | 0.735 | 0.622 | 13.39 |
| time_prof (ORACLE duration) | 0.621 | — | — | 15.52 |

**(1) RESOLUTION IS THE ONLY LEVER THAT WON — and only on the far camera.** 384 (on a new
416-px frame cache, `clips_grid_cache.py --size 416 --out ...`) gives **+0.021** over 256,
epoch-matched (both exactly 31 epochs / 3193 steps; 384 needed 240 min vs 80, ~7.6 min/epoch).
The gain is **CAM3 +0.039, CAM2 +0.003** — CAM3 is the distant view where the stream occupies
fewest pixels, so resolution helps exactly where the signal was pixel-starved. Mechanistically
coherent, and it says where to spend pixels next. Caveat: one fold, and fold-to-fold spread is
±0.04, so this is suggestive, not decisive — needs folds B/C/D to confirm.

**(2) PARTIAL FINE-TUNING BUYS NOTHING — the frozen-backbone premise is now TESTED, not
assumed.** Unfreezing the last 4/24 blocks (50.4M of 303.9M params, separate AdamW group at
lr 1e-5) gives **−0.009**. Verified before the run that gradients genuinely reach those blocks
and update them, and that block[-5] stays frozen — this is a real null, not broken plumbing.
1636 windows is far too little to improve a 300M-param representation. Consistent with
"head-init barely matters": the representation is already good enough and is doing the work.
**Answers the standing objection "you'd do better with fine-tuning" with a measurement.**
Note 384 + unfreeze does NOT fit in 16 GB at batch 16 (384 alone is 12.0/16.3 GB frozen).

**(3) DINOv3 — the first run was CRIPPLED BY MY OWN BUG; the corrected gap is −0.057, not
−0.170.** DINOv3 encodes each frame independently, so every frame emitted tokens with
identical positional content, and the head (3 permutation-equivariant self-attn blocks + a
permutation-INVARIANT 1-query cross-attn pool) was **frame-order blind — motion direction was
unrecoverable in principle**. Adding a fixed sinusoidal TIME stamp per frame
(`_dino_encoder.py::sincos_temporal`, scaled to 0.5x the batch token RMS; non-learned, so the
backbone stays honestly frozen) is worth **+0.113** (0.679 → 0.792).
- **Do NOT quote 0.679 as DINOv3's performance.** And the retracted claim "a linear V-JEPA
  ridge beats an attentive DINOv3 head" is FALSE once fixed: 0.792 > 0.766.
- V-JEPA still wins, but by 0.057 on one fold vs a ±0.04 spread — "decisively worse" is not
  supportable. Held fixed for the comparison: 2048 tokens x 1024-d (8 frames x 256 patches,
  mirroring tubelet_size=2), ImageNet norm, same head/init/schedule.
- **Lesson: a linear pilot under-predicts an attentive result.** The ridge pilot said
  DINOv3-best 0.669 vs V-JEPA 0.718 and predicted 0.75–0.78; the attentive run hit 0.792.

**(4) ROI CROP LOSES WITHIN-VIEW, 4th independent confirmation** (−0.045 here with a
fine-tuned backbone; −0.10 V-JEPA frozen; −0.075 DINOv3 ridge on ALL 8 temporal reps). So it
is not a frozen-features artefact. ROI's niche remains ONLY the unseen-camera case — and,
untested, the out-of-domain shortcut problem the eval set exposed (it is the natural fix for
"kettle-over-pot pose ⇒ predict fast", but that needs measuring ON the eval set, not fold A).

**DINOv3 TEMPORAL-REPRESENTATION PILOT (`dino_pilot.py`, CPU ridge on cached per-frame
features, experiment `pour_probe_dino_pilot`, 23 runs).** Cheap way to choose a representation
before spending 80-min runs. Flow @ lag 0.7, 4-fold OOF by trial, 2170 windows:

| temporal representation | dim | center | ROI |
|---|---|---|---|
| mean over frames (ORDER-BLIND) | 1024 | 0.453 | 0.382 |
| first+last concat | 2048 | 0.653 | 0.559 |
| **mean + signed endpoint diff** | 2048 | **0.669** | 0.594 |
| all frames concat (ordered) | 8192 | 0.626 | 0.524 |
| all diffs concat (ordered) | 7168 | 0.423 | 0.338 |
| *V-JEPA 2 mean-pool ridge (reference)* | 1024 | *0.718* | — |

- **Order alone is worth +0.216** (0.453 → 0.669) — this is what diagnosed the missing
  temporal embedding.
- **⚠ `mean-of-consecutive-diffs` TELESCOPES to `(last−first)/(T−1)`** — it is a pure ENDPOINT
  feature. A temporal-resolution sweep built on it returned identical R² at 4/8/16/24 Hz
  because of that algebraic identity, not because resolution is irrelevant. Do not reuse it as
  a "motion" feature.
- **FUSION: V-JEPA + DINOv3 = 0.719 vs V-JEPA alone 0.718.** DINOv3 carries essentially NO
  flow information V-JEPA lacks — closes off the "just ensemble them" route.

**REPRESENTATION STABILITY — dense 1-FRAME-STRIDE sliding-window inference (2026-08-08,
supervisor request; `clips_stability.py` + `clips_stability_figs.py`, experiment
`pour_probe_stability`, figures `datasets/eval/figs/fig_stability_*.png`).** Answers "does
a tiny change in input frames change the output?" — i.e. is any reported curve an accident
of where the window landed? Fold-A checkpoint on fold A's HELD-OUT trials (8/13/21/24),
1.0 s windows stepped by ONE frame (~33 ms) instead of the usual 0.5 s stride:
**6650 windows over 60 clips**.

⚠ **The frame overlap is NOT what it looks like.** Consecutive windows overlap 96.7% in TIME
SPAN, but the probe samples only 16 frames out of the 30-frame span (~every 2nd frame), so
consecutive windows land on INTERLEAVED sets — window 0 → frames [0,1,3,5,…,29], window 1 →
[1,2,4,…,30]. They share **1 of 16 actual frames**. So this is a stronger test than "almost
the same input": 15/16 of the images are different files showing a scene displaced by 33 ms.
Do not describe it as "shares 29/30 frames" (an earlier note did; it confuses span with
sampled frames).

| quantity | value |
|---|---|
| prediction spread (1–99 pct) | 123.3 g/s (sd 34.8) |
| \|Δpred\| for a 1-FRAME shift | mean **3.39**, median 1.84, p95 11.37 g/s |
| ground truth, same shift | mean 1.86, median 0.38 g/s |
| → as % of the model's own output range | **2.75%** |
| → as % of the SHUFFLED control | **10%** |

- **The headline number is the shuffled control, not the raw g/s.** Randomly permuting which
  window each prediction belongs to gives mean \|Δ\| = 33.7 g/s. The real 1-frame shift gives
  3.39 — **10% of that**. So the prediction is strongly determined by the input, not by where
  the window happened to land.
- **The change-vs-shift curve is the real evidence** (`fig_stability_summary.png`, left).
  Mean \|pred(t+k) − pred(t)\| rises smoothly: k=1 → 3.39, k=5 → 6.73, k=15 → 15.41,
  k=30 → 29.48 g/s, approaching the shuffled ceiling only at k≈30 (a full non-overlapping
  window). An unstable model would sit at the shuffled level from k=1. From k≈5 onward the
  model's curve tracks the GROUND TRUTH's own curve almost exactly — it changes at the rate
  the true signal changes.
- **Honest nuance: the model's 1-frame jitter is 1.82x the GT's** (3.39 vs 1.86 mean; medians
  1.84 vs 0.38). Do NOT claim "smoother than the signal". Much of that gap is an artefact of
  the GT, not model noise: the GT is a **quantised staircase** (1 g scale resolution, then
  isotonic regression per pour ⇒ long exactly-flat stretches and occasional jumps), so most
  of its steps are exactly zero. The model output is continuous. The fair statement is
  *"a one-frame input change moves the prediction by ~3% of its output range, one tenth of
  chance"*.
- **Holds for the 384 model too, slightly better** (same 60 clips / 6650 windows):
  \|Δpred\| mean **3.02** g/s (vs 3.39), p95 **9.71** (vs 11.37), **9.4%** of its shuffled
  control (vs 10.0%), jitter/GT **1.62x** (vs 1.82x). As a share of output range the two are
  identical (2.77% vs 2.75%) because 384's spread is tighter (109 vs 123 g/s). So the
  higher-resolution model is not bought at the cost of stability — it is marginally steadier.
- Practical corollary: the 0.5 s eval stride is not undersampling — the dense curve traces the
  same shape through the coarse samples (`fig_stability_curves.png`).
- ⚠ `clips_stability.py` must filter clips by trial BEFORE materialising frames;
  `ca.load_clips()` loads every clip's frame array (~18.6 GB across both cams at 416 px) and
  the process gets OOM-killed. Also: launching via `nohup ... &` means the harness tracks the
  LAUNCHER, not the python — two 384 runs once ran concurrently and starved each other of RAM.

**Key sub-findings (each independently confirmed):**
- **Water-transit lag** — the scale registers mass ~0.7 s AFTER the stream is visible (fall
  time + load-cell/display filtering). The R²-vs-lag curve is asymmetric (target sampled
  LATER helps, earlier hurts) → a real delay. Correcting it (`--lag_s 0.7`) lifts the ridge
  OOF flow R² by +0.09; volume is lag-insensitive. `clips_lag_sweep.py`.
  **Mechanics (verified 2026-07-27, `build_windows` lines 74-76):** frames are UNTOUCHED; only
  the target moves — `flow = (W(t1+lag) − W(t0+lag))/(t1−t0)`, `volume = W(tmid+lag)`. Sampling
  GT at t+0.7 ≡ sliding the GT curve 0.7 s EARLIER. Two scale facts worth stating when
  presenting: (a) at lag 0.7 on a 1.0 s window the frames seen and the mass increment predicted
  overlap by only **0.3 s (30%)**, so this is short-horizon prediction, not instantaneous
  readout; (b) the lag pushes 27% of windows past the end of the GT curve, where `np.interp`
  CLAMPS — **checked and benign**, because clips carry a median **1.13 s post-pour plateau**
  (only 8/121 shorter than the lag), so 96% of those windows have flow exactly 0 and the mean
  target error from clamping is 0.02 g/s against a 34 g/s mean.
- **View-robust both-cam probe** — one probe trained on CAM2+CAM3 holds both views
  (combined flow 0.85, CAM2 0.87, CAM3 0.83 — *above* CAM3's native 0.80). But **zero-shot
  CAM2→CAM3 transfer FAILS** (0.52): the attentive pooler is view-specialized (learns "look
  upper-left where the arm is"). Cross-view needs training on both views.
- **Head-init barely matters** (confirmed 3×: EK100 warm-start 0.85 ≈ random 0.84 ≈
  SoW-pretrain 0.83). With a frozen encoder, **the V-JEPA representation does the work**,
  not the pretrained pooler. Keep the warm-start (converges faster) but don't depend on it.
- **Absolute-total-mass tails are data-limited, not fixable by loss/calibration.** The
  SmoothL1 probe compresses extremes (under-predicts big/fast pours); integrating predicted
  flow → total mass gives median 14 g / mean 22 g error (slope 0.927). MSE loss, post-hoc
  calibration, and ridge-alpha sweeps all fail to improve held-out totals — the residual is
  aleatoric. **Only lever = more data.**
- **Container-size model is a NO-GO for absolute volume** (`clips_oracle_container.py`):
  even PERFECT oracle container identity adds ~0 (poured mass is set by pour *duration*, a
  free choice, not vessel capacity). Dropped. Crop-to-container survives only as a
  view-robustness idea for flow.
- **Crop-to-container ENABLES cross-view transfer (validated 2026-07-20).** Motion-ROI
  cropping works on CAM2 but fails on CAM3 (distant view → whole body dominates motion), so
  the ROI uses **GroundingDINO-tiny zero-shot** (`clips_roi_cache.py --backend detector`,
  no training), prompted with the manifest's known source/target vessel classes; ROI = union
  of best source+target box, or target-anchored if they're far apart. Drop-in ROI frame cache
  (`$POUR_ROI_FRAMES_DIR`), so the existing trainer/eval run unchanged. **Result (flow, lag 0,
  same {8,13,21,24} val split, `attn_flow_CAM2_roi_best.pt`):**
  | | center-crop | detector-ROI | Δ |
  |---|---|---|---|
  | CAM2-native (within-view) | 0.899 | 0.738 | −0.16 |
  | **CAM2→CAM3 zero-shot** | **0.524** | **0.706** | **+0.18** |

  **⚠ SUPERSEDED — RERUN AT LAG 0.7 (2026-07-27, `run_roi_lag.sh`, 2h46m, same split
  {8,13,21,24}).** All four cells redone with the lag on both arms, AND the native-view ridge
  recomputed at the same lag (the old 0.593 reference was a lag-0 artefact of the
  `baselines_on_split` bug above):
  | | center-crop | detector-ROI | native ridge (that view) |
  |---|---|---|---|
  | CAM2 within-view | **0.885** | **0.781** | 0.788 |
  | CAM2→CAM3 zero-shot | **0.507** | **0.735** | 0.764 |

  Both trainings converged (val tail 0.874±0.001 and 0.780±0.002).
  - **Crop cost shrinks, crop gain grows:** within-view −0.104 (was −0.161), cross-view **+0.228**
    (was +0.182). Cropping closes the view gap from 0.378 to **0.046** — the cropped probe is
    nearly view-invariant.
  - **The lag helps ONLY the cropped arm** (+0.043) and slightly hurts the full-scene one
    (−0.014). Consistent with the attention-map story: once cropped, the probe must read the
    actual stream, whose timing genuinely lags the scale; the full-scene probe leans on arm/body
    pose, a coarser cue that is lag-insensitive.
  - **BUT the honest conclusion REVERSES: zero-shot ROI transfer (0.735) does NOT beat a plain
    linear ridge trained on the target view (0.764).** The old "ROI transfer beats the CAM3-native
    baselines" claim only held because that baseline was un-lagged. **The crop's niche is narrower
    than advertised: an unseen angle with NO labelled data there.** With any labelled target-view
    data, a linear probe on it already wins.
  - Also: at lag 0.7 the ROI within-view probe (0.781) merely TIES the CAM2 linear ridge (0.788),
    while the center-crop probe (0.885) still beats it by +0.10.
  A clean **robustness-vs-accuracy trade**: the tight crop removes the view-specific shortcut
  ("look upper-left at the torso") → worse on its home view but transfers far better, and ROI
  transfer (0.706) is *above* the CAM3-native ridge/time_prof baselines (0.59). **So cropping
  is the right tool ONLY for the unseen-camera scenario. **⚠ BOTH ROI bars are lag-0** — the ROI
  arm has never been rerun with the 0.7 s lag correction (worth +0.09 elsewhere), so this
  comparison very likely UNDERSTATES the crop. Settling it needs two ~80-min runs (center-crop
  CAM2 @ lag 0.7 and ROI CAM2 @ lag 0.7) + two `clips_eval_crossview.py` evals; the ROI frame
  cache (`~/.cache/pour_probe/clips_frames288_roi`, 8 GB, both cams) already exists.
  Driver: `pouring/pour_probe/run_roi_lag.sh` (both trainings + both evals, sequential).
  **`clips_eval_crossview.py` hardcoded lag=0** in `build_windows` for BOTH the train-split
  normalization stats and the eval targets; it now takes `--lag_s`, which MUST match the lag the
  checkpoint was trained with. (No past number is invalidated — every previous cross-view eval
  was of a lag-0 checkpoint, so the hardcoded 0 was correct then.)** (deploy to a new angle without
  collecting/labelling data there); if you already have the target view's data, both-cam
  training (CAM3 0.83) still wins outright. Caveats: single split (±noise, but effect is
  large); CAM3→CAM2 direction + lag-0.7 rerun not tested; ROI is imperfect on tall-pour CAM3
  clips yet transfer improved anyway. `clips_eval_crossview.py --tag _roi`.

**Recipe gotchas that cost real GPU time (don't repeat):**
- **LR schedule must be wall-clock-driven.** Sizing the cosine from epoch-0's time overshot
  (val eval excluded, cold caches) → LR hit 0 with ~20-24% of budget left OR never annealed
  → val R² bounced and never plateaued. Fix: cosine driven by wall-clock fraction of the
  time budget + linear warmup over epoch 0 + checkpoint on a 3-epoch rolling-mean of val R².
  A converged run is a dead-flat plateau at LR=0.
- **`clips_train_attn.baselines_on_split()` used to IGNORE the lag — FIXED 2026-07-27.** It read
  targets straight from the mean-pool cache, which `clips_extract` wrote at **lag 0**, so every
  lagged run scored the probe on lag-0.7 targets and its printed `ridge_meanpool`/`time_prof`
  baselines on lag-0 targets. That flattered the probe badly: **the CAM3 ridge goes 0.593 → 0.764
  (+0.171) once the lag is applied**, and that single correction reversed the ROI cross-view
  conclusion (see the ROI section). It now takes `lag_s`/`window_s` and recomputes targets via
  `_retarget()` (same definitions as `clips_extract`: volume = weight at window centre, flow =
  ΔWeight across the window, both sampled at t+lag). `clips_eval_crossview.py` passes its
  `--lag_s` through. Verified: lag 0 reproduces 0.593/0.597 exactly, lag 0.7 gives 0.764/0.594.
  **Any pre-2026-07-27 logged baseline in `pour_probe_clips_attn` from a lagged run is lag-0 and
  must not be compared against that run's probe score.** The headline ladder is unaffected (its
  ridge 0.69 comes from `clips_cnn_baseline`, which applies LAG_FLOW=0.7 correctly).
- **`cv_r2`'s `Normalizer()` (L2) destroys low-dim features** — silently zeroed the raw-time
  and decoded-λ baselines. Use `normalize=False` for interpretable/low-dim features;
  high-dim CNN/V-JEPA rows are fine.

**Cross-modal (Sound of Water) & sim (SoW-data) arms — reference, not rivals:**
- SoW audio model on our clips: flow 0.65 (above the temporal prior + video CNN, **below**
  V-JEPA), volume 0.80 (barely beats the clock's 0.78 → NOT strong audio volume perception).
- Our probe on SoW's own videos: their target is model-derived AND their pours are
  constant-flow, so it's **clock-dominated** (time-only baseline 0.80–0.98 within-video);
  the probe only modestly beats the clock on transparent sets. A *positive by contrast* —
  our own dataset has VARYING flow, which is exactly why time can't cheat there and the
  own-data flow result is the real contribution.
- **WHY their between-video rate randomisation cannot save them from the clock (settled
  2026-07-27).** The winning baseline is **normalised** time `t/T`, not absolute time, and on
  their data `t/T` is *algebraically identical* to fill fraction: constant Q + fill to
  completion ⇒ `T = V/Q` ⇒ `V(t)/V = Qt/QT = t/T`, **Q cancels**. Randomising Q between videos
  changes T but `t/T` absorbs it exactly. Only WITHIN-pour rate variation breaks it, and they
  deliberately don't do that. Compounded by (a) the within-video metric mean-removes each
  video, discounting the amplitude `V_total` — the one thing `t/T` can't know; (b)
  `sow_baselines_on_split`'s `prof()` derives each val video's duration T from its OWN windows,
  an ORACLE their model never gets (they predict from a 25/50/75% prefix). **So this is a
  degeneracy of OUR repurposing, not a flaw in their paper — say it that way.**
  Measured (`pour_probe_sow_attn`, within-video R², volume, grouped CV by container):
  | subset | V-JEPA probe | norm-time poly4 |
  |---|---|---|
  | S1 transparent (11 cont.) | 0.163 | 0.058 |
  | S2 cylindrical (21 cont.) | 0.287 | 0.253 |
  | S2o opaque (8 cont.) | 0.749 | **0.976** |
  | S3 all shapes (45 cont.) | 0.803 | 0.799 |
  The probe wins only where the level is VISIBLE (S1) and loses by 0.23 on opaque containers —
  itself the evidence that the target is a clock. NOTE our own volume clock is a DIFFERENT and
  much weaker timer: raw ABSOLUTE time (`volume ≈ −37 + 50·t`, R² 0.778), not `t/T`.
- **MEASURED clock-explainability of both datasets (2026-07-26) — the earlier contrast was
  overstated for VOLUME.** Per-sequence R² of a straight line in time fitted to cumulative
  volume: **SoW median 0.989, 83% of videos >0.95; ours median 0.943, 41% >0.95.** So volume is
  near-linear in BOTH corpora (a difference of degree, not kind) — which is exactly why the clock
  is also strong on OUR volume. The datasets separate cleanly on **flow**, not volume: our flow
  is a real measured derivative with clock R² 0.00, while their flow target is a numerical
  derivative of a model estimate (both probe and baseline ≈0, documented degenerate).
  Verified from their own splits: **988/1010 pours labelled `flow_rate_appx=constant`, exactly
  ONE labelled non-constant.**
- **THE PAPER PDF IS LOCAL: `/home/casimir/Zotero/storage/DWENJFTV/`** — read
  `.zotero-ft-cache` (plain text, 83 KB, grep-able). Do NOT re-derive from the abstract or the
  website; both are useless. Key facts settled from the text (2026-07-27):
- **Did SoW address the clock? YES, and they SAY SO. Don't misrepresent them.** Two verbatim
  lines resolve the whole question:
  - Dataset section: *"Across videos, we **randomly vary the flow rate** but keep it
    **approximately constant within a single video**."*
  - Time-to-fill derivation: *"For time to fill, we **assume a constant flow rate** (since
    otherwise, one could pause pouring midway leading to ill-defined time to fill)."*
  So constant-within-pour is a deliberate, **load-bearing** design choice, not an oversight —
  their task is ill-defined without it. The randomisation is BETWEEN pours; our clock baseline
  exploits the WITHIN-pour axis. Both claims are correct; they are different axes.
  Independently confirmed model-free from `splits/*.csv`: same container → duration CV median
  **0.23**, fastest/slowest **2.3–4.3×**, absolute rate 24–95 mL/s.
- **Their flow-rate GT is a SINGLE SCALAR per video** (*"the ratio of the volume of container and
  the time it took to fill it completely"*), never a trace. That is the full explanation for why
  our per-frame flow run on their data was degenerate — the target does not exist in their data.
- **Time-to-fill has a strong no-listening prior they don't report.** Their metric is remaining
  time τ = T − t (Eq. 8), so a baseline predicting τ̂ = T̄ − t from the free audio length has MAE
  |T̄ − T|, flat across cut levels. Computed on their splits: **Test I container-mean prior
  2.11 s, Test II global prior 2.68 s** vs their co-supervised model 4.16/1.49/1.07 (Test I) and
  4.10/2.99/2.21 (Test II) at 25/50/75% heard. **At 25% heard, not listening beats the model on
  both test sets**; relative error is 31–83% of the target throughout. Their model does clearly
  win from 50% on Test I. Fair framing: same lesson we applied to ourselves (always report the
  trivial prior), not a refutation.
- **PROTOCOL MISMATCH to fix before quoting "audio beats video on volume":** the audio rows are
  frozen+ridge 4-fold; the V-JEPA volume 0.667 is the ATTENTIVE probe on ONE split. On the
  matched ridge protocol V-JEPA volume is 0.398. A 4-fold attentive volume run would make the
  comparison honest.
- **⚠ CORRECTION (2026-07-27): SoW DOES compare against Wilson et al. PSNN** (IROS 2019, DOI
  10.1109/IROS40897.2019.8968118) — their **Table 4**, mass estimation on Wilson's own data.
  Earlier notes saying "no head-to-head" were about *vision* methods and must not be read as "no
  PSNN comparison". Mean MAE (oz): Linear SVM 3.90, SoundNet-8 4.58, SoundNet-5 3.88, k-NN 2.85,
  TCN 2.45, **PSNN (audio+visual) 1.35, SoW (audio, LINEAR PROBE) 1.20**. SoW beat PSNN's
  supervised fine-tuning with a linear probe on frozen features.
- **→ THE BEST NEXT EXPERIMENT: put frozen V-JEPA 2 into that table.** It is a published
  benchmark with a public protocol (136 sequences, 6 containers × 2 liquids, per-container
  regressor, their splits), an existing baseline ladder, and **no video-foundation-model entry**
  — SoW explicitly note "visual information is not used in any form". Our frozen+linear-probe
  recipe is exactly what SoW used, so it drops straight in. This is the one place the project can
  produce a directly comparable published number. Cost: feature extraction over 136 sequences +
  6 ridges.

---

# BACKGROUND / HISTORY (V-JEPA anomaly-detection thread, pre-pivot)

Pre-05.07.2026 experiments. Kept for context; not the deliverable.

## `egoper_probe/` — frozen V-JEPA 2 procedural-error probe on EgoPER
The original centerpiece. Sliding-window frozen ViT-L features → logreg probe →
window/video-level error-detection AUC, mlflow. **Result (coffee, 68 vids): supervised
window ROC-AUC ≈ 0.75** (0.78 with a regularized SlowFast detail view) — frozen latents
linearly separate correct vs. erroneous windows. **One-class (normal-only) stayed ~0.60** —
unsupervised detection of subtle local errors was the hard open problem. EgoPER: 5 tasks,
per-video localized action-type labels (0=Normal, 1–4=error types); error = any
action_type≠0.

## `exprt_probe/` — anomaly + action labelling on the eXprt tea dataset
Second probe, on a **fixed third-person wide shot** (CAM1, opposite of egocentric
EPIC/EgoPER) — the main reason off-the-shelf EK100 transfers poorly. 40 trials = 8 classes ×
5 iters, **video-level labels only**. Findings: **8-way error-type ≈ 0.49 acc** (~4× chance,
EK100 warm-start helps); **binary anomaly only ~0.70** (bottlenecked by 5 normal videos + no
within-video localization). Frozen V-JEPA **embeddings carry a real action signal the
egocentric EK100 head misses** (video-LOO 0.74 native-space probe vs 0.05 zero-shot EK100);
augmentation → 0.71 on a genuine 6-way problem. Post-26.06 the chosen next thread was
**open-vocab action labelling via soft kNN in V-JEPA's own latent space** (design locked: do
NOT project into text space — lossy + confounds the backbone; CLIP text encoder only as a
symmetric label ruler) — drafted (`knn_label.py`) but paused when the project pivoted to
pouring.

## `egoper_vqa/` — zero-shot video-QA + EK100 action recognition (baseline/contrast)
Qwen2.5-VL-7B (4-bit) + a training-free SlowFast-v1 scheme on EgoPER long clips, and a
faithful zero-shot **V-JEPA 2 + EK100 anticipation head** check on EgoPER tea. The latter
scored **top-1 verb 0.61 / top-5 0.96** (vs ~0.18 random) — off-the-shelf V-JEPA reads
tea-making verbs zero-shot and flags anomalies (mug-in-microwave, stir-with-knife). The VQA
baseline hallucinated generic knowledge and missed real errors — motivated the frozen
feature probe over a QA decoder.

## `video_qa/` — Appendix E replication (V-JEPA 2 paper Sec. 14)
Custom LLaVA-style loop aligning the frozen V-JEPA 2 encoder with Qwen2.5-7B (QLoRA) for
video QA; 3-stage visual instruction tuning. **This dir is the source of the shared
`build_encoder`.** Deviations for 16 GB: QLoRA, small public datasets, 256px/8 frames.

## Presentation (`presentation/`)
Slidev deck, **split by concern**: `slides.md` is only headmatter + a title slide + `src:`
imports of `pages/*.md` (`00-arc`, `10-background`, `20-data`, `30-method`, `40-results`,
`50-ablations`, `60-crossmodal`, `70-outlook`). Edit one page file per topic; nothing else
needs touching. ~47 slides as of 2026-08-03. **`presentation/README.md` is the operational
guide** (build, page map, figure regeneration, run record) — read it first.
- **The deck builds from a BARE CLONE — keep it that way** (verified 2026-08-03 by cloning
  from GitHub with `datasets/` and `~/.cache` both absent: all 19 figures regenerate
  **byte-identical**, `slidev build` clean, 28 images bundled, no broken refs).
  `presentation/data/` (4.4 MB) mirrors the few non-git inputs the figures touch:
  `headline_preds.npz` (was a hardcoded `/home/casimir/.cache` path), `clips_manifest.csv`,
  three extracted video frames (`fig_inputs`, `fig_views_example` — substitutes for the
  410 MB clip set), and the 121 GT clip curves. `make_figs.py::data_path()` prefers the
  original and falls back to the bundle, so nothing changes on the workstation.
  **If you add a figure that reads from `datasets/` or a cache, add its input to `data/`.**
- **Assets are relative, NOT in `public/`.** Slidev/Vite's slide-import-guard rejects
  `/foo.png` absolute paths inside imported page files (`resolves outside server.fs.allow`),
  so pages reference `../figs/*.png` (generated) and `../assets/*.png` (copied QC figures).
- `make_figs.py` regenerates every summary chart into `figs/` AND syncs/crops the experiment
  QC pngs into `assets/` (incl. cropping the 8-row ROI QC sheet to 3 rows). Numbers in it are
  transcribed from mlflow + the analyses that were never logged as runs (lag sweep,
  calibration, oracle-container) — it is the single place to fix a number.
  The QC *sources* under `pouring/` are gitignored (`qc_*.png`), so the **copies in
  `presentation/assets/` are the tracked ones** (`.gitignore` carries an explicit
  `!presentation/assets/qc_*.png` exception); `sync_assets()` no-ops when sources are absent.
- `colorSchema: light` is forced in the headmatter; the matplotlib figures are white-background
  and look broken on slidev's default dark scheme.
- Build/preview: `npx slidev` (dev) or `npx slidev build`. No playwright installed, so
  `slidev export` fails; to screenshot, build then serve the dist through an SPA-fallback
  static server and drive `/usr/bin/chromium --headless --screenshot`.

## Final deck (`presentation_final/`)
The **talk deck** (14.08.2026), on a local `theme-tum` Slidev theme reproducing the TUM
pptx template pixel-for-pixel. Everything lives in one `slides.md` (`pages/` is empty).
`presentation_final/README.md` is the authority — how the theme was derived from the OOXML,
the layout table, and the gotchas.
- **Architecture figures are generated with PlotNeuralNet** (vendored MIT, `figs_src/`):
  `../.venv/bin/python figs_src/attentive_probe.py` → `public/attentive_probe.png`.
  **No TeX is installed system-wide** — the script finds `tectonic` (or `pdflatex`) on PATH;
  a standalone tectonic binary works fine, `pdftoppm` does the raster at 300 dpi.
  PlotNeuralNet quirks that cost time: `to_Conv(width=...)` needs `"{4,4,4}"` (braces, or
  pgfkeys reads the entries as keys), `n_filer` needs one entry PER concatenated box, and
  `to_input`'s node has no `-east` anchor (draw from a plain coordinate).
- **Citations: `bib.ts` + `<Cite>` / `<CiteFooter>` / `<References>`** (`components/`,
  added 2026-08-13). One entry per source in `bib.ts`; the number IS the entry's position
  there, so markers and the "Quellen" slide can't drift. `<Cite>` registers into
  `cite-registry.ts` keyed by page, `<CiteFooter>` prints that page's sources above the
  footer, `<References cols="2"/>` renders the full list. Unknown id → red `[?]`, never a
  build failure. Details in `presentation_final/README.md § Citations`. Metadata filled from
  **Zotero** (`~/Zotero/zotero.sqlite` — copy it first, the live db is locked while Zotero
  runs; items + `itemData`/`itemCreators` joins give authors/venue/DOI). **Not in the library
  and therefore unverified: LeCun 2022 JEPA, DINOv3, Grounding DINO; and Zotero holds only
  title+authors for V-JEPA 1 (Bardes) and EgoPER (Lee).**
- **Diagrams go in `layout: figure`, photos in `layout: image`** (figure added 2026-08-13,
  `theme-tum/layouts/figure.vue`). The pptx picture boxes start at 34.4 %/23.8 % and run to
  the slide edge, so a tall figure sits too low with its bottom labels under the footer.
  `figure` spans from the head text to 91.11 % and letterboxes — write a bare `<img>`, no
  wrapper div, no inline sizing. Its top edge is MEASURED from the rendered title/subtitle
  (a long subtitle wraps and overruns its 6.25 % box). Used by the V-JEPA, attentive-probe
  and Bland-Altman slides.
- **A wrapping title overlaps the next block** — every placeholder is pinned by a pptx
  percentage, so a two-line title grows down into whatever follows. Fixed 2026-08-13 for
  `cover`/`cover-photo`/`section` (title + info now flow inside one `.tum-cover-head` box)
  and for `figure` (measures the head text). **`default`/`two-cols`/`content-image`/`image`
  are still pinned** — a title long enough to wrap will collide with the subtitle there.
  Still true for hand-placed HTML: keep inline HTML on ONE line; a multi-line `<img …>`
  loses its attributes to the markdown parser (the `style` silently vanishes and the image
  overflows/clips).

## Misc
- (Add homeless notes here.)
