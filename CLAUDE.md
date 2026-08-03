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
needs touching. 34 slides as of 2026-07-26.
- **Assets are relative, NOT in `public/`.** Slidev/Vite's slide-import-guard rejects
  `/foo.png` absolute paths inside imported page files (`resolves outside server.fs.allow`),
  so pages reference `../figs/*.png` (generated) and `../assets/*.png` (copied QC figures).
- `make_figs.py` regenerates every summary chart into `figs/` AND syncs/crops the experiment
  QC pngs into `assets/` (incl. cropping the 8-row ROI QC sheet to 3 rows). Numbers in it are
  transcribed from mlflow + the analyses that were never logged as runs (lag sweep,
  calibration, oracle-container) — it is the single place to fix a number.
- `colorSchema: light` is forced in the headmatter; the matplotlib figures are white-background
  and look broken on slidev's default dark scheme.
- Build/preview: `npx slidev` (dev) or `npx slidev build`. No playwright installed, so
  `slidev export` fails; to screenshot, build then serve the dist through an SPA-fallback
  static server and drive `/usr/bin/chromium --headless --screenshot`.

## Misc
- (Add homeless notes here.)
