---
name: idp-presentation
description: Answer questions about the IDP SS26 project history and help build the final presentation deck. Use when the user asks about project stages, timeline, what was explored or considered in a stage, or works on the presentation (slides, deck, figures, stage summaries). Covers the full arc from V-JEPA 2 setup through the pouring flow/volume centerpiece to the final TUM-themed deck.
---

# IDP SS26 — Presentation Workflow

## Working contract

- **Brief summaries, no essays.** Results are less important than what happened /
  what was considered / what was explored.
- Answer stage questions on demand; do not dump the whole timeline unprompted.
- When the user says "stage N" or names a stage, give the compact stage summary
  below, expanded only as asked.

## The 5 stages (dates, what was explored/considered)

1. **Setup (Jun 2–9):** V-JEPA 2 exploration notebooks; `video_qa/` = Appendix-E
   replication of the V-JEPA 2 paper (arXiv 2506.09985, Sec. 14): frozen ViT-L +
   Qwen2.5-7B QLoRA, 3-stage visual instruction tuning. Produced the shared
   `build_encoder` used by every probe since. Deviations for 16 GB: QLoRA, small
   public datasets, 256px/8 frames. First Blackwell/torch-2.1.2 constraint hit
   (LLaVA-NeXT ruled out).
2. **EgoPER anomaly probe (Jun 10):** sliding-window frozen features → logreg,
   window/video-level AUC. Explored: one-class (Mahalanobis, kNN, token banks —
   stayed ~0.60), SlowFast detail view (spatial/temporal pooling, strong L2).
3. **eXprt tea probe (Jun 18–26):** third-person wide shot, 40 trials, video-level
   labels. Explored: EK100 head transfer, clip-vs-video level (clip fails),
   augmentation, open-vocab soft-kNN labelling (drafted, dropped on pivot).
   Also `egoper_vqa`: Qwen2.5-VL zero-shot VQA (hallucinated) + zero-shot
   V-JEPA+EK100 verb recognition.
4. **Pivot → pouring centerpiece (Jul 5–Aug 7):** anomaly thread dropped at the
   Jul 5 supervisor meeting; LOCKED centerpiece = pouring volume/flow from frozen
   V-JEPA 2. Surveyed datasets: UWLPD (mask proxy, no mL), SimLiquid (BlenderProc
   sim, set up + validated), Sound of Water (audio), own-lab recording. UWLPD
   mask-proxy smoke test R² 0.80. Then the data pipeline (Jul 13–16): own-lab
   recording (3 GoPro views, scale-OCR GT); 3 failed OCR backends
   (tesseract/segment/template) → own `lcd_ocr.py` (7-seg correlation); filter
   variants (ema/median/mono → monotonicity prior + isotonic); pour detectors
   (regions → plateau-chain + CUP BARRIER); interactive annotator web app.
   Result: 121 clips, 8–362 g, 18 trials. Then probe & results (Jul 17–20):
   ridge → attentive probe; water-transit lag discovery (+0.09); view transfer
   (CAM2→CAM3 fails); both-cam probe; CNN baselines (ResNet strawman ~0, Kinetics
   video CNNs 0.53); Sound-of-Water cross-modal; ROI cropping (motion fails on
   CAM3 → GroundingDINO); oracle-container NO-GO; head-init ablations (barely
   matters); LR-schedule fixes. Then the rigor pass (Jul 26–31): clock analysis
   (volume clock-dominated, flow not); fixed `baselines_on_split` lag bug; strict
   eval protocol with skill scores; SoW paper deep-dive (constant-flow design is
   load-bearing); 4-fold attentive volume correction; headline metrics in
   physical units (MAE, Bland-Altman). Then the final ablations (Aug 7):
   resolution 384 (+0.021, CAM3 +0.039); unfreezing last 4 blocks buys nothing
   (−0.009); DINOv3 + temporal embedding (0.792, order worth +0.113); ROI loses
   within-view (4th confirmation); DINOv3 fusion adds nothing.
5. **External validation (Aug 6):** 9 iPhone pours, unseen scenes. Shape
   transfers, totals don't (MAE 153 g); phantom-flow floor (no "not pouring"
   concept); trickle test IMG_0868 fails (5.6× over). Single-final-reading
   analysis (audio beats video).

## Where the detail lives

- `CLAUDE.md` — current project summary (post-Jul-20, the centerpiece + rigor +
  ablations). Read this first for any stage ≥ 6.
- `CLAUDE.md.bak` — full pre-Jul-20 history (stages 1–5, day-by-day).
- `git log --pretty=format:"%ad|%s" --date=short` — coarse commit timeline.
- mlflow (10 experiments, creation dates): `egoper_probe_coffee` (Jun 10),
  `exprt_anomaly` (Jun 18), `exprt_action_probe` (Jun 26), `pour_probe` (Jul 5),
  `pour_probe_clips` (Jul 17), `pour_probe_clips_attn` (Jul 17),
  `pour_probe_sow_attn` (Jul 19), `pour_probe_baselines` (Jul 19),
  `pour_probe_dino_pilot` (Aug 7). UI: `mlflow ui --backend-store-uri
  sqlite:////home/casimir/UNI/SS_26/idp/mlflow.db`. Run record as CSV in
  `mlflow_export/` (db is gitignored — GitHub secret scanner false positive).
- `presentation/README.md` — interim deck operational guide (build, page map,
  figure regeneration, experiment record).
- `presentation_final/README.md` — final TUM-themed deck: how the theme was
  derived from the pptx, layouts table, gotchas.

## Decks

- **`presentation/`** (interim): Slidev, `slides.md` = headmatter + `src:` imports
  of `pages/*.md` (00-arc, 10-background, 20-data, 30-method, 40-results,
  50-ablations, 60-crossmodal, 70-outlook). 19 figures via `make_figs.py`
  (numbers transcribed by hand from mlflow — single place to fix a number).
  `data/` mirrors non-git figure inputs; assets relative, NOT in `public/`.
  `colorSchema: light` forced. Build: `npx slidev` / `npx slidev build`.
- **`presentation_final/`** (final, TUM theme): local theme `theme-tum/`
  reproducing `HLU_Presentation_Template_Oct_2022_Modified_by_Tian.pptx`.
  `canvasWidth: 720` → 1 CSS px = 1 PowerPoint pt. Layouts: cover, cover-photo,
  default (+`lead:`), two-cols (`::right::`), content-image (`band:`), image
  (`size: full`), section. themeConfig: `chair` (3 affiliation lines), `footer`.
  Gotchas: no blank lines/# comments in headmatter; never type a layout prop as
  `string | boolean`; layout roots need `slidev-layout` class; export needs
  playwright-chromium (else build + headless chromium screenshot).
  Build: `pnpm dev` / `pnpm build`.

## Key numbers worth keeping straight (only where they are the point)

- Flow (the deliverable): attentive probe R² 0.81 ± 0.04 (4-fold OOF), MAE ~8.5 g/s;
  beats Kinetics video CNNs (0.53) by +0.28, time-profile prior (0.62), audio (0.65).
- Volume: clock-dominated — raw-time 0.78, V-JEPA 0.58, V-JEPA+clock 0.82.
- Total mass per clip: 23.2 g MAE (integrated flow); external set: 153 g (shape
  transfers, totals don't).
- Lag: 0.7 s water-transit (scale registers late); lag-0.7 is the standard.
- Head-init barely matters; resolution 384 is the only lever that won; fine-tuning
  and DINOv3 fusion buy nothing.
