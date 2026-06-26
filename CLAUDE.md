# IDP — Interdisziplinäres Projekt (SS26)

## General instructions
Update this file A LOT, every time you learn something new about this project.
Use the Misc section at the bottom if you don't have a good section for it.

## Notion Page
The canonical project page lives at:
https://app.notion.com/p/34830491519b800eb334c130d1478d73

Use `mcp__notion__notion-fetch` with that URL (or page ID `34830491-519b-800e-b334-c130d1478d73`) to get the latest state of the project, including meeting notes, topics, datasets, and tasks.

## Project Summary
Systematic evaluation of **V-JEPA models for anomaly detection** in video.

Key focus areas:
- Anomaly detection in action sequences (pouring water, making tea, MSAD dataset)
- Exploring V-JEPA-2 checkpoints and latent representations
- Visualization of attention maps / spatiotemporal tubelets
- Generalization limits and error analysis
- Possibly: hierarchical JEPA over different time horizons
- use mlflow for experiment tracking
- create a powerpoint slide each week before supervisor discussions

## Datasets
Tracked in the Datasets DB: https://app.notion.com/p/0d5ea22cf2e948018b6f59819cb75a4a
Main focus is EgoPER

### On-disk storage layout
Large datasets live on the **Storage HDD** (1.8 TB NTFS, label `Storage`, `/dev/sdb1`),
auto-mounted at `/mnt/storage` via an `/etc/fstab` `nofail` entry (UUID 3E5EC6754FC651A5,
ntfs3, uid/gid=1000). The 1 TB NVMe SSD (`/`) is kept free for active work.
`datasets/` contains **symlinks** into `/mnt/storage/datasets/` so all code paths are unchanged:
- `datasets/egoper` → `/mnt/storage/datasets/egoper` (214 GB, 816 files)
- `datasets/eXprt-Daten` → `/mnt/storage/datasets/eXprt-Daten` (135 GiB, 87,700 files;
  fetched via rclone from `tum-nas:/tumw/sgm/02_Studierende/Wallwitz/Dateneingang/eXprt-Daten`)

If the egoper/eXprt symlinks ever look broken, the Storage drive isn't mounted —
`sudo -A mount /mnt/storage` (use `sudo -A`; plain sudo hangs on the password prompt).

## Hardware
- RTX 5060ti 16 GB VRAM (for training)

## Direction (LOCKED)
Centerpiece = **frozen V-JEPA 2 feature probe for procedural-error detection on EgoPER**
(V-JEPA is non-negotiable; some light training is OK; SOTA is NOT the bar — need a working
result). Plan: cache sliding-window V-JEPA 2 ViT-L features → train a light head
(linear probe first; one-class/prototype variant as the more EgoPER-faithful stretch) →
eval window-level error-detection AUC/AP, tracked in **mlflow**. The `egoper_vqa/`
Qwen2.5-VL zero-shot QA stays as the *baseline/contrast*, NOT the deliverable.
Reuse `video_qa/model.py::build_encoder` (local vjepa2 pkg + `checkpoints/vitl.pt`,
`target_encoder` key; forward `(B,C,T,H,W)->(B,N,1024)`, tubelet=2, patch=16) — new dir `egoper_probe/`.

## EgoPER as Video QA — design notes 
Plan: adapt EgoPER (egocentric procedural error detection) to a **video-QA format**
rather than using EgoPER's native step-prototype pipeline. Goal: ask user-defined
questions about long (~10 min) egocentric videos containing subtle procedural errors.

Why the existing `video_qa/` (Appendix E) setup does NOT transfer as-is:
- **Temporal coverage is the dealbreaker.** 8 frames over ~600 s ≈ 1 frame / 75 s.
  Subtle errors (omission, wrong order/tool/ingredient, premature action) are
  brief and localized; sparse global sampling misses the evidence entirely.
- 256 px + avg-pool 1024→256 discards the fine spatial detail that distinguishes
  subtle object-state errors (lid on/off, salt vs sugar).
- QA decoder is the wrong head for EgoPER's frame/segment-level error labels.

Agreed direction:
- Keep frozen V-JEPA 2 encoder; switch to **sliding-window** clip features over the
  full video (dense temporal sampling), not 8 global frames.
- **The real constraint is the LLM token budget, not the encoder.** Encoder runs
  sequentially per-window (cheap, ~0.6 GB), so **precompute & cache features to disk**
  → removes encoder VRAM/latency from the train/inference loop. OOM + inference time
  scale with total visual tokens fed to the LLM (Qwen2.5-7B KV ≈ 56 KB/token).
  Naive 256 tok/clip × ~200 clips ≈ 51k tokens = non-starter on 16 GB.
- Fix = compress each clip to a handful of tokens. Target total visual budget in the
  **low thousands** (~1–4k tokens / video) → trainable w/ QLoRA + grad-ckpt + flash-attn.
- Inject per-clip **timestamps** into the compressed tokens so the model can *localize*
  errors, not just detect them.

### SlowFast-LLaVA-1.5 (arXiv 2503.18943, Apple) — candidate approach
Two-stream token pooling that directly fits "spatial detail vs temporal coverage":
- **Slow** pathway: few frames (e.g. 32), light spatial pool (2×2), keeps spatial detail.
- **Fast** pathway: many frames (e.g. 128), heavy spatial pool (4×4), motion/coverage.
- Their config: 128 input frames, ~9K total visual tokens; encoder = **Oryx-ViT** (p16),
  LLM = **Qwen2.5** (1B/3B/7B). ~1.79 s/video fwd on H100; ~65% tokens of Oryx1.5.
- Pooling is **parameter-free** (avg pool, no learned Q-former) → simpler/lower-VRAM
  than a resampler. Good default for 16 GB; reach for a learned resampler only if it
  underperforms on subtle cases.
- **Adaptation for our project:** don't adopt their per-frame Oryx-ViT encoder — apply
  the SlowFast *pooling scheme* on top of our cached V-JEPA 2 clip-token grid (V-JEPA
  encodes spatiotemporal tubelets, so Fast temporal res is bounded by window stride —
  keep stride tight so brief errors aren't straddled).
- For 16 GB, the **1B/3B** scales are the realistic inference targets (7B @ 9K tokens is tight).

**Model availability:** official SF-LLaVA-1.5 trained weights are
NOT discoverable on HuggingFace. `apple/ml-slowfast-llava` (GitHub, only `main`) is the
*original training-free v1* (arXiv 2407.15841) — ships no weights, uses LLaVA-NeXT
checkpoints + a frame-sampling trick. HF search `slowfast-llava` → only one unrelated
community model. So testing "their pretrained model" may require the v1 training-free
route (LLaVA-NeXT base) or a public stand-in long-video VLM.

### EgoPER on-disk layout (datasets/egoper/)
- Tasks: Coffee, Tea, Oatmeal, Pinwheels, Quesadilla (dir names capitalized; annotation
  keys lowercase). Per task: `trim_videos/*.mp4` (the full ~10-16 min recordings, ~15 fps,
  720×1280), frame `*.zip`s, `*_normal_actions.txt`, `*_chatgpt4omini_error.txt`.
- `annotation.json` = dict[task] with `action2idx`, `actiontype2idx`
  (**0=Normal, 1=Error_Modification, 2=Error_Slip, 3=Error_Correction, 4=Error_Addition**),
  and `segments`: per-video `{video_id, labels:{action[], action_type[], time_stamp[[s,e]],
  error_description[]}}`. Video has an error iff any action_type≠0; many segments ship only
  as frame zips (no .mp4) — filter to existing trim_videos. ~35 error + ~33 normal mp4s for Coffee.

### egoper_vqa/ek100_tea* — zero-shot V-JEPA 2 + EK100 action recognition on EgoPER tea (presentation)
Qualitative + quantitative check that **off-the-shelf V-JEPA 2 + the EK100 anticipation head reads
tea-making actions** (motivates using EgoPER's localized action labels for low-level action labelling,
since eXprt has only video-level scenario labels). Faithful pipeline: frozen ViT-L `target_encoder` +
`predictor` (both in `checkpoints/vitl.pt`) via the repo's `vit_encoder_predictor_concat_ar`
`AnticipativeWrapper` + trained EK100 `AttentiveClassifier` (`checkpoints/ek100-vitl-256.pt`,
`classifiers[0]`), anticipate 1.0 s, 32f@8fps@256px. EPIC verb/noun/action indices decoded by rebuilding
the dataloader's `enumerate(set(...))` dicts from `egoper_vqa/epic_meta/EPIC_100_train.csv` (+ verb/noun
class CSVs); **verbs decode reliably (97, identity), nouns/actions are the train-derived subset**.
- Files: `egoper_vqa/{ek100_tea.py (pipeline+decoders+load_clip), ek100_tea_viz.py (montage),
  ek100_tea_eval.py (verb-accuracy)}`. Tea videos extracted to `egoper_vqa/tea_clips/` (the
  `datasets/egoper` NTFS mount is **read-only** — extract `tea_videos.zip` to a writable dir). Figures →
  `egoper_vqa/figures/`.
- **Qualitative (montage):** predicts the right verbs (pour/take/insert/mix) on normal tea actions AND
  flags the anomalies — "put mug in **microwave**" → `turn-on/press`, noun microwave; "stir with **knife**"
  → noun `knife/scissors`. Only gap is EPIC vocab (tea→coffee, mug→cup, honey→oil).
- **Quantitative (`ek100_tea_eval.py`, 28 tea videos, 220 GT action segments, hand-built EPIC-validated
  `TEA_VERB_MAP`):** **top-1 verb acc 0.61, top-5 0.96** vs ~0.18 random — zero-shot, no fine-tuning.
  Per-action top-1 high for measure-water 0.95 / trash-teabag 0.92 / hold-cup 0.82 / pour 0.76; low for
  "place tea bag" 0.05 (predicts open/close of the wrapper; top-5=1.0) and "check water temp" 0.14
  (genuinely subtle). Caveat: anticipation +1 s alignment; verbs are the fair metric; map is subjective.

### egoper_vqa/ — video-QA inference baseline on EgoPER
Test dir for running a SOTA-class video LLM on EgoPER long clips with user-defined
questions, scored against GT error labels. Uses the project `.venv` (no new installs).
- Files: `egoper_vqa/{egoper.py (data+frame sampling), vqa.py (Qwen2.5-VL wrapper), inference.ipynb, README.md}`.
- **Model = `Qwen/Qwen2.5-VL-7B-Instruct`, 4-bit nf4** (native in transformers 5.10.2,
  runs on the Blackwell 5060 Ti, ~5-6 GB resident). Frames passed as PIL list → no
  `qwen_vl_utils` dependency.
- **Why not LLaVA-Video-7B-Qwen2 (the leaderboard pick):** LLaVA-NeXT pins torch==2.1.2 +
  a frozen old transformers commit → can't share this env's torch 2.12/cu132 + transformers
  5.x, AND torch 2.1.2 has no Blackwell (sm_120) kernels so it can't use the GPU at all.
  Isolated-venv off-ramp (with torch override) documented in the dir README.
- **SF-LLaVA distinction:** v1 (arXiv 2407.15841, the `apple/ml-slowfast-llava` repo) is
  *training-free* but bound to LLaVA-NeXT (torch==2.1.2 → no Blackwell kernels, can't run
  on the 5060 Ti). v1.5 (arXiv 2503.18943) is the *trained* successor with no public weights.
- **`vqa.ask_slowfast()`** implements the training-free SF-LLaVA *v1 scheme* (slow = few hi-res
  frames + fast = many lo-res frames, two video streams concatenated) directly on Qwen2.5-VL —
  no LLaVA-NeXT, no training, runs on this card. Defaults 16 slow @360×640 + 96 fast @200×200
  ≈ 4.8K tokens (~same VRAM as 32 uniform frames) but 112 frames coverage. Two-video plumbing
  verified with processor-only smoke test.
- **Procedure grounding (`egoper.procedure_text(task)`):** prepends a paper-accurate task-graph
  explanation + lists the steps, passed to the model via the `context=` arg of `vqa.ask`/`ask_slowfast`.
  Sources: `task_graph.txt` (per-task DAG: `Edges`/Start/End, node index = action id) topologically
  sorted, named from `<task>_normal_actions.txt` (`Action_N` index == action id in `action2idx`).
  Per EgoPER paper (Lee et al. CVPR 2024, Sec. 3.1): the task graph "encodes all possible ways the
  recipe could be made" (multiple valid orderings, not one fixed sequence); **error = any deviation
  from the graph** — Omission/Addition/Modification/Slip (+Correction). NOTE: EgoPER's own EgoPED
  method assumes NO task-graph access at train/test; feeding the graph to a VLM is our deviation, so
  it's not directly comparable to their benchmark numbers.
- **Observed :** model hallucinated generic coffee knowledge
  ("didn't tamp grounds" — not a pour-over step), missed all 3 real errors, AND false-positived the
  normal control. Motivates procedure grounding; watch whether it fixes the normal-clip false positive.
- Project stack confirmed: only `video_qa/train.py` (Qwen2.5 QLoRA, ok on ~4.43+) and
  `vjepa2/notebooks/vjepa2_demo.py` (HF `AutoModel`/`AutoVideoProcessor`, needs transformers
  ≥4.52 for V-JEPA 2) touch transformers; core training uses the *local* vjepa2 pkg + `checkpoints/vitl.pt`.

### egoper_probe/ — frozen V-JEPA 2 feature probe (THE deliverable)
The V-JEPA evaluation centerpiece. `extract.py` runs frozen V-JEPA 2 ViT-L over sliding
windows of EgoPER videos → mean-pooled 1024-d feature + `[start,end]` + error label
(window overlaps a GT error segment) cached to `features/<task>/<vid>.npz`. `probe.py`
trains a logreg probe (split by video via GroupShuffleSplit), reports window/video-level
ROC-AUC/AP, logs to mlflow. Reuses `video_qa/model.py::build_encoder` (`checkpoints/vitl.pt`).
- Run: `extract.py --task coffee --stride_s 2` (GPU; ~45 min for 68 coffee vids) then `probe.py --task coffee`.
- Feature = per-window mean-pool (simplest); upgrades = keep token grid / tiny temporal head.
- Supervised (uses error vids); one-class on normal-only vids = more EgoPER-faithful follow-up.
- `--stride_s` must be ≤ window length or brief errors fall between windows.
- **Extraction speed:** decode each video ONCE at reduced res (decord `width/height`), gather all
  unique frame indices in one batched `get_batch`, center-crop unique frames ONCE (pure slice, no
  PIL resize), normalize on GPU, encode in `--batch_windows` (32) GPU batches under **bf16 autocast**.
  This took 1 video from ~130s → ~45s (decode ~14s + encode ~30s); peak GPU ~5.3 GB. fp32 was the
  bottleneck, NOT decode or I/O — bf16 autocast on the frozen encoder is the key win.
- **RESULTS (2026-06-10, coffee, 68 vids → 14,744 windows, 3.9% error windows, 10-seed
  GroupShuffleSplit test_frac=0.3):**
  - `[supervised]` window ROC-AUC = **0.745 ± 0.040**, AP = 0.121 ± 0.035 (~3× base rate),
    video ROC-AUC = 0.678 ± 0.099 (noisy: only ~21 test vids/split).
  - `[one-class]` (fit normal-only, Mahalanobis on mean-pool feats) window ROC-AUC = 0.554 ± 0.052
    — barely above chance. Mean-pool + single Gaussian is too lossy for subtle local errors.
  - **Verdict: the V-JEPA thesis holds** — frozen latents linearly separate correct vs. erroneous
    windows at ROC-AUC ≈ 0.75. The working "SOMETHING" deliverable. One-class is the weak spot →
    next upgrade is keeping the token grid (not mean-pool) + kNN/prototype scoring, or a tiny
    temporal head. Logged to mlflow experiment `egoper_probe_coffee`.
- **SlowFast detail view :** `extract.py` now caches TWO views per window from one
  encoder forward (same 45s/video): `feats` (mean-pool 1024-d) + `feats_sf` (24,576-d SlowFast:
  slow = temporal-mean→4×4 spatial pool = 16 tokens preserving WHERE; fast = spatial-mean keeping
  all 8 temporal slices = 8 tokens preserving WHEN). `probe.py --view feats_sf`. Cache 1.5 GB/task.
  - Naive concat at C=1 OVERFITS (AUC 0.659 < 0.749 mean-pool). Strong L2 fixes it — sweep is
    monotonic: **C=0.001 → win AUC 0.779 ± 0.049, AP 0.163 ± 0.026** (vs mean-pool 0.749/0.132),
    vid AUC ~0.71. So spatial/temporal detail DOES carry extra error signal (+0.03 AUC/AP) when
    properly regularized. Best probe config = `--view feats_sf --C 0.001`.
  - One-class stays ~0.60 regardless of view/scorer: kNN-on-meanpool 0.606, kNN-on-SF 0.610,
    localized max-over-tokens cosine-kNN vs 60k normal-token bank 0.601 (all >> Mahalanobis 0.554,
    so kNN scorer itself was the only one-class win). Unsupervised detection of subtle errors
    remains the hard open problem — next ideas: per-position token banks, temporal context (window
    sequences), or task-step conditioning.
  - sklearn logreg on 24k-d is CPU-only, ~1 min/fit (~20 min for a 4×5 sweep); switch to a torch
    linear head on GPU if sweeping more.

### exprt_probe/ — anomaly classification on the eXprt tea dataset (V-JEPA 2 + EK100 transfer)
Second probe experiment, on the **eXprt** dataset (`datasets/eXprt-Daten/CAM1 Aufnahmen Patrick/`):
tea-making, PNG frame sequences 1886×1056 @ 20 fps (orig 23.96 Hz, downsampled), **40 trials = 8 classes × 5 iters**.
**VIEWPOINT (important): CAM1 is a FIXED THIRD-PERSON wide shot** (subject stands across the room, full
body visible) — NOT egocentric. This is the opposite of EPIC-Kitchens/EgoPER (head-mounted, hands fill
frame) and is the main reason off-the-shelf EK100 transfers poorly to eXprt (see qual finding below).
(`Normal` + `2tb 2stir`, `Spüli`, `glass and fork`, `no tea bag`, `not enough water`, `perplexity`,
`sequence`). Goal: classify anomaly vs normal (binary) + 8-way error type. Files mirror egoper_probe:
`dataset.py` (mapping), `extract.py` (token grids), `head.py` (EK100 head), `pool.py` (pooled feats),
`train.py` (probe), `README.md`. mlflow experiment `exprt_anomaly`.
- **Labels are video-level only** (one `start_time` per trial = wall-clock, NOT within-video
  localization). Trial→video had no id column + aborted re-takes (43 dirs vs 40 trials); PNGs carry no
  capture date (mtime = rclone copy time), so `dataset.py` maps via the **dir-name timestamp**:
  bucket each dir to the most-recent preceding trial start, keep the longest recording (drops aborts).
  Verified clean 40↔40, 5/class. Persisted to `exprt_probe/mapping.json` + a `video_id` column added
  to the CSV (frame folders untouched).
- **Model:** frozen V-JEPA 2 ViT-L (NOT AC — AC needs robot actions, no classifier head). Head =
  `AttentiveClassifier` (`vjepa2/src/models/attentive_pooler.py`) warm-started from the **EK100**
  attentive probe (`checkpoints/ek100-vitl-256.pt`, dl.fbaipublicfiles.com/vjepa2/evals/; pooler
  depth 4 / 16 heads / 1024-d, 49/49 params load, "action" query kept, fresh linear). EK100's verb/
  noun/action linears discarded. `--train_pooler` would fine-tune the pooler (needs token grids).
- **Pipeline note (perf):** caching full token grids `(n_clips,2048,1024)` fp16 = 9.8 GB; training a
  head over them is **I/O-bound** (GPU idle, ~70 min/cold-fold). Fix: `pool.py` runs the frozen pooler
  **once** → one 1024-d vector/clip (`pooled/`, 10 MB) → `train.py` is then seconds. `pool.py --pool
  {ek100,mean,rand}` makes EK100 / plain-mean-of-tokens / random-pooler caches.
- **Eval lesson (important):** **clip-level classification fails** — with whole-video labels and only
  5 videos/class the probe overfits recording-specific appearance (in-sample AUC ~1.0, **held-out OOF
  ~0.5**), and most "anomaly" clips are normal-looking tea-making. So the deliverable is **video-level**
  (`train.py --level video`): mean-pool a video's clip embeddings → one clean-labelled vector (40),
  LOO (binary) / stratified 5-fold (8-way). `--level clip` (StratifiedGroupKFold OOF) is kept to
  document the negative.
- **RESULTS (2026-06-18, torch linear probe, defaults lr 5e-3 / wd 1e-2):**
  | level | task | mean-pool | EK100-pooled | random-pooler |
  |---|---|---|---|---|
  | video | binary AUC | **0.60** | 0.34 | 0.52 |
  | video | 8-way acc (chance .125) | 0.39 | **0.49** | 0.43 |
  | clip | binary AUC | 0.53 | 0.51 | 0.53 |
  | clip | 8-way acc | 0.17 | 0.18 | 0.16 |
  - **8-way error-type ≈ 0.49 acc (~4× chance) is the strong result**, and **EK100 warm-start helps**
    (0.49 > 0.43 random > 0.39 mean) — the action-recognition pretraining transfers to error-type.
  - **Binary anomaly is only weakly above chance** (~0.60 torch; **~0.70 with sklearn LogReg** lbfgs/L2,
    which is more robust on 40 pts) and **high variance** (only 5 normal videos → LOO AUC SE ~±0.1).
    Interestingly the EK100 attentive pooler **hurts** binary (0.34) — it compresses toward action type
    and discards the subtle normal-vs-anomaly appearance cues that plain mean-pool keeps.
  - **Verdict:** frozen V-JEPA features carry real *error-type* signal at the video level; *anomaly
    detection* is hard here, bottlenecked by 5 normal videos + no within-video localization (vs EgoPER's
    localized labels → 0.745). Next: more normal videos / temporal localization; or a torch-vs-sklearn
    probe toggle (sklearn more robust at this N, torch gives the mlflow loss curve).

### exprt_probe/qual + ek100_label.py — qualitative action labelling on eXprt (zero-shot EK100)
Manual-annotation workflow (eXprt has no within-video action labels): render a few videos to local mp4
(`exprt_probe/qual/watch/`; `datasets/` is read-only, ffmpeg from PNGs at 20fps so player-clock = real
time), user writes start/stop+action in `exprt_probe/qual/annotations.csv`, then `ek100_label.py` runs the
faithful EK100 pipeline (reuses `egoper_vqa/ek100_tea.py`) over each segment → top verb/noun/action + a
montage, and writes predictions back as CSV columns.
- **KEY QUAL FINDING (zero-shot EK100, ~20 hand-annotated eXprt segments):** transfers **much worse than
  on EgoPER tea** (which scored top-5 verb 0.96). Most predictions collapse to generic `take/put/close`
  and the noun fixates on **`maker:coffee`** (the coffee machine in the background) — because **CAM1 is
  third-person/wide**, so hand-object actions are tiny and the scene dominates. Only big unambiguous cues
  land (one `pour:water`; `bottle`/`oil` for soap; `bag` for teabag). **Cause = viewpoint mismatch, not a
  code bug.**
- **FROZEN-EMBEDDING PROBE (`action_probe.py`, mlflow `exprt_action_probe`):** answers "do the V-JEPA
  *embeddings* carry the action even though the egocentric EK100 *head* fails?" Map each annotated segment
  → an EK100-verb class; logreg on the frozen V-JEPA encoder mean-pool (1024-d). **Leave-one-VIDEO-out =
  0.74** (== segment-LOO; permutation null 0.30, **p<0.003**) vs zero-shot EK100 top-1 **0.05** / majority
  0.42. So the embeddings DO carry a real, cross-video-generalizing signal the head misses — NOT just
  within-video overfitting (video-LOO controls it; in-sample 1.0 is vacuous in 1024-d).
  **Caveats (narrow pilot):** only 19 segments / 1 subject / same room+camera; 4 of 6 classes are
  singletons (LOO can't get them → caps acc ~0.79), so this really shows **pour-vs-put** separability, not
  6-way recognition. Next: label more (populate `take/open/close/mix` + anomaly actions soap/fork/2-teabag),
  ideally a 2nd subject/session to test scene generalization, then video-LOO becomes a real multiclass number.

## Misc

### video_qa/ — Appendix E (Video QA) replication
Custom LLaVA-style training loop aligning the frozen V-JEPA 2 ViT-L encoder
(`checkpoints/vitl.pt`, embed_dim 1024) with Qwen2.5-7B-Instruct (4-bit QLoRA)
for video question answering. Replicates V-JEPA 2 (arXiv 2506.09985) Sec. 14 /
Appendix E: 3-stage visual instruction tuning (1: projector only on image
captions; 2: +LoRA on image QA; 3: +LoRA on video QA).
- Files: `video_qa/{model,dataset,collate,train}.py`, `video_qa/configs/stage{1,2,3}.yaml`, `video_qa/README.md`
- Imports the sibling `vjepa2/` package (reuses `robust_checkpoint_loader`, `vision_transformer.vit_large`, `CSVLogger`). `vjepa2/` is a pristine **git submodule** (facebookresearch/vjepa2 @ 204698b, no local changes — `bitsandbytes` comes from the root `pyproject.toml`, not vjepa2's requirements.txt).
- Encoder always frozen. Visual tokens (8 frames@256px → 1024 patch tokens, avg-pooled to 256 via `spatial_pool_stride: 2`) spliced into text at `IMAGE_TOKEN_INDEX = -200` (LLaVA convention). Loss = CE on assistant tokens only.
- Needs `bitsandbytes` installed in `.venv` (only for the LLM; encoder/projector/dataset logic tested without it). Sanity check: `.venv/bin/python video_qa/train.py --config video_qa/configs/stage1.yaml --dry_run`.
- Deviations from paper (16 GB VRAM): QLoRA instead of full FT, small public datasets, 256px/8 frames default.
