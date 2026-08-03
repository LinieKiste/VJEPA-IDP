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
- **AUGMENTED probe (`action_probe_aug.py`, mlflow run `augmented_k128`):** pixel-space augmentation
  (horizontal mirror + temporal clip-length jitter 0.8-1.2x + random 256-crop from a 288 frame), re-encoded
  through the frozen encoder, k=128 aug views/segment; train only on augmented TRAIN videos, test the clean
  held-out video. **Video-LOO 0.74 -> 0.79** (perm null 0.28, p<0.005) on the FIRST label set (19 seg, 4
  singleton classes). 0.79 was the data-imposed ceiling there: pour 7/7 and put 8/8 perfect (augmentation
  fixed the one put->take miss); the only remaining errors were the 4 single-example classes LOO can't recover.
  PERF NOTE: encoding 2432 clips = ~7 min, but the 200-rep permutation at k=128 dragged total runtime to
  ~98 min — cap perm reps / use a lighter null next time.
- **More labels (2026-06-26, 24 seg, all 6 classes now >=2 videos; `annotations.csv` is now a single free-text
  `label` col, schema-robust in both probe scripts):** removing the singleton cap turns it into a genuine 6-way
  problem (chance 0.21, majority 0.33). **Augmented video-LOO = 0.71 (no-aug 0.62, perm null 0.21, p=0.000,
  k=128 perm=50, runtime ~34 min).** Headline is NOT "0.79->0.71": the old 0.79 was effectively pour-vs-put
  binary (singletons unrecoverable); 0.71 is a real multiclass number. Per-class (LOO confusion): pour 7/7,
  open 2/2, put 7/8 perfect/strong; **mix 1/3, close 0/2, take 0/2 still fail** — 2 videos is the bare minimum
  that LETS LOO learn a class, but for visually-subtle small-hand-motion classes (close lid / take mug, both
  look like a generic "put") holding one out leaves only 1 training video. **Next minimal step: a THIRD video
  each for close + take** (so LOO leaves 2); pour/put/open are already solid.

### Direction after 26.06.2026 supervisor meeting (Notion notes + design dialogue)
Supervisor's 5 short prompts: **CLIP for videos/actions**, **soft labelling**, **EK100 finetuning
possible gains**, **AC / auxiliary (sensor) inputs**, **EK100 outputs anpassen**. Mapped to state:
- **Chosen next thread = open-vocabulary action labelling** (fixes the #1 documented eXprt limitation:
  EK100's fixed EPIC vocab can't express tea nouns — mug/teabag/honey). Model that runs in this env:
  `XCLIPModel`/`SiglipModel`/`CLIPModel` all import fine in transformers 5.10.2 (X-CLIP is video-native
  zero-shot action recognition — the natural pick; SigLIP per-frame is the fallback).
- **KEY DESIGN DECISION (locked): do NOT project V-JEPA embeddings into text space.** That step is
  lossy AND makes the text encoder a confound (you'd measure "how well V-JEPA fits CLIP's geometry",
  not V-JEPA itself). Instead: **soft kNN in V-JEPA's OWN latent space** — retrieve nearest labelled
  segments by cosine on the full frozen embedding (zero backbone info loss), aggregate neighbours'
  human free-text labels similarity-weighted (open-vocab, real nouns). CLIP text encoder is used ONLY
  as a label-vs-label ruler; it sits identically on prediction and GT sides so it cancels across
  comparisons and can't bias the backbone score (DINO-style kNN eval, soft + open-vocab variant).
  Swap the retrieval space (V-JEPA vs X-CLIP-video vs random) under the SAME ruler to isolate the
  backbone's contribution. Metrics (video-LOO): soft relevance@k, pred-label cosine, coarse top-1
  (comparable to the 0.71 probe), shuffle null. Honest limit: native-kNN open-vocab only covers phrases
  already in the gallery (grows as you annotate); genuinely-unseen nouns would need the lossy projection.
  Soft *targets over a fixed class set* were considered and rejected — user wants similarity-weighted
  neighbourhood info + open vocab, not label-smoothed 6-way. (Impl `knn_label.py` drafted but not yet
  committed/run — user paused before running.)
- **AC (V-JEPA 2-AC) gives NO gain for labelling/retrieval:** AC freezes the SAME encoder and only adds
  an action-conditioned *predictor* (needs robot end-effector actions we don't have; trained on Droid
  robot data, won't transfer to third-person kitchen video; it's a planning/MPC tool, not a
  representation tool). Retrieval embeddings are identical to plain V-JEPA 2. AC would only matter for a
  *different* task — action-conditioned predictive anomaly detection with a fabricated action signal
  (ties to the "auxiliary inputs" note), a domain-transfer gamble, not this labeller.
- **Auxiliary inputs = there are none natively** (eXprt is CAM1 PNG frames only); the open question is
  whether to *fabricate* one (best-motivated: a hand-trajectory/pose channel concatenated to frozen
  V-JEPA feats — targets exactly the failing close/take small-hand-motion classes) and ablate its effect.
  Caveat: any video-derived auxiliary adds no new information, only explicit inductive bias.
- Submission will need a **Gantt diagram** (per Notion).

## Direction after 05.07.2026 — Pouring volume/flow estimation (LOCKED, NEW centerpiece)
Supervisor + user chose **ONE** direction to shrink the solution space: **pouring volume / flow
estimation from a frozen V-JEPA 2 backbone** (dropped the V-JEPA-latent *anomaly* thread — the
relevant anomalies are trajectory-based and would need hierarchical JEPA + a hand-trajectory
regressor via a Cosy transform, too big for the IDP). Pouring is "the most innovative field, could
make a nice small publication." Plan (supervisor): **pretrain a probe on simulation volume data →
fine-tune on own-lab data** (rig is plug&play in their lab). All work now lives under **`pouring/`**
(consolidated this session — repo had too many open experiments; `pour_probe/` moved to
`pouring/pour_probe/`, its `ROOT` is now `parents[2]`).

### Datasets for pouring (survey)
- **UWLPD** (UW Liquid Pouring Dataset, Schenck & Fox) — downloaded to `datasets/UWLPD/` = the
  **Real Robot Dataset**, `large_bowls` subset: 5 zips (source→target `bowl←{bottle,cup,mug}`,
  `fruitBowl←{bottle,cup}`), 36 conditions each = **180 sequences** (~61 GB, kept zipped). Condition
  = fill`{empty,30,60,90%}`×profile`{dump,hold,partial}`×motion`{minimal,moderate,high}` (from
  `render_v3/sim_args.txt`; **despite "sim_args"/ROS topics these are real Baxter recordings, actually
  a person pouring in a 3rd-person kitchen**). Per frame (~490 @ 640×480): `data*.jpg` RGB,
  **`ground_truth*.png` = binary liquid mask (liquid = BLUE `(0,0,255,255)` on transparent black —
  `convert("L")` collapses it to ~29 and ERASES it; count `max(RGB)>127`)**. **No mL here** (fill %
  is fixed 30/60/90) → volume/flow **must be derived from the mask** (proxy). The clean **mL** trace
  lives only in UWLPD's separate *Simulated* dataset (`bowl_volume.csv`, per-frame m³) — requested
  from connor.schenck@gmail.com, **not yet in hand**.
- **SimLiquid** (github.com/Jiaviz/SimLiquid, "A Simulation-Based Liquid Perception Pipeline") —
  cloned to `pouring/SimLiquid/`, **SET UP + VALIDATED (2026-07-05)**. A **BlenderProc** renderer that
  generates liquid-in-cup images (960×600 RGB + depth/normals + COCO/BOP) with **per-cup volume labels
  in mL** (`volumes.txt`, mesh `bm.calc_volume()*1e6`). **This is the clean-volume sim-pretrain source.**
  Setup done:
  - `blenderproc` installed via `uv pip` into the **project `.venv`** (NOT a new conda env — user
    prefers one venv per project). BlenderProc auto-installed its own **Blender 4.2.1** to
    `~/blender/` on first run (+ tqdm into Blender's python).
  - Assets: `blenderproc download haven liquid_render/assets/hdri --types {hdris,textures}
    --resolution 1k` → 977 HDRIs (1.5 GB) + 213 complete textures (0.6 GB), 2.1 GB total. Config's
    `hdri_path: assets/hdri` = the haven root (both `hdris/` + `textures/` live under it; the README's
    `assets/haven` name is just cosmetic). Textures capped (full set ~20 GB); code only needs a few
    (random desk material, `self.textures` line ~225). `liquids.blend` copied from `~/Downloads/` to
    `liquid_render/blender_projs/liquids.blend`.
  - **Code fix (committed to the working tree):** `liquid_render.py` imported
    `object_print3d_utils.mesh_helpers` — that 3D-Print-Toolbox addon is **not bundled in Blender 4.2**.
    Inlined its `bmesh_copy_from_object` (world-transform + apply-modifiers bmesh copy → `calc_volume`);
    only thing SimLiquid used from it.
  - **Render VALIDATED:** `blenderproc run liquid_render.py --cfg cfgs/liquid.yaml -et hdri -ns 3 -pn 2
    -o outputs/samples` → photoreal 960×600 images + `volumes.txt` (varied cups: coke/milk/water bottle,
    wine/shot glass, mug; liquids water/juice/wine/milk; e.g. milk 777.3 mL, water 1585 mL), ~1–3 s/frame
    GPU. Samples in `outputs/samples/`. Full dataset = `./render.sh` (1000 scenes ×10 poses ≈ 10k images,
    several hours) — **not yet run** (user paused; 2026-07-06).
  - **pacman Blender CANNOT drive BlenderProc — don't retry:** system Blender is 5.1.2/py3.14 (pacman
    `17:5.1.2-1`); BlenderProc 2.8.0 needs **portable Blender 4.2/py3.11** with the bundled-python layout
    `<blender>/<ver>/python/bin/`. Arch's package builds against system python → no such layout →
    `--custom-blender-path` can't work. So SimLiquid keeps the auto-downloaded portable 4.2.1 in
    `~/blender/` (user confirmed OK; pacman 5.1.2 untouched for other use).
- Also noted by supervisor: MultimodalPouring (github.com/lianghongzhuo, requested), Schenck's
  "Perception & Reasoning about Liquids" 4.5M-image set. Own self-made dataset likely the endgame.
- **Own-lab pouring dataset (recorded 2026-07-13)** — downloaded to `datasets/pouring-lab/`
  (**real dir on the SSD**, NOT a /mnt/storage symlink — active postprocessing work + the ntfs3
  write-hang rules out a 9 GB sustained write to the HDD). Source =
  `tum-nas:/tumw/sgm/02_Studierende/Wallwitz/Dateneingang/eigene Experimente/` (rclone). 3 GoPro
  views `CAM1/2/3`, **75 `GX*.MP4`** full-res videos (9.2 GB total). Only the MP4s + CSVs were
  pulled; the NAS also has `GL*.LRV` (low-res proxy duplicates) + `*.THM` (thumbnails), skipped
  (~2.2 GB, still on the NAS). **Ground truth = OCR of a display in frame**: per-video CSV
  `frame_number,timecode,display_ocr,ocr_confidence` — but only ONE exists so far
  (`CAM1/GX011267.csv`, a pilot); the rest presumably get generated in postprocessing.
- **NAS layout for processing (2026-07-15): originals are READ-ONLY, outputs go to
  `Datenverarbeitung`.** Processed/new files → `tum-nas:/tumw/sgm/02_Studierende/Wallwitz/
  Datenverarbeitung/` (created next to `Dateneingang`); NEVER write into `Dateneingang/eigene
  Experimente/`. All 227 original files there carry the DOS read-only attribute (NAS refuses
  overwrite/delete with NT_STATUS_ACCESS_DENIED) — set via `smbclient //nas.ads.mwn.de/tumw -U
  "ads.mwn.de\\ge35ral%$(rclone reveal <obscured pass from rclone config dump>)"` + `setmode <f> +r`
  (rclone can't set SMB attrs; needs the TUM VPN up; undo with `setmode <f> -r`). Local copy
  `datasets/pouring-lab/` is `chmod -R a-w` too. Integrity manifest (size+modtime of all 227):
  `pouring/nas_manifest_eigene-Experimente.txt` — `rclone lsl` the NAS folder and diff to detect
  any accidental change.
- **`OCR_Scale_REader/`** = supervisor's repo (cloned 2026-07-15): GoPro rename + undistort +
  scale-display OCR pipeline (generates the per-frame `display_ocr` CSVs).
- **`pouring/clip_split/`** = split trials into per-pour clips (gated workflow, plan in
  `~/.claude/plans/silly-cooking-sketch.md`). `trials.py` → `trials.csv` (27 trials; user marked
  8 `exclude`, filled `source_obj`/`target_obj` = kettle/teapot/bottle → blue_mug/white_mug/
  glass/ikea_glass; **19 usable**). `run_ocr.py --roi` (interactive ROI per CAM1 video →
  `rois.json`) then `--trials N`/`--all` → `ocr/<stem>.csv` @ ~30 Hz. `detect_pours.py` → events +
  `qc/trace_*.png`. **SCALE PHYSICS (critical, from user):** cup tared on scale ≈0 g (sometimes
  2 g) → pour ramps up → plateau = poured-mass GT → cup removed: scale goes NEGATIVE but OCR
  can't read the minus sign → bogus large positive spike (e.g. -347→"347"). A real pour must
  rise out of a STABLE ~0 baseline; rises from plateau/chaos = removal artifacts, always ignored;
  clip must end before plateau end (removal start). Validated on pilot GX011267 = exactly 1 pour,
  142 g. **Trace cleaning (final, user-chosen) = MONOTONICITY PRIOR (`--filter mono`, default):**
  while pouring, true weight never decreases → sample valid iff ≥ trailing-window rolling median
  − tol (else dropout reading LOW: lost leading digits) AND ≤ leading-window rolling median + tol
  (else digit-BLENDING spike reading HIGH — these exist, per user). TWO window scales (0.15 s +
  0.4 s; lower bound = max of past medians, upper = min of future medians): long window survives
  long dropouts, short window catches dips right after a level change (pour onset). Final step =
  **isotonic regression** (sklearn, non-decreasing) over each detected pour interval → w_f inside
  a pour is monotone BY CONSTRUCTION (single-frame ambiguities like 23→2→24 can't survive). Robust order statistics, no σ
  estimation; catches BOTH spike signs (the EMA z-score approach couldn't kill positive blends).
  Sustained wrong levels (unsigned-negative removal stretch) survive the filter by design — the
  zero-baseline event rule discards them downstream. `--filter ema` (supervisor's sketch, tuned:
  negative-only reject, trimmed σ w/ 1 g floor, 2 passes) and `--filter median` kept for A/B.
  **OCR runs at FULL native rate (every frame, 120 Hz on CAM1)** — user wants max resolution, no
  subsampling. Detection boundaries were near-identical across all filter configs tested. **CAM1 =
  scale cam @ 119.88 fps; CAM2/3 @ 29.97 fps.**
  **OCR BACKEND (Gate C finding 2026-07-15): supervisor's backends FAIL on the wide-shot CAM1
  crops (~170×80 px display)** — tesseract reads 3–25% of frames w/ garbage >500 g (7-seg digits =
  disconnected strokes; pilot trial 6 only worked because its CAM1 was a close-up), segment_ocr's
  Otsu breaks on uneven display lighting, and the template backend assumes a fixed grid but the
  display DRIFTS several px within a trial (scale nudged between pours). Fix = **own
  `clip_split/lcd_ocr.py` (`--backend lcd`, now default)**: static camera → per-pixel background
  model = p90 over 150 gain-normalized sampled frames (unlit segment = BRIGHT state; p90 survives
  mostly-lit segments AND hand shadows — a plain median does NOT), read on the continuous ratio
  image (bg−frame)/bg, per-cell correlation against 10 synthetic 7-seg templates over ±8 px window
  (absorbs drift), cells auto-calibrated by multi-scale "8"-template scan of the lit-ever (p90−p10)
  image; blank cell = low lit-mass; ANY unreadable non-blank cell rejects the whole frame (silent
  digit drops would turn 356 into 35). Valid rates 46%/88%/69% (trials 5/25/26; ≥99% plausible),
  ~2 ms/frame = ~60 s/video (tesseract: 20–30 min). Needs `tesseract-data-eng` (pacman) only for
  the legacy backend — Arch ships tesseract without eng data.
  **DETECTOR (redesigned on the dense full-batch traces, 2026-07-16): `--detector chain` (default)
  = plateau-chain.** The Gate-C region logic (rise from stable ~0) broke on dense traces: pours
  start from small STANDING levels (residual water, no re-tare), slow spurt-pours read as
  staircases of <5 g steps, and transition misreads fed fake evidence. Chain model: stable
  plateaus → pour = maximal ASCENDING chain from a settled start (dwell ≥ baseline_win) to a
  settled end (dwell ≥ settle_dwell 2.5 s); weight = end − start level (true poured mass even
  un-tared); VIRTUAL start plateaus at falling-gap turnaround minima (fast cadence = baseline too
  short to register); chain breakers: falls, ramp-evidence failure for steps >100 g (raw mid-range
  readings ≥10 spread ≥0.35 s — removal jumps flip in ~1 display refresh), drawdown >8 g on a
  0.3 s-median (scale RINGING after removal overshoots: 304→383→356 unsigned), and the **CUP
  BARRIER**: any step LANDING at the trial's cup constant (modal high plateau, e.g. 346/356 —
  unsigned −cup after every removal) is the removal; ends the pour at the previous plateau
  (rescued trial 25's 278 g pour whose chain ran into the removal). `near_cup_level` flag marks
  events ending ≈cup for audit. `--detector regions` (Gate-C logic + auto-baseline) kept — more
  robust on low-validity traces; **trial 5's events come from the Gate-C-validated run** (the
  45%-valid blurry trace flip-flops between detector versions).
  **Full-batch state (Gate D packet): OCR 92.1% valid overall (13/19 trials ≥99.6%), 115 events →
  `events.csv` w/ `exclude` col: trial 4 excluded (blurry portrait, 2.9% valid, scattered misreads
  — single ~193 g warm-up pour, hand-labelable), trial 5 pour 6 excluded (truncated at video end,
  user call). = 113 usable pour clips, 8–362 g, all 12 source→target combos.** Weights sanity:
  per-trial lists in the Gate D report; audit via `qc/boundaries_*.png` + `qc/trace_*.png`.
  **INTERACTIVE ANNOTATOR (user-requested, replaces the static Gate D audit): `annotate.py` +
  `annotate_ui.html`** — stdlib-only local web app (`.venv/bin/python pouring/clip_split/annotate.py`
  → localhost:8765; Range-capable video serving so the browser scrubs CAM1 natively). Trial sidebar
  w/ completion badges → CAM1 video + zoomable trace timeline (canvas): auto-detected events as
  draggable spans (edges drag; [ / ] snap to playhead), per-event weight input, add-event-at-playhead
  (n), completed (c) / excluded toggles, delete. State = `annotations.json` (initialized once from
  events.csv; events.csv untouched; autosaved, atomic writes). **Weight override semantics: per-clip
  GT curve rescaled w' = base + (w−base)·W_user/(plateau−base) (monotone-preserving; live orange
  preview in the UI); measured rise < 5 g (garbage trace) → synthetic smoothstep ramp 0→W_user at
  export.** `cut_clips.py` now prefers `annotations.json` when present and cuts ONLY completed,
  non-excluded events (user-set weight authoritative, curves rescaled as above). NOTE: user marked
  trial 4 `exclude` in trials.csv themselves (16 annotatable trials + 5/25/26 = 18 in the tool).
  **Per-sample OCR corrections (user-requested full control):** OCR edit mode (`e`) — drag on the
  timeline selects a sample range (click = one ~1/30 s display sample), set value / mark invalid /
  clear; patches stored as time-ranges in `ocr_overrides.json` (OCR CSVs never modified), the mono
  filter RE-RUNS server-side on every patch and the fresh curve returns live. User-set samples are
  PROTECTED from filter rejection (`protect` mask in `detect_pours.load_trace`, which auto-applies
  `ocr_overrides.json` whenever loading by path — detect_pours re-runs AND cut_clips' per-clip GT
  CSVs pick corrections up with zero extra wiring). Raw OCR points drawn as blue dots (yellow =
  corrected, purple @top = off-scale garbage); playhead readout shows t / trace / adjusted / raw
  per frame for frame-by-frame verification against the video.
  **Two reader bugs found via user's trial-26 report (2026-07-16, both fixed in lcd_ocr.py):**
  (1) near-tie digit confusion — "6" vs "8" differ in ONE segment so whole-cell correlation is a
  coin flip (348 read as 346/347 for ~1900 frames); fix = when top-2 templates within 0.10, sample
  only the DIFFERING segments directly (max lit-fraction over ±3 px alignment offsets — a lit
  segment is found at some offset, a dark gap never is). (2) **hand/teapot SHADOWS over the display
  lit up blank cells in the weak ratio mask** → correlation garbage → whole frames rejected →
  trial 26 had NO readings in the post-removal zero stretches; fix = blank test on STRONG mass
  (ratio > ratio_t+0.08, threshold 0.03): shadows never exceed +0.08 (measured 0.000), the weakest
  blurry digit stroke does (0.07). DEAD END (do not retry): a "display slid a cell-pitch" theory →
  frame registration against the bg model — registering vs a mixed-position p98 background
  misaligns good frames, and aligning to an arbitrary reference frame can push digits outside the
  ROI; reverted. Final batch: trial 26 67.5→99.8% valid, trial 5 45→58%, trial 24 96→100%,
  overall 95.2%; 16/19 trials ≥99.2%.
  **DATASET FINISHED (2026-07-16): 121 pour clips, 8–362 g (median 140 g), 2.6–8.7 s, 18 trials,
  all 12 modality combos** — `datasets/pouring_processed/clips/{CAM2,CAM3,csv}/NNNN.*` +
  `clips_manifest.csv` + `README.md` (410 MB), uploaded to
  `tum-nas:.../Wallwitz/Datenverarbeitung/pouring_clips/` (+ `provenance/` = trials.csv,
  annotations.json, ocr_overrides.json). **Final GT CSV = 2 cols `t_s,weight`** (user choice):
  weight = poured mass since clip start, BASELINE-SUBTRACTED + clamped to [0,W] → runs exactly
  0 → annotated final mass (== manifest `weight_g`) for every clip incl. the ~6 residual-baseline
  clips (trial 22 sat at ~6 g, trial 9 pour 3 at 4 g — consecutive pours into an un-emptied cup).
  `cut_clips.py --full-csv` regenerates the 6-col provenance variant (t_s,timecode,wallclock,
  weight_g_raw,weight_g_filtered[absolute reading],ocr_confidence); `--csv-only` rewrites
  CSVs+manifest without re-encoding videos. User annotated ALL 121 events (11 added manually, 133 OCR
  patches, weights human-verified); Gate E sync preview approved. GT quality: every clip's filtered
  curve is monotone BY CONSTRUCTION (isotonic inside each exported window — `--csv-only` flag
  regenerates CSVs+manifest without re-encoding) and its rise equals the annotated weight exactly.
  **This is the fine-tune GT for the pour_probe (sim-pretrain → own-lab fine-tune plan).**
  Downstream stages IMPLEMENTED + smoke-tested on the
  pilot (trial 6): `qc_boundaries.py` (Gate D contact sheets: CAM1 thumbs at clip_start/rise/
  plateau/clip_end per pour), `cut_clips.py` (Stage 4: final layout `datasets/pouring_processed/
  clips/{CAM2,CAM3,csv}/NNNN.*` + `clips_manifest.csv`; clip ids global over events.csv ordered by
  trial wallclock; duration-mismatch → window cropped so it fits EVERY cam; libx264 CRF17 re-encode,
  audio kept for sync checks; timestamps in metadata only — `-metadata creation_time` (mp4 stores
  UTC, whole seconds) + `-timecode` stream tag (carries the millis), both verified on a real cut;
  per-clip GT CSV re-runs the exact mono+isotonic cleaning via `detect_pours.build_parser()`;
  honors an `exclude` column in events.csv from the Gate D audit), `qc_sync.py` (Gate E: CAM1|CAM2|
  CAM3 hstack preview of one pour — CAM1 appears here ONLY, never in final clips). Remaining: user
  ROI session (`run_ocr.py --roi`) → Gate C (OCR 2–3 trials) → full OCR batch → Gates D/E → cut →
  rclone to Datenverarbeitung.
  **`datasets/pouring-lab-renamed/`** = working copy of pouring-lab with
  `video_processing/rename_gopro_videos.py` applied: `YYYYMMDD_HHMMSS_<millis>_GX*.mp4`
  (frame-accurate embedded-timecode start times → cross-camera sync is visible in the names;
  CAM1/2/3 starts differ by <100 ms). Pilot CSV renamed alongside its video. WATCH OUT: the
  script's `main()` is invoked TWICE at the bottom (copy-paste bug) — pipe exactly one `yes`
  into `--execute` or files get double-prefixed on the second pass.

### pouring/pour_probe/ — frozen V-JEPA 2 pouring flow/volume regression (UWLPD)
Mirrors `exprt_probe/`: sliding-window frozen ViT-L token grids (`extract.py`, reuses
`video_qa/model.py::build_encoder`) → frozen attentive pooler (`head.py`/`pool.py`, `AttentiveClassifier`
`num_outputs=1`, EK100 warm-start) → SmoothL1 **regression** linear probe (`train.py`), GroupKFold by
sequence, mlflow `pour_probe`. Windows in FRAME units (real-robot fps unreliable): 32-frame span→16
sampled, stride 16. `dataset.py` reads frames straight from the zips; `attn_map.py` renders the pooler
cross-attention overlay (Stage-1 interpretability gate). **Target = mask-derived PROXY** (`flow` = mean
liquid-mask area/window; `volume` = running-max area) — a smoke test until the real mL arrives.
- **RESULTS (2026-07-05, all 180 seqs → 5,399 windows, GroupKFold-5 by sequence, OOF):**
  | target | features | R² | MAE/baseline |
  |---|---|---|---|
  | flow   | mean-pool  | **0.80** | 0.37 |
  | flow   | EK100-pool | 0.775 | 0.40 |
  | volume | mean-pool  | 0.773 | 0.40 |
  | volume | EK100-pool | 0.754 | 0.41 |
  Frozen V-JEPA linearly predicts the liquid flow/volume proxy at **R²≈0.75–0.80** (MAE ~0.4× the
  predict-mean baseline), generalizing across held-out sequences → **the representation carries pouring
  signal**. Plain **mean-pool ≥ frozen EK100 pooler** (same as exprt binary — the action pooler
  compresses away appearance detail). Attention map of the *frozen* pooler is **diffuse** (not liquid-
  localized) — expected, it wasn't trained on pouring; faithful Stage-1 gate needs `--train_pooler` (a
  pooler trained on the real mL target, deferred). Proxy result, not the deliverable.
- **INFRA LESSON (important):** the `/mnt/storage` **NTFS (ntfs3) driver HANGS on sustained writes** —
  process goes uninterruptible `D`-state in `do_truncate`, un-killable, corrupts the in-flight file
  (crashed extraction at 130/180 and pooling at ~89/180, twice). **Reads are fine.** Fix: **all pouring
  caches now on the SSD** at `/home/casimir/.cache/pour_probe/{features,pooled,pooled_mean}` (22 GB;
  override via `$POUR_FEATURES_DIR`/`$POUR_POOLED_ROOT`). Don't write datasets/caches to `/mnt/storage`.

### pouring/pour_probe/ — OWN-LAB clips probe (real gram-weight GT — THE deliverable result)
`clips_extract.py` + `clips_train.py` + `clips_viz.py` = probe of frozen V-JEPA 2 on the finished
own-lab clip dataset (`datasets/pouring_processed/clips/`, real weight GT). `clips_extract.py`:
sliding 1.0 s windows (stride 0.5 s, 16 frames, short-side-256+center-crop — the pour sits centered),
frozen ViT-L → **mean-pool 1024-d** (fast "does it work" path); per-window targets from the per-clip
`t_s,weight` curve — `volume` = weight at window centre (cumulative poured mass), `flow` = Δweight/window
(g/s). Cache `~/.cache/pour_probe/clips_feats/{CAM2,CAM3}/<id>.npz` (5.5 MB, both cams, ~3 s/clip).
`clips_train.py`: ridge probe, **GroupKFold by TRIAL** (clips of one trial share scene/container),
mlflow `pour_probe_clips`. Honest baselines (appearance methods get NO clock): `motion` = frame-diff
energy, `time` = window-centre time only, `mean` = floor, `vjepa_shuf` = train-label-permuted null.
- **RESULTS (2026-07-17, 121 clips / 18 trials, GroupKFold-6 by trial, OOF, ridge α=100, CAM2):**
  | target | method | R² | MAE/base |
  |---|---|---|---|
  | **flow** | **vjepa mean-pool** | **0.712** | **0.49** |
  | flow | time_prof (norm-time poly4) | 0.622 | 0.53 |
  | flow | motion-energy | 0.183 | 0.87 |
  | flow | time (raw, linear) | 0.005 | 1.00 |
  | flow | shuffle null | −0.321 | — |
  | volume | time_prof | 0.563 | 0.56 |
  | volume | time (raw, linear) | 0.784 | 0.40 |
  | volume | vjepa mean-pool | 0.364 | 0.73 |
  **HEADLINE: frozen V-JEPA reads pouring FLOW RATE from appearance at R²≈0.71 (MAE 13 g/s, ~½ the
  predict-mean baseline), decisively beating a motion-energy baseline (0.18) and edging the STRONG
  temporal-profile prior (0.62); shuffle collapses (−0.32) → real, appearance-driven, generalizes to
  held-out trials.** Predicted flow curves track the GT bell shape per clip (`qc_probe_flow_CAM2.png`).
  **CORRECTION to an earlier overstatement:** "time carries no flow signal" was WRONG — it was true
  only for RAW time + a LINEAR model (0.005; flow is bell-shaped so a line is flat). The FAIR temporal
  baseline = time normalized to [0,1] WITHIN each clip + deg-4 poly (the average pour profile) →
  R²=0.62 on flow, close to V-JEPA. V-JEPA's edge over it is modest BUT real for two reasons the number
  hides: (1) `time_prof` needs the clip's start/end (the pour segmentation) to normalize — V-JEPA needs
  no boundaries; (2) `time_prof` only predicts the MEAN profile, can't tell a fast pour from a slow one
  at the same phase — V-JEPA sees the actual stream. So the honest claim is "V-JEPA beats naive motion
  4× and matches/edges a boundary-cheating temporal prior, with capabilities the prior lacks."
  **VOLUME is time-dominated**: raw-time-linear alone gets 0.78 (cumulative mass ↑ monotonically with
  time), V-JEPA only 0.36 — absolute fill-level from a wide 3rd-person shot is hard; volume is better
  read off the clock. Adding CAM3 didn't help flow (both-cams 0.64 < CAM2 0.71); CAM2 is the better view.
- **UWLPD comparison (loose, NOT apples-to-apples):** the 2026-07-05 UWLPD run got flow R²≈0.80 /
  volume 0.773 (mean-pool) — but those targets were mask-derived PROXIES (`flow`=mean visible-liquid
  PIXEL AREA/window, `volume`=running-max area), not real grams. So both experiments say "frozen
  V-JEPA linearly predicts a pouring signal at R²~0.7–0.8 across held-out groups", but they measure
  different quantities (pixel area vs Δweight/s). It's a consistency signal, not a benchmark match.
  Next comparisons: MLP head (recover the damped flow peaks — ridge is linear), EK100 attentive pooler
  (needs token-grid re-extraction), CAM3-alone, then SimLiquid sim-pretrain → own-lab fine-tune.
- **ATTENTIVE PROBE = the real V-JEPA 2 eval protocol (2026-07-17, `clips_train_attn.py`, mlflow
  `pour_probe_clips_attn`):** full `AttentiveClassifier` (depth-4 pooler: 3 self-attn Blocks +
  CrossAttentionBlock w/ 1 query, 16 heads, 1024-d, `complete_block=True`) + linear(1024→1),
  pooler warm-started 49/49 from the EK100 action query (`head.py`). Trained ON TOP of the frozen
  ViT-L with the **encoder IN-LOOP** so augmentation is pixel-space: h-flip (mirror the pour) +
  random 256-crop from a cached 288 frame + ±0.1 s temporal jitter. AdamW lr 1e-3 cosine, SmoothL1,
  ~1.5 min/epoch, **41 epochs / 61 min**. Split = 4 held-out TRIALS (8/13/21/24, spans all sources),
  818 train / 267 val windows; baselines evaluated on the SAME trials.
  - **RESULT (flow, held-out val): attentive probe R²=0.899, MAE 6.99 g/s** — vs ridge mean-pool
    0.765 / 12.7, temporal-profile 0.597 / 16.2, predict-mean −0.01 / 28.1. Val R² climbs smoothly to
    a 0.897–0.899 plateau over the last ~8 epochs (converged, not overfitting; best epoch 37).
    Predicted flow curves now TRACK THE GT BELL PEAKS (`qc_attn_flow_CAM2.png`) — the nonlinear
    attentive head recovers what linear ridge damped. **So the real V-JEPA 2 probe reads pouring flow
    at R²≈0.90 (MAE 7 g/s), a large jump over the linear quick-check (+0.13 R², MAE ~halved), and far
    above any non-V-JEPA baseline.** Infra: 288-frame cache `~/.cache/pour_probe/clips_frames288/`
    (4.5 GB, `clips_grid_cache.py`); best ckpt `~/.cache/pour_probe/attn_flow_CAM2_best.pt`; encoder
    in-loop peak ~9.7 GB VRAM. Not yet done: volume target, k-fold CV of the attentive probe (single
    split → val is 4 trials, so ±noise; the ridge 6-fold OOF was 0.712 for context).
  - **VIEW / TRANSFER STUDY (2026-07-17) — complete picture, all on the same held-out trials:**
    | probe | R² | MAE |
    |---|---|---|
    | CAM2 native (train+eval CAM2) | 0.899 | 7.0 g/s |
    | CAM3 native (train+eval CAM3) | 0.804 | 10.4 g/s |
    | CAM2 → CAM3 zero-shot transfer | 0.524 | 15.9 g/s |
    | CAM3-native ridge / time_prof baselines | 0.593 / 0.597 | — |
    1. **Both views support a strong probe on their own** (0.80–0.90), each well above baselines —
       so the frozen features carry the pour signal from either angle. **CAM3 is modestly WEAKER
       and less stable** than CAM2 (peaks R²≈0.80 at epoch 15 then wobbles 0.72–0.78; CAM2 climbed
       smoothly to a 0.90 plateau) — CAM3 is the harder angle (more head-on/distant, body occlusion).
    2. **Zero-shot CAM2→CAM3 transfer FAILS** (0.524, *below* CAM3's own linear ridge 0.593) — the
       attentive probe is VIEW-SPECIALIZED. `clips_attn_map.py` shows why: the CAM2-trained pooler's
       cross-attention (rendered on CAM3 frames, `qc_attn_map_transfer_flow.png`) is pulled to the
       person's TORSO/upper body — a learned "look upper-left where the arm is" habit that catches the
       kettle in CAM2's framing but lands off the pour in CAM3's. (Caveat: attn maps are diffuse +
       only the final cross-attn layer, so suggestive not definitive; the numbers are the proof.)
    So: a per-view probe works (~0.80–0.90); a cross-view probe needs training on BOTH views (or a
    view-invariance term) — the CAM2 model won't translate to a new angle unadapted.
    Files: `clips_train_attn.py --cam CAM3`, `clips_eval_crossview.py`, `clips_attn_map.py`,
    `clips_viz_attn.py`; ckpts `~/.cache/pour_probe/attn_flow_{CAM2,CAM3}_best.pt`.
  - **VIEW-ROBUST (both-cam) PROBE + WATER-TRANSIT LAG (2026-07-17):** user dropped the per-view
    thread → train ONE probe on CAM2+CAM3 pooled (`clips_train_attn.py --cam both`, `build_windows`
    now takes a `cams` dict {cam:{clip:data}} tagging each window with its cam; per-camera eval of the
    best ckpt is the robustness test). A single pooled model holds both views: **both-cam flow (lag 0,
    60 min) = combined R² 0.812, CAM2 0.827 / CAM3 0.797** — CAM3 ≈ its native 0.804, CAM2 gives back
    some of its native 0.899 (price of one model vs the view-specialized transfer that FAILED at 0.524).
    Both baselines (ridge 0.647 / time_prof 0.631) far below.
  - **WATER-TRANSIT LAG (the physics fix):** the scale registers poured mass ~0.7 s AFTER the stream is
    visible (fall time ~0.2 s + the digital scale's own load-cell filtering & few-Hz display refresh —
    our GT is OCR of that display), so the weight-derived flow GT is a DELAYED copy of what V-JEPA sees.
    `clips_lag_sweep.py` MEASURES it for free: the cached features don't depend on the lag, only the
    target does, so hold features fixed, recompute flow targets sampled `t+tau`, refit the SAME OOF
    6-fold trial-grouped ridge per tau. **The R²-vs-tau curve is ASYMMETRIC — sampling the target LATER
    helps, earlier hurts monotonically (pure smoothing would be symmetric) → a real transit delay, not
    noise-averaging.** Optima: pooled both-cam **tau*=+0.70 s (ridge OOF R² 0.640→0.730, +0.090)**;
    per-view CAM2 +0.65 (0.711→0.772), CAM3 +0.80 (0.616→0.723); **volume is unaffected** (~0, slow
    monotone integral dominated by the clock — only the flow DERIVATIVE, with sharp onset/offset, cares).
    `--lag_s` added to `clips_train_attn.build_windows` (shifts only the target to t+lag; default 0 =
    non-breaking; lagged runs get distinct ckpt/run names `attn_flow_both_lag0.7`).
  - **TRAINING-RECIPE FIX (2026-07-17, important — the first both-cam runs were mis-trained):** the
    early both-cam runs' val R² BOUNCED 0.66–0.83 to the end and never plateaued. Metric autopsy (mlflow
    train_loss/lr/val_r2 across all runs) found the cause: the cosine LR **never annealed** — the old
    code sized `T_max` from epoch-0's *train-loop-only* time (val eval excluded), which badly overshot on
    the 2× both-cam set (predicted 38 epochs, ran 24), so LR ended at ~3e-4 (31% of peak) and the model
    kept taking big steps forever. Proof: the CAM2 single-cam run annealed LR fully to 0 and converged to
    a FLAT 0.892–0.899 plateau (σ≈0.003); the both-cam runs (LR ended high) bounced. Two more faults: NO
    LR warmup → AdamW jumps straight to 1e-3 on the warm-started pooler → train_loss spikes to 22–131 at
    step 0–4 → first 1–2 epochs wasted (negative val R²); and "best" = argmax over a noisy 4-trial val =
    a lucky-epoch cherry-pick. **Fixes (all in `clips_train_attn.py`):** (1) linear warmup over epoch 0
    (0→peak) kills the startup spike — epoch-0 val R² went −0.24 → **+0.67**; (2) cosine sized from the
    REAL full-epoch wall time (incl. val) at 92% of budget, so LR reaches 0 ~2 epochs early → a stable
    low-LR plateau tail; (3) checkpoint on the 3-epoch ROLLING MEAN of val R², not a single spike;
    (4) report a FRESH eval of the saved ckpt (combined + per-cam) + a `val_tail_mean±std` so convergence
    is visible. Peak LR kept at 1e-3 (CAM2 proved it converges when warmed+annealed — the schedule was
    the bug, not the magnitude).
  - **FINAL lag-corrected both-cam probe (tau=0.70, FIXED recipe, 3 h → converged by ~132 min):**
    **combined R² 0.854, CAM2 0.874 / CAM3 0.834, val tail 0.854 ± 0.000** (`attn_flow_both_lag0.7_best.pt`,
    epoch 35). The bouncing is GONE — dead-flat plateau for 18 epochs at LR=0, exactly like the CAM2 run.
    vs the mis-trained 90-min run (0.829, CAM2 0.831/CAM3 0.827, bouncing): **+0.025 combined AND real
    convergence.** The pooled model now gets CAM2 to 0.874 (near its native 0.899) and **CAM3 to 0.834 —
    ABOVE CAM3's own native single-cam probe (0.804)**: one view-robust model is genuinely strong on both
    angles, and the earlier "CAM2 pays a price for pooling" was largely a training artifact, not a ceiling.
    Baselines on the same split: ridge mean-pool 0.647 / time_prof 0.631 / predict-mean ~0. **The clean
    quantitative proof the LAG helps is still the OOF ridge sweep (+0.090); the attentive run confirms it
    carries to the full probe.** Note: converged ~132 min then 50 min at LR=0 unchanged → ~2 h suffices.
    Files: `clips_lag_sweep.py` (+ `qc_lag_sweep_flow_{CAM2,CAM3,both}.png`).
  - **WEIGHT-RECONSTRUCTION viz (`clips_viz_weight.py` → `qc_weight_recon_CAM2.png`):** integrate the
    probe's predicted flow → cumulative poured mass, overlay on GT weight for held-out clips (reconstruction
    placed at window-centre+lag to undo the training lag). Shape + onset timing track GT cleanly; **integrated
    TOTAL mass vs GT = median 14 g / mean 22 g error over 30 val clips (GT 31-335 g)**. HONEST: the probe
    compresses extremes (under-predicts big/fast pours, over-predicts the tiny one) — SmoothL1 on per-window
    flow regresses to the mean and integration accumulates that bias; instantaneous flow shape is excellent,
    absolute total drifts on the tails.
  - **COMPRESSION INVESTIGATION (2026-07-18, `clips_bias_diag.py` / `clips_attn_slope.py` /
    `clips_calibrate.py`) — the short-over/long-under bias is largely IRREDUCIBLE; DON'T re-try these.**
    Measured as a slope (pred-total ~ a + b*GT-total; b<1 = compression). Deliverable SmoothL1 model:
    total MAE **23.0 g**, slope **0.927** (mild — the eye-catching worst clips are the tails of a mostly-good
    fit). Findings:
    (1) **Ridge alpha sweep (cheap proxy, OOF 6-fold):** less regularization reduces the flow-shrinkage
    (flow slope 0.69→0.83) BUT plateaus at ~0.83 and never reaches 1.0 even as held-out R² collapses to
    NEGATIVE — proof the residual is aleatoric (absolute g/s not fully visible from a wide 3rd-person shot),
    not timidity. In-sample slope DOES →1.0 (0.995) as alpha→0 while OOF stalls at 0.83 = textbook
    overfitting-vs-generalization split (`clips_bias_diag.py`).
    (2) **MSE + lower weight_decay retrain (120 min, `attn_flow_both_lag0.7_mse_best.pt`):** de-hedging
    worked on bias (total slope 0.927→0.960) but is a NET LOSS — total MAE 23→31 g, flow R² 0.854→0.799,
    converged to a WORSE 0.740 plateau. Bias-variance trade went the wrong way: committing harder on an
    ambiguous view = committing to noise. `--loss {smoothl1,mse}` added to clips_train_attn (mse tag
    `_mse`; also folded warmstart status into the ckpt tag so `_noWS` runs don't clobber).
    (3) **Post-hoc calibration (linear rescale) does NOT help on held-out data:** leave-one-val-trial-out
    on the deliverable 23→28 g (WORSE); nested-CV on the ridge OOF (121 clips) 35.4→35.5 g (no change,
    slope 0.848→0.809). The bias isn't a clean global slope you can correct — it's per-clip/ per-container
    idiosyncratic + the aleatoric floor.
    **VERDICT: the SmoothL1 deliverable (total MAE 23 g, slope 0.927) is at the achievable frontier for
    this data. Neither a stricter loss nor calibration improves held-out total-mass accuracy. The only real
    lever is MORE DATA (more containers/trials/views) — instantaneous flow (R² 0.854) is already strong; it's
    the absolute-total tails that are data-limited.** SmoothL1 stays the deliverable; MSE ckpt kept for record.
  - **RIGOR BATCH (2026-07-18, 6×80-min runs, `--val_trials`/`--fold` args added to clips_train_attn):**
    - **Flow probe 4-fold CV (error bars on the headline):** disjoint folds covering all 18 trials —
      A{8,13,21,24}=0.849, B{7,9,11,12}=0.800, C{5,15,16,25,26}=0.834, D{17,18,20,22,27}=0.763 →
      **combined flow R² = 0.81 ± 0.04** (CAM2 0.86 ± 0.04, CAM3 0.77 ± 0.04). The single-split 0.854 was
      the favorable end of the fold spread, not a fluke; fold D is hardest; CAM2 beats CAM3 by ~0.09 every
      fold. Ckpts `attn_flow_both_lag0.7_fold{A,B,C,D}_best.pt`.
    - **Warm-start vs SCRATCH pooler (80-min, split A):** EK100 warm-start combined 0.849 (CAM2 0.898/CAM3
      0.801) vs random-init 0.843 (CAM2 0.848/CAM3 0.838) — **a wash on final performance (+0.006)** despite
      a huge epoch-0 head start (0.669 vs 0.003). Scratch CATCHES UP by convergence; warm-start just relocates
      strength (better CAM2, worse CAM3). So the EK100 transfer is NOT essential here (pooler learns pouring
      from random init given the data) but converges faster and doesn't hurt → keep it, don't depend on it.
      Ckpt `attn_flow_both_lag0.7_noWS_best.pt`.
    - **VOLUME target (attentive, split A, lag 0 — volume is lag-insensitive):** combined R² **0.667**
      (CAM2 0.685/CAM3 0.648, MAE 36.8 g) — beats temporal-profile 0.577 and ridge mean-pool 0.450, so the
      attentive V-JEPA reads real fill signal beyond the clock. But volume is the HARDER target (0.67 vs flow
      0.81; absolute fill is hard from a wide shot). LOOSE END: raw-time-linear volume baseline (~0.78 in the
      earlier 6-fold ridge OOF) wasn't computed on this split, so "V-JEPA vs clock for volume" is unresolved
      — add raw-time to `baselines_on_split` for a clean call. Ckpt `attn_volume_both_best.pt`.
  - **PRETRAINED-CNN BASELINE (2026-07-19, `clips_cnn_baseline.py`, user asked for a solid small-data
    baseline):** frozen ImageNet ResNet-50 (torchvision IMAGENET1K_V2, 2048-d) + linear probe, SAME
    frozen-backbone + ridge + 4-fold-by-trial-CV protocol as V-JEPA. Per window (8 frames) pool features
    mean (appearance) or mean++std (adds temporal variation — Schenck & Fox IJRR2018 found temporal info
    is essential for liquid perception). **RESULT: flow R² ≈ 0.00 (mean) / 0.01 (mean+std), volume ≈ 0 —
    held-out CV. NOT a bug:** ResNet fits flow IN-SAMPLE at 0.91 (volume 0.98) so features have capacity,
    and my exact CV pipeline reproduces V-JEPA mean-pool 0.686±0.057 (positive control). So the ImageNet
    CNN MEMORIZES per-trial appearance but carries no cross-container flow signal (per-frame features =
    static object identity, not motion). **V-JEPA decisively beats it: attentive 0.81±0.04, mean-pool
    ridge 0.69, vs CNN 0.00.** The defensible baseline the writeup needs — the in-sample control proves
    the CNN got a fair shot; the gap is the video model's spatiotemporal transfer. Related lit: Schenck &
    Fox "Perceiving & Reasoning about Liquids w/ FCNs" (arXiv 1703.01564, the UWLPD lineage, CNN+temporal);
    "The Sound of Water" (Bagad et al. arXiv 2411.11222, infers pouring RATE from AUDIO/pitch — same target,
    other modality); SimLiquid (sim mL). Cache `~/.cache/pour_probe/clips_cnn_feats/resnet50/`. Possible
    stronger-CNN variant if needed: feed frame-DIFFERENCES through the CNN (explicit motion input).
  - **FAIR TEMPORAL-CNN BASELINE (2026-07-19, `clips_cnn3d_baseline.py`) — the ResNet-50 row was a
    STRAWMAN; use this one in the writeup.** A per-frame 2D CNN structurally cannot represent flow, so
    ~0.00 proved nothing. Frozen Kinetics-400 VIDEO CNNs, same frozen+ridge+4-fold-by-trial protocol,
    same 288 frame cache + center-crop-256 pixels as V-JEPA (r2plus1d_18 @112 and @128, s3d @224; one
    forward per window, 16 frames):
    | backbone | flow R² | volume R² |
    |---|---|---|
    | r2plus1d_18 @112 | 0.525 ± 0.039 | 0.236 |
    | r2plus1d_18 @128 | 0.530 ± 0.032 | 0.242 |
    | s3d @224 | 0.540 ± 0.089 | 0.206 |
    | **V-JEPA 2 attentive** | **0.81 ± 0.04** | 0.667 |
    | V-JEPA 2 ridge mean-pool | 0.69 | 0.450 |
    | ResNet-50 per-frame (strawman) | ~0.00 | ~0.00 |
    So a credible video CNN reaches **~0.53 flow** — the honest gap to V-JEPA is **+0.28 R²**, not +0.81.
    Resolution is NOT the confound (112 vs 128 vs s3d@224 all ~0.53). Volume is where V-JEPA's margin is
    widest (0.67 vs 0.24). Cache `~/.cache/pour_probe/clips_cnn3d_feats/`; all rows in mlflow experiment
    **`pour_probe_baselines`** (ResNet-50 rows retro-logged there too).
  - **BUG FOUND + FIXED in `clips_cnn_baseline.cv_r2` (2026-07-19):** the pipeline started with
    `Normalizer()` (L2 row-normalization), which is right for high-dim CNN embeddings but **destroys
    low-dimensional features** — a 1-D feature becomes the constant 1. It silently zeroed the raw-time
    baseline (0.78 → −0.01) and the decoded-λ audio row (0.27 → 0.04). `cv_r2(..., normalize=False)` now
    skips it; use that for any interpretable/low-dim feature. High-dim CNN/V-JEPA rows are unaffected.

### pouring/pour_probe/ — ORACLE-CONTAINER diagnostic (2026-07-20): container-size model is a NO-GO for absolute volume
User was bugged by the absolute poured-mass ("scale") error and proposed a separate container-size
model (train on SoW/CORSMAL/SimLiquid) to disambiguate absolute volume, OR crop-to-container for
scale invariance. `clips_oracle_container.py` tests the go/no-go cheaply: give the volume probe
PERFECT container identity (oracle one-hot of the target vessel, same OOF-GroupKFold-by-trial as
clips_train) and measure the lift. **RESULT (volume, CAM2, 121 clips): container identity ALONE =
win R² 0.012; vjepa+cont = 0.365 vs vjepa 0.364 (+0.001); time+cont 0.776 vs time 0.784; the
multiplicative form (time×cont per-container rescale) is WORSE (0.756).** So oracle container size
adds ~nothing. **Cause is fundamental, not low container diversity:** per-container mean poured mass
is 139/178/185/146 g (spread ~46 g) but WITHIN-container std ~80–110 g and every container gets
8 g→270–360 g pours — poured mass is set by the pour DURATION (a free choice), not by capacity, which
is a loose non-binding upper bound. So knowing container size cannot tell you how much was poured →
a container-size model can't fix absolute volume here (and a NOISY predicted size would do less).
Corollaries: (1) volume is clock-dominated (time R² 0.78 vs vjepa 0.36); absolute fill from a wide
3rd-person shot is the aleatoric floor already documented. (2) The BEST absolute-mass estimate is
INTEGRATING the strong flow probe (weight-recon median 14 g / mean 22 g), not any volume/size probe;
its residual is data-limited, not size-ignorance. **Container-size model DROPPED as a volume fix**
(CORSMAL only worth it as a standalone capacity estimator, a separate mini-project that would NOT
reduce poured-mass error). **Crop-to-container SURVIVES but for a different problem:** not absolute
volume (crop discards scale, scale doesn't help), but VIEW/DISTANCE ROBUSTNESS of the FLOW probe —
candidate fix for the CAM2→CAM3 zero-shot failure (0.524, pooler learned "look upper-left at torso").

### pouring/pour_probe/ — CROP-TO-CONTAINER prototype (2026-07-20): motion-ROI works on CAM2, fails on CAM3
Follow-up to the oracle-container no-go: the surviving idea was cropping to the pour region to fix the
FLOW probe's cross-view transfer failure (CAM2->CAM3 zero-shot 0.524; pooler learned "look upper-left
at torso"). `clips_roi_cache.py` = drop-in replacement for the clips_frames288 cache (same npz schema)
that crops each frame to a per-clip MOTION ROI (weighted 5-95 pctile bbox of mean |frame_t - frame_{t-1}|,
largest-connected-blob to reject edge distractors, down-biased toward the vessel, squared+expanded,
floored). No detection model, no new deps. Point `$POUR_ROI_FRAMES_DIR`→ it and `$POUR_FRAMES288_DIR`
at the same → the existing `clips_train_attn.py` + `clips_eval_crossview.py` run UNCHANGED. QC (`--qc`
→ `qc_roi_crop.png`, box+motion map over 6 clips × both cams; the heavy re-cache/train is GATED on QC).
- **QC FINDING (go/no-go for the crop route): motion-ROI is a clean win on CAM2 (box lands tightly on
  hand+source vessel+target mug — real scale normalization) but CANNOT isolate the container on CAM3.**
  CAM3 is the distant head-on view, so the pourer's WHOLE BODY dominates the motion energy → the ROI
  frames "person+table," not the pour. So CAM2 crops are pour close-ups, CAM3 crops are person-in-scene:
  a content mismatch that muddies any transfer test. Cause is the SAME reason CAM3 is the hard view.
  (The largest-blob + down-bias tweaks fixed the one hard failure — clip 0073 CAM3, box yanked to a
  left-edge distractor — and tightened boxes, but can't make motion find a small distant container.)
- **Implication:** a fair cross-view crop test needs an actual container localizer on CAM3 — an
  off-the-shelf OPEN-VOCAB DETECTOR (GroundingDINO zero-shot "kettle. mug. glass. teapot.", no training)
  is the clean route; motion alone is insufficient on the distant view. **Also resurfaced (important):
  the pragmatic robustness fix ALREADY EXISTS** — the both-cam trained probe gets CAM3 to 0.834 (ABOVE
  its native 0.804); crop's unique payoff is transfer to a genuinely UNSEEN angle, which we can't measure
  with only 2 fixed cameras (the CAM2->CAM3 zero-shot 0.524 is the one available proxy).
- **USER CHOSE the detector-ROI path (2026-07-20). `clips_roi_cache.py --backend detector` (now default):**
  GroundingDINO-tiny (`IDEA-Research/grounding-dino-tiny`, transformers 5.10.2, zero-shot, no training,
  ~700 MB) localizes the vessels. Uses the manifest's KNOWN source_obj/target_obj to prompt for the right
  classes per clip (`SRC_PROMPT`/`TGT_PROMPT`) so it doesn't grab the background electric kettle / spare
  bottles. ROI = union of best source + best target box IF close; if they're far apart (spurious source),
  anchor on the TARGET container alone (reliable scale anchor) expanded upward to keep the stream; motion
  fallback if nothing found. **QC PASSED (`qc_roi_crop_detector_v2.png`): object-anchored crops in BOTH
  views — tight on CAM2, and on CAM3 they frame the container/table region instead of the whole standing
  body (the exact fix motion couldn't do). A few tall-pour CAM3 clips (0069/0086) are looser but still
  vessel-anchored, not fixed-frame.** Cache: `$POUR_ROI_FRAMES_DIR` (~/.cache/pour_probe/clips_frames288_roi,
  drop-in for clips_frames288, same npz schema). EXPERIMENT QUEUED (~90 min): build full ROI cache both
  cams → retrain attn flow probe on ROI CAM2 (lag 0, tag_extra=roi → ckpt attn_flow_CAM2_roi_best.pt,
  matches the baseline recipe so the CROP is the only variable) → `clips_eval_crossview.py --tag _roi`
  (POUR_FRAMES288_DIR=ROI cache). Compare ROI CAM2-native vs 0.899 (does cropping cost accuracy?) and ROI
  CAM2->CAM3 vs 0.524 (does cropping enable transfer?).

### pouring/pour_probe/ — Sound of Water (cross-modal audio baseline + external dataset)
`third_party/SoundOfWater` = their MIT repo, cloned **for reference only** (never on `sys.path` — it
imports pytorch_lightning/librosa). `sow_model.py` re-implements `Wav2Vec2WithTimeEncoding` +
`WavelengthWithTime` as plain `nn.Module`s with identical attribute names and loads their checkpoint
(`checkpoints/sow/dsr9mf13_ep100_...pth`, 378 MB, HF `bpiyush/sound-of-water-models`) with
**`strict=True`** — the correctness gate. Config from their `demo/util.py`: 64 axial/radial bins,
readout = softmax @ `linspace(0,100,64)` cm (soft-argmax; argmax would give a staircase).
- **GATE PASSED:** strict load clean; on a SoW video λ falls 86.1 → 9.1 cm with Spearman(t,λ) = **−0.995**
  (air column shortens as the vessel fills). λ(0)=86 cm ⇒ 21.5 cm air column vs the annotated 19.7 cm
  net height — physically consistent.
- **`sow_physics.py` — cleaner formulation than the paper's:** for volume poured SINCE CLIP START the end
  correction β and the container height H **cancel**, leaving `V_poured(t) = (πR²/4)·[λ(0) − λ(t)]` and
  `Q(t) = −(πR²/4)·dλ/dt` — **only the radius R is needed**. (ρ_water=1 ⇒ mL ≡ g, so these are directly
  comparable to our scale GT.) Savitzky-Golay ~0.5 s smoothing before differentiating; running-max clamp
  since water only enters.
- **W2a — SoW audio model on OUR clips (`clips_sow_baseline.py`, frozen + ridge, same 4 folds):**
  | feature | flow R² | volume R² |
  |---|---|---|
  | wav2vec 768-d mean (both cams) | **0.648 ± 0.041** | **0.802 ± 0.068** |
  | axial 64-bin head | 0.324 | 0.423 |
  | decoded λ (3-d) | 0.272 | 0.353 |
  | *V-JEPA 2 attentive (video)* | *0.81* | *0.667* |
  | *raw-time-linear control* | *0.221* | *0.778* |
  | *time_prof control* | *0.569* | *0.557* |
  **Reading it honestly: (a) for FLOW the audio FM gets 0.648 — well above the temporal prior (0.569) and
  the video CNN (0.53), but below V-JEPA 2 (0.81), so the video model still wins on our data. (b) for
  VOLUME audio's 0.802 barely beats the CLOCK (raw time = 0.778), so it is NOT evidence of strong audio
  volume perception — cumulative mass is mostly predictable from elapsed time.** CAM2 vs CAM3 differ
  little (0.60/0.63 flow) despite different mics. Their zero-shot physics readout (λ) carries real signal
  (0.27/0.35) even though our mugs are non-cylindrical and R is uncalibrated — the W2b measured-GT
  validation is still blocked on caliper measurements.
- **Note on their claims:** SoW claim SOTA *for audio only* and run **no vision comparison**, so this is a
  cross-modal reference point, not a rival method (see `RELATED_WORK.md`).
- **W3 — our probe on THEIR data (`sow_targets.py`, `sow_grid_cache.py`, `clips_train_attn.py --dataset
  sow`):** target is **model-derived** (their audio estimate), so R² measures video-probe ↔ audio-physics
  agreement, NOT ground truth. 790 videos → targets decoded; sanity gate: median V(T) = 273 mL = **79% of
  container capacity**, 94% plausible, 41 implausible dropped. Subsets (grouped by CONTAINER for CV):
  S1 cylindrical+transparent 218/11 containers, S2 all cylindrical 317/21, S2o opaque-only 89/8,
  S3 all shapes 780/45. **Did NOT apply a strict taper cut** — it drops 5 of 11 transparent containers,
  and a container-grouped split needs containers more than a perfect cylinder.
  - **Frame cache gotcha (important):** SoW videos are PORTRAIT 270×480 and the container sits at the
    BOTTOM (median annotated box centre = 80% of frame height) — a short-side-resize + center crop, which
    is what our own clips use, **crops the container out entirely**. `sow_grid_cache.py` instead takes a
    full-width square anchored on `annotations/container_bboxes/<vid>_box.npy` (container just above the
    bottom edge, stream visible above). Verified visually.
  - **JPEG-in-npz** (q92, lazy `JpegFrames` decode + shared thread pool since cv2 releases the GIL):
    780 videos = **4.3 GB** instead of ~66 GB raw — necessary given only ~16 GB free RAM. Decode is
    7 ms/window (0.4 min/epoch); the run is GPU-bound at ~0.13 s/window, same as the clips runs.
  - Trainer additions: `--dataset {clips,sow} --subset --val_frac --split_seed`; SoW baselines =
    `time_prof` / `predict_mean` / `container_mean` + **`mean_removed_r2`** (per-video means removed, to
    prove the score isn't just between-video offsets). mlflow experiment **`pour_probe_sow_attn`**.
  - **METRIC LESSON (2026-07-19, cost ~30 min of GPU — don't repeat):** the first S1 run showed val
    R² = −4.6 then −8.4 and looked broken. It wasn't a bug. Two compounding causes: (1) the uniformly
    random container split put **85% of val windows on ONE container** (video counts per container run
    1..46), and (2) that container's volume range was narrow (val sd 80 mL vs train 248 mL) — R²'s
    denominator IS that val variance, so a small bias becomes a huge negative score. **Absolute mL across
    containers is intrinsically hard anyway: it requires inferring the unseen container's physical size
    from pixels.** Fixes, both in place: the val split is now **balanced by video count** (greedy
    largest-first to the target fraction → several containers, realistic spread), and the SoW
    **selection + headline metric is the within-video (mean-removed) R²**, which is scale-free and asks
    the question we care about (does the probe track filling over time). Plain R² and MAE stay logged.
    Signal that this is right: at the 2-min smoke test, plain R² was −0.217 while within-video R² was
    already **0.703**.
  - `sow_grid_cache.load_clips(subset)` used to ACCEPT and silently IGNORE `subset` (the trainer
    filtered separately); it now actually filters. Watch for this when writing analysis scripts.
  - **RESULTS (2026-07-20, attentive probe, wall-clock LR schedule, within-video R² = the metric).
    THE HEADLINE IS A NEAR-NULL — read the baseline column, not the probe column.**
    | subset | target | probe within-vid R² | **time-only baseline** | probe edge |
    |---|---|---|---|---|
    | S1 transparent | volume | 0.163 | 0.058 | **+0.105** |
    | S2 all cylindrical | volume | 0.287 | 0.253 | +0.034 |
    | S2o opaque only | volume | 0.749 | **0.976** | **−0.227** |
    | S3 all shapes (45 cont.) | volume | 0.803 | 0.799 | +0.005 |
    | S2 all cylindrical | flow | −0.07 | −0.13 | (degenerate) |
    **The SoW volume target is CLOCK-DOMINATED.** Their pours are labelled `flow_rate_appx=constant`
    (988/1010), so V(t) within a video is ~linear in time and a normalized-time poly-4 predicts it almost
    perfectly (0.80–0.98 within-video). The probe only modestly beats the clock on TRANSPARENT sets
    (S1 +0.105) and is a wash on S3 (+0.005). **CORRECTION to an earlier over-claim: S2o (opaque) is NOT
    a win — the probe's 0.749 looked impressive alone, but the time baseline gets 0.976, so the probe is
    0.227 BELOW the clock there.** Makes sense: opaque + constant-flow → the level is invisible AND
    perfectly time-predictable, so a visual guess is strictly noisier than elapsed time. **S2 flow is a
    documented degenerate:** near-constant per video → nothing within-video to predict (both probe and
    baseline ≈ 0). So W3 = external validation that mostly reveals *their* target is too easy for a clock.
    This is a POSITIVE for the thesis by contrast: on OUR clips flow VARIES, so time can't cheat
    (time_prof flow 0.62 vs V-JEPA 0.85) — that's exactly why the own-dataset varied-flow result is the
    real contribution and SoW volume is not. **Do NOT spend the S3 rerun slot** — a longer run just
    converges harder onto a clock-explainable target.
  - **STAGED-TRANSFER ARM (SoW-pretrain → fine-tune on our gram-GT clips, flow, both cams, lag 0.7,
    converged flat val tail 0.826 ± 0.007) — the clean, useful result.** Third arm of the head-init
    ablation:
    | head init | flow R² (our clips, both cams) |
    |---|---|
    | EK100 action pretrain | 0.854 |
    | random init | 0.843 |
    | **SoW pouring pretrain** | **0.831** |
    On-domain pouring pretraining does **NOT** beat off-domain action pretraining or even random — it is
    marginally BELOW both. Confirms (third independent time) that **with a frozen encoder the head init
    barely matters; the V-JEPA representation does the work**, not the pretrained pooler. `--init_ckpt`
    added to `clips_train_attn.py` for staged transfer (loads a trained head as init; kept SEPARATE from
    joint training because the two corpora's targets differ in kind — measured grams vs audio-model
    estimate — and in lag ~0.7 s vs ~0, so one normalization/lag can't serve both).
  - **LR-SCHEDULE FIX (2026-07-20, important — supersedes the epoch-estimate cosine):** the cosine length
    was estimated from EPOCH 0's wall time ×0.92, but epoch 0 is the slowest (cold caches), so it
    overshot per-epoch cost, sized the schedule too short, and **LR hit 0 with ~20-24% of the budget still
    to run** (measured 24% dead-zero steps on S2o) — those steps do nothing (AdamW update and decoupled
    weight decay both scale by LR). Fix: the cosine is now driven by **wall-clock fraction of the time
    budget** (`sched_state` holds `t_start`/`warmup_end_t`/`budget_s`), so LR reaches 0 exactly when the
    loop stops — no estimate, full budget spent training. Warmup stays the first epoch. Verified in
    production: transfer run LR = 0 at 91.4/90 min with a flat 0.826 plateau; no dead tail.
  - **INFRA: two MLflow stores got created** because MLflow's default sqlite path is CWD-relative and I
    ran scripts from `pour_probe/` while history lived at the repo root. `mlflow_util.py` now PINS an
    absolute store (`sqlite:////…/idp/mlflow.db`); all scripts route through `mlflow_util.setup()`.
    `mlflow_migrate.py` consolidates a stray store into the canonical one (idempotent, copies full metric
    history). **Always launch `mlflow ui --backend-store-uri sqlite:////home/casimir/UNI/SS_26/idp/mlflow.db`.**
    Caveat learned the hard way: editing the trainer MID-CHAIN changes which store later fresh
    subprocesses use — patch between chains, not during.
  - **ENV NOTE:** `nvidia-smi` currently errors `Failed to initialize NVML: Driver/library version
    mismatch` (a background driver package update bumped userspace NVML while the old kernel module stays
    loaded). **torch/CUDA are UNAFFECTED** (GPU matmul + training work fine); `nvidia-smi` needs a REBOOT
    to resync. Monitor GPU via mlflow/nvidia-smi-free means until then.

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
