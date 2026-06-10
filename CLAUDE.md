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

## Datasets
Tracked in the Datasets DB: https://app.notion.com/p/0d5ea22cf2e948018b6f59819cb75a4a
Main focus is EgoPER

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
