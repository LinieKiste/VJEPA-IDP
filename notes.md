1. Headline result (lead with this)

- Frozen V-JEPA 2 ViT-L features linearly separate correct vs. erroneous windows on EgoPER Coffee: window ROC-AUC 0.745 ± 0.040 (AP 0.121 ≈ 3× the 3.9% base rate; video-level 0.678 ± 0.099 but noisy with only ~21 test videos per split). 68 videos → 14,744 sliding windows, 10-seed GroupShuffleSplit by video, tracked in mlflow.
- Framing: the V-JEPA thesis holds — error evidence survives a frozen encoder + mean-pool + logistic regression. This is the agreed "working SOMETHING" deliverable.
- The one-class variant failed (Mahalanobis on mean-pooled features, ROC-AUC 0.554 ≈ chance) — worth discussing why (see §4) and whether the more EgoPER-faithful normal-only setting is worth rescuing.

2. Papers read / evaluated

- SlowFast-LLaVA-1.5 (arXiv 2503.18943, Apple): two-stream parameter-free token pooling (few hi-res "slow" frames + many lo-res "fast" frames) — directly addresses the spatial-detail vs. temporal-coverage tradeoff for ~10 min videos. Catch: no public weights (verified — only the training-free v1 repo exists, which is bound to LLaVA-NeXT / torch 2.1.2 and can't run on the Blackwell GPU). I reimplemented the v1 scheme on Qwen2.5-VL instead (ask_slowfast: 16 slow @360×640 + 96 fast @200×200 ≈ 4.8K tokens, 112 frames coverage).
- LongVA (long-context transfer from language to vision): the alternative philosophy — extend the LLM's context so you can feed more frames instead of compressing tokens. Discussion point: on 16 GB VRAM the KV-cache cost (~56 KB/token for Qwen2.5-7B) makes token compression, not context extension, the realistic path — but LongVA is the right citation for the "why not just more frames" question.
- LongVideoBench / long-video evaluation for V-JEPA: relevant as the framing for whether sparse frame sampling can ever catch brief, localized procedural errors — supports the core design argument (8 frames over 600 s ≈ 1 frame / 75 s misses the evidence entirely), which is exactly why the probe uses dense sliding windows.
- EgoPER paper itself (Lee et al., CVPR 2024): important caveat to raise — their EgoPED method assumes no task-graph access at train/test. The VQA baseline feeds the task graph to the VLM, so it's not directly comparable to their benchmark numbers. The supervised probe also deviates (uses error videos for training). Worth aligning with the supervisor on which comparison claim is honest to make.

3. Implementation attempts & engineering findings

- egoper_probe/ (the deliverable): sliding-window feature extraction + cached .npz features + logreg probe. Key engineering win: extraction went ~130 s → ~45 s per video — the bottleneck was fp32 encoding, not decode/IO; bf16 autocast on the frozen encoder + decode-once-at-reduced-res + batched windows fixed it. Peak GPU only ~5.3 GB, so there's headroom.
- egoper_vqa/ (baseline/contrast): Qwen2.5-VL-7B in 4-bit, frames as PIL list, runs in ~5–6 GB. Zero-shot result without grounding was bad in an instructive way: the model hallucinated generic coffee knowledge ("didn't tamp the grounds" — not a pour-over step), missed all 3 real errors, and false-positived the normal control. Built procedure grounding (task graph topologically sorted + named steps prepended as context) in response — open question whether it fixes the normal-clip false positive.

4. What I discarded, and why

- VL-JEPA: no public weights, but would probably be the perfect fit
- LLaVA-Video-7B-Qwen2 (the leaderboard pick): pins torch 2.1.2, which has no Blackwell (sm_120) kernels — literally cannot use the GPU. Same reason SF-LLaVA v1's official repo is unusable here.
- SF-LLaVA-1.5 pretrained weights: don't exist publicly; only the pooling idea is portable.
- 8-global-frames video-QA setup (the Appendix E video_qa/ pipeline) for EgoPER: temporal coverage is the dealbreaker for subtle, brief errors; also 256 px + heavy pooling discards the spatial detail that distinguishes lid-on/lid-off-class errors.
- Mean-pool + single-Gaussian one-class scoring: ROC-AUC 0.554 — mean-pooling over a whole window washes out subtle local errors, and one Gaussian can't model the normal manifold. Discarded as-is, but the setting (normal-only training) is worth keeping.

5. What might work next (options to get supervisor input on)

- Keep the token grid instead of mean-pooling + kNN/prototype scoring per token — the most direct fix for the one-class failure, and closer to EgoPER's own prototype approach.
- Tiny temporal head over the cached window features (the "light training" that's in scope) — could lift both supervised and video-level numbers.
- SlowFast-style pooling on top of cached V-JEPA tokens feeding a small LLM (1B/3B Qwen2.5) — the bridge between the probe and the VQA track, with timestamps injected for error localization, not just detection.
- Scale to the other 4 tasks (Tea, Oatmeal, Pinwheels, Quesadilla) to show the coffee result generalizes — cheap now that extraction is fast.
- Grounded VQA baseline numbers — finish scoring ask_slowfast + procedure grounding against GT labels so the probe-vs-VLM contrast is quantitative.

- supervised vs. one-class — which framing should the writeup center; (c) how much effort the VQA contrast deserves given it's not the deliverable.



visualisierung
powerpoints
