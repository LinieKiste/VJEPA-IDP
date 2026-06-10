# IDP — V-JEPA 2 for Procedural-Error Detection in Video (SS26)

Interdisciplinary project: systematic evaluation of frozen **V-JEPA 2** video
representations for **anomaly / procedural-error detection**, centered on the
[EgoPER](https://github.com/robert80203/EgoPER_official) dataset (egocentric cooking
videos with subtle procedural errors).

**Headline result:** frozen V-JEPA 2 ViT-L features linearly separate correct vs.
erroneous procedure windows on EgoPER Coffee at window ROC-AUC ≈ 0.78 (SlowFast-pooled
view, L2-regularized logistic probe; mean-pool view 0.75). See `egoper_probe/README.md`.

## Layout
- **`egoper_probe/`** — the centerpiece: sliding-window V-JEPA 2 feature extraction +
  linear/one-class probes for window-level error detection, tracked with mlflow.
- **`egoper_vqa/`** — baseline/contrast: zero-shot video-QA on EgoPER with Qwen2.5-VL
  (4-bit), incl. a training-free SlowFast-LLaVA-style two-stream scheme and task-graph
  procedure grounding.
- **`video_qa/`** — replication of V-JEPA 2 (arXiv 2506.09985) Appendix E: LLaVA-style
  visual instruction tuning aligning the frozen encoder with Qwen2.5-7B via QLoRA.
- **`vjepa2/`** — git submodule: [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
  (model code; checkpoints not included).

## Setup
```bash
git clone --recurse-submodules <this-repo>
uv sync                                   # Python 3.10, torch cu132, see pyproject.toml
# place V-JEPA 2 ViT-L checkpoint at checkpoints/vitl.pt
# place EgoPER at datasets/egoper/ (see CLAUDE.md for expected layout)
```

Run the probe pipeline:
```bash
.venv/bin/python egoper_probe/extract.py --task coffee   # cache features (GPU, ~50 min)
.venv/bin/python egoper_probe/probe.py --task coffee --view feats_sf --C 0.001
.venv/bin/mlflow ui                                      # inspect runs
```

Hardware target: single RTX 5060 Ti (16 GB, Blackwell/sm_120).
