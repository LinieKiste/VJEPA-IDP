# What is done and what is open

<div class="grid grid-cols-2 gap-8 mt-6 text-sm">
<div>

### Done

- Own pouring dataset, 121 clips, gram-level ground truth, on the NAS
- OCR reader, trace cleaner, pour detector, annotator, clip cutter
- Frozen V-JEPA 2 flow probe with 4-fold group CV
- Eight baselines across three modalities
- Eight ablations, each with a clear verdict
- Everything reproducible from mlflow

</div>
<div>

### Open

- 4-fold attentive **volume**, so the audio comparison is protocol-matched
- CAM3 → CAM2, the untested transfer direction, and a 4-fold version of the ROI comparison
- **Enter the Wilson et al. mass benchmark**, the one place our probe gets a directly comparable published number
- **More data** is the only lever left on total-mass accuracy
- SimLiquid: renderer validated, the 10k render and sim-pretrain arm not run
- A third camera angle, to measure genuine unseen-view transfer instead of the CAM2-to-CAM3 proxy

</div>
</div>

---

# Takeaways

<div class="mt-6">

1. **A frozen video foundation model reads a physical quantity off raw pixels.** Flow rate at R² 0.81 ± 0.04, from a wide third-person shot. It holds across held-out trials, containers and views.

2. **The gap is the video model, not the head.** +0.28 over a Kinetics video CNN on identical pixels and folds. Head initialisation is a wash three ways over.

3. **Know which target is actually a vision problem.** Flow is. Absolute volume is mostly a clock, and saying so is a result.

4. **The remaining error is aleatoric.** Loss, calibration, regularisation and oracle container identity all fail to move held-out totals.

</div>

<div class="text-sm opacity-70 mt-10">

Code: `pouring/clip_split` (data), `pouring/pour_probe` (probe, baselines, ablations) &middot; runs: `mlflow.db`, experiments `pour_probe*`

</div>
