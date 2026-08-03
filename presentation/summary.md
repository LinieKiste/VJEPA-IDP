---
theme: default
title: Pouring probe, four-slide summary
info: IDP SS26, Casimir Wallwitz
layout: cover
transition: none
mdc: true
colorSchema: light
---

# Frozen V-JEPA 2 reads pouring flow rate

Everything since the pouring dataset, in four slides

<div class="text-sm opacity-60 mt-8">

121 clips &middot; 18 trials &middot; 12 container combos &middot; gram-level scale ground truth

</div>

---

# 1. The result

<img src="./figs/baselines_flow.png" class="max-h-80 mx-auto mt-2" />

<div class="grid grid-cols-3 gap-6 text-sm mt-4">
<div>

**R² 0.81 ± 0.04**, MAE 7 g/s. Four folds group by trial, so each fold holds out whole containers and scenes.

</div>
<div>

Encoder stays frozen. Only a light attentive head trains. The gap is the representation, not the head.

</div>
<div>

Beats the fair video CNN by **+0.28** and the audio model by **+0.16**. A per-frame image CNN scores 0.00.

</div>
</div>

---

# 2. Flow is a vision problem, volume is a clock

<img src="./figs/protocol.png" class="w-full mt-1" />

<div class="grid grid-cols-3 gap-6 text-sm mt-3">
<div>

**Volume:** elapsed time alone gets 0.78. V-JEPA alone gets 0.37. Cumulative mass rises with the clock, so the clock wins.

</div>
<div>

**Vision still adds.** V-JEPA plus clock reaches 0.82, a **+0.19 skill** gain over the clock. Removing each pour's own mean changes neither verdict.

</div>
<div>

**Flow needs vision.** A straight line gets 0.24. Even a time profile handed the oracle duration gets 0.59. Integrating predicted flow gives total mass to **14 g median**.

</div>
</div>

---

# 3. What the ablations settled

<div class="grid grid-cols-2 gap-6 mt-3">
<div>

<img src="./figs/roi.png" class="w-full" />

<div class="text-xs opacity-70 mt-2">

Cropping trades accuracy for robustness. Its niche is a new angle with **no labels at all**. Given any labeled data there, a linear ridge on it already wins.

</div>

</div>
<div class="text-xs">

| question | verdict |
|---|---|
| Is the ground truth aligned? | **No.** 0.7 s water-transit lag, worth +0.09 |
| Does the pretrained head matter? | **No.** Three inits tie |
| One probe, both views? | **Yes.** It lifts the weak view |
| Transfer to an unseen view? | **No.** Cropping mostly fixes it |
| Does container size help volume? | **No**, even with a perfect oracle |
| Can a better loss fix the tails? | **No.** The residual is aleatoric |
| Does audio beat it? | **On volume only.** Flow stays ours |

</div>
</div>

---

# 4. Position, and the next number worth having

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

<img src="./figs/sow_crossmodal.png" class="w-full" />

<div class="text-xs opacity-60 text-center mt-1">

Sound of Water audio model, our clips, matched protocol

</div>

</div>
<div class="text-sm">

- Audio wins volume (0.80), loses flow (0.65). Our contribution sits on **flow**
- Their own dataset cannot test us. Constant rate plus fill to completion makes `t/T` **identical** to fill fraction. A clock scores up to 0.98 there
- **Open:** enter the Wilson et al. mass benchmark. Published protocol, existing baseline ladder, **no video-foundation-model entry**
- **More data** is the only lever left on absolute totals

</div>
</div>

<div class="text-sm opacity-70 mt-6">

Code: `pouring/pour_probe` &middot; runs: `mlflow.db`, experiments `pour_probe*`

</div>
