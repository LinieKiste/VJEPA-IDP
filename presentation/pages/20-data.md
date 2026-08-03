# The pivot: pouring flow and volume

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

### Why this target

- Continuous and dense in time, one label per 1 s window
- **Physically measured**, not annotated by opinion
- No prior work reports poured-mass or flow **regression** from third-person video
- Cheap to record more of

</div>
<div>

### Why it is hard

- Wide third-person shot, the stream is a few pixels wide
- Water is transparent, specular, fast
- Held-out containers and held-out scenes
- No public dataset with real gram-level ground truth

</div>
</div>

<div class="mt-8">

**So we recorded our own.** 2026-07-13, three synchronised GoPros, one pointed at a kitchen scale.

</div>

---

# Where this sits in the literature

<div class="text-sm mt-4">

| work | modality | output | relation to us |
|---|---|---|---|
| Wilson et al., **PSNN** (IROS 2019) | audio + visual | poured weight, 0.2 oz classes | closest prior task. Its data **is** a usable benchmark, see below |
| Bagad et al., **Sound of Water** (2024) | audio only | container height, flow rate, time to fill | audio SOTA, **no vision comparison**. We use it as a baseline |
| Schenck & Fox (IJRR 2018) | vision | per-pixel liquid mask | UWLPD lineage. Finds temporal information is essential |
| Lin et al. (ICCV 2023) | vision | liquid stream tracking | tracks the stream, not the mass |

</div>

<div class="grid grid-cols-2 gap-8 text-sm mt-5">
<div>

**Gap:** no prior work regresses **flow rate** in physical units from a third-person camera. For flow there is no leaderboard to climb, so the deck is built out of baselines instead.

</div>
<div>

**But for mass there is one, and we should use it.** Sound of Water linear-probes on Wilson's data and reports a full baseline table. Their protocol is public: 136 sequences, 6 containers, per-container regressor. See the next slide.

</div>
</div>

---

# There is a benchmark we can actually enter

<div class="grid grid-cols-2 gap-8 mt-4">
<div class="text-sm">

Sound of Water, Table 4. Mass estimation from pouring sound on **Wilson et al.'s** data, mean MAE in ounces:

| method | modality | MAE (oz) |
|---|---|---|
| SoundNet-8 | audio | 4.58 |
| k-NN | audio | 2.85 |
| TCN | audio | 2.45 |
| PSNN (Wilson et al.) | **audio + visual** | 1.35 |
| Sound of Water | audio | **1.20** |
| **frozen V-JEPA 2** | **video** | **?** |

</div>
<div class="text-sm">

### Why this is the right next experiment

- A published table with an established protocol, and **no video-foundation-model entry in it**
- Sound of Water beat PSNN's supervised fine-tuning with a **linear probe**, which is exactly our recipe
- They note "visual information is not used in any form", so the video slot is open
- Wilson's data has real weight annotations and is downloadable

<div class="mt-4 opacity-70">

Cost: extract frozen features over 136 sequences, fit 6 per-container ridges. Days, not weeks.

</div>

</div>
</div>

---

# Recording setup

<img src="../figs/views_example.png" class="w-full mt-2" />

<div class="grid grid-cols-3 gap-6 text-sm mt-3">
<div>

**CAM1**, 119.88 fps<br>
Scale display, ground truth only, never fed to the model

</div>
<div>

**CAM2**, 29.97 fps<br>
Side view, the pour fills the frame

</div>
<div>

**CAM3**, 29.97 fps<br>
Front view, distant, whole body visible

</div>
</div>

<div class="text-sm opacity-70 mt-3">

18 trials &middot; 3 source vessels into 4 targets &middot; 12 combinations

</div>

---

# Ground truth: OCR of the scale, then physics

<img src="../assets/trace_example.png" class="w-full mt-2" />

<div class="grid grid-cols-2 gap-8 text-sm mt-3">
<div>

- Off-the-shelf OCR fails on a 170x80 px 7-segment display in a wide shot
- Own reader: static camera, per-pixel background model, per-cell correlation against synthetic 7-segment templates
- **95.2% valid readings**, 16 of 19 trials above 99.2%, 2 ms per frame

</div>
<div>

- **Monotonicity prior:** while pouring, true weight never decreases
- Rolling-median bounds reject dropouts and digit blends, then isotonic regression per pour
- Removing the cup drives the scale negative, the display drops the minus sign, producing a fake spike (red bands)

</div>
</div>

---

# From trials to clips: a gated pipeline

```mermaid {scale: 0.68}
flowchart LR
  A["19 recordings<br/>3 cameras"] --> B["lcd_ocr.py<br/>per-frame weight"]
  B --> C["mono filter<br/>+ isotonic"]
  C --> D["plateau-chain<br/>pour detector"]
  D --> E["annotator<br/>121 events, verified"]
  E --> F["cut_clips.py<br/>121 clips x 2 views"]
```

<div class="grid grid-cols-2 gap-8 text-sm mt-6">
<div>

- Pour = maximal ascending chain between two settled plateaus
- "Cup barrier" rule ends a pour before the removal step
- Every clip weight was checked by hand in a local web UI
- One recording had no usable pours, so 18 trials survive

</div>
<div>

- Per-clip ground truth is 2 columns: `t_s, weight`
- Poured mass since clip start, baseline subtracted
- **Monotone by construction**, total equals the annotated mass

</div>
</div>

---

# The finished dataset

<img src="../figs/dataset.png" class="w-full mt-4" />

<div class="grid grid-cols-3 gap-6 text-sm mt-6">
<div>

**121 clips**, 410 MB<br>
2 views each, plus GT curve

</div>
<div>

Flow **varies within and across pours**, which is exactly what makes the clock a weak baseline here

</div>
<div>

Uploaded to the TUM NAS under `Datenverarbeitung/pouring_clips/`

</div>
</div>

---

# The four datasets, and what each can actually tell us

<div class="text-xs mt-2">

| dataset | what it is | labels | how we used it | the catch |
|---|---|---|---|---|
| **Own-lab pouring** <br> *ours, 2026-07-13* | 121 clips, 18 trials, 12 container combos, 2 synced views at 1080p | **poured mass in grams**, OCR'd from a scale on a third camera. 8 to 362 g | train and evaluate the probe. **The deliverable** | one room, one person, one session |
| **Sound of Water** <br> *Bagad et al. 2024* | 1010 smartphone videos, 48 containers. 1000 clean, **780 used** | **no measured volume.** Container geometry plus fill time only | 1. their frozen **audio** model as a cross-modal baseline **on our clips** <br> 2. our probe **on their videos** | see below |
| **UWLPD** <br> *Schenck & Fox* | 36 real pouring scenes, 640×480, per-frame binary liquid masks | **fill % only** (30 / 60 / 90), no mL | mask-derived proxy, smoke test of the pipeline | no volume labels. The mL trace is in their separate simulated set, requested, not in hand |
| **SimLiquid** <br> *BlenderProc renderer* | photoreal liquid-in-cup renders | **per-cup volume in mL**, clean by construction | renderer validated. **10k render not run** | sim-to-real gap untested |

</div>

<div class="text-xs opacity-70 mt-2">

**EK100** is not a pouring dataset. It supplies the optional warm start for the attentive head, and the ablation shows warm start ≈ random.

</div>

---

# Why Sound of Water cannot validate us

<div class="grid grid-cols-2 gap-8 mt-4 text-sm">
<div>

### What it is

- The strongest published model for **pouring from audio**, and the closest work to ours
- 1010 videos, 48 containers, ceramic / glass / paperboard / plastic / steel, hot and cold water
- Their target is **derived from their own audio model**, not measured. So our probe on their data measures *agreement with an audio-physics estimate*, not ground truth

</div>
<div>

### Why it is not a test of flow

- **988 of 1010 pours are labelled constant flow rate. Exactly one is labelled non-constant**
- Constant rate + fill to completion gives `T = V/Q`, so `V(t)/V = t/T` and **the rate cancels**. Normalised time *is* fill fraction, and a clock scores up to **0.98**
- Their flow ground truth is a **single scalar per video** (container volume ÷ fill time), never a trace

</div>
</div>

<div class="text-sm mt-4">

**This is a property of our repurposing, not a flaw in their paper.** Constant-within-pour is deliberate and load-bearing for them: their time-to-fill task is ill-defined without it, and they say so. Their randomisation is **between** pours. Our clock baseline exploits the **within**-pour axis. Both claims are correct, on different axes.

</div>

---

# Know the data before trusting the number

<img src="../figs/inputs.png" class="w-full mt-1" />

<div class="grid grid-cols-3 gap-5 text-xs mt-2">
<div>

**What the probe may use**

- **RGB only**, 1.0 s window, 16 frames, 256 px
- Two synchronised views, CAM2 side and CAM3 distant
- Audio exists, used **only** by the cross-modal baseline

</div>
<div>

**Shortcuts we ruled out**

- **Read the scale?** No. The LCD faces the overhead OCR camera. At 256 px the scale is ~30×15 px
- **Memorise the scene?** No. Folds group **by trial**
- **Guess from the vessel?** No. A perfect container oracle adds **+0.001**

</div>
<div>

**Shortcuts that are real**

- **Look at the arm, not the stream.** The pooler learns "upper left". CAM2→CAM3 collapses to 0.04 on one fold
- **Duration leaks mass.** Clip length correlates **0.81** with total poured

</div>
</div>

<div class="text-xs opacity-60 mt-2">

What the data genuinely cannot say: fill level inside an opaque vessel, and how much the person *intended* to pour. Poured mass is set by pour duration, a free choice, not by the vessel.

</div>
