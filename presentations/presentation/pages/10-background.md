# Background: V-JEPA 2 + EK100 for action labelling

EgoPER tea, egocentric, zero-shot (no fine-tuning)

- Frozen V-JEPA 2 features plus the EK100 action head, applied off the shelf
- Top-5 verb accuracy 0.96 over 220 ground-truth action segments
- Reads the right actions: pour, take, insert, stir
- Flags anomalies too: microwave (added step) and knife (wrong tool)
- Main limitation is the EPIC vocabulary (no tea, mug, teabag), not perception

---

# Background: transfer to eXprt (third-person view)

- eXprt is a fixed third-person wide shot, not egocentric
- Off-the-shelf EK100 head fails: top-1 verb 0.05, fixates on the background
- A probe on the frozen V-JEPA embeddings recovers the action signal
- Leave-one-video-out accuracy 0.74, permutation null 0.30, p < 0.003
- 6-way action probe with augmentation reaches 0.71

---

# Background: eXprt probe, qualitative examples

<img src="../assets/ek100_examples_6.png" style="max-height: 82%; margin: 0 auto; display: block;" />

<div class="text-xs text-gray-500 text-center">

Green border = verb correct, red = wrong, orange = noun misclassified (no tea nouns in EPIC). Only large, unambiguous cues survive the distant view.

</div>

---

# Why the anomaly thread stalled

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

### EgoPER procedural errors

- Supervised window ROC-AUC **0.75**
- One-class (normal-only) stuck at **~0.60**
- Subtle local errors are the open problem

</div>
<div>

### eXprt tea anomalies

- 8-way error type **0.49** accuracy (4x chance)
- Binary anomaly only **~0.70**
- Bottleneck: 5 normal videos, video-level labels only

</div>
</div>

<div class="mt-8">

**Diagnosis:** the frozen features carry real signal; the *labels* are coarse, few, and video-level.

**Consequence (05.07 supervisor meeting):** pivot to a target that is continuous, dense in time, and physically measured.

</div>
