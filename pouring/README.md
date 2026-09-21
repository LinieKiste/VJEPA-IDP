# pouring/ — from lab recordings to a flow-rate probe

Two stages. Run everything from the repo root.

```
raw GoPro trials (NAS: Dateneingang/)
   │  clip_split/   read the scale display → find each pour → human review → cut clips
   ▼
datasets/pouring_processed/clips/{CAM2,CAM3,csv}/NNNN.*   (121 pours; NAS: Datenverarbeitung/pouring_clips/)
   │  pour_probe/   cache frames/features → train probes → evaluate → figures and demo videos
   ▼
results in mlflow (mlflow_export/*.csv) + figures used by the decks
```

| Folder | Contents |
|---|---|
| [`clip_split/`](clip_split/README.md) | Ground-truth pipeline: scale OCR, pour detection, the interactive annotator, clip cutting. It also holds the human-verified annotations. |
| [`pour_probe/`](pour_probe/README.md) | Everything model-side: V-JEPA 2 and DINOv3 probes, baselines (CNNs, audio, time priors), evaluation protocol, external test set, demo videos. |
| `SimLiquid/` | Submodule: BlenderProc liquid renderer, intended for simulation pretraining. Set up but not used in the results. |
| `nas_manifest_eigene-Experimente.txt` | Integrity manifest (file list and sizes) of the raw recordings on the NAS. |

**The recording setup.** 18 trials, 12 source→target container combinations
(kettle/teapot/bottle → mug/glass), three GoPros: CAM1 films the scale display (the ground
truth) and CAM2/CAM3 are the two wide views the probe sees. GT per clip is the poured mass
over time (`t_s,weight`), baseline-subtracted and monotone.
