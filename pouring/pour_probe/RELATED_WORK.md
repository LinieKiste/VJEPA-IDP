# Related work: pouring volume / flow / mass estimation — methods & datasets

Scope for OUR use case: estimate **poured amount over time** (mass in g / flow rate g·s⁻¹ /
volume) from a **third-person camera**, small real dataset, frozen-backbone + light-probe
protocol. Compiled from the related-work section of *The Sound of Water* (Bagad et al. 2024,
arXiv 2411.11222) plus follow-up searches (2026-07-19).

## TL;DR / recommendation
- **No prior work reports poured-MASS REGRESSION (R² / MAE in grams) from third-person video.**
  The closest visual-ish method (Wilson et al.) does audio-visual weight **classification** (0.2 oz
  bins); *Sound of Water* does **audio-only** height/flow. So our frozen-video-probe **flow-rate
  regression** sits in a genuinely open niche — good for the writeup, and nothing "beats" it.
- **Sound of Water claims SOTA only for AUDIO** ("first to demonstrate human-like capabilities…
  from sound alone"). **No head-to-head vs any vision method.** Their only quantitative comparison
  is linear-probing on Wilson et al.'s audio-visual data. → their headline does **not** compete
  with a vision approach; treat their model as a **cross-modal baseline**, not a rival.
- **Best next baseline (feasible now — our clips have AAC audio):** run the *Sound of Water*
  pretrained audio encoder as a **frozen-feature baseline on our own clips** (linear probe → flow,
  same 4-fold-by-trial CV as V-JEPA/ResNet). This is the natural "audio foundation model vs video
  foundation model" comparison and directly analogous to the ResNet baseline.

## Methods — ordered by relevance to "poured amount over time"

### A. Closest task (predict poured mass / rate over time)
| method | modality | output | notes |
|---|---|---|---|
| **Wilson, Sterling, Lin — PSNN (IROS 2019)** | audio+visual CNN | poured **weight** (0.2 oz **classes**), liquid, container, overflow | THE closest prior task. Classification not regression; mel-spectrogram + video frames. [proj](http://gamma.cs.unc.edu/PSNN/) |
| **Bagad, Tapaswi, Snoek, Zisserman — The Sound of Water (2024)** | audio (wav2vec2 pitch) | container height, **flow rate**, level, time-to-fill | Audio-only SOTA; physics assumes **cylindrical** resonating air column. Pretrained model on HF `bpiyush/sound-of-water-models`. |

### B. Vision liquid perception (detect/track → volume via pixels)
| method | output | notes |
|---|---|---|
| **Schenck & Fox — FCNs for Liquids (IJRR 2018, arXiv 1703.01564)** | pixel liquid mask → volume by counting | UWLPD lineage. Key finding: **temporal info is essential** (matches our ResNet-fails result). |
| **Schenck & Fox — Visual Closed-Loop Pouring (ICRA 2017, arXiv 1610.02610)** | target-volume control | earlier, control-oriented |
| **Narasimhan et al.** | self-supervised liquid tracking | tracks stream, not mass |
| **Lin et al. (ICCV 2023)** | liquid stream tracking during pouring | recent vision stream tracker |
| **Transparent Liquid Segmentation for Robotic Pouring (CoRL 2022)** | transparent-liquid mask | segmentation for hard (clear) liquids |

### C. Simulation-based volume
| method | output | notes |
|---|---|---|
| **Huang et al. — SimLiquid / SimLiquid20k (J. Field Robotics 2024)** | volume (mL) | YOLO multi-task on fully-synthetic data → **~5% real-world volume error**. We have the renderer (`pouring/SimLiquid`). Reference number to beat/cite. |

### D. Multimodal / rich-sensory (vision + audio + force/haptics/IMU)
| method | notes |
|---|---|
| **Liang et al. — Liquid Pouring Monitoring via Rich Sensory Inputs (ECCV 2018, arXiv 1808.01725)** | vision+audio+haptics; stage/overflow monitoring |
| Liu / Zheng / Wu et al. | dynamic liquid state from force-torque, hand trajectory, IMU + vision/audio |

### E. Audio / tactile level & height (non-visual references)
| method | notes |
|---|---|
| **Making Sense of Audio Vibration for Liquid Height (arXiv 1903.00650)** | audio→height in robotic pouring |
| **Understanding Dynamic Auditory & Tactile Perception for Water Filling Level (IJSR 2024)** | audio+tactile filling level |
| **RoboCAP (arXiv 2405.07423)** | capacitive-sensing pour classification/precision |

## Datasets — relevant to us
| dataset | size | modality | labels | for us |
|---|---|---|---|---|
| **Sound of Water** (bpiyush/sound-of-water) | **805 vids / 50 containers**, 1.48 GB | audio+video | container height, flow (physics) | DOWNLOADED → `datasets/sound-of-water`. Secondary vision benchmark + their model as audio baseline. |
| **Wilson et al. 2019** | 500+ vids, only 276 pouring, 4 containers | audio+video | poured weight | closest label to ours (mass); small container variety |
| **UWLPD** (we have) | real + sim | RGB + liquid masks; sim has mL | pixel masks / mL | our earlier proxy-flow experiments |
| **SimLiquid** (we have renderer) | synthetic (10k+ renderable) | RGB+depth+normals | **mL** | our planned sim-pretrain source |
| **Liquid Content Detection benchmark (Sensors 2023)** | — | RGB | transparent-container fill | transparent-container niche |
| **OUR own-lab clips** | 121 clips / 18 trials, 12 container combos | video (+audio) | **real grams over time** | the deliverable; unusually clean per-frame mass GT |

## Where our contribution sits
- Everyone above is either **robotics/control**, **pixel detection**, **classification**, or
  **audio-only**. A **frozen video-foundation-model probe doing continuous flow-rate regression on
  third-person clips with dense gram-accurate GT** is not covered → clean niche.

### Measured comparison (2026-07-19) — one protocol: frozen backbone + probe, 4-fold CV by trial
| method | modality | flow R² | volume R² |
|---|---|---|---|
| **V-JEPA 2 + attentive probe (ours)** | video | **0.81 ± 0.04** | 0.667 |
| V-JEPA 2 + ridge mean-pool | video | 0.69 | 0.450 |
| Sound of Water wav2vec (frozen) | audio | 0.648 | 0.802 |
| s3d / r2plus1d_18 (Kinetics-400) | video | ~0.53 | ~0.24 |
| SoW decoded λ (zero-shot physics) | audio | 0.272 | 0.353 |
| time_prof (normalized-time poly-4) | — | 0.569 | 0.557 |
| raw time (linear) | — | 0.221 | 0.778 |
| ResNet-50 per-frame (strawman) | video | ~0.00 | ~0.00 |

**How to state this honestly in the writeup:**
1. The headline is **flow**, where V-JEPA 2 (0.81) beats every alternative: the fair temporal-CNN
   baseline by +0.28, the audio FM by +0.16, and the boundary-cheating temporal prior by +0.24.
2. **Do not claim a win on volume.** Cumulative poured mass is largely predictable from elapsed time
   (raw-time linear = 0.778), so audio's 0.802 and our 0.667 are both in clock territory. Volume is the
   wrong quantity to argue perception with.
3. The ResNet-50 ~0.00 row is a **structural** result (a per-frame 2D CNN cannot represent a temporal
   quantity), not a baseline anyone should be impressed by beating — report r2plus1d/s3d instead.
4. SoW's own claim is SOTA **for audio**, with no vision comparison, so they are a cross-modal reference
   point rather than a competitor.

### Our probe on the SoW dataset (2026-07-20) — a near-null that reinforces point 2
We ran the V-JEPA attentive probe on the SoW videos (target = their audio-physics V(t), grouped by
container). Within-video R² vs the time-only baseline:

| subset | probe | time-only | edge |
|---|---|---|---|
| transparent | 0.16 | 0.06 | +0.10 |
| all cylindrical | 0.29 | 0.25 | +0.03 |
| **opaque only** | 0.75 | **0.98** | **−0.23** |
| all shapes | 0.80 | 0.80 | +0.00 |

**SoW pours are constant-flow, so V(t) is ~linear in time and the clock predicts it near-perfectly.** The
probe beats the clock only on transparent containers, is a wash on the full set, and is *worse* than the
clock on opaque (invisible level + perfectly time-predictable → vision only adds noise). This is the same
"volume is clock territory" point on an external dataset, and it is exactly why **our varied-flow own
dataset is the contribution**: there the clock gets 0.62 on flow and V-JEPA 0.85, because the pour rate
actually changes and time cannot stand in for perception.

### Head-init ablation (frozen encoder, our clips, flow, both cams)
| init | flow R² |
|---|---|
| EK100 action pretrain | 0.854 |
| random | 0.843 |
| SoW pouring pretrain | 0.831 |

Initialization is a wash (spread 0.02) — **with a frozen backbone the representation, not the pretrained
head, carries the signal.** On-domain (SoW) pretraining does not beat off-domain (EK100) or random.
