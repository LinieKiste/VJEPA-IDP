---
theme: ./theme-tum
title: Anwendungen von JEPA im Kontext von Aktivitäten des täglichen Lebens
info: Interdisziplinäres Projekt SS26 — Abschlusspräsentation
canvasWidth: 720
aspectRatio: 16/9
colorSchema: light
transition: none
mdc: true
drawings:
  persist: false
themeConfig:
  chair:
    - Professur für Sportgeräte und –materialien
    - TUM School of Engineering and Design
    - Technische Universität München
  footer: Casimir Wallwitz | IDP SS26
hideInToc: true
---

Casimir Wallwitz

Technische Universität München

TUM School of Engineering and Design

Garching, 14. August 2026

<!--
Cover slide = slideLayout7 "1_Start" on slideMaster4: affiliation block top
left, wordmark top right, Uhrenturm sketch bottom right, no footer, no page
number. The three affiliation lines come from the `tumChair` headmatter.
-->

---
layout: default
title: Agenda
hideInToc: true
---

<Toc maxDepth="1" />

---
layout: section
level: 1
title: "JEPA: Joint-Embedding Predicitve Architecture"
---


---
layout: image
level: 2
size: full
crop: false
title: "JEPA: Architektur"
subtitle: funktioniert für Videos, Bilder, Sprache...
---

<img src="/JEPA_arch.png" alt="JEPA architecture" />

<CiteFooter id="vjepa" mode="short" />


---
layout: figure
level: 2
title: V-JEPA
---

<img src="/VJEPA_arch.png" alt="V-JEPA-2-Architektur: Encoder, EMA-Encoder, Predictor und L1-Verlust auf maskierten Videoframes" />

<CiteFooter id="vjepa2" mode="short" />


---
layout: figure
level: 2
title: "Attentive Probe: Architektur"
---

<img src="/attentive_probe.png" alt="Attentive-Probe-Architektur: eingefrorener Encoder, Attentive Pooler mit lernbarem Query, lineare Regression" />


---
layout: section
level: 1
title: Anomalieerkennung
---

---
layout: default
level: 2
title: Anomalieerkennung
subtitle: Problemstellung
---

- Eingabe: Video von Ausführung einer Alltagsaufgabe mit evtl. eingebauten Fehlern
- Ausgabe variabel: Timestamps, Binär, Fehlerklassen-labels, Text...
- Oft unterschiedliche Problemstellungen und Lösungsansätze in der Literatur
- Lange Videos sind resourcenintensiv!
- Nicht jeder Fehler kann im Datensatz vorkommen
- Definition von "Fehler" oft unklar


---
layout: two-cols
level: 2
title: SlowFast-LLaVA
subtitle: Effizientere Video-Inputs
---

#### Idee

- State-of-the-art für lange Videos
- spart Speicher
- kombiniert wenige, hochauflösende Bilder mit vielen niedrigauflösenden
- Details werden sichtbar, Kontext über längere Zeit bleibt

::right::

<img src="/SFLLaVA.png" alt="SlowFast-LLaVA" />

<CiteFooter id="slowfastllava" />

---
layout: default
level: 2
title: EgoPER
subtitle: Datensatz und Framework
---

- Problemstellung: Fehlererkennung ohne Fehler als Trainingsdaten
- Egozentrisches Fehler-Dataset für prozedurale Aufgaben, 386 Videos<Cite id="egoper" />
- Methode (EgoPED) nutzt Action-Segmentation + kontrastive Schritt-Prototypen
- Abweichung vom Prototyp ⇒ Fehler.

<br>
Eigene Experimente auf dem Datensatz (qualitativ)

- Zero-Shot-VQA (Qwen2.5-VL) als Vergleich
- V-JEPA+EK100-Verb-Erkennung
- eingefrorene V-JEPA-2-Features + MLP layer und SlowFast-inputs


---
layout: image
level: 2
size: full
crop: false
title: EgoPER
subtitle: Ergebnisse der V-JEPA-2-Probe
---

<img src="/egoper_roc.png" alt="ROC-Kurve EgoPER" />


---
layout: section
level: 1
title: Schüttvolumen mit V-JEPA
---

---
layout: default
level: 2
title: Schüttvolumen - Problemstellung
---

<video src="/pour_cam3_0001.mp4" controls autoplay loop muted style="width: 100%; border-radius: 4px;"></video>


---
layout: default
level: 2
title: Schüttvolumen
---

- Relevant in feinteiligeren Anomalieerkennungs-frameworks
- Bestehende Methoden meistens auf gut kontrollierten Szenarien
- Audio hilfreich

<br>

| Datensatz | Domäne |
| --- | --- |
| **UWLPD** (Schenck & Fox)<Cite id="uwlpd" /> | Gießen mit Roboterarm, RGB + Liquid Masks, nur Füllstand % |
| **Sound of Water** (Bagad et al. 2025)<Cite id="sow" /> | Smartphone-Videos, Audio + Video, nimmt gleichmäßigen Fluss und vollständige Füllung des Behälters an |
| **SimLiquid** (Huang et al. 2024)<Cite id="simliquid,blenderproc" /> | synthetisch (BlenderProc), nur statische Bilder |
| **PSNN** (Wilson et al. 2019)<Cite id="psnn" /> | Audio + Video, Gewichtsklassen (0.2 oz) |
| **Eigener Datensatz** | 121 Clips, Schüttvolumen gewogen |

<CiteFooter mode="short" />

---
layout: default
level: 2
title: Versuchsaufbau
---

- 2 Kameras: Eine näher, die Andere weiter entfernt
- Gemessen wird das Gewicht der geschütteten Flüssigkeit mit einer Haushaltswaage
- Eine Kamera filmt die 7-Segment-Anzeige der Waage
- Gewicht wird via OCR ausgelesen, Fehler werden per Hand korrigiert
- Verzögerung von Bild zu Messung: empirisch auf 0.7s festgelegt

<!-- img mit tool-->

---
layout: default
level: 2
title: Trainingsprotokoll
---

- frozen Backbone: V-JEPA 2 ViT-L
- V-JEPA: 1-Sekündige Clips, jeweils 16 Bilder
- "Attentive Probe" als regression head
- Regression auf Flussrate
- Training: SmoothL1-Loss, AdamW, linearer Warmup + Cosine-Decay, Checkpoint auf Rolling-Mean val-R²
- Auswertung nach 4-Fold Validierung


---
layout: default
level: 2
title: Ergebnisse
---

- Attentive Probe: Flow R² 0.81 ± 0.0, MAE ~8.5 g/s, schlägt Kinetics-Video-CNN (0.53) um +0.28 und Zeit-Profil (0.62)
- Volumen ist stark von Videolänge abhängig: Zeitmessung allein 0.78, V-JEPA 0.58, V-JEPA + Zeit 0.8
- Höhere Auflösung (256->384) verbessert Ergebnis (+0.021, CAM3 +0.039)
- Unfreezing der letzten 4 Blöcke bringt nichts (−0.009)
- DINOv3: + temporal embedding 0.792, ROI-Crop verliert within-view
- Fusion V-JEPA + DINOv3 bringt nichts


---
layout: section
level: 1
title: Demo-Videos
hideInToc: true
---


---
layout: image
size: full
level: 2
title: Eigener Datensatz
---

<video src="/demo_eigen_schnell.mp4" controls muted style="height: 234px; width: auto; max-width: 100%; display: block; margin: 0 auto 6px; border-radius: 4px;"></video>


---
layout: image
level: 2
title: Langsames Schütten
size: full
---

<video src="/demo_extern_langsam.mp4" controls muted style="height: 234px; width: auto; max-width: 100%; display: block; margin: 0 auto 6px; border-radius: 4px;"></video>


---
layout: default
level: 2
title: Sound of Water
---

<video src="/demo_sow_vergleich.mp4" controls muted style="height: 234px; width: auto; max-width: 100%; display: block; margin: 0 auto 6px; border-radius: 4px;"></video>


---
layout: figure
level: 2
title: Bland-Altman
subtitle: Schüttvolumen pro Schüttvorgang — V-JEPA-2-Attentive-Probe gegen Waage, n = 121
---

<img src="/bland_altman.png" alt="Bland-Altman-Diagramm: Differenz zwischen vorhergesagtem und gewogenem Schüttvolumen gegen deren Mittelwert" />


---
layout: two-cols
level: 1
title: Future Work
---


Anomaliedetektion
- Größerer Datensatz
- V-JEPA als z.B. Action Classifier in EgoPED

::right::

Schüttvolumen
- Größerer Datensatz
- mehr flow-variation
- mehr Negativbeispiele
- Auflösung vergrößern
- Architektur anpassen (helper inputs, preprocessing)


---
layout: default
level: 1
title: Quellen
hideInToc: true
---

<References cols="2" />


---
layout: image
level: 2
title: Flow ohne Schütten
size: full
hideInToc: true
---

<video src="/demo_yt_manhattan.mp4" controls muted style="height: 234px; width: auto; max-width: 100%; display: block; margin: 0 auto 6px; border-radius: 4px;"></video>

<CiteFooter id="manhattanwalk" mode="short" />


---
layout: figure
level: 2
title: ROI-Crop
subtitle: Crop auf die Gefäße — bringt Stabilität auf ungesehener Kameraansicht (3 Folds, Punkte = Folds)
hideInToc: true
---

<img src="/roi_crop.png" alt="ROI-Crop-Ergebnisse: Center-Crop gegen Detektor-ROI, within-view und ungesehene Ansicht" />


---
layout: figure
level: 2
title: EgoPED
hideInToc: true
---

<img src="/EgoPED.png" alt="EgoPED-Architektur: Action-Segmentation und kontrastive Schritt-Prototypen für prozedurale Fehlererkennung" />

<CiteFooter id="egoper" mode="short" />
