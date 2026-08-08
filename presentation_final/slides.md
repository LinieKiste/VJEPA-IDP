---
theme: ./theme-tum
title: IDP Abschlusspräsentation
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

<img src="/JEPA_arch.jpg" alt="JEPA architecture" />


---
layout: image
size: full
crop: false
level: 2
title: V-JEPA
---

<img src="/VJEPA_arch.jpg" alt="JEPA architecture" />


---
layout: default
level: 2
title: "Attentive Probe: Architektur"
---

<!-- `layout: default`, not `layout: image`: the picture boxes of the image
     layouts run to the slide edge and this diagram's bottom labels would sit
     under the footer. The content box stops at 91.3 %. -->
<div style="height:100%;display:flex;align-items:center;justify-content:center"><img src="/attentive_probe.png" alt="Attentive-Probe-Architektur: eingefrorener Encoder, Attentive Pooler mit lernbarem Query, lineare Regression" style="max-width:100%;max-height:100%;object-fit:contain" /></div>


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

<img src="/SFLLaVA.jpg" alt="SlowFast-LLaVA" />

---
layout: default
level: 2
title: EgoPER
subtitle: Datensatz und Framework
---

- Problem: Fehlererkennung ohne Fehler als Trainingsdaten
- Datensatz: egozentrisches Fehler-Dataset für prozedurale Aufgaben, 386 Videos
- Methode (EgoPED): Action-Segmentation + kontrastive Schritt-Prototypen; Abweichung vom Prototyp ⇒ Fehler.

<br>
Eigene Experimente auf dem Datensatz

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
| **UWLPD** (Schenck & Fox) | Gießen mit Roboterarm, RGB + Liquid Masks, nur Füllstand % |
| **Sound of Water** (Bagad et al. 2024) | Smartphone-Videos, Audio + Video, nimmt gleichmäßigen Fluss und vollständige Füllung des Behälters an |
| **SimLiquid** (Huang et al. 2024) | synthetisch (BlenderProc), nur statische Bilder |
| **PSNN** (Wilson et al. 2019) | Audio + Video, Gewichtsklassen (0.2 oz) |
| **Eigener Datensatz** | 121 Clips, Schüttvolumen gewogen |

---
layout: default
level: 2
title: Ergebnisse
---

- Attentive Probe: Flow R² 0.81 ± 0.0, MAE ~8.5 g/s, schlägt Kinetics-Video-CNN (0.53) um +0.28 und Zeit-Profil (0.62)
- Volumen ist stark von Videolänge abhängig: Uhr allein 0.78, V-JEPA 0.58, V-JEPA + Uhr 0.8
- Höhere Auflösung (256->384) verbessert Ergebnis (+0.021, CAM3 +0.039)
- Unfreezing der letzten 4 Blöcke bringt nichts (−0.009)
- DINOv3: + temporal embedding 0.792, ROI-Crop verliert within-view
- Fusion V-JEPA + DINOv3 bringt nichts


---
layout: section
level: 1
title: Demo-Videos
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
layout: default
level: 2
title: Bland-Altman
subtitle: Schüttvolumen pro Schüttvorgang — V-JEPA-2-Attentive-Probe gegen Waage, n = 121
---

<div style="height:100%;display:flex;align-items:center;justify-content:center"><img src="/bland_altman.png" alt="Bland-Altman-Diagramm: Differenz zwischen vorhergesagtem und gewogenem Schüttvolumen gegen deren Mittelwert" style="max-width:100%;max-height:100%;object-fit:contain" /></div>


---
layout: default
level: 1
title: Future Work
---

- Größerer Datensatz
- mehr flow-variation
- mehr Negativbeispiele
- Auflösung vergrößern
- Architektur anpassen (helper inputs, preprocessing)
