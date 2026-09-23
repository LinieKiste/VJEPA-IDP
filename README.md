# IDP SS26: Exploration von JEPA-Architekturen zur Anomalidetektion in Alltagsaktivitäten

Sammlung verschiedener Experimente mit V-JEPA.
Das Projekt besteht aus 2 Hauptphasen: Anomalidetektion mit V-JEPA und Schüttvolumenschätzung.

Zur Anomaliedetektion gehören:
`egoper_probe/`,
`egoper_vqa/`,
`exprt_probe/` und
`video_qa/`. Code zur Schüttvolumenschätzung ist in `pouring/`.

`mlflow.db` enthält [mlflow](https://mlflow.org/)-kompatible Daten zu Training-runs und anderen Experimenten.

## Layout

| Ordner | Inhalt |
|---|---|
| **`pouring/`** | `clip_split/` ist zur Datenverarbeitung; `pour_probe/` trainiert und evaluiert die Probes. Siehe [`pouring/README.md`](pouring/README.md). |
| **`presentations/presentation_final/`** | Abschlusspräsentation. evtl. hilfreich zum Einstieg. |
| `mlflow.db` | mlflow-Datenbank mit verschiedenen Ergebnissen zu den Experimenten. |
| `egoper_probe/` | V-JEPA-2-Probe zur Anomalieerkennung auf EgoPER. |
| `exprt_probe/` | Wie `egoper_probe`, aber auf dem eXprt-Datensatz. |
| `egoper_vqa/` | Zero-Shot-Video-QA-Baseline (Qwen2.5-VL) und ein Zero-Shot-Check des EK100-Action-Heads. |
| `video_qa/` | Replikation von V-JEPA 2 Appendix E (Encoder plus LLM). **`video_qa/model.py::build_encoder` ist der Encoder-Loader, den alle anderen Ordner verwenden.** |
| `CLAUDE.md` | Kontinuerliche Updates über den Verlauf und Status des Projekts. |

## Datensätze

Code erwartet alle Datensätze unter `datasets/`

| Pfad | Wofür | Woher |
|---|---|---|
| `datasets/pouring_processed/clips/` | Eigene 121 Schütt-Clips. Trainings- und Evaluationsgrundlage von `pour_probe/`. | TUM NAS |
| `datasets/sound-of-water/` | `pouring`. Audio-Vergleichsbaseline und Gegenprobe auf fremden Daten. | [Sound of Water](https://github.com/bpiyush/SoundOfWater) (Piyush et al.). |
| `datasets/egoper/` | `egoper_probe/` und `egoper_vqa/`. | [EgoPER](https://github.com/robert80203/EgoPER_official). |
| `datasets/eXprt-Daten/` | `exprt_probe/`. | Lehrstuhl |

## Setup

```bash
git clone --recurse-submodules <this-repo>     # OCR_Scale_REader braucht Repo-Zugriff
uv sync                                        # Python 3.10, torch cu132 (für Blackwell-GPUs, für andere Architekturen evtl. anpassen)
```

- **Checkpoints:** `checkpoints/vitl.pt` (V-JEPA 2 ViT-L, Key `target_encoder`) und optional
  `checkpoints/ek100-vitl-256.pt` (EK100-Probe als Warmstart). Beide aus dem
  [V-JEPA-2-Release](https://github.com/facebookresearch/vjepa2).
- **Caches:** Frame- und Feature-Caches (mehrere zehn GB) landen in `$POUR_CACHE`,
  Standard `~/.cache/pour_probe`. Am besten auf einer SSD.
- **mlflow:** `mlflow ui --backend-store-uri sqlite:///$PWD/mlflow.db`.

Getestet auf einer RTX 5060 Ti 16 GB. Alle Skripte vom Repo-Root aus starten.

## Hauptergebnisse reproduzieren

```bash
P=.venv/bin/python; D=pouring/pour_probe
$P $D/clips_extract.py                    # Features cachen (GPU)
$P $D/clips_eval_protocol.py --cam both   # Ridge-Probe + Baselines (CPU, ~1 min)
```

Die attentive Probe (Hauptergebnis, ~80 min GPU pro Fold) und alle weiteren Skripte:
[`pouring/pour_probe/README.md`](pouring/pour_probe/README.md).
