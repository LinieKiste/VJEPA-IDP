# clip_split/ — ground truth: from lab trials to per-pour clips

The raw recordings are 3-camera GoPro trials. CAM1 films a kitchen scale's 7-segment
display, and its reading over time *is* the ground truth. The pipeline runs in gated
stages, with a human check between them:

| # | Script | Does | Output |
|---|---|---|---|
| 1 | `trials.py` | Groups the renamed CAM1/2/3 videos into trials (by wall-clock start) | `trials.csv` |
| 2 | `run_ocr.py` → `lcd_ocr.py` | Reads the scale display on every CAM1 frame. `lcd_ocr.py` is our own reader (per-pixel background model + 7-seg template correlation); the supervisor's backends failed on this small display. | `ocr/<stem>.csv` (not in git, regenerable) |
| 3 | `detect_pours.py` | Cleans the trace (`--filter mono`: monotone prior + isotonic regression) and finds the pours (`--detector chain`) | `events.csv`, `qc/trace_*.png` |
| 4 | `qc_boundaries.py`, `qc_sync.py` | Contact sheets and a 3-camera sync preview, for checking the detected boundaries | `qc/` |
| 5 | `annotate.py` + `annotate_ui.html` | Local web app (`http://localhost:8765`): video plus trace, drag the pour spans, override the weight, correct OCR misreads | `annotations.json`, `ocr_overrides.json` |
| 6 | `cut_clips.py` | Cuts every accepted pour into CAM2/CAM3 clips plus a GT curve CSV | `datasets/pouring_processed/clips/` |

**Human-curated files; do not regenerate or overwrite them:**
- `annotations.json`: all 121 pour events, boundaries and weights verified by hand
- `ocr_overrides.json`: manual OCR corrections, stored as time ranges
- `rois.json`: location of the scale display in each CAM1 video
- `trials.csv`: trial ↔ video mapping, the container pair, exclusion flags
- `events.csv`: detector output; it seeds `annotations.json` and carries the `exclude` column

`run_ocr.py` needs the `OCR_Scale_REader` submodule at the repo root (private repo).

**Scale physics worth knowing:** the cup is tared to ≈0 g, the pour ramps up, and the plateau
is the poured mass. Lifting the cup off makes the reading go negative, but the OCR can't see
the minus sign, so that shows up as a bogus spike. A real pour always rises from a stable
~0 baseline.
