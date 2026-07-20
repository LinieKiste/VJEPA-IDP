#!/usr/bin/env python3
"""Batch-drive the supervisor's scale-OCR pipeline over the CAM1 (scale-facing) videos.

Two modes:
  --roi              interactive: pops a window per CAM1 video that has no cached ROI;
                     draw a box around the scale display (ENTER/SPACE = confirm,
                     c = skip). ROIs are cached to rois.json (original-pixel coords),
                     so every later OCR run is headless.
  --trials N [N ..]  run OCR for these trial_ids (needs cached ROIs)
  --all              run OCR for every non-excluded trial with a cached ROI

OCR output: pouring/clip_split/ocr/<cam1_stem>.csv  (columns from OCR_Scale_REader:
frame_number, timecode, display_ocr, ocr_confidence). Every frame is OCR'd
(full native rate, 120 Hz on CAM1) — maximum temporal resolution for the GT trace.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "OCR_Scale_REader" / "video_processing"))
sys.path.insert(0, str(HERE))

import ocr_pipeline  # noqa: E402
from ocr_pipeline import process_video  # noqa: E402
from lcd_ocr import LcdBackend  # noqa: E402

CAM1_DIR = ROOT / "datasets" / "pouring_processed" / "CAM1"
OCR_DIR = HERE / "ocr"
ROIS_JSON = HERE / "rois.json"
TRIALS_CSV = HERE / "trials.csv"

MAX_DISPLAY_W = 1600


def load_trials(include_excluded=False):
    rows = list(csv.DictReader(open(TRIALS_CSV)))
    trials = []
    for r in rows:
        if not r["cam1_file"]:
            continue
        if not include_excluded and "exclude" in (r.get("flags") or ""):
            continue
        trials.append(r)
    return trials


def load_rois():
    return json.loads(ROIS_JSON.read_text()) if ROIS_JSON.exists() else {}


def grab_frame(video_path, frac=0.5):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def roi_session(trials):
    rois = load_rois()
    todo = [t for t in trials if Path(t["cam1_file"]).stem not in rois]
    print(f"{len(todo)} videos need an ROI ({len(rois)} cached).")
    for k, t in enumerate(todo, 1):
        stem = Path(t["cam1_file"]).stem
        frame = grab_frame(CAM1_DIR / t["cam1_file"])
        if frame is None:
            print(f"WARN: cannot read {t['cam1_file']}")
            continue
        h, w = frame.shape[:2]
        scale = min(1.0, MAX_DISPLAY_W / w)
        disp = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
        title = f"[{k}/{len(todo)}] trial {t['trial_id']} {stem} - draw box on scale display, ENTER=ok c=skip"
        r = cv2.selectROI(title, disp, showCrosshair=True)
        cv2.destroyAllWindows()
        if r == (0, 0, 0, 0):
            print(f"  skipped {stem}")
            continue
        roi = [int(round(v / scale)) for v in r]
        rois[stem] = {"roi": roi, "resolution": [w, h]}
        ROIS_JSON.write_text(json.dumps(rois, indent=2))
        print(f"  {stem}: roi={roi} (cached)")
    print(f"Done. {len(load_rois())} ROIs in {ROIS_JSON}")


def ocr_trials(trials, args):
    rois = load_rois()
    OCR_DIR.mkdir(exist_ok=True)
    for t in trials:
        stem = Path(t["cam1_file"]).stem
        if stem not in rois:
            print(f"SKIP trial {t['trial_id']} ({stem}): no ROI cached — run --roi first")
            continue
        out_csv = OCR_DIR / f"{stem}.csv"
        if out_csv.exists() and not args.overwrite:
            print(f"SKIP trial {t['trial_id']} ({stem}): {out_csv.name} exists")
            continue
        fps = float(t["cam1_fps"])
        interval = 1  # every frame: full native temporal resolution
        state = {"t0": time.time(), "last": 0.0}

        def cb(ev, state=state, tid=t["trial_id"]):
            if ev["level"] == "progress":
                now = time.time()
                if now - state["last"] > 5 and ev["total_frames"]:
                    state["last"] = now
                    frac = ev["frame"] / ev["total_frames"]
                    print(f"  trial {tid}: {100 * frac:5.1f}%  ({now - state['t0']:.0f}s)", flush=True)
            elif ev["level"] in ("warning", "error"):
                print(f"  trial {tid} {ev['level'].upper()}: {ev['message']}", flush=True)

        print(f"trial {t['trial_id']} ({stem}): fps={fps} interval={interval} "
              f"backend={args.backend} -> {out_csv.name}", flush=True)
        if args.backend == "lcd":
            # stateful per-video backend: calibrates a background model + digit
            # cells from the video itself, then registers under BACKENDS['lcd']
            backend_obj = LcdBackend(CAM1_DIR / t["cam1_file"], rois[stem]["roi"],
                                     grid=rois[stem].get("grid"))
            ocr_pipeline.BACKENDS["lcd"] = backend_obj
            qc = HERE / "qc" / f"lcd_calib_{stem}.png"
            cv2.imwrite(str(qc), backend_obj.qc_overlay())
            print(f"  lcd calib: {len(backend_obj.cells)} cells {backend_obj.cw}x{backend_obj.ch} "
                  f"fit_corr {backend_obj.fit_corr:.2f} -> {qc.name}", flush=True)
        process_video(
            CAM1_DIR / t["cam1_file"], tuple(rois[stem]["roi"]), out_csv,
            frame_interval=interval, progress_cb=cb,
            roi_source_resolution=tuple(rois[stem]["resolution"]),
            invert=args.invert, backend=args.backend,
        )
        print(f"  done in {time.time() - state['t0']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--roi", action="store_true", help="interactive ROI collection")
    mode.add_argument("--trials", nargs="+", type=str, help="trial_ids to OCR")
    mode.add_argument("--all", action="store_true", help="OCR all non-excluded trials")
    ap.add_argument("--backend", default="lcd", choices=["lcd", "tesseract", "segment"],
                    help="lcd = our background-model+template-correlation reader "
                         "(see lcd_ocr.py; tesseract/segment fail on the small display crops)")
    ap.add_argument("--invert", action="store_true", help="bright-on-dark display")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    trials = load_trials()
    if args.roi:
        roi_session(trials)
    else:
        if args.trials:
            trials = [t for t in trials if t["trial_id"] in set(args.trials)]
        ocr_trials(trials, args)


if __name__ == "__main__":
    main()
