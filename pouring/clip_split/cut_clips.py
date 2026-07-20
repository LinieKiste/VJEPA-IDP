#!/usr/bin/env python3
"""Cut per-pour clips (CAM2 + CAM3) and per-clip GT scale-trace CSVs (Stage 4).

Final dataset layout (user-specified; CAM1 is the GT instrument, its video does
NOT ship):

    datasets/pouring_processed/clips/
      CAM2/0001.mp4 ...    CAM3/0001.mp4 ...    csv/0001.csv ...
      clips_manifest.csv

Clip ids are numbered sequentially over ALL (non-excluded) events in events.csv,
ordered by trial wallclock start then pour index — so run this on the COMPLETE
events.csv for final numbering (a --trials pilot run keeps the same global ids).

Per event:
  - CAM1-time window [clip_start_s, clip_end_s] from detect_pours; cropped so it
    fits inside EVERY camera (crop-to-shortest rule for duration-mismatch trials)
  - cam-local window = CAM1 window - cam offset (trials.csv, filename timecodes)
  - re-encode (frame-accurate): libx264 CRF 17, audio kept for sync checks
  - timestamps live in METADATA only: creation_time (clip wallclock start) +
    timecode stream tag; exact values also in the manifest
  - csv/NNNN.csv = the cleaned CAM1 scale trace cut to the window.
    Default (final GT): two columns `t_s,weight` (weight = cleaned, monotone,
    rescaled to the human-verified final mass). `--full-csv` writes the
    6-column provenance format instead (t_s, timecode, wallclock,
    weight_g_raw, weight_g_filtered, ocr_confidence).

Respects an optional `exclude` column in events.csv (Gate D audit): non-empty
value = event skipped and no clip id consumed.

Usage: cut_clips.py [--events events.csv] [--trials N ...] [--dry-run]
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "datasets" / "pouring_processed"
CLIPS = DATA / "clips"
OCR_DIR = HERE / "ocr"
TRIALS_CSV = HERE / "trials.csv"

sys.path.insert(0, str(HERE))
import detect_pours  # noqa: E402

FINAL_CAMS = ("CAM2", "CAM3")
TZ_SUFFIX = "+02:00"  # recordings 2026-07-13, Europe/Berlin (CEST); filenames are local time
MIN_CLIP_S = 1.5  # cropped windows shorter than this are dropped (flagged)


def load_trials():
    """cam1_stem -> trial row (parsed floats, wallclock datetime)."""
    out = {}
    for r in csv.DictReader(open(TRIALS_CSV)):
        if not r["cam1_file"]:
            continue
        r["start_dt"] = datetime.strptime(r["start_wallclock"], "%Y-%m-%d %H:%M:%S.%f")
        out[Path(r["cam1_file"]).stem] = r
    return out


def timecode_str(dt, fps):
    base = int(round(fps))
    ff = min(base - 1, int(round(dt.microsecond / 1e6 * base)))
    return f"{dt:%H:%M:%S}:{ff:02d}"


def filtered_trace(ocr_csv, fps):
    """Re-run the exact detect_pours cleaning (mono filter + isotonic fit)."""
    args = detect_pours.build_parser().parse_args([str(ocr_csv)])
    df = detect_pours.load_trace(ocr_csv, fps, args)
    events, _ = detect_pours.find_events(df, args)
    detect_pours.apply_monotone_fit(df, events)
    return df


def cut_video(src, local_a, dur, out, creation_dt, fps, dry):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{local_a:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-crf", "17", "-preset", "medium",
        "-c:a", "aac", "-movflags", "+faststart",
        "-metadata", f"creation_time={creation_dt.isoformat()}{TZ_SUFFIX}",
        "-timecode", timecode_str(creation_dt, fps),
        str(out),
    ]
    if dry:
        print("   DRY:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def load_annotations():
    """Events from the interactive annotator (annotations.json) in events.csv
    row format. Only COMPLETED, non-excluded events ship; the user-set
    weight_g is authoritative (per-clip curves get rescaled to match it)."""
    ann = json.loads((HERE / "annotations.json").read_text())
    rows, skipped = [], 0
    for stem, entry in ann.items():
        evs = sorted((e for e in entry["events"] if not e["excluded"]),
                     key=lambda e: e["clip_start_s"])
        for k, e in enumerate(evs, 1):
            if not e["completed"]:
                skipped += 1
                continue
            rows.append({
                "cam1_stem": stem, "pour_idx": str(k),
                "clip_start_s": str(e["clip_start_s"]), "clip_end_s": str(e["clip_end_s"]),
                "weight_g": str(e["weight_g"]), "flags": e.get("flags", ""),
                "exclude": "",
            })
    if skipped:
        print(f"NOTE: {skipped} non-excluded events are not marked completed -> not cut")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", type=Path, default=HERE / "events.csv")
    ap.add_argument("--trials", nargs="+", type=str, help="cut only these trial_ids (numbering stays global)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, cut nothing")
    ap.add_argument("--csv-only", action="store_true",
                    help="regenerate GT CSVs + manifest only (videos untouched)")
    ap.add_argument("--full-csv", action="store_true",
                    help="write the 6-column provenance CSV (t_s,timecode,wallclock,weight_g_raw,"
                         "weight_g_filtered,ocr_confidence) instead of the default final GT (t_s,weight)")
    args = ap.parse_args()

    trials = load_trials()
    if (HERE / "annotations.json").exists():
        print("using annotations.json (interactive annotator output)")
        events = list(load_annotations())
    else:
        events = list(csv.DictReader(open(args.events)))

    # global ordering + numbering (skip Gate-D excluded events)
    events = [e for e in events if not e.get("exclude")]
    events.sort(key=lambda e: (trials[e["cam1_stem"]]["start_dt"], int(e["pour_idx"])))
    for i, e in enumerate(events, 1):
        e["clip_id"] = f"{i:04d}"

    if args.trials:
        want = set(args.trials)
        events = [e for e in events if trials[e["cam1_stem"]]["trial_id"] in want]

    if not args.dry_run:
        for sub in (*FINAL_CAMS, "csv"):
            (CLIPS / sub).mkdir(parents=True, exist_ok=True)

    trace_cache = {}
    manifest = []
    for ev in events:
        stem = ev["cam1_stem"]
        tr = trials[stem]
        a, b = float(ev["clip_start_s"]), float(ev["clip_end_s"])

        # crop-to-shortest: the CAM1-time window must fit inside every camera
        lo, hi = 0.0, float(tr["cam1_duration_s"])
        for cam in FINAL_CAMS:
            key = cam.lower()
            if not tr[f"{key}_file"]:
                print(f"SKIP clip {ev['clip_id']} (trial {tr['trial_id']} pour {ev['pour_idx']}): missing {cam}")
                break
            off = float(tr[f"{key}_offset_ms"]) / 1000.0
            lo = max(lo, off)
            hi = min(hi, off + float(tr[f"{key}_duration_s"]))
        else:
            a2, b2 = max(a, lo), min(b, hi)
            cropped = (a2 > a + 1e-3) or (b2 < b - 1e-3)
            if b2 - a2 < MIN_CLIP_S:
                print(f"SKIP clip {ev['clip_id']} (trial {tr['trial_id']} pour {ev['pour_idx']}): "
                      f"window [{a2:.2f},{b2:.2f}] too short after crop-to-shortest")
                continue
            start_dt = tr["start_dt"] + timedelta(seconds=a2)
            print(f"clip {ev['clip_id']}: trial {tr['trial_id']} pour {ev['pour_idx']} "
                  f"[{a2:.2f},{b2:.2f}]s {ev['weight_g']} g"
                  + ("  (cropped-to-shortest)" if cropped else ""))

            if not args.csv_only:
                for cam in FINAL_CAMS:
                    key = cam.lower()
                    off = float(tr[f"{key}_offset_ms"]) / 1000.0
                    cut_video(DATA / cam / tr[f"{key}_file"], a2 - off, b2 - a2,
                              CLIPS / cam / f"{ev['clip_id']}.mp4", start_dt,
                              float(tr[f"{key}_fps"]), args.dry_run)

            # per-clip GT CSV from the CAM1 OCR trace
            if not args.dry_run:
                if stem not in trace_cache:
                    trace_cache[stem] = filtered_trace(OCR_DIR / f"{stem}.csv", float(tr["cam1_fps"]))
                df = trace_cache[stem]
                seg = df[(df["t"] >= a2) & (df["t"] <= b2)]
                # honor the annotated final weight: rescale the curve so the
                # poured mass (plateau - base) equals weight_g while staying
                # monotone; garbage traces (measured rise < 5 g) fall back to
                # a synthetic smoothstep ramp 0 -> weight_g
                wf = seg["w_f"].to_numpy().astype(float)
                tt = seg["t"].to_numpy()
                W = float(ev["weight_g"])
                base = 0.0
                if len(wf):
                    # monotone by construction inside the clip (the detector's
                    # isotonic pass only covered ITS windows, not user-annotated ones)
                    from sklearn.isotonic import IsotonicRegression
                    wf = IsotonicRegression(increasing=True).fit_transform(tt, wf)
                    base = float(np.median(wf[tt <= a2 + 0.6]))
                    plat = float(np.median(wf[tt >= b2 - 0.6]))
                    if abs(plat - base) >= 5.0:
                        # rescale so the pour's rise equals the annotated mass;
                        # wf stays the ABSOLUTE reading (base .. base+W)
                        wf = base + (wf - base) * (W / (plat - base))
                    else:
                        # unreadable trace: synthetic ramp, no meaningful baseline
                        x = np.clip((tt - (a2 + 0.15 * (b2 - a2))) / (0.7 * (b2 - a2)), 0, 1)
                        wf = W * x * x * (3 - 2 * x)
                        base = 0.0
                # final GT curve = poured mass since clip start (baseline removed,
                # clamped to [0, W] so it runs exactly 0 .. annotated final mass
                # for every clip; clamp only trims sub-gram rescale rounding)
                wf_gt = np.clip(wf - base, 0.0, W)
                with open(CLIPS / "csv" / f"{ev['clip_id']}.csv", "w", newline="") as f:
                    wr = csv.writer(f)
                    if args.full_csv:
                        # provenance/debug format: raw reading + timing + confidence
                        wr.writerow(["t_s", "timecode", "wallclock",
                                     "weight_g_raw", "weight_g_filtered", "ocr_confidence"])
                        for i, (_, r) in enumerate(seg.iterrows()):
                            wc = tr["start_dt"] + timedelta(seconds=float(r["t"]))
                            wr.writerow([
                                round(float(r["t"]) - a2, 4), r["timecode"],
                                wc.isoformat(sep=" ", timespec="milliseconds"),
                                "" if np.isnan(r["w_raw"]) else r["w_raw"],
                                round(float(wf[i]), 1), r["ocr_confidence"],
                            ])
                    else:
                        # final GT format: time + the cleaned monotone poured mass
                        wr.writerow(["t_s", "weight"])
                        for i, (_, r) in enumerate(seg.iterrows()):
                            wr.writerow([round(float(r["t"]) - a2, 4), round(float(wf_gt[i]), 1)])

            flags = ";".join(x for x in [ev.get("flags", ""), "cropped_to_shortest" if cropped else ""] if x)
            manifest.append({
                "clip_id": ev["clip_id"], "trial_id": tr["trial_id"], "pour_idx": ev["pour_idx"],
                "source_obj": tr["source_obj"], "target_obj": tr["target_obj"],
                "weight_g": ev["weight_g"],
                "wallclock_start": start_dt.isoformat(sep=" ", timespec="milliseconds"),
                "duration_s": round(b2 - a2, 3),
                "cam1_window_s": f"{a2:.3f}-{b2:.3f}",
                "cam2_src": tr["cam2_file"], "cam3_src": tr["cam3_file"],
                "cam1_ocr_csv": f"{stem}.csv", "flags": flags,
            })

    if manifest and not args.dry_run:
        out = CLIPS / "clips_manifest.csv"
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            wr.writeheader()
            wr.writerows(manifest)
        print(f"\n{len(manifest)} clips -> {CLIPS}  (manifest: {out.name})")
    elif args.dry_run:
        print(f"\nDRY RUN: {len(manifest)} clips would be cut -> {CLIPS}")


if __name__ == "__main__":
    main()
