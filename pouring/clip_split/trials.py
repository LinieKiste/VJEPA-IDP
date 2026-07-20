#!/usr/bin/env python3
"""Group the renamed pouring-lab videos (CAM1/2/3) into trials.

Each trial = one recording session started near-simultaneously on all three GoPros.
Filenames encode frame-accurate start times: YYYYMMDD_HHMMSS_mmm_GX######.mp4
(from OCR_Scale_REader/video_processing/rename_gopro_videos.py).

Videos whose start times lie within CLUSTER_WINDOW_S of each other form one trial.
CAM1 is the scale-facing camera (ground-truth instrument); offsets of CAM2/3 are
reported relative to CAM1 in milliseconds.

Output: pouring/clip_split/trials.csv (modality column left blank for manual fill).
"""

import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets" / "pouring_processed"
OUT = Path(__file__).resolve().parent / "trials.csv"

CAMS = ["CAM1", "CAM2", "CAM3"]
NAME_RE = re.compile(r"^(\d{8})_(\d{6})_(\d{3})_(GX\w+)\.mp4$")
CLUSTER_WINDOW_S = 5.0  # CAM3 of one trial was started 3.6 s after CAM1/2; real trial gaps are >80 s


def ffprobe_info(path):
    """Return (fps, duration_s) for a video."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    data = json.loads(subprocess.run(cmd, capture_output=True, check=True).stdout)
    fps = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            num, den = stream["r_frame_rate"].split("/")
            fps = float(num) / float(den)
            break
    duration = float(data.get("format", {}).get("duration", "nan"))
    return fps, duration


def scan_videos():
    videos = []
    for cam in CAMS:
        for f in sorted((DATA / cam).glob("*.mp4")):
            m = NAME_RE.match(f.name)
            if not m:
                print(f"WARN: unparseable name skipped: {cam}/{f.name}")
                continue
            date_s, time_s, ms_s, gx = m.groups()
            start = datetime.strptime(date_s + time_s, "%Y%m%d%H%M%S").replace(
                microsecond=int(ms_s) * 1000
            )
            videos.append({"cam": cam, "file": f.name, "path": f, "start": start, "gx": gx})
    return sorted(videos, key=lambda v: v["start"])


def cluster(videos):
    trials = []
    current = []
    for v in videos:
        if current and (v["start"] - current[0]["start"]).total_seconds() > CLUSTER_WINDOW_S:
            trials.append(current)
            current = []
        current.append(v)
    if current:
        trials.append(current)
    return trials


def main():
    videos = scan_videos()
    print(f"Found {len(videos)} videos:", {c: sum(v['cam'] == c for v in videos) for c in CAMS})

    rows = []
    for tid, group in enumerate(cluster(videos), start=1):
        by_cam = {}
        for v in group:
            if v["cam"] in by_cam:
                print(f"WARN trial {tid}: duplicate {v['cam']} video {v['file']} "
                      f"(already have {by_cam[v['cam']]['file']})")
            by_cam.setdefault(v["cam"], v)

        anchor = by_cam.get("CAM1", group[0])
        row = {"trial_id": tid, "start_wallclock": anchor["start"].isoformat(sep=" ", timespec="milliseconds")}
        flags = []
        for cam in CAMS:
            v = by_cam.get(cam)
            if v is None:
                row[f"{cam.lower()}_file"] = ""
                row[f"{cam.lower()}_offset_ms"] = ""
                row[f"{cam.lower()}_fps"] = ""
                row[f"{cam.lower()}_duration_s"] = ""
                flags.append(f"missing_{cam}")
                continue
            fps, dur = ffprobe_info(v["path"])
            row[f"{cam.lower()}_file"] = v["file"]
            row[f"{cam.lower()}_offset_ms"] = round((v["start"] - anchor["start"]).total_seconds() * 1000)
            row[f"{cam.lower()}_fps"] = round(fps, 3)
            row[f"{cam.lower()}_duration_s"] = round(dur, 2)
        durs = [row[f"{c.lower()}_duration_s"] for c in CAMS if row[f"{c.lower()}_duration_s"] != ""]
        if durs and max(durs) - min(durs) > 5:
            flags.append("duration_mismatch")
        row["flags"] = ";".join(flags)
        row["modality"] = ""
        rows.append(row)
        print(f"trial {tid:02d}  {row['start_wallclock']}  "
              + "  ".join(f"{c}={by_cam[c]['gx'] if c in by_cam else '---'}" for c in CAMS)
              + (f"  [{row['flags']}]" if row["flags"] else ""))

    fieldnames = list(rows[0].keys())
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} trials -> {OUT}")


if __name__ == "__main__":
    main()
