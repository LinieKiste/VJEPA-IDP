#!/usr/bin/env python3
"""Local web tool for reviewing/fixing pour annotations (Gate D, interactive).

    .venv/bin/python pouring/clip_split/annotate.py   ->  http://localhost:8765

Left: trial list (completion badges). Main: CAM1 video (native scrubbing).
Bottom: zoomable trace timeline with the auto-detected events as draggable
spans; select an event to adjust its boundaries (drag edges or set to the
playhead), fix the final weight, add missed pours, exclude bogus ones, and
mark each clip as completed.

Weight overrides: the per-clip GT curve is rescaled so it still makes sense —
w' = base + (w - base) * W_user / (plateau - base) within the clip (positive
scaling keeps monotonicity; the curve then ends at base + W_user). If the
measured rise is garbage (|plateau-base| < 5 g, e.g. the blurry trial 4), the
export falls back to a smooth synthetic ramp 0 -> W_user. The same preview is
drawn live in the UI (orange overlay).

State lives in pouring/clip_split/annotations.json (initialized from
events.csv on first run; events.csv is never modified). cut_clips.py will read
annotations.json once the audit is done.

Endpoints: /api/trials, /api/trial/<stem> (trace + events),
POST /api/events/<stem> (save), /video/<stem>.mp4 (Range-capable).
No third-party deps (stdlib http.server + the repo's detect_pours filtering).
"""

import csv
import json
import sys
import threading

import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAM1_DIR = ROOT / "datasets" / "pouring_processed" / "CAM1"
OCR_DIR = HERE / "ocr"
EVENTS_CSV = HERE / "events.csv"
ANNOT_JSON = HERE / "annotations.json"
UI_HTML = HERE / "annotate_ui.html"

sys.path.insert(0, str(HERE))
import detect_pours  # noqa: E402

PORT = 8765
_lock = threading.Lock()


def load_trials():
    trials = []
    for r in csv.DictReader(open(HERE / "trials.csv")):
        if not r.get("cam1_file"):
            continue
        if "exclude" in (r.get("flags") or ""):
            continue
        stem = Path(r["cam1_file"]).stem
        trials.append({
            "trial_id": r["trial_id"], "stem": stem,
            "source_obj": r.get("source_obj") or "?",
            "target_obj": r.get("target_obj") or "?",
            "fps": float(r["cam1_fps"]), "duration_s": float(r["cam1_duration_s"]),
        })
    return trials


def init_annotations(trials):
    """Build annotations.json from events.csv (only for stems not yet present)."""
    ann = json.loads(ANNOT_JSON.read_text()) if ANNOT_JSON.exists() else {}
    by_stem = {}
    if EVENTS_CSV.exists():
        for r in csv.DictReader(open(EVENTS_CSV)):
            by_stem.setdefault(r["cam1_stem"], []).append(r)
    changed = False
    for t in trials:
        if t["stem"] in ann:
            continue
        events = []
        for k, r in enumerate(by_stem.get(t["stem"], []), 1):
            events.append({
                "id": f"{t['stem'][-8:]}_{k}",
                "clip_start_s": float(r["clip_start_s"]),
                "clip_end_s": float(r["clip_end_s"]),
                "weight_g": float(r["weight_g"]),
                "auto_weight_g": float(r["weight_g"]),
                "completed": False,
                "excluded": bool(r.get("exclude")),
                "exclude_reason": r.get("exclude") or "",
                "source": "auto",
                "flags": r.get("flags") or "",
            })
        ann[t["stem"]] = {"trial_id": t["trial_id"], "events": events}
        changed = True
    if changed:
        ANNOT_JSON.write_text(json.dumps(ann, indent=1))
    return ann


def trace_payload(stem, fps):
    """Filtered + raw trace (user OCR overrides applied + re-filtered),
    decimated 4x -> ~30 Hz (display updates ~10 Hz, nothing is lost)."""
    csv_path = OCR_DIR / f"{stem}.csv"
    args = detect_pours.build_parser().parse_args([str(csv_path)])
    df = detect_pours.load_trace(csv_path, fps, args)  # auto-applies ocr_overrides.json
    patches = detect_pours.load_overrides_for(stem)
    tt = df["t"].to_numpy()
    edited = np.zeros(len(tt), bool)
    for p in patches:
        edited |= (tt >= p["t0"]) & (tt <= p["t1"])
    t = tt[::4]
    w = df["w_f"].to_numpy()[::4]
    raw = df["w_raw"].to_numpy()[::4]
    return {"t": [round(float(x), 3) for x in t],
            "w": [round(float(x), 1) for x in w],
            "raw": [None if not (x == x) else round(float(x), 1) for x in raw],
            "edited": [bool(x) for x in edited[::4]],
            "patches": patches}


class Handler(BaseHTTPRequestHandler):
    trials = None
    traces = {}

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/" or p == "/index.html":
            body = UI_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/api/trials":
            ann = json.loads(ANNOT_JSON.read_text())
            out = []
            for t in self.trials:
                evs = ann.get(t["stem"], {}).get("events", [])
                live = [e for e in evs if not e["excluded"]]
                out.append({**t, "n_events": len(live),
                            "n_completed": sum(1 for e in live if e["completed"])})
            self._json(out)
        elif p.startswith("/api/trial/"):
            stem = p.rsplit("/", 1)[1]
            t = next((x for x in self.trials if x["stem"] == stem), None)
            if t is None:
                return self._json({"error": "unknown trial"}, 404)
            if stem not in self.traces:
                self.traces[stem] = trace_payload(stem, t["fps"])
            ann = json.loads(ANNOT_JSON.read_text())
            self._json({"trial": t, "trace": self.traces[stem],
                        "events": ann[stem]["events"]})
        elif p.startswith("/video/"):
            self._video(p.rsplit("/", 1)[1])
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        p = self.path.split("?")[0]
        if p.startswith("/api/ocr/"):
            # replace the OCR correction patches for one video, re-filter,
            # return the fresh trace
            stem = p.rsplit("/", 1)[1]
            t = next((x for x in self.trials if x["stem"] == stem), None)
            if t is None:
                return self._json({"error": "unknown trial"}, 404)
            n = int(self.headers.get("Content-Length", 0))
            patches = json.loads(self.rfile.read(n))
            ovr_file = HERE / "ocr_overrides.json"
            with _lock:
                ovr = json.loads(ovr_file.read_text()) if ovr_file.exists() else {}
                ovr[stem] = patches
                tmp = ovr_file.with_suffix(".tmp")
                tmp.write_text(json.dumps(ovr, indent=1))
                tmp.replace(ovr_file)
            self.traces.pop(stem, None)  # invalidate -> next build re-filters
            self.traces[stem] = trace_payload(stem, t["fps"])
            self._json({"ok": True, "trace": self.traces[stem]})
        elif p.startswith("/api/events/"):
            stem = p.rsplit("/", 1)[1]
            n = int(self.headers.get("Content-Length", 0))
            events = json.loads(self.rfile.read(n))
            with _lock:
                ann = json.loads(ANNOT_JSON.read_text())
                if stem not in ann:
                    return self._json({"error": "unknown trial"}, 404)
                ann[stem]["events"] = events
                tmp = ANNOT_JSON.with_suffix(".tmp")
                tmp.write_text(json.dumps(ann, indent=1))
                tmp.replace(ANNOT_JSON)
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def _video(self, name):
        f = CAM1_DIR / name
        if not f.exists() or f.suffix != ".mp4":
            return self._json({"error": "no video"}, 404)
        size = f.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = min(int(b), size - 1) if (a and b) else end
        length = end - start + 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(f, "rb") as fh:
            fh.seek(start)
            left = length
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)


def main():
    trials = load_trials()
    init_annotations(trials)
    Handler.trials = trials
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"annotation tool: http://localhost:{PORT}   ({len(trials)} trials)")
    print("state: pouring/clip_split/annotations.json  (Ctrl-C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
