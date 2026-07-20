#!/usr/bin/env python3
"""LCD scale-display reader: background-model + template-correlation backend.

Why this exists (Gate C finding, 2026-07-15): on the wide-shot CAM1 videos the
display crop is ~170x80 px and both OCR_Scale_REader backends fail there —
tesseract reads 3-25% of frames (7-segment digits are disconnected strokes) and
segment_ocr's Otsu binarization breaks on the uneven display lighting. The
supervisor's template backend assumes a FIXED digit grid, but the display
drifts several px within a trial (scale gets nudged between pours).

This reader exploits what those don't:
  * static camera -> per-pixel BACKGROUND MODEL: p90 over ~150 sampled frames
    (a segment's unlit state is its BRIGHT state; p90 survives segments that
    are lit most of the time AND hands/shadows passing over the display).
    Each frame is gain-normalized (median brightness) before comparison, so
    global illumination changes cancel.
  * reading works on the CONTINUOUS ratio image (bg-frame)/bg — no per-frame
    binarization, no contour fragmentation.
  * per cell, all 10 synthetic seven-segment digit templates are correlated
    (cv2 TM_CCOEFF_NORMED) over a +-SEARCH px window -> display drift is
    absorbed per frame. Best-correlating digit wins; low correlation on a
    non-blank cell rejects the WHOLE frame (no silent digit drops — a dropped
    digit would silently turn 356 into 35).
  * cells are auto-calibrated by scanning an "8" template over the "lit-ever"
    ratio image (p90-p10) at multiple scales; peaks = digit cells. Verified
    on trials 5/25/26 incl. the blurry portrait video.

Segment geometry (SEGMENT_BOXES / DIGIT_PATTERNS) is reused from the
supervisor's segment_ocr.py. Drop-in compatible with ocr_pipeline.process_video
as a *stateful* backend object (register in ocr_pipeline.BACKENDS per video):
preprocess_crop(frame, roi, invert) -> gray crop, read_digits(crop) -> (text, conf).

Validation (full videos, vs tesseract): trial 5 valid 46% (was 25% w/ 9.5%
garbage >500 g), trial 25 88.5% (was 3.3%), trial 26 69% (was n/a); >=99%
of readings plausible (<=500 g); ~2-3 ms/frame (~50x faster than tesseract).
"""

import cv2
import numpy as np

# supervisor's segment geometry (run_ocr.py puts OCR_Scale_REader/video_processing
# on sys.path before importing us)
from segment_ocr import SEGMENT_BOXES, SEGMENT_ORDER, DIGIT_PATTERNS

CALIB_FRAMES = 150      # frames sampled for the background model
BG_PCT = 98             # background = per-pixel p98 (unlit segments are bright; 98 not 90
                        # because a short video can show one value for >10% of its frames —
                        # trial 4 sat at "193" for ~65% and p90 absorbed the lit digits)
LOW_PCT = 10            # lit-ever contrast partner for calibration
ASPECT = 0.55           # seven-segment digit width/height
ANCHOR_SEARCH = 14      # px global drift search for the consensus fallback
CELL_SEARCH_PRIMARY = 8  # px per-cell search, primary independent pass
CELL_SEARCH = 3         # px residual per-cell search after the rigid offset is known
CORR_T = 0.50           # min correlation to accept a digit
# Blank test uses STRONG lit mass (ratio > ratio_t + BLANK_MARGIN): a hand /
# teapot SHADOW darkens part of the display and lights up the weak ratio mask
# over blank cells (trial 26's "no zeros after removal" bug), but shadows
# never push the ratio much past threshold — real digit strokes always do
# (measured: shadows 0.000 strong mass, weakest blurry digit 0.07).
BLANK_MARGIN = 0.08
BLANK_STRONG = 0.03     # strong-mass fraction below which a cell is blank
MAX_CELLS = 4

_DIGIT_BITS = {d: bits for bits, d in DIGIT_PATTERNS.items()}


def _digit_template(d, cw, ch):
    img = np.zeros((ch, cw), np.float32)
    for bit, name in zip(_DIGIT_BITS[d], SEGMENT_ORDER):
        if not bit:
            continue
        fx0, fy0, fx1, fy1 = SEGMENT_BOXES[name]
        img[int(fy0 * ch):max(int(fy0 * ch) + 1, int(fy1 * ch)),
            int(fx0 * cw):max(int(fx0 * cw) + 1, int(fx1 * cw))] = 1.0
    k = max(3, (min(cw, ch) // 8) * 2 + 1)
    return cv2.GaussianBlur(img, (k, k), k / 4)


def _gain_norm(gray):
    return gray.astype(np.float32) * (128.0 / max(float(np.median(gray)), 1.0))


class LcdBackend:
    """Stateful per-video backend (calibrates itself on construction).

    grid: optional manual override [bx, by, bw, bh, n] = digit-block box in
    crop coords + cell count, for videos where the auto "8"-scan fails (e.g.
    the blurred portrait trial 4). Cells = n equal divisions of the block.
    """

    def __init__(self, video_path, roi, grid=None):
        self.roi = tuple(roi)
        self.grid = grid
        self._calibrate(str(video_path))

    def _calibrate(self, path):
        x, y, w, h = self.roi
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stack = []
        for fi in np.linspace(0, total - 1, CALIB_FRAMES).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, f = cap.read()
            if not ok:
                continue
            stack.append(_gain_norm(cv2.cvtColor(f[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)))
        cap.release()
        S = np.stack(stack)
        self.bg = np.percentile(S, BG_PCT, 0)
        lit_ever = np.clip((self.bg - np.percentile(S, LOW_PCT, 0))
                           / np.maximum(self.bg, 1), 0, 0.6).astype(np.float32)

        if self.grid is not None:
            # manual digit-block box -> n equal cells
            bx, by, bw, bh, ncells = self.grid
            pitch = bw / ncells
            self.cw, self.ch = int(round(pitch * 0.85)), int(bh)
            self.cells = [(int(round(bx + (i + 1) * pitch - self.cw)), int(by))
                          for i in range(ncells)]
            c = float("nan")
        else:
            # multi-scale "8" scan -> cell size + positions
            best = None
            for ch in range(int(0.45 * h), int(0.98 * h)):
                cw = int(round(ch * ASPECT))
                if cw >= w // 2:
                    break
                res = cv2.matchTemplate(lit_ever, _digit_template("8", cw, ch), cv2.TM_CCOEFF_NORMED)
                c = float(res.max())
                if best is None or c > best[0]:
                    best = (c, ch, cw, res)
            c, self.ch, self.cw, res = best
            peaks, r = [], res.copy()
            while len(peaks) < MAX_CELLS:
                _, mx, _, (px, py) = cv2.minMaxLoc(r)
                if mx < 0.45 * c:
                    break
                peaks.append((px, py))
                r[:, max(0, px - int(0.8 * self.cw)):px + int(0.8 * self.cw)] = -1
            peaks.sort()
            self.cells = peaks
        ot, _ = cv2.threshold((lit_ever * 255).astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.ratio_t = max(0.08, 0.6 * ot / 255)
        self.templates = {d: _digit_template(d, self.cw, self.ch) for d in "0123456789"}
        self.fit_corr = c
        self.lit_ever = lit_ever  # kept for QC overlays

    # --- ocr_pipeline backend contract ---------------------------------
    def preprocess_crop(self, frame_bgr, roi, invert=False):
        x, y, w, h = roi
        return cv2.cvtColor(frame_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

    def read_digits(self, gray_crop):
        ratio = np.clip((self.bg - _gain_norm(gray_crop)) / np.maximum(self.bg, 1),
                        0, 0.5).astype(np.float32)
        # primary: independent per-cell read at the calibrated positions
        text, conf = self._read_cells(ratio, 0, 0, CELL_SEARCH_PRIMARY)
        if conf >= 0:
            return text, conf
        # fallback — moderate display drift (scale nudged >8 px): rigid
        # consensus offset = median of each cell's best-match displacement
        # over a wide window
        H, W = ratio.shape
        offs = []
        for cx0, cy0 in self.cells:
            wy0, wx0 = max(0, cy0 - ANCHOR_SEARCH), max(0, cx0 - ANCHOR_SEARCH)
            win = ratio[wy0:min(H, cy0 + self.ch + ANCHOR_SEARCH),
                        wx0:min(W, cx0 + self.cw + ANCHOR_SEARCH)]
            bc, bo = -1.0, None
            for tm in self.templates.values():
                res = cv2.matchTemplate(win, tm, cv2.TM_CCOEFF_NORMED)
                _, mx, _, (lx, ly) = cv2.minMaxLoc(res)
                if mx > bc:
                    bc, bo = mx, (wx0 + lx - cx0, wy0 + ly - cy0)
            if bc >= CORR_T:
                offs.append(bo)
        if not offs:
            return '', -1.0
        dx = int(np.median([o[0] for o in offs]))
        dy = int(np.median([o[1] for o in offs]))
        return self._read_cells(ratio, dx, dy, CELL_SEARCH)

    def _read_cells(self, ratio, dx, dy, search):
        H, W = ratio.shape
        digits, confs, started = [], [], False
        for cx0, cy0 in self.cells:
            cx, cy = cx0 + dx, cy0 + dy
            inner = ratio[max(0, cy):max(0, cy + self.ch), max(0, cx):max(0, cx + self.cw)]
            if inner.size == 0 or float((inner > self.ratio_t + BLANK_MARGIN).mean()) < BLANK_STRONG:
                if started:      # blank cell INSIDE the number = garbled frame
                    return '', -1.0
                continue
            wy0, wx0 = max(0, cy - search), max(0, cx - search)
            win = ratio[wy0:min(H, cy + self.ch + search), wx0:min(W, cx + self.cw + search)]
            scored = []          # (corr, digit, match loc)
            for d, tm in self.templates.items():
                res = cv2.matchTemplate(win, tm, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(res)
                scored.append((float(mx), d, loc))
            scored.sort(reverse=True)
            best_c, best_d, best_loc = scored[0]
            if best_c < CORR_T:  # unreadable non-blank cell -> reject whole frame
                return '', -1.0
            # near-tie disambiguation: digits differing in ONE segment (6/8,
            # 8/0, 9/8...) correlate almost identically because 13/14 segments
            # match — decide by sampling only the DIFFERING segments directly
            # (trial 26: display "348" read "346" at 6=0.594 vs 8=0.589)
            if scored[1][0] > best_c - 0.10:
                second_d = scored[1][1]
                bits_a, bits_b = _DIGIT_BITS[best_d], _DIGIT_BITS[second_d]
                gx, gy = wx0 + best_loc[0], wy0 + best_loc[1]
                votes = 0
                for bit_a, bit_b, name in zip(bits_a, bits_b, SEGMENT_ORDER):
                    if bit_a == bit_b:
                        continue
                    fx0, fy0, fx1, fy1 = SEGMENT_BOXES[name]
                    # the match loc can be ~3 px off the true digit — a LIT
                    # segment is found at some offset (max), a dark region
                    # stays dark at all of them
                    frac = 0.0
                    for oy in range(-3, 4):
                        for ox in (-1, 0, 1):
                            sx0 = gx + ox + int(fx0 * self.cw)
                            sx1 = max(sx0 + 1, gx + ox + int(fx1 * self.cw))
                            sy0 = gy + oy + int(fy0 * self.ch)
                            sy1 = max(sy0 + 1, gy + oy + int(fy1 * self.ch))
                            seg = ratio[max(0, sy0):sy1, max(0, sx0):sx1]
                            if seg.size:
                                frac = max(frac, float((seg > self.ratio_t).mean()))
                    votes += 1 if (frac >= 0.4) == bool(bit_a) else -1
                if votes < 0:
                    best_d = second_d
            started = True
            digits.append(best_d)
            confs.append(best_c)
        if not digits:
            return '', -1.0
        return ''.join(digits), 100.0 * min(confs)

    def qc_overlay(self):
        """Calibration QC image: lit-ever ratio with fitted cells."""
        vis = cv2.cvtColor((self.lit_ever * 400).clip(0, 255).astype(np.uint8),
                           cv2.COLOR_GRAY2BGR)
        for cx0, cy0 in self.cells:
            cv2.rectangle(vis, (cx0, cy0), (cx0 + self.cw, cy0 + self.ch), (0, 0, 255), 1)
        return cv2.resize(vis, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
