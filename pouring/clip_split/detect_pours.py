#!/usr/bin/env python3
"""Detect pouring events in scale-OCR weight traces (CSV from OCR_Scale_REader).

Protocol per trial video: cup on scale, tared to ~0 (sometimes reads ~2 g) -> weight
ramps up while pouring -> plateau (pour done, plateau value = poured mass GT) ->
cup removed: the scale goes NEGATIVE (-(cup+content)) but the OCR cannot read the
minus sign, so removal shows up as a large bogus POSITIVE spike (e.g. -347 -> "347").
Then the cup is emptied off-scale, returned, re-tared to ~0 for the next pour.

Consequences for detection:
  - one pour per region, and a real pour must RISE OUT OF A STABLE ~0 BASELINE;
    any rise starting from a plateau or from unstable chaos is a removal artifact
    and is IGNORED (never an event, never inside a clip)
  - the clip ends at the END of the plateau at the latest (removal chaos starts there)

Per input CSV:
  - parse display_ocr -> grams, mask unreadable frames (confidence < 0 / non-numeric),
    interpolate + centered median filter
  - find regions where weight > start threshold (short below-threshold dips bridged);
    event = region start -> first sustained plateau; require stable near-zero baseline
    directly before the region start
  - clip window = [rise_start - pad, plateau_start + pad], clamped to plateau end
    (minus margin) and video bounds

Outputs: events.csv (one row per pour) + qc/trace_<video>.png per input.
Usage: detect_pours.py CSV [CSV ...] [--out events.csv] [--qc-dir qc/]
"""

import argparse
import csv as csv_mod
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAM1_DIR = ROOT / "datasets" / "pouring_processed" / "CAM1"


def video_fps(stem):
    """fps of the CAM1 video matching a CSV stem (ffprobe)."""
    video = CAM1_DIR / f"{stem}.mp4"
    if not video.exists():
        return None
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video)]
    data = json.loads(subprocess.run(cmd, capture_output=True, check=True).stdout)
    for s in data["streams"]:
        if s.get("codec_type") == "video":
            num, den = s["r_frame_rate"].split("/")
            return float(num) / float(den)
    return None


def apply_ocr_overrides(df, patches, fps):
    """Apply user OCR corrections (annotate tool, ocr_overrides.json) to a raw
    OCR dataframe IN PLACE. Each patch = {t0, t1, value}; value None = mark the
    range unreadable. Returns a 'protect' mask: user-set samples are truth and
    must never be rejected by the filters."""
    t = df["frame_number"] / fps
    protect = pd.Series(False, index=df.index)
    for p in patches:
        m = (t >= p["t0"]) & (t <= p["t1"])
        if p.get("value") is None:
            df.loc[m, "display_ocr"] = ""
            df.loc[m, "ocr_confidence"] = -1.0
            protect[m] = False
        else:
            df.loc[m, "display_ocr"] = str(p["value"])
            df.loc[m, "ocr_confidence"] = 100.0
            protect[m] = True
    return protect


def load_overrides_for(stem):
    """User OCR corrections for one video (empty list if none)."""
    f = HERE / "ocr_overrides.json"
    if not f.exists():
        return []
    return json.loads(f.read_text()).get(stem, [])


def load_trace(csv_path, fps, args, df=None, protect=None):
    if df is None:
        df = pd.read_csv(csv_path, dtype={"display_ocr": str})
        patches = load_overrides_for(Path(csv_path).stem)
        if patches:
            protect = apply_ocr_overrides(df, patches, fps)
    df["t"] = df["frame_number"] / fps
    w = pd.to_numeric(df["display_ocr"], errors="coerce")
    w[df["ocr_confidence"] < 0] = np.nan
    df["w_raw"] = w

    if args.filter == "mono":
        # monotonicity prior: while pouring, true weight never decreases. A valid
        # sample must be >= the recent past (else it's a dropout reading low) and
        # <= the near future (else it's a digit-blending spike reading high).
        # Rolling MEDIANS of the trailing/leading windows make both tests robust
        # to other spikes inside the windows. The removal fall violates the prior
        # by construction; its samples get rejected + interpolated into a ramp
        # down, which the region/plateau-end logic handles as before.
        # Two scales: the long window is robust against long dropouts, the short
        # window catches violations right after a level change (e.g. a dip back
        # to 0 just after pour onset, where the long past median is still 0).
        # Lower bound = max of past medians, upper bound = min of future medians.
        dt = float(np.median(np.diff(df["t"])))
        lower = None
        upper = None
        for win_s in (args.mono_win_short, args.mono_win):
            win = max(3, int(round(win_s / dt)))
            past = w.rolling(win, min_periods=1).median().shift(1)
            future = w[::-1].rolling(win, min_periods=1).median()[::-1].shift(-1)
            lower = past if lower is None else np.maximum(lower, past)
            upper = future if upper is None else np.minimum(upper, future)
        outlier = ((w < lower - args.mono_tol) | (w > upper + args.mono_tol)) & w.notna()
        if protect is not None:
            outlier &= ~protect  # user-corrected samples are ground truth
        df["ema"] = np.nan
        df["zstar"] = np.nan
        df["outlier"] = outlier
        df["mono_past"] = lower
        df["mono_future"] = upper
        df["w_f"] = w.mask(outlier).interpolate(limit_direction="both")
    elif args.filter == "ema":
        # supervisor's method: x - EMA -> z-score the residual -> reject |y*| > 3.
        # Short OCR misreads (dropped digits) become huge residual spikes; sustained
        # levels (plateaus, removal spikes) are tracked by the EMA and survive.
        # Rejection is NEGATIVE-ONLY: the scale can never read below the true
        # weight mid-pour, so downward spikes are always OCR dropouts, while
        # positive residuals (EMA lag on a genuine ramp) are real. Sigma comes
        # from iterated symmetric trimming (a few huge spikes would otherwise
        # inflate it and hide smaller dropouts), floored at the scale resolution
        # (digital display -> most residuals are exactly 0). The whole
        # EMA->reject cycle runs twice: pass 2's EMA, computed on the cleaned
        # trace, hugs the true staircase, exposing dropouts that pass 1's
        # spike-dragged EMA absorbed.
        dt = float(np.median(np.diff(df["t"])))
        halflife = max(1.0, args.ema_halflife / dt)
        w_work = w.copy()
        outlier = pd.Series(False, index=w.index)
        for _ in range(args.passes):
            ema = w_work.ewm(halflife=halflife, ignore_na=True).mean()
            resid = w_work - ema
            keep = resid.notna()
            trimmed = pd.Series(False, index=resid.index)
            for _ in range(args.z_iters):
                r = resid[keep & ~trimmed]
                zstar = (resid - r.mean()) / max(r.std(), args.sigma_floor)
                new = zstar.abs() > args.z_thresh
                if new.equals(trimmed):
                    break
                trimmed = new
            new_out = zstar < -args.z_thresh
            if protect is not None:
                new_out &= ~protect
            outlier |= new_out
            w_work = w.mask(outlier)
        df["ema"] = ema
        df["zstar"] = zstar
        df["outlier"] = outlier
        df["w_f"] = w_work.interpolate(limit_direction="both")
    else:  # legacy median filter
        df["ema"] = np.nan
        df["zstar"] = np.nan
        df["outlier"] = False
        filled = w.interpolate(limit_direction="both")
        df["w_f"] = filled.rolling(args.median_k, center=True, min_periods=1).median()
    return df


def rolling_range(w, half):
    """max-min over a centered window of 2*half+1 samples."""
    s = pd.Series(w)
    return (s.rolling(2 * half + 1, center=True, min_periods=1).max()
            - s.rolling(2 * half + 1, center=True, min_periods=1).min()).to_numpy()


def find_events_regions(df, args):
    """Gate-C-validated region detector: rise out of a stable low baseline.

    Assumes each pour gets its own above-baseline region (re-tare between
    pours). Breaks when pours start from standing levels or share regions —
    but it is MORE robust on low-validity traces (e.g. the blurry portrait
    trial 5 at 45% valid), where the plateau-chain detector fragments.
    Selected via --detector regions (used per-trial where it wins the audit).
    """
    t = df["t"].to_numpy()
    w = df["w_f"].to_numpy()
    w_raw = df["w_raw"].to_numpy()
    dt = float(np.median(np.diff(t)))
    n = len(t)

    def sec(x):
        return max(1, int(round(x / dt)))

    stable = rolling_range(w, sec(args.stable_win) // 2) < args.stable_tol
    base0 = float(np.nanmedian(w[w <= np.nanpercentile(w, 20)]))
    thr = base0 + args.start_thresh

    above = w > thr
    raw_regions = []
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            raw_regions.append((i, j))
            i = j
        else:
            i += 1
    regions = []
    for r in raw_regions:
        if regions and t[r[0]] - t[regions[-1][1] - 1] < args.gap_bridge:
            regions[-1] = (regions[-1][0], r[1])
        else:
            regions.append(r)
    regions = [(i0, i1) for i0, i1 in regions if t[i1 - 1] - t[i0] >= args.region_min_dur]

    events, artifacts = [], []
    for i0, i1 in regions:
        b0 = max(0, i0 - sec(args.baseline_win))
        base = w[b0:i0]
        baseline_ok = (
            len(base) > 0
            and float(np.nanmedian(base)) < thr
            and float(np.nanmax(base) - np.nanmin(base)) < 3 * args.stable_tol
        )
        if not baseline_ok:
            artifacts.append((t[i0], t[i1 - 1]))
            continue
        k, min_stab = i0, sec(args.stable_min_dur)
        plateau = None
        while k < i1:
            if stable[k] and np.all(stable[k:min(k + min_stab, i1)]):
                plateau = k
                break
            k += 1
        flags = []
        if plateau is None:
            plateau = i1 - 1
            flags.append("truncated")
        level = float(np.median(w[plateau:min(plateau + sec(args.stable_win), i1)]))
        weight = level - float(np.nanmedian(base))
        if weight > args.step_min_weight:
            wr = w_raw[i0:plateau + 1]
            mids = int(np.sum((wr > thr + 15) & (wr < 0.7 * level)))
            if mids < args.step_min_mids:
                artifacts.append((t[i0], t[i1 - 1]))
                continue
        pe = plateau
        while pe < i1 and stable[pe] and abs(w[pe] - level) < args.stable_tol:
            pe += 1
        if pe >= n - 1:
            flags.append("ends_at_plateau")
        clip_a = max(t[i0] - args.pad, 0.0)
        clip_b = min(t[plateau] + args.pad, t[pe - 1] - args.drop_margin, t[-1])
        events.append({
            "rise_t": round(t[i0], 3), "plateau_t": round(t[plateau], 3),
            "clip_start_s": round(clip_a, 3), "clip_end_s": round(clip_b, 3),
            "weight_g": round(weight, 1),
            "plateau_end_t": round(t[pe - 1], 3),
            "flags": ";".join(flags),
        })
    return events, artifacts


def find_events(df, args):
    """Plateau-graph event detection.

    The old region-from-zero logic (rise out of a stable ~0 baseline) broke on
    dense traces: pours can start from a small STANDING level (residual water,
    imperfect tare), consecutive pours can share one above-zero region, and
    dense transition misreads fed fake 'intermediates' to the old step rule.

    New model: segment the filtered trace into stable PLATEAUS (level + span);
    every consecutive plateau pair (A -> B) with level_B - level_A >= thresh is
    a pour candidate, weight = level_B - level_A. A candidate is a REAL pour
    only with ramp evidence: raw readings strictly between the two levels,
    temporally spread (a hand pour crosses the mid-range over >= ramp_min_dur;
    a cup removal 0 -> unsigned-cup flips within ~1 display refresh, and its
    transition misreads cluster near the final value). Small deltas (< step
    weight gate) skip the evidence test — their mid-range barely exists.
    Falling transitions (removal, re-tare, emptying) are never events.
    """
    t = df["t"].to_numpy()
    w = df["w_f"].to_numpy()
    w_raw = df["w_raw"].to_numpy()
    dt = float(np.median(np.diff(t)))
    n = len(t)

    def sec(x):  # seconds -> samples
        return max(1, int(round(x / dt)))

    stable = rolling_range(w, sec(args.stable_win) // 2) < args.stable_tol

    # stable runs -> plateaus (min duration, merged if same level across a gap)
    plateaus = []  # [i0, i1, level, virtual]
    i = 0
    min_len = sec(args.stable_min_dur)
    while i < n:
        if stable[i]:
            j = i
            while j < n and stable[j]:
                j += 1
            if j - i >= min_len:
                level = float(np.median(w[i:j]))
                if plateaus and abs(level - plateaus[-1][2]) < args.stable_tol \
                        and t[i] - t[plateaus[-1][1] - 1] < args.gap_bridge:
                    plateaus[-1][1] = j
                    plateaus[-1][2] = float(np.median(w[plateaus[-1][0]:j]))
                else:
                    plateaus.append([i, j, level, False])
            i = j
        else:
            i += 1

    # VIRTUAL plateaus: a pour often starts before the post-tare baseline had
    # time to register as a stable plateau (fast experiment cadence). Such a
    # pour hides inside a falling gap (removal-fall -> turnaround -> new ramp):
    # insert the gap's turnaround minimum as a zero-dwell start plateau.
    half = sec(0.15)
    augmented = []
    for a, b in zip(plateaus, plateaus[1:]):
        augmented.append(a)
        g0, g1 = a[1], b[0]
        if g1 - g0 < 2:
            continue
        k_min = g0 + int(np.argmin(w[g0:g1 + 1]))
        m = float(np.median(w[max(0, k_min - half):k_min + half + 1]))
        if min(a[2], b[2]) - m >= args.start_thresh:
            augmented.append([k_min, k_min + 1, m, True])
    if plateaus:
        augmented.append(plateaus[-1])
    plateaus = augmented

    # trial's cup constant: after (almost) every pour the removed cup shows as
    # unsigned -cup = one recurring high level. Events ENDING near it are the
    # residual removal artifacts the rules above can miss (ringing, chain into
    # removal) -> flagged 'near_cup_level' for the Gate D audit, not dropped
    # (a real pour can coincidentally end near the cup weight).
    from collections import Counter
    high = [p[2] for p in plateaus if not p[3] and p[2] > 150]
    cup_level = None
    if high:
        binned = Counter(int(round(v / 4)) for v in high)
        b, cnt = binned.most_common(1)[0]
        if cnt >= 3:
            cup_level = 4 * b

    def step_is_ramp(a1, b0, la, delta):
        """Ramp evidence for one rising step.

        All rises: the filtered trace must be monotone through the transition
        (max drawdown < 2*stable_tol). Scale RINGING after a cup removal
        overshoots and settles back (e.g. 304 -> 383 -> 356 = unsigned
        -(cup) with spring oscillation), which no real pour does.

        Big rises (> step_min_weight) additionally need raw readings spread
        across the mid-range over >= ramp_min_dur: a removal 0 -> unsigned-cup
        flips within ~1 display refresh and its transition misreads cluster
        near the final value."""
        seg_f = pd.Series(w[a1:b0 + 1]).rolling(sec(0.3), center=True, min_periods=1).median().to_numpy()
        # (0.3 s median: single-frame OCR dips the mono filter left behind must
        # not trigger the drawdown test; ringing levels last ~0.5 s and survive)
        if len(seg_f) and float(np.max(np.maximum.accumulate(seg_f) - seg_f)) > 2 * args.stable_tol:
            return False
        if delta <= args.step_min_weight:
            return True
        lo, hi = la + 0.15 * delta, la + 0.85 * delta
        seg = w_raw[a1:b0 + 1]
        mid_idx = np.nonzero((seg > lo) & (seg < hi))[0]
        span = t[a1 + mid_idx[-1]] - t[a1 + mid_idx[0]] if len(mid_idx) else 0.0
        return len(mid_idx) >= args.step_min_mids and span >= args.ramp_min_dur

    def dwell(p):
        return t[p[1] - 1] - t[p[0]]

    # a pour = maximal ASCENDING chain of plateaus (slow spurt-pours read as a
    # staircase of tiny steps at 120 Hz) from a settled start level to a settled
    # end level. Chains break on falls, on jump steps without ramp evidence
    # (removals), and on long dwells (the cup settling after the pour).
    events, artifacts = [], []
    k = 0
    while k < len(plateaus) - 1:
        start = plateaus[k]
        # need a settled pre-pour level; a virtual turnaround minimum is a
        # valid start by construction (it IS the moment the cup was set down)
        if not start[3] and dwell(start) < args.baseline_win:
            k += 1
            continue
        # extend the ascending chain
        j = k
        while j + 1 < len(plateaus):
            a, b = plateaus[j], plateaus[j + 1]
            delta = b[2] - a[2]
            if delta <= -args.stable_tol:  # fall: removal / re-tare / emptying
                break
            # cup barrier: a step LANDING at the trial's unsigned-cup constant
            # is the removal (incl. ringing rises and small-delta steps from a
            # high pour level that the ramp rules can't separate) — end the
            # pour at the previous plateau
            if cup_level is not None and abs(b[2] - cup_level) <= 8:
                if delta >= args.start_thresh:
                    artifacts.append((t[a[1] - 1], t[b[0]]))
                break
            if not step_is_ramp(a[1], b[0], a[2], delta):
                if delta >= args.start_thresh:
                    artifacts.append((t[a[1] - 1], t[b[0]]))
                break
            j += 1
            if dwell(plateaus[j]) >= args.settle_dwell:
                break  # settled end level
        end = plateaus[j]
        weight = end[2] - start[2]
        if weight < args.start_thresh:
            k = max(j, k + 1)
            continue
        flags = []
        if end[1] >= n - 1:
            flags.append("ends_at_plateau")  # video ends on this level
        if cup_level is not None and abs(end[2] - cup_level) <= 8:
            flags.append("near_cup_level")   # audit: possibly a removal artifact
        clip_a = max(t[start[1] - 1] - args.pad, 0.0)
        clip_b = min(t[end[0]] + args.pad, t[end[1] - 1] - args.drop_margin, t[-1])
        events.append({
            "rise_t": round(t[start[1] - 1], 3), "plateau_t": round(t[end[0]], 3),
            "clip_start_s": round(clip_a, 3), "clip_end_s": round(clip_b, 3),
            "weight_g": round(weight, 1),
            "plateau_end_t": round(t[end[1] - 1], 3),
            "flags": ";".join(flags),
        })
        k = max(j, k + 1)
    return events, artifacts


def apply_monotone_fit(df, events):
    """Hard monotonicity: isotonic (non-decreasing) fit of w_f inside each pour
    interval [rise, plateau_end]. Guarantees zero drops within a pour; the trace
    outside pours (baseline, removal phase) is left untouched."""
    from sklearn.isotonic import IsotonicRegression

    t = df["t"].to_numpy()
    w = df["w_f"].to_numpy().copy()
    for ev in events:
        i0 = int(np.searchsorted(t, ev["clip_start_s"]))
        i1 = int(np.searchsorted(t, ev["plateau_end_t"])) + 1
        seg = slice(i0, i1)
        w[seg] = IsotonicRegression(increasing=True).fit_transform(t[seg], w[seg])
    df["w_f"] = w


def plot_trace(df, events, artifacts, args, out_png, title):
    if args.filter == "ema":
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                      height_ratios=[2.2, 1])
        # residual panel, mirroring the supervisor's sketch: y* with +-z bands
        ax2.plot(df["t"], df["zstar"], "-", lw=0.7, color="steelblue", label="y* = (y-mean)/std")
        ax2.axhline(0, color="gray", lw=0.6)
        ax2.axhline(-args.z_thresh, color="crimson", ls=":", lw=0.9)
        out = df["outlier"]
        ax2.plot(df["t"][out], df["zstar"][out], "o", ms=4, mfc="none", mec="red",
                 label=f"rejected (y* < -{args.z_thresh:g})")
        ax2.set_ylabel("residual y* (σ)")
        ax2.set_xlabel("t (s, CAM1 video time)")
        ax2.legend(loc="lower left", fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.set_xlabel("t (s, CAM1 video time)")
    masked = df["w_raw"].isna()
    ax.plot(df["t"], pd.to_numeric(df["display_ocr"], errors="coerce"),
            ".", ms=2, color="lightgray", label="raw OCR (incl. masked)")
    ax.plot(df["t"][~masked], df["w_raw"][~masked], ".", ms=2, color="steelblue", label="valid OCR")
    if args.filter == "ema":
        ax.plot(df["t"], df["ema"], "-", lw=0.8, color="darkorange", alpha=0.7, label="EMA")
    elif args.filter == "mono":
        ax.plot(df["t"], df["mono_past"], "-", lw=0.6, color="darkorange", alpha=0.6,
                label="past median (lower bound)")
        ax.plot(df["t"], df["mono_future"], "-", lw=0.6, color="purple", alpha=0.5,
                label="future median (upper bound)")
    if args.filter in ("ema", "mono"):
        rej = df["outlier"]
        ax.plot(df["t"][rej], df["w_raw"][rej], "x", ms=5, color="red", label="rejected")
    ax.plot(df["t"], df["w_f"], "-", lw=1.5, color="crimson", label="filtered")
    ax.axhline(args.start_thresh, color="gray", ls=":", lw=0.8)
    for a0, a1 in artifacts:
        ax.axvspan(a0, a1, color="red", alpha=0.12)
        ax.annotate("removal artifact\n(unsigned negative)", xy=(a0, ax.get_ylim()[1] * 0.75),
                    fontsize=7, color="darkred")
    for k, ev in enumerate(events, 1):
        ax.axvspan(ev["clip_start_s"], ev["clip_end_s"], color="green", alpha=0.15)
        ax.axvline(ev["rise_t"], color="green", ls="--", lw=0.8)
        ax.axvline(ev["plateau_t"], color="orange", ls="--", lw=0.8)
        ax.axvline(ev["plateau_end_t"], color="red", ls="--", lw=0.8)
        label = f"pour {k}: {ev['weight_g']} g" + (f" [{ev['flags']}]" if ev["flags"] else "")
        ax.annotate(label, xy=(ev["clip_start_s"], ax.get_ylim()[1] * 0.9),
                    fontsize=8, color="darkgreen")
    ax.set_ylabel("scale reading (g)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "events.csv")
    ap.add_argument("--qc-dir", type=Path, default=HERE / "qc")
    ap.add_argument("--start-thresh", type=float, default=5.0, help="g above zero = pour region")
    ap.add_argument("--region-min-dur", type=float, default=0.8, help="s, discard shorter blips")
    ap.add_argument("--stable-win", type=float, default=0.5, help="s window for stability check")
    ap.add_argument("--stable-tol", type=float, default=4.0, help="g max range within window")
    ap.add_argument("--stable-min-dur", type=float, default=0.4, help="s stability to call plateau")
    ap.add_argument("--gap-bridge", type=float, default=0.4, help="s, merge regions split by OCR dips")
    ap.add_argument("--baseline-win", type=float, default=0.8, help="s of stable ~0 required before a rise")
    ap.add_argument("--pad", type=float, default=1.0, help="s context before rise / after plateau")
    ap.add_argument("--drop-margin", type=float, default=0.3, help="s clip must end before drop-off")
    ap.add_argument("--step-min-weight", type=float, default=100.0,
                    help="g, ramp-evidence check for rises above this (no hand pour adds 40 g in one display refresh)")
    ap.add_argument("--step-min-mids", type=int, default=10,
                    help="min raw mid-range readings for a heavy rise to count as a real pour")
    ap.add_argument("--ramp-min-dur", type=float, default=0.35,
                    help="s, mid-range readings of a real pour span at least this long "
                         "(a removal jump flips within ~1 display refresh)")
    ap.add_argument("--settle-dwell", type=float, default=2.5,
                    help="s, a plateau at least this long ends the pour (cup settled); "
                         "shorter intermediate plateaus are spurt pauses within one pour")
    ap.add_argument("--detector", choices=["chain", "regions"], default="chain",
                    help="chain = plateau-chain (dense traces, pours from standing levels); "
                         "regions = Gate-C rise-from-baseline (more robust on low-validity traces)")
    ap.add_argument("--filter", choices=["mono", "ema", "median"], default="mono",
                    help="trace cleaning: monotone-pour prior (reject below-past / above-future), "
                         "EMA-residual z-score, or legacy median")
    ap.add_argument("--mono-win", type=float, default=0.4, help="s long trailing/leading median window")
    ap.add_argument("--mono-win-short", type=float, default=0.15, help="s short window (onset-local violations)")
    ap.add_argument("--mono-tol", type=float, default=2.0, help="g tolerance for monotonicity tests")
    ap.add_argument("--ema-halflife", type=float, default=0.15, help="s EMA halflife")
    ap.add_argument("--sigma-floor", type=float, default=1.0, help="g, lower bound for sigma (scale resolution)")
    ap.add_argument("--passes", type=int, default=2, help="full EMA->reject cycles (2nd pass EMA is spike-free)")
    ap.add_argument("--z-thresh", type=float, default=3.0, help="reject samples with |y*| above this")
    ap.add_argument("--z-iters", type=int, default=20, help="max sigma-trim iterations (stops on convergence)")
    ap.add_argument("--median-k", type=int, default=7, help="only for --filter median")
    return ap


def main():
    args = build_parser().parse_args()

    args.qc_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for csv_path in args.csvs:
        stem = csv_path.stem
        fps = video_fps(stem)
        if fps is None:
            print(f"WARN: no CAM1 video for {stem}, skipping (need fps)")
            continue
        df = load_trace(csv_path, fps, args)
        detector = find_events if args.detector == "chain" else find_events_regions
        events, artifacts = detector(df, args)
        if args.filter == "mono":
            apply_monotone_fit(df, events)
        out_png = args.qc_dir / f"trace_{stem}.png"
        plot_trace(df, events, artifacts, args, out_png,
                   f"{stem}  ({len(events)} pours, {len(artifacts)} removal artifacts)")
        print(f"{stem}: {len(events)} pours, {len(artifacts)} artifacts  -> {out_png.name}")
        for k, ev in enumerate(events, 1):
            print(f"   pour {k}: rise {ev['rise_t']:.2f}s plateau {ev['plateau_t']:.2f}s "
                  f"clip [{ev['clip_start_s']:.2f}, {ev['clip_end_s']:.2f}] "
                  f"weight {ev['weight_g']} g {ev['flags']}")
            rows.append({"cam1_stem": stem, "pour_idx": k, **ev})

    if rows:
        with open(args.out, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} events -> {args.out}")


if __name__ == "__main__":
    main()
