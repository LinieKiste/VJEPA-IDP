"""Side-by-side demo videos: the pour on the right, live model readouts on the left.

One rendered mp4 per source pour. The left panel shows, at every instant, what each
frozen-backbone probe *currently believes* the flow rate is and how much liquid it thinks
has been poured so far; the right panel shows the frames those numbers came from, with the
256 px centre crop the encoders actually see outlined.

Three probes, all trained on the SAME fold-A split (val trials 8/13/21/24) so the
comparison is matched and every own-lab source below is genuinely held out:

    V-JEPA 2 attentive   attn_flow_both_lag0.7_foldA_best.pt            video
    DINOv3 attentive     attn_flow_both_dinov3_lag0.7_foldA_tpe_best.pt video (per-frame
                         backbone + the sinusoidal time stamp; without it the head is
                         frame-order blind, see _dino_encoder.sincos_temporal)
The audio row (frozen SoW wav2vec2 -> our ridge) is WIRED UP but switched off for every
source: it is not SoW's own method, and mixing it in invited exactly that misreading. SoW
is compared properly on ITS OWN data instead -- `infer_sow` / `--sow`, where their
published Eq. (6) pipeline runs end-to-end. Re-enable per source by putting "sow" back in
that source's models tuple.

Ground truth. Own-lab clips carry the full scale trace, so GT flow is drawn exactly as
``build_windows`` defines the training target (a 1.0 s difference of the weight curve
sampled at +LAG_FLOW). The external iPhone pours have ONE number each -- the final poured
mass -- so their GT is a horizontal line on the cumulative panel and "n/a" on the
instantaneous one.

    .venv/bin/python pouring/pour_probe/eval_videos.py --infer     # GPU, caches preds
    .venv/bin/python pouring/pour_probe/eval_videos.py --render    # CPU, writes mp4s
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

import clips_train_attn as ca
from clips_cnn_baseline import FOLDS, LAG_FLOW

ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = ROOT / "datasets/pouring_processed/clips"
EXT_DIR = ROOT / "datasets/eval/videos"
EXT_GT = ROOT / "datasets/eval/gt.csv"
CACHE = Path("/home/casimir/.cache/pour_probe")
PRED_NPZ = CACHE / "eval_videos_preds.npz"
OUT_DIR = ROOT / "datasets/eval/demo_videos"

FOLD = "A"
VAL_TRIALS = FOLDS[FOLD]
VJEPA_CKPT = CACHE / "attn_flow_both_lag0.7_foldA_best.pt"
DINO_CKPT = CACHE / "attn_flow_both_dinov3_lag0.7_foldA_tpe_best.pt"
SOW_CACHE = Path(os.environ.get("POUR_SOW_FEATS_DIR", CACHE / "clips_sow_feats"))
SOW_EXT_CACHE = CACHE / "eval_videos_sow_ext"

WINDOW_S, STRIDE_S, NUM_FRAMES, CROP = 1.0, 0.25, 16, 256

# The sources. Own-lab ids are fold-A trials only (held out for every probe here).
#
# The audio row runs ONLY on the own-lab clips. On the external pours it would be a ridge
# extrapolating past its 2.6-8.7 s fitting range (SoW injects ABSOLUTE time into the
# features) on an unseen microphone and room -- three stacked shifts, so its number would
# say nothing about audio pouring estimation. SoW gets a fair hearing on its OWN data
# instead, via `infer_sow` below.
SOURCES = [
    # (kind, key, cam, models, blurb)
    ("clip", "0047", "CAM2", ("vjepa", "dinov3"),
     "Eigenes Labor, Teekanne->Tasse, 31 g in 3,3 s (9,3 g/s) - der langsame Schüttvorgang"),
    ("clip", "0016", "CAM2", ("vjepa", "dinov3"),
     "Eigenes Labor, Wasserkocher->Glas, 335 g in 5,7 s (59 g/s) - der schnelle Schüttvorgang"),
    ("clip", "0045", "CAM3", ("vjepa", "dinov3"),
     "Eigenes Labor, Teekanne->Tasse, 244 g in 7,0 s, ferne Kameraansicht"),
    ("ext", "IMG_0866", None, ("vjepa", "dinov3"),
     "EXTERN: Wasserkocher->Topf, 895 g - außerhalb der Trainingsdomäne"),
    ("ext", "IMG_0868", None, ("vjepa", "dinov3"),
     "EXTERN: Wasserkocher->Topf, langsames Schütten, 128 g (~10 g/s)"),
    # YouTube stress test: a street walk with NO pouring at all -- the honest question is
    # whether the probes report ~0 g/s everywhere. Downloaded 2026-08-09 from
    # youtube.com/watch?v=1aedKShR1rA ("Manhattan Evening Walk", 640x360, first 15 s).
    # No scale GT exists; gt_total=None -> GT row shows k.A.
    ("yt", "manhattan_15s", None, ("vjepa", "dinov3"),
     "EXTERN: YouTube Straßenszene (Manhattan), KEIN Schüttvorgang - sollte 0 g/s melden"),
]

# --- the Sound-of-Water comparison, on THEIR data -----------------------------------
# Their pours fill the vessel to the brim, which is the boundary condition l(T)=0 that
# Eq. (6) `R = lambda(T)/4beta` needs -- so here their full no-measurement pipeline runs
# as published, and our probe can be compared against it properly. On our own pours that
# route is closed: the vessel never fills, so lambda(T) reads the leftover air column
# (measured: the same mug over 48 pours yields R = 1.7-11.4 cm).
SOW_ITEM = "VID_20240417_000535_2.2_8.0"     # container_7, 5.8 s, 199 mL, mono_frac 1.00
SOW_SUBSET = "S1"                            # cylindrical + transparent: level is VISIBLE
SOW_CKPT = CACHE / "attn_volume_sowS1_v_best.pt"
SOW_BETA = 0.62                              # their fixed cylindrical end correction


# --------------------------------------------------------------------------- sources

def ext_gt():
    """gt.csv is a bare column of finals with no filename column; row order == sorted
    filename order (confirmed by the user 2026-08-06)."""
    rows = np.atleast_1d(np.loadtxt(EXT_GT, delimiter=","))
    if rows.ndim == 2:
        rows = rows[:, -1]
    vids = sorted(EXT_DIR.glob("*.MOV"))
    return {v.stem: float(g) for v, g in zip(vids, rows)}


def decode_square(path, size=288):
    """(N,size,size,3) uint8 -- short-side resize then centre crop, the training pipeline."""
    import decord
    vr = decord.VideoReader(str(path))
    fps = float(vr.get_avg_fps())
    h, w = vr[0].shape[:2]
    scale = size / min(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    vr = decord.VideoReader(str(path), width=nw, height=nh)
    fr = vr.get_batch(np.arange(len(vr))).asnumpy()
    y0, x0 = (nh - size) // 2, (nw - size) // 2
    return fr[:, y0:y0 + size, x0:x0 + size], fps


def load_source(kind, key, cam):
    """-> dict(frames(N,288,288,3), fps, dur, gt_t, gt_w, gt_total, video_path, title)."""
    if kind == "clip":
        a = np.load(ca.FRAMES_DIR / cam / f"{key}.npz", allow_pickle=True)
        frames, fps = a["frames"], float(a["fps"])
        return {"frames": frames, "fps": fps, "dur": len(frames) / fps,
                "gt_t": a["gt_t"], "gt_w": a["gt_w"],
                "gt_total": float(a["gt_w"][-1]), "trial": str(a["trial_id"]),
                "video_path": CLIPS_DIR / cam / f"{key}.mp4", "cam": cam}
    if kind == "yt":
        return load_yt(key)
    frames, fps = decode_square(EXT_DIR / f"{key}.MOV")
    return {"frames": frames, "fps": fps, "dur": len(frames) / fps,
            "gt_t": None, "gt_w": None, "gt_total": ext_gt()[key], "trial": None,
            "video_path": EXT_DIR / f"{key}.MOV", "cam": None}


def load_yt(key):
    """YouTube stress-test videos: no scale GT, so gt_total is NaN (the GT row shows
    'k.A.' and no reference line is drawn)."""
    path = EXT_DIR / f"{key}.mp4"
    frames, fps = decode_square(path)
    return {"frames": frames, "fps": fps, "dur": len(frames) / fps,
            "gt_t": None, "gt_w": None, "gt_total": np.nan, "trial": None,
            "video_path": path, "cam": None}


def windows(dur):
    """Window mid-times and the (t0,t1) spans, dense stride for a smooth readout."""
    spans, t0 = [], 0.0
    while True:
        t1 = min(dur, t0 + WINDOW_S)
        spans.append((t0, t1))
        if t1 >= dur:
            break
        t0 += STRIDE_S
    return spans


def windows_mid(dur):
    return np.array([(a + b) / 2 for a, b in windows(dur)])


def cumulative(tmid, flow, dur):
    """Running trapezoid integral of a flow curve -> poured mass in g, at ``tmid``.

    Integrating between window CENTRES silently drops the first and last half-window
    (0.5 s each here), which biases every total LOW -- the `totals_from_flow` convention
    noted in CLAUDE.md, worth ~12 g on our clips. So the flow is held constant out to the
    clip bounds [0, dur] and the integral starts there instead. Shared with the renderer
    so the printed totals and the plotted curves are the same numbers.
    """
    if len(tmid) < 2:
        return np.zeros_like(flow)
    t = np.concatenate([[0.0], tmid, [dur]])
    f = np.concatenate([[flow[0]], flow, [flow[-1]]])
    c = np.concatenate([[0.0], np.cumsum(np.diff(t) * (f[1:] + f[:-1]) / 2)])
    return np.interp(tmid, t, c)


def total(tmid, flow, dur):
    return float(np.trapezoid(np.concatenate([[flow[0]], flow, [flow[-1]]]),
                              np.concatenate([[0.0], tmid, [dur]])))


def gt_curves(src, tmid):
    """GT flow/volume on the probes' own clock: exactly the training-target definition
    (build_windows), i.e. sampled at t + LAG_FLOW. None for the external pours."""
    if src["gt_t"] is None:
        return None, None
    gt, gw = src["gt_t"], src["gt_w"]
    flow, vol = [], []
    for t0, t1 in windows(src["dur"]):
        flow.append((np.interp(t1 + LAG_FLOW, gt, gw) - np.interp(t0 + LAG_FLOW, gt, gw))
                    / max(t1 - t0, 1e-3))
        vol.append(np.interp((t0 + t1) / 2 + LAG_FLOW, gt, gw))
    return np.asarray(flow), np.asarray(vol)


# --------------------------------------------------------------------------- probes

def fold_target_stats(lag=LAG_FLOW):
    """(ymean, ystd) over fold A's TRAINING windows -- the normalisation both attentive
    checkpoints were trained with. Reads only GT curves from the cache, never frames."""
    vals = []
    for cam in ("CAM2", "CAM3"):
        for f in sorted((ca.FRAMES_DIR / cam).glob("*.npz")):
            a = np.load(f, allow_pickle=True)
            if str(a["trial_id"]) in VAL_TRIALS:
                continue
            n, fps, gt, gw = len(a["frames"]), float(a["fps"]), a["gt_t"], a["gt_w"]
            t0 = 0.0
            while True:
                t1 = min(n / fps, t0 + WINDOW_S)
                vals.append((np.interp(t1 + lag, gt, gw) - np.interp(t0 + lag, gt, gw))
                            / max(t1 - t0, 1e-3))
                if t1 >= n / fps:
                    break
                t0 += 0.5                       # training stride, not the demo stride
    v = np.asarray(vals, np.float32)
    return float(v.mean()), float(v.std() + 1e-6)


def load_probe(backbone, ckpt, device="cuda"):
    """(encoder, head) for one backbone -- loaded ONCE and reused across all sources."""
    from head import build_head
    if backbone == "dinov3":
        import _dino_encoder
        enc = _dino_encoder.load_encoder(img_size=CROP, num_frames=NUM_FRAMES, device=device)
    else:
        from _encoder import load_encoder
        enc = load_encoder(img_size=CROP, num_frames=NUM_FRAMES, device=device)
    head = build_head(1).to(device)
    head.load_state_dict(torch.load(ckpt, map_location=device))
    head.eval()
    return enc, head


@torch.no_grad()
def run_attentive(src, enc, head, ymean, ystd, device="cuda", bs=8):
    """Dense sliding-window flow prediction in g/s from one attentive checkpoint."""
    frames, fps, N = src["frames"], src["fps"], len(src["frames"])
    # frames may be a lazy JpegFrames view (SoW cache), which has no .shape
    m = (np.asarray(frames[0]).shape[0] - CROP) // 2
    idxs = [np.clip((np.linspace(t0, t1, NUM_FRAMES) * fps).astype(int), 0, N - 1)
            for t0, t1 in windows(src["dur"])]
    mean, std = ca.MEAN.to(device), ca.STD.to(device)
    out = []
    for s in range(0, len(idxs), bs):
        chunk = np.stack([frames[i][:, m:m + CROP, m:m + CROP] for i in idxs[s:s + bs]])
        tok = ca.encode(enc, np.ascontiguousarray(chunk), device, mean, std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = head(tok).squeeze(1).float().cpu().numpy()
        out.append(p * ystd + ymean)
    return np.concatenate(out)


def sow_window_feats(feats, dur, spans):
    """Mean-pool the ~49 fps wav2vec series over each window (clips_sow_baseline.window_feats)."""
    F = len(feats)
    X = []
    for t0, t1 in spans:
        a = int(np.clip(t0 / dur * F, 0, F - 1))
        b = int(np.clip(np.ceil(t1 / dur * F), a + 1, F))
        X.append(feats[a:b].astype(np.float32).mean(0))
    return np.stack(X)


def fit_sow_ridge():
    """Refit the Sound-of-Water flow ridge on the NON-fold-A own-lab windows, using the
    cached per-frame wav2vec features. Same pipeline as clips_cnn_baseline.cv_r2
    (L2 row-norm + StandardScaler + Ridge); alpha picked by CV over folds B/C/D so
    fold A never touches the selection."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler

    X, y, trial = [], [], []
    for cam in ("CAM2", "CAM3"):
        clips = ca.load_clips(cam)
        wins = ca.build_windows({cam: clips}, WINDOW_S, 0.5, NUM_FRAMES, LAG_FLOW)
        cache = {}
        for w in wins:
            cid = w["clip"]
            if cid not in cache:
                f = SOW_CACHE / cam / f"{cid}.npz"
                cache[cid] = np.load(f) if f.exists() else None
            a = cache[cid]
            if a is None:
                continue
            X.append(sow_window_feats(a["feats"], float(a["dur"]),
                                      [(w["t0"], w["t1"])])[0])
            y.append(w["flow"]); trial.append(w["trial"])
    X, y, trial = np.stack(X), np.asarray(y), np.asarray(trial)
    keep = ~np.isin(trial, list(VAL_TRIALS))
    Xtr, ytr, gtr = X[keep], y[keep], trial[keep]

    def build(alpha):
        return make_pipeline(Normalizer(), StandardScaler(), Ridge(alpha=alpha))

    best, best_r2 = None, -9e9
    for alpha in (1, 10, 100, 1e3, 1e4, 1e5):
        r2s = []
        for f in ("B", "C", "D"):
            te = np.isin(gtr, list(FOLDS[f]))
            p = build(alpha).fit(Xtr[~te], ytr[~te]).predict(Xtr[te])
            r2s.append(r2_score(ytr[te], p))
        if np.mean(r2s) > best_r2:
            best, best_r2 = alpha, float(np.mean(r2s))
    pipe = build(best).fit(Xtr, ytr)
    print(f"  SoW ridge: alpha={best:g} (inner CV R2 {best_r2:+.3f}), "
          f"{len(Xtr)} train windows")
    return pipe


@torch.no_grad()
def sow_feats_for(src, kind, key, cam, device="cuda"):
    """Per-frame wav2vec features for one source: reuse the own-lab cache, extract and
    cache for the external MOVs."""
    if kind == "clip":
        a = np.load(SOW_CACHE / cam / f"{key}.npz")
        return a["feats"].astype(np.float32), float(a["dur"])
    SOW_EXT_CACHE.mkdir(parents=True, exist_ok=True)
    out = SOW_EXT_CACHE / f"{key}.npz"
    if not out.exists():
        import sow_model as sm
        model = sm.load_sow_model(device=device)
        wav = sm.load_audio(src["video_path"])
        _, feats = sm.predict_axial(model, wav, device=device)
        np.savez(out, feats=feats.astype(np.float16), dur=wav.shape[-1] / sm.SR)
        del model
        torch.cuda.empty_cache()
    a = np.load(out)
    return a["feats"].astype(np.float32), float(a["dur"])


# --------------------------------------------------------------------------- driver

def infer(device="cuda"):
    ymean, ystd = fold_target_stats()
    print(f"fold {FOLD} target norm: mean {ymean:.2f} g/s, std {ystd:.2f} g/s")
    needs_sow = any("sow" in m for _, _, _, m, _ in SOURCES)
    sow = fit_sow_ridge() if needs_sow else None

    print("\ndecoding sources ...")
    srcs = {}
    for kind, key, cam, models, blurb in SOURCES:
        srcs[key] = load_source(kind, key, cam)
        print(f"  {key:<10} {srcs[key]['dur']:5.1f} s @ {srcs[key]['fps']:.2f} fps  |  {blurb}")

    preds = {key: {} for key in srcs}
    for backbone, ckpt in (("vjepa", VJEPA_CKPT), ("dinov3", DINO_CKPT)):
        print(f"\n--- {backbone} ({ckpt.name})")
        enc, head = load_probe(backbone, ckpt, device)
        for kind, key, cam, models, _ in SOURCES:
            if backbone not in models:
                continue
            p = run_attentive(srcs[key], enc, head, ymean, ystd, device)
            preds[key][backbone] = p
            print(f"  {key:<10} peak {p.max():6.1f} g/s   integral "
                  f"{total(windows_mid(srcs[key]['dur']), p, srcs[key]['dur']):7.1f} g")
        del enc, head
        torch.cuda.empty_cache()

    print("\n--- sound of water (audio)"
          if needs_sow else "\n--- sound of water (audio): no source requests it")
    for kind, key, cam, models, _ in SOURCES:
        if "sow" not in models:
            continue
        feats, adur = sow_feats_for(srcs[key], kind, key, cam, device)
        p = sow.predict(sow_window_feats(feats, adur, windows(srcs[key]["dur"])))
        preds[key]["sow"] = p
        print(f"  {key:<10} peak {p.max():6.1f} g/s   integral "
              f"{total(windows_mid(srcs[key]['dur']), p, srcs[key]['dur']):7.1f} g")

    # merge, don't replace: infer_sow() writes its own keys into the same file
    store = dict(np.load(PRED_NPZ, allow_pickle=True)) if PRED_NPZ.exists() else {}
    print(f"\n{'source':<12}{'GT g':>8}{'V-JEPA':>9}{'DINOv3':>9}{'SoW':>9}")
    for kind, key, cam, models, _ in SOURCES:
        src, tmid = srcs[key], windows_mid(srcs[key]["dur"])
        gf, gv = gt_curves(src, tmid)
        tag = f"{kind}_{key}" + (f"_{cam}" if cam else "")
        store[f"{tag}__tmid"] = tmid
        for k, p in preds[key].items():
            store[f"{tag}__{k}"] = p
        store[f"{tag}__gt_flow"] = gf if gf is not None else np.array([np.nan])
        store[f"{tag}__gt_vol"] = gv if gv is not None else np.array([np.nan])
        store[f"{tag}__gt_total"] = np.array([src["gt_total"]])
        store[f"{tag}__dur"] = np.array([src["dur"]])
        store[f"{tag}__fps"] = np.array([src["fps"]])
        tot = {k: total(tmid, p, src["dur"]) for k, p in preds[key].items()}
        cells = "".join(f"{tot[k]:9.0f}" if k in tot else f"{'-':>9}"
                        for k in ("vjepa", "dinov3", "sow"))
        gtcell = f"{src['gt_total']:8.0f}" if not np.isnan(src["gt_total"]) else f"{'-':>8}"
        print(f"{key:<12}{gtcell}{cells}")

    PRED_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PRED_NPZ, **store)
    print(f"\nwrote {PRED_NPZ}")


# ------------------------------------------------------- SoW's own data (volume, mL)

def sow_split(subset=SOW_SUBSET, val_frac=0.25):
    """Reproduce clips_train_attn.main's container split for the sow dataset at
    --split_seed 0 (no permutation): containers largest-first until val_frac is met."""
    import sow_grid_cache as sg
    clips = sg.load_clips(subset)
    counts = {}
    for c in clips.values():
        counts[c["trial"]] = counts.get(c["trial"], 0) + 1
    val, acc, tgt = set(), 0, val_frac * sum(counts.values())
    for k in sorted(counts, key=lambda k: (-counts[k], k)):
        if acc >= tgt:
            break
        val.add(k); acc += counts[k]
    return clips, val


@torch.no_grad()
def infer_sow(device="cuda"):
    """One SoW video, three volume estimates in mL: their physics with the MEASURED
    radius (the target our probe was trained against), their physics with the radius
    ESTIMATED from the audio via Eq. (6), and our V-JEPA attentive probe."""
    import sow_physics as sp
    clips, val = sow_split()
    print(f"sow[{SOW_SUBSET}]: {len(clips)} videos, held-out containers {sorted(val)}")
    clip = clips[SOW_ITEM]
    if clip["trial"] not in val:
        raise SystemExit(f"{SOW_ITEM} is container {clip['trial']}, NOT held out")
    print(f"  {SOW_ITEM}  container {clip['trial']} (held out)")

    tr = [w for w in ca.build_windows({"sow": clips}, WINDOW_S, 0.5, NUM_FRAMES, 0.0)
          if w["trial"] not in val]
    y = np.asarray([w["volume"] for w in tr], np.float32)
    ymean, ystd = float(y.mean()), float(y.std() + 1e-6)
    print(f"  target norm from {len(tr)} training windows: "
          f"mean {ymean:.1f} mL, std {ystd:.1f} mL")

    a = np.load(Path("/home/casimir/.cache/pour_probe/sow_targets") / f"{SOW_ITEM}.npz",
                allow_pickle=True)
    lam, r_meas = a["lam"], float(a["r_cm"])
    r_est = float(lam[-1] / (4 * SOW_BETA))          # their Eq. (6), no measurement
    v_meas, _ = sp.decode_lambda_to_volume(lam, r_meas)
    v_est, _ = sp.decode_lambda_to_volume(lam, r_est)
    t_phys = sp.frame_times(len(lam))
    print(f"  radius: measured {r_meas:.2f} cm | estimated from lambda(T) {r_est:.2f} cm "
          f"({r_est / r_meas:.2f}x) -> volume scales by that squared")

    dur = len(clip["frames"]) / clip["fps"]
    src = {"frames": clip["frames"], "fps": clip["fps"], "dur": dur}
    enc, head = load_probe("vjepa", SOW_CKPT, device)
    tmid = windows_mid(dur)
    v_probe = run_attentive(src, enc, head, ymean, ystd, device)
    del enc, head
    torch.cuda.empty_cache()

    print(f"\n  final volume (mL):  GT(measured R) {np.interp(dur, t_phys, v_meas):6.1f}"
          f" | SoW(estimated R) {np.interp(dur, t_phys, v_est):6.1f}"
          f" | V-JEPA probe {v_probe[-1]:6.1f}")

    store = dict(np.load(PRED_NPZ, allow_pickle=True)) if PRED_NPZ.exists() else {}
    tag = f"sow_{SOW_ITEM}"
    store.update({f"{tag}__tmid": tmid, f"{tag}__vjepa": v_probe,
                  f"{tag}__t_phys": t_phys, f"{tag}__lam": lam,
                  f"{tag}__v_meas": v_meas, f"{tag}__v_est": v_est,
                  f"{tag}__r_meas": np.array([r_meas]), f"{tag}__r_est": np.array([r_est]),
                  f"{tag}__dur": np.array([dur]), f"{tag}__fps": np.array([clip["fps"]]),
                  f"{tag}__container": np.array([clip["trial"]])})
    np.savez(PRED_NPZ, **store)
    print(f"\nwrote {PRED_NPZ}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", action="store_true", help="GPU pass, cache predictions")
    ap.add_argument("--render", action="store_true", help="CPU pass, write the mp4s")
    ap.add_argument("--sow", action="store_true",
                    help="the Sound-of-Water comparison video on THEIR data instead")
    ap.add_argument("--only", default="", help="comma-separated source keys to restrict to")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.only:
        keep = set(args.only.split(","))
        SOURCES = [s for s in SOURCES if s[1] in keep]
    if args.infer:
        infer_sow(args.device) if args.sow else infer(args.device)
    if args.render:
        import eval_videos_render as r
        if args.sow:
            r.render_sow(SOW_ITEM, PRED_NPZ, OUT_DIR)
        else:
            r.render_all(SOURCES, PRED_NPZ, OUT_DIR)
    if not (args.infer or args.render):
        ap.error("pass --infer and/or --render")
