"""Build per-video regression targets for the *Sound of Water* dataset by running their
frozen audio model and decoding its wavelength prediction into poured volume.

    audio -> SoW model -> lambda(t) -> (physics) -> V_poured(t) [mL], Q(t) [mL/s]

The SoW videos ship no measured volume — only container geometry. So the target here is
**model-derived**, and every number produced downstream measures *agreement between a
video probe and an audio-physics estimate*, not agreement with ground truth. That is
still a meaningful external test (the audio estimate is independent of the video
evidence, and is the published SOTA for this quantity), but it must be stated as such.
Validation of the audio estimate against gram-accurate mass is a separate experiment on
our own clips.

Subsets (all `clean == yes`, grouped by container for CV):
  S1  cylindrical + transparent  — physics valid AND the level is visible on camera
  S2  cylindrical (adds opaque)  — physics valid, level often NOT visible
  S3  all shapes                 — cylinder approximation; more data, noisier target
  S2o cylindrical + opaque only  — the sharpest "level is invisible" test

Note we do NOT apply a strict taper cut: it drops 5 of 11 transparent containers, and a
container-grouped split needs containers more than it needs a perfect cylinder. Residual
taper acts as a per-container scale error on the target — a real limitation, recorded in
the manifest as the `taper` column so it can be checked against the residuals.

Usage:
    .venv/bin/python pouring/pour_probe/sow_targets.py --build
    .venv/bin/python pouring/pour_probe/sow_targets.py            # sanity report
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd

import sow_physics as sp

ROOT = Path(__file__).resolve().parents[2]
SOW = ROOT / "datasets/sound-of-water"
CACHE = Path(os.environ.get("POUR_SOW_TARGETS_DIR",
                            "/home/casimir/.cache/pour_probe/sow_targets"))
SPLITS = ("train", "test_I", "test_II", "test_III")


def metadata():
    """Union of their split CSVs, de-duplicated to one row per video."""
    d = pd.concat([pd.read_csv(SOW / "splits" / f"{s}.csv") for s in SPLITS])
    d = d.drop_duplicates("item_id").reset_index(drop=True)
    d["meas"] = d["measurements"].apply(ast.literal_eval)
    d["r_cm"] = d["meas"].apply(sp.radius_cm)
    d["taper"] = d["meas"].apply(sp.taper)
    return d


def subsets(d):
    clean = d[d["clean"] == "yes"]
    cyl = clean[clean["shape"] == "cylindrical"]
    return {"S1": cyl[cyl["visibility"] == "transparent"],
            "S2": cyl,
            "S2o": cyl[cyl["visibility"] == "opaque"],
            "S3": clean}


def build(device="cuda"):
    import torch
    import sow_model as sm
    d = metadata()
    model = sm.load_sow_model(device=device)
    CACHE.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm
    rows = []
    for _, r in tqdm(d.iterrows(), total=len(d), desc="sow targets"):
        vid = SOW / r["file_name"]
        out = CACHE / f"{r['item_id']}.npz"
        if not vid.exists():
            continue
        if out.exists():
            a = np.load(out)
            rows.append(dict(item_id=r["item_id"], v_end=float(a["v"][-1]),
                             mono_frac=float(a["mono_frac"])))
            continue
        wav = sm.load_audio(vid)
        with torch.no_grad():
            dur = wav.shape[-1] / sm.SR
            o = model(wav.reshape(1, 1, 1, -1).to(device),
                      torch.tensor([[[0.0, dur]]], dtype=torch.float32))
            axial = o["axial"][0, 0].float().cpu().numpy()
        lam = axial @ np.linspace(0, sm.W_MAX, sm.N_BINS)
        # fraction of the RAW decode that is already non-increasing: a quality signal
        # that survives the monotone clamp we apply for the actual target
        v_raw, _ = sp.decode_lambda_to_volume(lam, r["r_cm"], clamp_monotone=False)
        mono_frac = float((np.diff(v_raw) >= -1e-9).mean())
        v, q = sp.decode_lambda_to_volume(lam, r["r_cm"])
        t = sp.frame_times(len(v))
        np.savez(out, t=t.astype(np.float32), v=v.astype(np.float32),
                 q=q.astype(np.float32), lam=lam.astype(np.float32),
                 mono_frac=mono_frac, r_cm=float(r["r_cm"]),
                 container=str(r["container_id"]), dur=float(dur))
        rows.append(dict(item_id=r["item_id"], v_end=float(v[-1]), mono_frac=mono_frac))
    return pd.DataFrame(rows)


def report():
    """Sanity gate: is the decoded target physically plausible?"""
    d = metadata().set_index("item_id")
    recs = []
    for f in sorted(CACHE.glob("*.npz")):
        a = np.load(f, allow_pickle=True)
        if f.stem not in d.index:
            continue
        r = d.loc[f.stem]
        cap = np.pi * r["r_cm"] ** 2 * r["meas"].get("net_height", np.nan)
        recs.append(dict(item_id=f.stem, container=str(r["container_id"]),
                         shape=r["shape"], visibility=r["visibility"],
                         taper=r["taper"], v_end=float(a["v"][-1]),
                         mono_frac=float(a["mono_frac"]), capacity=cap,
                         frac_cap=float(a["v"][-1]) / cap if cap == cap else np.nan,
                         dur=float(a["dur"])))
    t = pd.DataFrame(recs)
    print(f"=== decoded targets: {len(t)} videos ===")
    print(f"  raw decode already monotone: median {t.mono_frac.median():.2%} of frames "
          f"({(t.mono_frac > 0.9).mean():.0%} of videos >90%)")
    print(f"  poured volume V(T): median {t.v_end.median():.0f} mL "
          f"[{t.v_end.quantile(0.05):.0f}-{t.v_end.quantile(0.95):.0f}]")
    print(f"  V(T) as fraction of container capacity: median {t.frac_cap.median():.2f} "
          f"({(t.frac_cap.between(0.05, 1.25)).mean():.0%} within a plausible 0.05-1.25)")
    print(f"  implausible (V(T)>1.25x capacity or <5 mL): "
          f"{int(((t.frac_cap > 1.25) | (t.v_end < 5)).sum())} videos\n")
    for name, sub in subsets(metadata()).items():
        s = t[t.item_id.isin(sub.item_id)]
        print(f"  {name:<4} n={len(s):>4}  containers={s.container.nunique():>3}  "
              f"V(T) med={s.v_end.median():>6.0f} mL  frac_cap={s.frac_cap.median():.2f}  "
              f"taper med={s.taper.median():.3f}")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.build:
        build(args.device)
    report()


if __name__ == "__main__":
    main()
