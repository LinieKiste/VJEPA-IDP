"""Train the V-JEPA 2 attentive-probe (AttentiveClassifier, EK100 warm-start) for
pouring flow/volume regression on the own-lab clips — the full eval-protocol probe,
not the linear ridge quick-check.

Architecture (head.py) = the V-JEPA 2 evaluation probe: a depth-4 attentive pooler
(3 self-attention blocks + a cross-attention block with one learnable query,
16 heads, 1024-d) followed by a linear layer, here num_outputs=1 for regression.
The pooler is warm-started from the Epic-Kitchens-100 attentive probe (its "action"
query); the linear regression head is fresh. The frozen V-JEPA 2 ViT-L encoder runs
IN THE LOOP so we can augment in pixel space (h-flip + random 256-crop from 288 +
temporal jitter) — the standard frozen-backbone eval recipe.

Split: hold out a fixed set of TRIALS for validation (clips of a trial share
scene/container). Trains for a wall-clock budget (--minutes). Logs per-step train
loss + per-epoch val loss/R²/MAE to mlflow (pour_probe_clips_attn), and prints the
ridge/temporal/motion baselines on the SAME held-out trials for a fair comparison.

Usage:
    .venv/bin/python pour_probe/clips_train_attn.py --target flow --minutes 60
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from _encoder import load_encoder
from head import build_head, set_pooler_trainable, warm_start_from_ek100

FRAMES_DIR = Path(os.environ.get("POUR_FRAMES288_DIR",
                                 "/home/casimir/.cache/pour_probe/clips_frames288"))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

# fixed validation trials — span source (kettle/teapot/bottle) x target variety
VAL_TRIALS = {"8", "13", "21", "24"}


def load_clips(cam):
    """clip_id -> dict(frames(N,288,288,3) uint8, fps, gt_t, gt_w, trial)."""
    clips = {}
    for f in sorted((FRAMES_DIR / cam).glob("*.npz")):
        a = np.load(f, allow_pickle=True)
        clips[f.stem] = {"frames": a["frames"], "fps": float(a["fps"]),
                         "gt_t": a["gt_t"], "gt_w": a["gt_w"], "trial": str(a["trial_id"])}
    return clips


def build_windows(cams, window_s, stride_s, num_frames, lag_s=0.0):
    """Flat list of window dicts (cam, clip, trial, frame range, targets) over all
    cameras in the ``cams`` dict {cam_name: {clip_id: clip}}.

    ``lag_s`` shifts the WEIGHT-derived target later in time to correct for the
    water-transit + scale-display delay (the scale registers mass ~0.7 s after the
    stream is visible; clips_lag_sweep.py measures it). The FRAMES are unchanged; only
    the target is sampled at t+lag_s, so the probe learns visual-flow -> true
    instantaneous flow with the right timing. lag_s=0 = raw (uncorrected) alignment."""
    wins = []
    for cam, clips in cams.items():
        for cid, c in clips.items():
            dur = len(c["frames"]) / c["fps"]
            t0 = 0.0
            while True:
                t1 = min(dur, t0 + window_s)
                vol = float(np.interp((t0 + t1) / 2 + lag_s, c["gt_t"], c["gt_w"]))
                flow = float((np.interp(t1 + lag_s, c["gt_t"], c["gt_w"])
                              - np.interp(t0 + lag_s, c["gt_t"], c["gt_w"])) / max(t1 - t0, 1e-3))
                wins.append({"cam": cam, "clip": cid, "trial": c["trial"], "t0": t0, "t1": t1,
                             "volume": vol, "flow": flow})
                if t1 >= dur:
                    break
                t0 += stride_s
    return wins


def sample_window(clip, w, num_frames, train, rng):
    """Return (num_frames, 256, 256, 3) uint8 with augmentation if train."""
    frames, fps, N = clip["frames"], clip["fps"], len(clip["frames"])
    t0, t1 = w["t0"], w["t1"]
    if train:                                   # temporal jitter ±0.1 s
        j = (t1 - t0) * 0.1
        t0 = max(0.0, t0 + rng.uniform(-j, j)); t1 = min(N / fps, t1 + rng.uniform(-j, j))
    idx = np.clip((np.linspace(t0, t1, num_frames) * fps).astype(int), 0, N - 1)
    fr = frames[idx]                            # (T,288,288,3)
    if train:                                   # random 256-crop from 288
        y0, x0 = rng.integers(0, 33), rng.integers(0, 33)
    else:
        y0 = x0 = 16                            # center crop
    fr = fr[:, y0:y0 + 256, x0:x0 + 256]
    if train and rng.random() < 0.5:            # horizontal flip (mirror the pour)
        fr = fr[:, :, ::-1]
    return np.ascontiguousarray(fr)


@torch.no_grad()
def encode(enc, batch_u8, device, mean, std):
    x = torch.from_numpy(batch_u8).to(device)
    x = x.permute(0, 4, 1, 2, 3).float().div_(255.0)
    x = (x - mean) / std
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return enc(x).float()                   # (B,N,1024)


def batches(items, bs, shuffle, rng):
    order = rng.permutation(len(items)) if shuffle else np.arange(len(items))
    for s in range(0, len(items), bs):
        yield [items[i] for i in order[s:s + bs]]


def run_eval(enc, head, cams, wins, target, num_frames, device, mean, std, ymean, ystd, bs=16):
    head.eval()
    preds, ys = [], []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for batch in batches(wins, bs, False, rng):
            fr = np.stack([sample_window(cams[w["cam"]][w["clip"]], w, num_frames, False, rng)
                           for w in batch])
            tok = encode(enc, fr, device, mean, std)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = head(tok).squeeze(1).float().cpu().numpy()
            preds.append(p * ystd + ymean); ys.append([w[target] for w in batch])
    preds = np.concatenate(preds); ys = np.concatenate(ys)
    return r2_score(ys, preds), mean_absolute_error(ys, preds), preds, ys


def sow_baselines_on_split(wins, target, va_groups):
    """Mandatory controls for the Sound-of-Water runs, on the same held-out containers.

    `container_mean` is the key one: predicting a held-out container's mean target is
    impossible without seeing it, so it is evaluated as the TRAIN-set global mean per
    unseen container — i.e. it collapses to predict-mean for new containers, and any
    gap above it must come from within-video temporal evidence.

    `mean_removed_r2` reports the attentive/temporal signal AFTER subtracting each
    video's own mean, isolating within-video variation from between-video offsets —
    the degeneracy that a near-constant flow target would otherwise reward.
    """
    from sklearn.preprocessing import PolynomialFeatures
    tr = [w for w in wins if w["trial"] not in va_groups]
    va = [w for w in wins if w["trial"] in va_groups]
    ytr = np.asarray([w[target] for w in tr]); yva = np.asarray([w[target] for w in va])
    out = {}
    # normalized-time polynomial profile (very strong for a constant-flow V(t))
    def prof(ws):
        dur = {}
        for w in ws:
            dur[w["clip"]] = max(dur.get(w["clip"], 0.0), w["t1"])
        return np.asarray([[(w["t0"] + w["t1"]) / 2 / max(dur[w["clip"]], 1e-6)] for w in ws])
    P = PolynomialFeatures(4, include_bias=False)
    r = Ridge(alpha=1.0).fit(P.fit_transform(prof(tr)), ytr)
    pv = r.predict(P.transform(prof(va)))
    out["time_prof"] = (r2_score(yva, pv), mean_absolute_error(yva, pv))
    # predict-mean / container-mean (identical for unseen containers, reported for clarity)
    pm = np.full_like(yva, ytr.mean())
    out["predict_mean"] = (r2_score(yva, pm), mean_absolute_error(yva, pm))
    out["container_mean"] = out["predict_mean"]
    # The SAME baselines under the within-video metric the probe is scored on —
    # without these, a within-video R2 is uninterpretable (V(t) rises monotonically,
    # so "time" could in principle explain most of the within-video variation).
    cl = [w["clip"] for w in va]
    out["time_prof_withinvid"] = (mean_removed_r2(pv, yva, cl), float("nan"))
    out["predict_mean_withinvid"] = (mean_removed_r2(pm, yva, cl), float("nan"))
    return out


def within_video_corr(preds, ys, clips):
    """Mean per-video Pearson r between prediction and target.

    Scale-free companion to the mean-removed R2: it asks only whether the predicted
    curve has the right SHAPE within each video, ignoring amplitude. Useful because a
    time-profile baseline is penalised on amplitude (it cannot know an unseen
    container's size) while the probe can infer size from appearance — so a gap in
    mean-removed R2 partly reflects that advantage, whereas correlation isolates shape.
    """
    preds, ys, clips = np.asarray(preds), np.asarray(ys), np.asarray(clips)
    rs = []
    for c in np.unique(clips):
        m = clips == c
        if m.sum() < 3 or np.std(preds[m]) < 1e-9 or np.std(ys[m]) < 1e-9:
            continue
        rs.append(np.corrcoef(preds[m], ys[m])[0, 1])
    return float(np.mean(rs)) if rs else float("nan")


def mean_removed_r2(preds, ys, clips):
    """R² after removing each video's own mean from both prediction and target."""
    preds, ys, clips = np.asarray(preds), np.asarray(ys), np.asarray(clips)
    p, y = preds.copy(), ys.copy()
    for c in np.unique(clips):
        m = clips == c
        p[m] -= p[m].mean(); y[m] -= y[m].mean()
    return r2_score(y, p)


def _retarget(cids, tmid, target, lag_s, window_s):
    """Per-window target resampled at t+lag, matching clips_extract's definitions:
    volume = weight at the window centre, flow = dWeight across the window."""
    import clips_train as ct
    CLIPS = ct.CLIPS
    out = np.zeros(len(tmid), np.float32)
    for c in set(cids.tolist()):
        m = cids == c
        d = np.loadtxt(CLIPS / "csv" / f"{c}.csv", delimiter=",", skiprows=1)
        tgt, wgt = d[:, 0], d[:, 1]
        t = tmid[m].astype(np.float64) + lag_s
        if target == "volume":
            out[m] = np.interp(t, tgt, wgt)
        else:
            out[m] = (np.interp(t + window_s / 2, tgt, wgt)
                      - np.interp(t - window_s / 2, tgt, wgt)) / window_s
    return out


def baselines_on_split(cam, target, va_trials, lag_s=0.0, window_s=1.0):
    """Ridge on mean-pool feats + temporal-profile + predict-mean, evaluated on the
    SAME held-out val trials (uses the mean-pool cache from clips_extract).

    `lag_s` MUST match the lag the probe was trained with. The mean-pool cache stores
    targets computed at lag 0, so without this the baselines are scored against
    lag-0 targets while the probe is scored against lagged ones — which flatters the
    probe badly (the lag is worth ~+0.17 to the CAM3 ridge on our split). Targets are
    recomputed here exactly as clips_extract does, but sampled at t+lag.
    """
    import clips_train as ct
    from sklearn.preprocessing import PolynomialFeatures
    X, y, groups, cids, tmid = ct.load_both(target) if cam == "both" else ct.load(cam, target)
    if lag_s:
        y = _retarget(cids, tmid, target, lag_s, window_s)
    is_va = np.asarray([g in va_trials for g in groups])
    out = {}
    # ridge on V-JEPA mean-pool
    m, s = X[~is_va].mean(0), X[~is_va].std(0) + 1e-6
    r = Ridge(alpha=100).fit((X[~is_va] - m) / s, y[~is_va])
    pv = r.predict((X[is_va] - m) / s)
    out["ridge_meanpool"] = (r2_score(y[is_va], pv), mean_absolute_error(y[is_va], pv))
    # temporal profile (normalized-time poly4)
    tnorm = np.zeros_like(tmid)
    for c in set(cids.tolist()):
        mc = cids == c; lo, hi = tmid[mc].min(), tmid[mc].max()
        tnorm[mc] = (tmid[mc] - lo) / max(hi - lo, 1e-6)
    tp = PolynomialFeatures(4).fit_transform(tnorm[:, None])
    r = Ridge(1.0).fit(tp[~is_va], y[~is_va]); pv = r.predict(tp[is_va])
    out["time_prof"] = (r2_score(y[is_va], pv), mean_absolute_error(y[is_va], pv))
    # predict-mean floor
    base = np.full(int(is_va.sum()), y[~is_va].mean())
    out["mean"] = (r2_score(y[is_va], base), mean_absolute_error(y[is_va], base))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["flow", "volume"], default="flow")
    ap.add_argument("--cam", default="CAM2", choices=["CAM2", "CAM3", "both"])
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--window_s", type=float, default=1.0)
    ap.add_argument("--stride_s", type=float, default=0.5)
    ap.add_argument("--lag_s", type=float, default=0.0,
                    help="shift weight target later to correct water-transit/scale lag "
                         "(clips_lag_sweep.py finds ~0.7s for flow; 0=uncorrected)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--loss", choices=["smoothl1", "mse"], default="smoothl1",
                    help="mse punishes peak-flow misses harder (less mean-hedging) than "
                         "SmoothL1's Huber tail; see clips_bias_diag.py")
    ap.add_argument("--warmstart", action="store_true", default=True)
    ap.add_argument("--no_warmstart", dest="warmstart", action="store_false")
    ap.add_argument("--val_trials", default=",".join(sorted(VAL_TRIALS)),
                    help="comma-separated held-out trial ids (for k-fold CV)")
    ap.add_argument("--fold", default="", help="label appended to ckpt/run name for k-fold runs")
    ap.add_argument("--dataset", choices=["clips", "sow"], default="clips",
                    help="clips = our own-lab gram-GT clips; sow = Sound-of-Water videos "
                         "with audio-physics-derived volume targets (grouped by container)")
    ap.add_argument("--subset", default="S2", help="sow subset: S1/S2/S2o/S3")
    ap.add_argument("--val_frac", type=float, default=0.25,
                    help="sow: fraction of CONTAINERS held out for validation")
    ap.add_argument("--split_seed", type=int, default=0, help="sow: container split seed")
    ap.add_argument("--init_ckpt", default="",
                    help="initialize the head from a trained checkpoint (staged transfer, "
                         "e.g. Sound-of-Water pretrain -> own-clips fine-tune)")
    ap.add_argument("--tag_extra", default="", help="suffix for ckpt/run names")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.dataset == "sow":
        import sow_grid_cache as sg
        clips = sg.load_clips(args.subset)
        cam_list = ["sow"]
        cams = {"sow": clips}
        # Hold out whole CONTAINERS: clips of one container share geometry+appearance,
        # so a video-level split would leak the very thing we want to generalize over.
        # The split is BALANCED BY VIDEO COUNT rather than uniformly random: container
        # video counts are very uneven (1..46), and a random draw put 85% of the val
        # windows on a single container whose volume range was narrow — which makes the
        # R2 denominator tiny and swings the score to -8 on modest bias. Greedily
        # assigning containers (largest first) to reach the target val fraction gives a
        # val set with several containers and a realistic spread.
        counts = {}
        for c in clips.values():
            counts[c["trial"]] = counts.get(c["trial"], 0) + 1
        order = sorted(counts, key=lambda k: (-counts[k], k))
        rng0 = np.random.default_rng(args.split_seed)
        order = [order[i] for i in rng0.permutation(len(order))] if args.split_seed else order
        target = args.val_frac * sum(counts.values())
        val_trials, acc = set(), 0
        for k in order:
            if acc >= target:
                break
            val_trials.add(k); acc += counts[k]
        print(f"sow[{args.subset}]: {len(clips)} videos / {len(counts)} containers; "
              f"held out {sorted(val_trials)} ({acc} videos, "
              f"{acc / sum(counts.values()):.0%})", flush=True)
    else:
        cam_list = ["CAM2", "CAM3"] if args.cam == "both" else [args.cam]
        cams = {c: load_clips(c) for c in cam_list}
        val_trials = set(args.val_trials.split(","))
    wins = build_windows(cams, args.window_s, args.stride_s, args.num_frames, args.lag_s)
    dstag = "" if args.dataset == "clips" else f"sow{args.subset}_"
    tag = (f"{dstag}{args.cam if args.dataset == 'clips' else 'v'}"
           f"{'' if args.lag_s == 0 else f'_lag{args.lag_s:g}'}"
           f"{'' if args.loss == 'smoothl1' else '_' + args.loss}"
           f"{'' if args.warmstart else '_noWS'}{('_' + args.fold) if args.fold else ''}"
           f"{('_' + args.tag_extra) if args.tag_extra else ''}")
    tr = [w for w in wins if w["trial"] not in val_trials]
    va = [w for w in wins if w["trial"] in val_trials]
    print(f"cams {cam_list} | {len(tr)} train / {len(va)} val windows | "
          f"val trials {sorted(val_trials)}", flush=True)

    enc = load_encoder(img_size=256, num_frames=args.num_frames, device=args.device)
    head = build_head(1).to(args.device)
    if args.init_ckpt:
        # Staged transfer: start from a head trained on ANOTHER pouring corpus (e.g.
        # Sound-of-Water) instead of the EK100 action pooler. Kept as a separate stage
        # rather than mixing corpora in one batch — the two datasets' targets differ in
        # kind (measured grams vs an audio-model estimate) and in lag (~0.7 s scale
        # delay vs none), so a single normalization/lag cannot be right for both.
        head.load_state_dict(torch.load(args.init_ckpt, map_location=args.device))
        print(f"head initialized from {args.init_ckpt}")
    elif args.warmstart:
        n, tot = warm_start_from_ek100(head)
        print(f"pooler warm-started {n}/{tot} from EK100")
    set_pooler_trainable(head, True)
    mean, std = MEAN.to(args.device), STD.to(args.device)

    ytr = np.asarray([w[args.target] for w in tr], np.float32)
    ymean, ystd = float(ytr.mean()), float(ytr.std() + 1e-6)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.MSELoss() if args.loss == "mse" else nn.SmoothL1Loss()
    rng = np.random.default_rng(0)

    import mlflow_util
    mlflow_util.setup("pour_probe_clips_attn" if args.dataset == "clips"
                      else "pour_probe_sow_attn")
    run = mlflow.start_run(run_name=f"attn_{args.target}_{tag}")
    mlflow.log_params({"target": args.target, "cam": args.cam, "arch": "AttentiveClassifier",
                       "depth": 4, "heads": 16, "embed_dim": 1024, "warmstart_ek100": args.warmstart,
                       "lag_s": args.lag_s, "loss": args.loss,
                       "lr": args.lr, "weight_decay": args.weight_decay, "batch": args.batch,
                       "window_s": args.window_s, "stride_s": args.stride_s,
                       "num_frames": args.num_frames, "minutes": args.minutes,
                       "n_train_win": len(tr), "n_val_win": len(va),
                       "dataset": args.dataset,
                       "subset": args.subset if args.dataset == "sow" else "",
                       "val_trials": ",".join(sorted(val_trials)), "fold": args.fold})

    ckpt_path = FRAMES_DIR.parent / f"attn_{args.target}_{tag}_best.pt"
    steps_per_epoch = (len(tr) + args.batch - 1) // args.batch

    # LR schedule = linear warmup over epoch 0 (kills the AdamW startup loss spikes on
    # the warm-started pooler) -> cosine decay to 0, driven by WALL-CLOCK FRACTION of the
    # time budget. Earlier this estimated a step count from epoch 0's wall time (x0.92);
    # but epoch 0 is the SLOWEST (cold caches / first-batch overhead), so the estimate
    # overshot per-epoch cost, sized the cosine too short, and LR hit 0 with ~20% of the
    # budget still to run — those steps did nothing. Tying the cosine to elapsed time
    # instead needs no estimate and reaches 0 exactly when the loop stops, so the whole
    # budget is spent training. (It still finishes on a low-LR plateau, which is what let
    # the CAM2 run converge flat instead of bouncing.)
    warmup_steps = steps_per_epoch
    sched_state = {"t_start": None, "warmup_end_t": None, "budget_s": args.minutes * 60}

    def lr_lambda(s):
        if s < warmup_steps:
            return (s + 1) / warmup_steps                      # 0 -> peak over epoch 0
        ts = sched_state["t_start"]
        if ts is None:
            return 1.0                                          # loop not started yet
        if sched_state["warmup_end_t"] is None:                 # mark end of warmup once
            sched_state["warmup_end_t"] = time.time()
        warm_s = sched_state["warmup_end_t"] - ts
        prog = min(1.0, (time.time() - sched_state["warmup_end_t"])
                   / max(1.0, sched_state["budget_s"] - warm_s))
        return 0.5 * (1.0 + math.cos(math.pi * prog))           # peak -> 0 at budget end

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    t_start = time.time(); step = 0; epoch = 0
    sched_state["t_start"] = t_start                            # drives the wall-clock cosine
    best = {"r2": -1e9}; val_hist = []

    while time.time() - t_start < args.minutes * 60:
        ep_t0 = time.time()
        head.train()
        for batch in batches(tr, args.batch, True, rng):
            fr = np.stack([sample_window(cams[w["cam"]][w["clip"]], w, args.num_frames, True, rng)
                           for w in batch])
            tok = encode(enc, fr, args.device, mean, std)
            y = torch.tensor([(w[args.target] - ymean) / ystd for w in batch],
                             dtype=torch.float32, device=args.device).unsqueeze(1)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = head(tok)
                loss = crit(pred, y)
            loss.backward(); opt.step(); sched.step()
            if step % 10 == 0:
                mlflow.log_metric("train_loss", loss.item(), step=step)
                mlflow.log_metric("lr", opt.param_groups[0]["lr"], step=step)
            step += 1
        # per-epoch val (combined)
        r2, mae, pv_e, yv_e = run_eval(enc, head, cams, va, args.target, args.num_frames,
                                       args.device, mean, std, ymean, ystd)
        mlflow.log_metric("val_r2", r2, step=step)
        mlflow.log_metric("val_mae", mae, step=step)
        # On the SoW data the plain R2 is unreliable as a SELECTION metric: absolute
        # volume depends on the (unseen) container's physical size, and a held-out
        # container with a narrow volume range gives R2 a tiny denominator. The
        # within-video (mean-removed) R2 asks the question we actually care about —
        # does the probe track the filling over time — and is scale-free, so it is the
        # selection + headline metric here. Plain R2 is still logged and reported.
        if args.dataset == "sow":
            mr_e = mean_removed_r2(pv_e, yv_e, [w["clip"] for w in va])
            mlflow.log_metric("val_mean_removed_r2", mr_e, step=step)
            sel = mr_e
        else:
            mr_e = float("nan")
            sel = r2
        el = time.time() - t_start
        if epoch == 0:
            print(f"  (warmup {warmup_steps} steps; cosine -> 0 by wall-clock budget "
                  f"{args.minutes:g} min)", flush=True)
        # checkpoint on the 3-epoch ROLLING MEAN of the selection metric (not a single
        # lucky epoch)
        val_hist.append(sel)
        sm = float(np.mean(val_hist[-3:]))
        extra = "" if args.dataset == "clips" else f"  within-vid R²={mr_e:.3f}"
        print(f"epoch {epoch:2d} step {step:5d} [{el/60:.1f}m]  "
              f"val R²={r2:.3f} (sm {sm:.3f})  MAE={mae:.2f}{extra}  "
              f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        if sm > best["r2"]:
            best = {"r2": sm, "raw": r2, "mae": mae, "epoch": epoch, "step": step}
            torch.save(head.state_dict(), ckpt_path)
        epoch += 1

    tail = val_hist[-5:]
    print(f"  val tail (last {len(tail)} ep): mean {np.mean(tail):.3f} ± {np.std(tail):.3f} "
          f"(low std = converged plateau)", flush=True)
    mlflow.log_metric("val_tail_mean", float(np.mean(tail)))
    mlflow.log_metric("val_tail_std", float(np.std(tail)))

    # reload the saved checkpoint, report a FRESH eval of it (combined + PER-CAMERA on
    # the held-out trials — the robustness test). Every number below is the actual
    # saved weights, not the noisy selection metric.
    head.load_state_dict(torch.load(ckpt_path))
    unit = ("g/s" if args.target == "flow" else "g") if args.dataset == "clips" else \
           ("mL/s" if args.target == "flow" else "mL")
    r2_comb, mae_comb, pv, yv = run_eval(enc, head, cams, va, args.target, args.num_frames,
                                         args.device, mean, std, ymean, ystd)
    mlflow.log_metric("ckpt_val_r2", r2_comb); mlflow.log_metric("ckpt_val_mae", mae_comb)
    percam = {}
    if args.dataset == "clips":
        for c in cam_list:
            vc = [w for w in va if w["cam"] == c]
            r2c, maec, _, _ = run_eval(enc, head, cams, vc, args.target, args.num_frames,
                                       args.device, mean, std, ymean, ystd)
            percam[c] = (r2c, maec)
            mlflow.log_metric(f"val_r2_{c}", r2c); mlflow.log_metric(f"val_mae_{c}", maec)
    else:
        # within-video R²: strips per-video offsets so a near-constant target cannot
        # inflate the score through between-video variance alone
        # run_eval iterates `va` in order (shuffle=False), so preds align with it
        cl = [w["clip"] for w in va]
        mr = mean_removed_r2(pv, yv, cl)
        wc = within_video_corr(pv, yv, cl)
        mlflow.log_metric("mean_removed_r2", mr)
        mlflow.log_metric("within_video_corr", wc)
        percam["mean-removed (within-vid)"] = (mr, float("nan"))
        percam["within-vid corr r"] = (wc, float("nan"))

    # baselines on the same held-out groups
    base = (baselines_on_split(args.cam, args.target, val_trials, args.lag_s, args.window_s)
            if args.dataset == "clips"
            else sow_baselines_on_split(wins, args.target, val_trials))
    for name, (r2, mae) in base.items():
        mlflow.log_metric(f"{name}_r2", r2); mlflow.log_metric(f"{name}_mae", mae)
    mlflow.log_metric("best_val_r2", best["r2"]); mlflow.log_metric("best_val_mae", best["mae"])
    mlflow.end_run()

    where = args.cam if args.dataset == "clips" else f"sow/{args.subset}"
    print(f"\n=== {args.target} [{where}] attentive probe (V-JEPA 2 eval) | "
          f"{epoch} epochs / {step} steps / {(time.time()-t_start)/60:.0f} min ===")
    print(f"  ckpt = epoch {best['epoch']} (3-ep rolling-mean peak {best['r2']:.3f}); "
          f"val tail {np.mean(tail):.3f} ± {np.std(tail):.3f}")
    print(f"  {'method':<22} {'R2':>7} {'MAE':>9}")
    print(f"  {'attn ckpt (combined)':<22} {r2_comb:>7.3f} {mae_comb:>7.2f}{unit}")
    for c, (r2c, maec) in percam.items():
        lbl = ("attn on " + c) if args.dataset == "clips" else ("attn " + c)
        print(f"  {lbl:<22} {r2c:>7.3f} {maec:>7.2f}{unit}")
    for name, (r2, mae) in base.items():
        print(f"  {name:<22} {r2:>7.3f} {mae:>7.2f}{unit}")


if __name__ == "__main__":
    main()
