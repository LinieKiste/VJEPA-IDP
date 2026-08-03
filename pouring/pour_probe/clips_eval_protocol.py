"""Stricter evaluation protocol for the own-lab clips, borrowed from what we already
demanded of the Sound-of-Water runs but never applied to our own data.

Motivation. Our headline R2 uses the GLOBAL mean as its denominator. For `volume` the
target is monotone and near-linear inside every pour, so that denominator rewards any
method that can draw a rising line, and the clock duly wins (0.78). Reporting "V-JEPA
0.40 vs clock 0.78" then says almost nothing about perception. `sow_baselines_on_split`
already fixes this for the SoW runs via mean-removed within-video R2 plus the trivial
controls scored under the SAME metric; this script applies that discipline to our clips
and adds a skill score.

Three metrics per method, OOF over the 4 trial-grouped folds:

  r2_global    R2 against the global train mean.               [what we report today]
  r2_within    R2 after removing each CLIP's own mean from
               prediction and target. Kills the between-pour
               offset, so only within-pour shape counts.       [the SoW metric]
  skill        1 - SSE_model / SSE_reference, i.e. the
               fraction of the best trivial method's squared
               error that this method removes. Negative means
               worse than doing the trivial thing.             [the honest headline]

Reference for `skill` is the strongest NON-VISUAL method on that target, chosen per
target rather than fixed, so the probe is always compared against the best thing you
could do without looking.

Methods. `raw_time` is causal (elapsed seconds only). `time_prof` needs each clip's
DURATION to normalise, which is an oracle at test time and is flagged as such.

Usage:
    .venv/bin/python pouring/pour_probe/clips_eval_protocol.py
    .venv/bin/python pouring/pour_probe/clips_eval_protocol.py --cam both
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

import clips_train_attn as ca
from clips_cnn_baseline import FOLDS, LAG_FLOW


def within_removed(pred, y, cids):
    """Subtract each clip's own mean from prediction and target."""
    p, t = np.asarray(pred, float).copy(), np.asarray(y, float).copy()
    for c in np.unique(cids):
        m = cids == c
        p[m] -= p[m].mean()
        t[m] -= t[m].mean()
    return p, t


def r2(pred, y):
    y = np.asarray(y, float)
    ss = ((y - pred) ** 2).sum()
    return 1.0 - ss / ((y - y.mean()) ** 2).sum()


def sse(pred, y):
    return float(((np.asarray(y, float) - pred) ** 2).sum())


def norm_time(cids, tmid):
    """t / clip duration. Needs the clip's full extent, so it is an ORACLE feature."""
    tn = np.zeros_like(tmid, dtype=float)
    for c in np.unique(cids):
        m = cids == c
        lo, hi = tmid[m].min(), tmid[m].max()
        tn[m] = (tmid[m] - lo) / max(hi - lo, 1e-6)
    return tn


def oof(X, y, groups, fit_predict):
    """Out-of-fold predictions over the 4 trial-grouped folds."""
    pred = np.zeros(len(y), float)
    for trials in FOLDS.values():
        va = np.isin(groups, list(trials))
        if not va.any():
            continue
        pred[va] = fit_predict(X[~va], y[~va], X[va])
    return pred


def ridge_fp(alpha):
    def f(Xtr, ytr, Xva):
        m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
        return Ridge(alpha=alpha).fit((Xtr - m) / s, ytr).predict((Xva - m) / s)
    return f


def linear_fp(Xtr, ytr, Xva):
    return LinearRegression().fit(Xtr, ytr).predict(Xva)


def unit_metrics(pred, y, unit):
    """Physical-unit metrics. R2 hides bias and is scaled by whatever spread the test
    set happens to have; these are all in the units the quantity is measured in."""
    e = np.asarray(pred, float) - np.asarray(y, float)
    a = np.abs(e)
    return {
        "MAE": a.mean(),
        "medAE": np.median(a),
        "P90AE": np.percentile(a, 90),
        "bias": e.mean(),                       # systematic over/under-shoot
        "nMAE%": 100 * a.mean() / np.abs(y).mean(),
    }


def totals_from_flow(pred_flow, y_flow, cids, tmid):
    """Integrate the predicted flow curve per clip -> total poured mass in grams.

    This is the quantity a user actually wants, and unlike per-window volume it has NO
    within-pour axis for a clock to exploit: one number per pour. Overlapping windows
    make a naive sum double-count, so integrate the curve with the trapezoid rule over
    the window centres instead.
    """
    P, T = [], []
    for c in np.unique(cids):
        m = cids == c
        o = np.argsort(tmid[m])
        t = tmid[m][o]
        if len(t) < 2:
            continue
        P.append(np.trapz(np.asarray(pred_flow)[m][o], t))
        T.append(np.trapz(np.asarray(y_flow)[m][o], t))
    return np.asarray(P), np.asarray(T)


def tolerance_table(pred, true, bands=(10, 25, 50)):
    """Share of pours landing within +-X g. The most directly readable metric we have:
    '8 in 10 pours are within 25 g' needs no statistics background to interpret."""
    a = np.abs(np.asarray(pred, float) - np.asarray(true, float))
    return {f"within {b} g": 100 * (a <= b).mean() for b in bands}


def bland_altman(pred, true):
    """Measurement-agreement framing (Bland & Altman 1986): the honest question for an
    instrument is not 'does it correlate' but 'how far off can one reading be'.
    Returns bias and the 95% limits of agreement, all in grams."""
    d = np.asarray(pred, float) - np.asarray(true, float)
    return d.mean(), d.mean() - 1.96 * d.std(ddof=1), d.mean() + 1.96 * d.std(ddof=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="CAM2", choices=["CAM2", "CAM3", "both"])
    args = ap.parse_args()

    import clips_train as ct

    for target in ("flow", "volume"):
        lag = LAG_FLOW if target == "flow" else 0.0
        X, y, groups, cids, tmid = (ct.load_both(target) if args.cam == "both"
                                    else ct.load(args.cam, target))
        if lag:
            y = ca._retarget(cids, tmid, target, lag, 1.0)

        tn = norm_time(cids, tmid)
        poly = PolynomialFeatures(4).fit_transform(tn[:, None])
        clock = tmid[:, None].astype(float)

        preds = {
            "predict_mean": oof(clock, y, groups,
                                lambda a, b, c: np.full(len(c), b.mean())),
            "raw_time (causal clock)": oof(clock, y, groups, linear_fp),
            "time_prof (ORACLE duration)": oof(poly, y, groups, ridge_fp(1.0)),
            "V-JEPA ridge": oof(X, y, groups, ridge_fp(100.0)),
            "V-JEPA + clock": oof(np.hstack([X, clock]), y, groups, ridge_fp(100.0)),
        }

        trivial = ["predict_mean", "raw_time (causal clock)", "time_prof (ORACLE duration)"]
        ref = max(trivial, key=lambda k: r2(preds[k], y))
        ref_sse = sse(preds[ref], y)

        print(f"\n=== {target}  [{args.cam}, lag {lag}, {len(y)} windows, "
              f"4-fold OOF by trial] ===")
        print(f"  skill reference = best non-visual method = {ref}")
        print(f"  {'method':<30} {'r2_global':>10} {'r2_within':>10} {'skill':>8}")
        for name, p in preds.items():
            pw, yw = within_removed(p, y, cids)
            mark = " <- ref" if name == ref else ""
            print(f"  {name:<30} {r2(p, y):>10.3f} {r2(pw, yw):>10.3f} "
                  f"{1.0 - sse(p, y) / ref_sse:>8.3f}{mark}")

        # how much of the variance is within-pour at all
        _, yw = within_removed(y, y, cids)
        frac = yw.var() / y.var()
        print(f"  within-pour share of total variance: {frac:.2f}")

        # ---- physical-unit metrics, same predictions, no R2 anywhere
        unit = "g/s" if target == "flow" else "g"
        print(f"\n  --- in {unit}, not R² ---")
        print(f"  {'method':<30} {'MAE':>8} {'medAE':>8} {'P90AE':>8} {'bias':>8} {'nMAE%':>7}")
        for name, p in preds.items():
            m = unit_metrics(p, y, unit)
            print(f"  {name:<30} {m['MAE']:>8.2f} {m['medAE']:>8.2f} {m['P90AE']:>8.2f} "
                  f"{m['bias']:>+8.2f} {m['nMAE%']:>6.0f}%")

        # ---- the deliverable metric: total poured mass per clip, in grams
        if target == "flow":
            print(f"\n  === PER-CLIP TOTAL MASS (integrate the flow curve), grams ===")
            print(f"  {'method':<30} {'MAE':>7} {'medAE':>7} {'bias':>7} "
                  f"{'≤10g':>6} {'≤25g':>6} {'≤50g':>6} {'95% limits of agreement':>26}")
            for name, p in preds.items():
                Pt, Tt = totals_from_flow(p, y, cids, tmid)
                m = unit_metrics(Pt, Tt, "g")
                tol = tolerance_table(Pt, Tt)
                b, lo, hi = bland_altman(Pt, Tt)
                print(f"  {name:<30} {m['MAE']:>7.1f} {m['medAE']:>7.1f} {m['bias']:>+7.1f} "
                      f"{tol['within 10 g']:>5.0f}% {tol['within 25 g']:>5.0f}% "
                      f"{tol['within 50 g']:>5.0f}% {f'{lo:+.0f} to {hi:+.0f} g':>26}")
            print(f"  (n = {len(np.unique(cids))} clips, "
                  f"true totals mean {totals_from_flow(y, y, cids, tmid)[1].mean():.0f} g)")


if __name__ == "__main__":
    main()
