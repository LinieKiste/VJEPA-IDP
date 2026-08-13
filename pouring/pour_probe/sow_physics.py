"""Physics decode: Sound-of-Water wavelength lambda(t) -> poured volume V(t) and flow Q(t).

Model (Bagad et al., arXiv 2411.11222): a filling vessel is an air column closed at the
water surface and open at the top, resonating at

    lambda(t) = 4 * (l(t) + beta*R)

where l(t) is the air-column length, R the radius, and beta*R an end correction. Hence
the water level rises as h(t) = H - l(t), and for a cylinder the volume of water is
V_water(t) = pi*R^2*(H - l(t)).

**We only ever need volume POURED SINCE THE START**, and for that both the end correction
and the container height cancel:

    V_poured(t) = V_water(t) - V_water(0)
                = pi*R^2 * [l(0) - l(t)]
                = (pi*R^2/4) * [lambda(0) - lambda(t)]

so beta and H drop out and **only the radius R is needed**. That is a strictly weaker
requirement than their published formulation (which needs H as well), and it removes an
error source — worth stating in the writeup.

Correction (2026-08-08): an earlier version of this note said their formulation needs a
*per-container* beta. It does not — `shared/utils/physics.py` uses a fixed beta per
SHAPE (0.62 cylindrical, 1.28 semi-conical). The real gap is elsewhere: they do not need
a measured R either, because `estimate_cylinder_radius` recovers it from the audio as
R = lambda(T)/(4*beta). **That estimator assumes l(T)=0 — the vessel is full at the end
of the recording.** Our pours stop at an arbitrary fill, so lambda(T) reads the leftover
air column, not the radius: over 48 pours into the SAME mug it returns R = 1.7-11.4 cm
(within-container variance 28x the between-container variance, corr with poured mass
-0.48). So on our data R must be supplied; on THEIR data the estimator works as
published (measured 2.88 cm vs 2.49 cm estimated on the demo video).

Correspondingly the instantaneous flow is

    Q(t) = dV/dt = -(pi*R^2/4) * dlambda/dt      [cm^3/s = mL/s]

Since rho_water = 1 g/mL, mL and grams are interchangeable, which is what makes these
predictions directly comparable to our scale-measured GT.

Caveat carried by every number this module produces: it assumes a **cylindrical**
resonator. For tapered/conical containers R varies with fill level and V(t) is biased;
the SoW annotations mark shape, so filter on it when the bias matters.
"""
from __future__ import annotations

import numpy as np

RATE = 49.0          # SoW backbone frame rate


def smooth(x, win_s=0.5, rate=RATE, polyorder=2):
    """Savitzky-Golay smoothing of a per-frame series (odd window, >= polyorder+2).

    lambda(t) is a frame-wise expectation over 64 bins, so it carries high-frequency
    jitter that a raw derivative would amplify into noise dominating the flow signal.
    """
    from scipy.signal import savgol_filter
    w = int(round(win_s * rate))
    w = max(polyorder + 2, w + (w + 1) % 2)          # force odd
    if len(x) <= w:
        return x.copy()
    return savgol_filter(x, w, polyorder)


def radius_cm(measurements):
    """Effective radius from a SoW `measurements` dict (cm).

    Uses the mean of top/bottom diameters when both exist — for a mildly tapered
    container that is the radius at mid-fill, which minimises the cylinder-approximation
    error over the pour rather than at one end.
    """
    d_top = measurements.get("diameter_top")
    d_bot = measurements.get("diameter_bottom")
    ds = [d for d in (d_top, d_bot) if d is not None]
    if not ds:
        raise KeyError("no diameter in measurements")
    return float(np.mean(ds)) / 2.0


def taper(measurements):
    """Relative taper |d_top - d_bot| / mean(d) — 0 for a true cylinder."""
    d_top, d_bot = measurements.get("diameter_top"), measurements.get("diameter_bottom")
    if d_top is None or d_bot is None:
        return 0.0
    return abs(d_top - d_bot) / (0.5 * (d_top + d_bot))


def decode_lambda_to_volume(lam, r_cm, win_s=0.5, rate=RATE, clamp_monotone=True):
    """lambda(t) [cm] -> (V_poured [mL], Q [mL/s]) per frame.

    `clamp_monotone`: water only ever enters the vessel, so V must be non-decreasing.
    A running maximum removes decode jitter that would otherwise show up as bursts of
    negative flow. Disable to inspect the raw decode.
    """
    lam_s = smooth(np.asarray(lam, dtype=np.float64), win_s, rate)
    area_q = np.pi * r_cm ** 2 / 4.0                 # pi R^2 / 4
    v = area_q * (lam_s[0] - lam_s)
    if clamp_monotone:
        v = np.maximum.accumulate(v)
    q = np.gradient(v) * rate
    return v, q


def frame_times(n, rate=RATE):
    return np.arange(n) / rate
