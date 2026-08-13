"""Renderer for eval_videos.py -- composites the pour and the live probe readouts.

Layout (1600x900): the video on the right with the 256 px centre crop the encoders see
outlined; on the left a numeric readout block over two synchronised time plots
(instantaneous flow, cumulative poured mass). A cursor sweeps all panels together, so a
viewer can see the frame that produced any number.

Everything is drawn once and then updated via set_data/set_text -- redrawing the axes per
frame is ~10x slower and there are a few thousand frames to write.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle

# One colour per row, reused between the readout text and the curves.
COL = {"gt": "#111111", "vjepa": "#d62728", "dinov3": "#1f77b4", "sow": "#2ca02c"}
# The audio row is OUR ridge on SoW's frozen wav2vec2 features, not their method: their
# published readout decodes the wavelength through Eq. (6), which needs the vessel to be
# filled to the brim. Naming it "Sound of Water" would credit them with a number their
# method never produced.
# On-screen text is GERMAN (the talk is in German); code and comments stay English.
NAME = {"gt": "Ground Truth (Waage)", "vjepa": "V-JEPA 2 (attentiv)",
        "dinov3": "DINOv3 (attentiv)", "sow": "SoW wav2vec2 + unsere Ridge"}
DISPLAY_SHORT_SIDE = 400          # decode height for the right panel


FMT2, FMT_CLOCK = "{:.2f}", "{:5.2f}"


def de(x, fmt="{:.1f}"):
    """Format a number with a German decimal comma (on-screen text is German)."""
    return fmt.format(x).replace(".", ",")


FIG_W, FIG_H, DPI = 16.0, 9.0, 100


def decode_display(path, short_side=DISPLAY_SHORT_SIDE):
    """(N,h,w,3) uint8 at native fps, aspect preserved -- what the viewer sees."""
    import decord
    vr = decord.VideoReader(str(path))
    h, w = vr[0].shape[:2]
    scale = short_side / min(h, w)
    vr = decord.VideoReader(str(path), width=int(round(w * scale)),
                            height=int(round(h * scale)))
    return vr.get_batch(np.arange(len(vr))).asnumpy(), float(vr.get_avg_fps())


from eval_videos import cumulative          # noqa: E402  (shared integration convention)


def render_one(kind, key, cam, models, blurb, d, video_path, out_path):
    MODELS = tuple(models)
    frames, vfps = decode_display(video_path)
    tmid = d["tmid"]
    dur = float(d["dur"][0])
    gt_total = float(d["gt_total"][0])
    gt_flow = d["gt_flow"] if not np.isnan(d["gt_flow"]).all() else None
    gt_vol = d["gt_vol"] if not np.isnan(d["gt_vol"]).all() else None

    cum = {m: cumulative(tmid, d[m], dur) for m in MODELS}
    gt_cum = gt_vol if gt_vol is not None else None

    # Readout + flow on the left, video + cumulative on the right. The video is 16:9, so
    # giving it a full-height column would leave half that column empty.
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.22], height_ratios=[1.12, 1.0],
                          left=0.055, right=0.985, top=0.905, bottom=0.075,
                          wspace=0.16, hspace=0.30)
    ax_txt = fig.add_subplot(gs[0, 0]); ax_txt.axis("off")
    ax_flow = fig.add_subplot(gs[1, 0])
    ax_vid = fig.add_subplot(gs[0, 1]); ax_vid.axis("off")
    ax_cum = fig.add_subplot(gs[1, 1])

    title = f"{key}   -   {blurb}"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.968)

    # ---------------------------------------------------------------- video panel
    im = ax_vid.imshow(frames[0])
    h, w = frames[0].shape[:2]
    side = 256 / 288 * min(h, w)                 # the 256 crop, in display pixels
    ax_vid.add_patch(Rectangle(((w - side) / 2, (h - side) / 2), side, side,
                               fill=False, ec="#ff9900", lw=2.0, ls="--"))
    ax_vid.text(0.5, -0.03, "orange = der 256-px-Mittenausschnitt, den die Encoder sehen",
                transform=ax_vid.transAxes, ha="center", va="top",
                fontsize=10, color="#666666")

    # ---------------------------------------------------------------- flow panel
    lines_flow = {}
    if gt_flow is not None:
        lines_flow["gt"] = ax_flow.plot(tmid, gt_flow, color=COL["gt"], lw=3.0,
                                        label=NAME["gt"], zorder=5)[0]
    for m in MODELS:
        lines_flow[m] = ax_flow.plot(tmid, d[m], color=COL[m], lw=2.0, label=NAME[m])[0]
    ax_flow.set_xlim(0, dur); ax_flow.set_xlabel("Zeit (s)")
    ax_flow.set_ylabel("Flussrate (g/s)")
    ax_flow.set_title("Flussrate", fontsize=12, color="#444444", loc="left")
    ax_flow.axhline(0, color="#bbbbbb", lw=0.8)
    ax_flow.grid(alpha=0.25)
    cur_flow = ax_flow.axvline(0, color="#888888", lw=1.5)
    dots_flow = {k: ax_flow.plot([], [], "o", color=COL[k], ms=8, zorder=6)[0]
                 for k in lines_flow}

    # ---------------------------------------------------------------- cumulative panel
    for m in MODELS:
        ax_cum.plot(tmid, cum[m], color=COL[m], lw=2.0)
    if gt_cum is not None:
        ax_cum.plot(tmid, gt_cum, color=COL["gt"], lw=3.0, zorder=5)
    if not np.isnan(gt_total):
        ax_cum.axhline(gt_total, color=COL["gt"], ls=":", lw=2.0)
        ax_cum.text(dur * 0.99, gt_total, f"Endwert Ground Truth {gt_total:.0f} g ",
                    ha="right", va="bottom", fontsize=11, color=COL["gt"], fontweight="bold")
    ax_cum.set_xlim(0, dur); ax_cum.set_ylabel("Schüttvolumen (g)")
    ax_cum.set_xlabel("Zeit (s)")
    ax_cum.set_title("Masse insgesamt",
                     fontsize=12, color="#444444", loc="left")
    ax_cum.grid(alpha=0.25)
    cur_cum = ax_cum.axvline(0, color="#888888", lw=1.5)
    dots_cum = {m: ax_cum.plot([], [], "o", color=COL[m], ms=8, zorder=6)[0]
                for m in MODELS}

    # ---------------------------------------------------------------- readout block
    rows = ["gt"] + list(MODELS)
    clock = ax_txt.text(0.00, 1.10, "t = 0.00 s", fontsize=15, color="#333333",
                        va="top", ha="left", family="monospace", fontweight="bold")
    ax_txt.text(0.72, 1.10, "Flussrate g/s", fontsize=12, color="#666666", va="top", ha="right")
    ax_txt.text(1.00, 1.10, "Schüttvolumen g", fontsize=12, color="#666666", va="top", ha="right")
    ax_txt.plot([0, 1], [0.99, 0.99], color="#cccccc", lw=1.0, clip_on=False)
    txt = {}
    for i, k in enumerate(rows):
        y = 0.82 - i * 0.235
        ax_txt.text(0.00, y, NAME[k], fontsize=14, color=COL[k], va="center",
                    fontweight="bold" if k == "gt" else "normal")
        txt[(k, "flow")] = ax_txt.text(0.72, y, "-", fontsize=19, color=COL[k],
                                       va="center", ha="right", family="monospace")
        txt[(k, "cum")] = ax_txt.text(1.00, y, "-", fontsize=19, color=COL[k],
                                      va="center", ha="right", family="monospace")
    if gt_flow is None:
        ax_txt.text(0.0, -0.14, "* kein Gewichtsverlauf für diesen Schüttvorgang - nur die Endmasse "
                                "wurde gemessen", fontsize=10, color="#666666", va="top")
    ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)

    def val(arr, t):
        return float(np.interp(t, tmid, arr))

    n_out = int(dur * vfps)
    writer = FFMpegWriter(fps=vfps, bitrate=4000,
                          metadata={"title": title, "artist": "pour_probe"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(out_path), DPI):
        for i in range(n_out):
            t = i / vfps
            im.set_data(frames[min(i, len(frames) - 1)])
            cur_flow.set_xdata([t, t]); cur_cum.set_xdata([t, t])
            clock.set_text(f"t = {de(t, FMT_CLOCK)} s")

            if gt_flow is not None:
                gf, gc = val(gt_flow, t), val(gt_cum, t)
                txt[("gt", "flow")].set_text(de(gf, "{:6.1f}"))
                txt[("gt", "cum")].set_text(f"{gc:6.0f}")
                dots_flow["gt"].set_data([t], [gf])
            else:
                txt[("gt", "flow")].set_text("  k.A.")
                txt[("gt", "cum")].set_text(f"{gt_total:6.0f}*" if not np.isnan(gt_total)
                                            else "  k.A.")
            for m in MODELS:
                f_, c_ = val(d[m], t), val(cum[m], t)
                txt[(m, "flow")].set_text(de(f_, "{:6.1f}"))
                txt[(m, "cum")].set_text(f"{c_:6.0f}")
                dots_flow[m].set_data([t], [f_]); dots_cum[m].set_data([t], [c_])
            if gt_cum is not None:
                dots_cum.setdefault("gt", ax_cum.plot([], [], "o", color=COL["gt"],
                                                      ms=8, zorder=7)[0])
                dots_cum["gt"].set_data([t], [val(gt_cum, t)])
            writer.grab_frame()
    plt.close(fig)
    print(f"  wrote {out_path}  ({n_out} frames, {dur:.1f} s)")


def render_all(sources, pred_npz, out_dir):
    a = np.load(pred_npz, allow_pickle=True)
    root = Path(__file__).resolve().parents[2]
    for kind, key, cam, models, blurb in sources:
        tag = f"{kind}_{key}" + (f"_{cam}" if cam else "")
        if f"{tag}__tmid" not in a.files:
            print(f"  skip {key}: no cached predictions (run --infer first)")
            continue
        d = {f.split("__")[1]: a[f] for f in a.files if f.startswith(tag + "__")}
        vp = (root / f"datasets/pouring_processed/clips/{cam}/{key}.mp4" if kind == "clip"
              else root / f"datasets/eval/videos/{key}.MOV")
        if kind == "yt":
            vp = root / f"datasets/eval/videos/{key}.mp4"
        render_one(kind, key, cam, models, blurb, d, vp, Path(out_dir) / f"{tag}.mp4")


# ------------------------------------------------- SoW's own data: volume in mL

SOW_NAME = {"gt": "Ground Truth: SoW-Physik, GEMESSENER Radius",
            "est": "SoW-Physik, Radius aus Audio geschätzt",
            "vjepa": "V-JEPA 2 (attentiv, unser Modell)"}
SOW_COL = {"gt": "#111111", "est": "#2ca02c", "vjepa": "#d62728"}


def sow_frames(item_id):
    """Decode the cached 288 px container crop -- exactly the pixels the probe sees,
    so no crop box has to be reconstructed in the original video's coordinates."""
    import cv2
    a = np.load(Path("/home/casimir/.cache/pour_probe/sow_frames288") / f"{item_id}.npz",
                allow_pickle=True)
    fr = np.stack([cv2.imdecode(b, cv2.IMREAD_COLOR)[:, :, ::-1] for b in a["jpegs"]])
    return fr, float(a["fps"])


def render_sow(item_id, pred_npz, out_dir):
    """The comparison that our own pours cannot support: on a SoW video the vessel is
    filled to the brim, so Eq. (6) `R = lambda(T)/4beta` is valid and their published
    pipeline runs end-to-end with no measured container. The lambda panel shows where
    that radius comes from."""
    a = np.load(pred_npz, allow_pickle=True)
    tag = f"sow_{item_id}"
    if f"{tag}__tmid" not in a.files:
        raise SystemExit(f"no cached SoW predictions for {item_id} (run --sow --infer)")
    d = {f.split("__")[1]: a[f] for f in a.files if f.startswith(tag + "__")}
    tmid, t_phys, lam = d["tmid"], d["t_phys"], d["lam"]
    dur = float(d["dur"][0])
    r_meas, r_est = float(d["r_meas"][0]), float(d["r_est"][0])
    curves = {"gt": (t_phys, d["v_meas"]), "est": (t_phys, d["v_est"]),
              "vjepa": (tmid, d["vjepa"])}

    frames, vfps = sow_frames(item_id)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.22], height_ratios=[1.12, 1.0],
                          left=0.055, right=0.985, top=0.865, bottom=0.075,
                          wspace=0.16, hspace=0.30)
    ax_txt = fig.add_subplot(gs[0, 0]); ax_txt.axis("off")
    ax_v = fig.add_subplot(gs[1, 0])
    ax_vid = fig.add_subplot(gs[0, 1]); ax_vid.axis("off")
    ax_lam = fig.add_subplot(gs[1, 1])

    title = f"Sound-of-Water-Datensatz - {item_id}"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.975)
    fig.text(0.5, 0.930, f"{str(d['container'][0])}, transparenter Zylinder, "
                         f"BIS ZUM RAND GEFÜLLT  -  nicht im Training unseres Modells",
             ha="center", fontsize=12, color="#444444")

    im = ax_vid.imshow(frames[0])
    h, w = frames[0].shape[:2]
    side = 256 / 288 * min(h, w)
    ax_vid.add_patch(Rectangle(((w - side) / 2, (h - side) / 2), side, side,
                               fill=False, ec="#ff9900", lw=2.0, ls="--"))
    ax_vid.text(0.5, -0.03, "gecachter Behälter-Ausschnitt; orange = die 256 px, die der Encoder sieht",
                transform=ax_vid.transAxes, ha="center", va="top",
                fontsize=10, color="#666666")

    for k, (tt, vv) in curves.items():
        ax_v.plot(tt, vv, color=SOW_COL[k], lw=3.0 if k == "gt" else 2.0,
                  zorder=5 if k == "gt" else 3)
    ax_v.set_xlim(0, dur); ax_v.set_xlabel("Zeit (s)"); ax_v.set_ylabel("Schüttvolumen (mL)")
    ax_v.set_title("Schüttvolumen", fontsize=12, color="#444444", loc="left")
    ax_v.grid(alpha=0.25)
    cur_v = ax_v.axvline(0, color="#888888", lw=1.5)
    dots_v = {k: ax_v.plot([], [], "o", color=SOW_COL[k], ms=8, zorder=6)[0] for k in curves}

    ax_lam.plot(t_phys, lam, color="#7d3c98", lw=2.0)
    ax_lam.plot([t_phys[-1]], [lam[-1]], "o", color="#7d3c98", ms=9)
    ax_lam.annotate(f"$\\lambda(T)$ = {de(lam[-1])} cm  $\\Rightarrow$  "
                    f"$R=\\lambda(T)/4\\beta$ = {de(r_est, FMT2)} cm\n"
                    f"gemessener R = {de(r_meas, FMT2)} cm   "
                    f"(Verhältnis {de(r_est / r_meas, FMT2)})",
                    # the audio runs a hair past the video, so the true endpoint can sit
                    # outside xlim -- annotate() clips silently in that case
                    xy=(min(float(t_phys[-1]), dur), float(lam[-1])), xycoords="data",
                    xytext=(0.97, 0.72), textcoords="axes fraction",
                    ha="right", fontsize=11, color="#7d3c98", fontweight="bold",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="->", color="#7d3c98", lw=1.5,
                                    connectionstyle="arc3,rad=-0.25"))
    ax_lam.set_xlim(0, dur); ax_lam.set_xlabel("Zeit (s)")
    ax_lam.set_ylabel("Wellenlänge $\\lambda$ (cm)")
    ax_lam.set_title("Die von SoW dekodierte Tonhöhe - der Radius folgt aus ihrem ENDWERT",
                     fontsize=12, color="#444444", loc="left")
    ax_lam.grid(alpha=0.25)
    cur_lam = ax_lam.axvline(0, color="#888888", lw=1.5)
    dot_lam = ax_lam.plot([], [], "o", color="#7d3c98", ms=8, zorder=6)[0]

    clock = ax_txt.text(0.00, 1.10, "t = 0.00 s", fontsize=15, color="#333333",
                        va="top", ha="left", family="monospace", fontweight="bold")
    ax_txt.text(1.00, 1.10, "Volumen mL", fontsize=12, color="#666666", va="top", ha="right")
    ax_txt.plot([0, 1], [0.99, 0.99], color="#cccccc", lw=1.0, clip_on=False)
    txt = {}
    for i, k in enumerate(("gt", "est", "vjepa")):
        y = 0.80 - i * 0.26
        ax_txt.text(0.00, y, SOW_NAME[k], fontsize=14, color=SOW_COL[k], va="center",
                    fontweight="bold" if k == "gt" else "normal")
        txt[k] = ax_txt.text(1.00, y, "-", fontsize=20, color=SOW_COL[k],
                             va="center", ha="right", family="monospace")
    ax_txt.text(0.0, -0.02, "Ihre Radius-Schätzung setzt voraus, dass das Gefäß am Ende VOLL "
                            "ist. Unsere Schüttvorgänge füllen nie eines,\ndaher ist dieser Weg bei "
                            "uns versperrt - dieselbe Tasse liefert über 48 Schüttvorgänge R = 1,7-11,4 cm.",
                fontsize=10, color="#666666", va="top")
    ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)

    n_out = int(dur * vfps)
    writer = FFMpegWriter(fps=vfps, bitrate=4000,
                          metadata={"title": item_id, "artist": "pour_probe"})
    out_path = Path(out_dir) / f"{tag}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(out_path), DPI):
        for i in range(n_out):
            t = i / vfps
            im.set_data(frames[min(i, len(frames) - 1)])
            cur_v.set_xdata([t, t]); cur_lam.set_xdata([t, t])
            clock.set_text(f"t = {de(t, FMT_CLOCK)} s")
            for k, (tt, vv) in curves.items():
                y = float(np.interp(t, tt, vv))
                txt[k].set_text(f"{y:6.0f}"); dots_v[k].set_data([t], [y])
            dot_lam.set_data([t], [float(np.interp(t, t_phys, lam))])
            writer.grab_frame()
    plt.close(fig)
    print(f"  wrote {out_path}  ({n_out} frames, {dur:.1f} s)")
