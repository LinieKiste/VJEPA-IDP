"""Generate the attentive-probe architecture figure with PlotNeuralNet.

Draws the probe head (`pouring/pour_probe/head.py` = V-JEPA 2's
`AttentiveClassifier`) behind a frozen encoder. Kept deliberately generic — no
sample frame, no concrete dimensions — so the picture holds for any clip and any
frozen token source:

    input clip
      -> frozen encoder        (no gradients)
      -> N x D tokens
      -> 3 x self-attention block         |
      -> cross-attention against 1 learnable query   > trained probe
      -> 1 x D pooled vector              |
      -> linear                           |
      -> scalar prediction

    python figs_src/attentive_probe.py     # -> public/attentive_probe.png

Needs a LaTeX engine on PATH (`tectonic`, else `pdflatex`) and `pdftoppm`
(poppler) for the PNG. PlotNeuralNet (MIT) is vendored in `plotneuralnet/`.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "plotneuralnet"))

from pycore.tikzeng import (  # noqa: E402
    to_Conv, to_begin, to_connection, to_end, to_generate, to_head,
)

OUT_PNG = HERE.parent / "public" / "attentive_probe.png"
DPI = 300

# TUM corporate palette, same hex values as theme-tum/styles/tokens.css
COLORS = r"""
\definecolor{tumblue}{HTML}{0065BD}
\definecolor{tumbluedark}{HTML}{003359}
\definecolor{tumbluemid}{HTML}{64A0C8}
\definecolor{tumbluelight}{HTML}{98C6EA}
\definecolor{tumgreen}{HTML}{A2AD00}
\definecolor{tumorange}{HTML}{E37222}
\definecolor{tumbeige}{HTML}{DAD7CB}
\usetikzlibrary{fit,backgrounds,decorations.pathreplacing,calc}
"""


CAP_Y = -3.9   # every block caption sits on this baseline, whatever the box height
GRP_Y = -6.0   # group titles below the dashed frames

# The pics' own `caption=` hangs 25pt under each box, so boxes of different heights
# would put their captions at different heights, and the xlabel/zlabel numbers
# collide with them. All dimension labels are therefore left blank on the pics and
# captions are drawn separately, anchored to CAP_Y; each is a named node so the
# group frames can `fit` around them.


def caption(name, title, sub="", width=3.2):
    body = rf"\textbf{{{title}}}" + (rf"\\[1pt]{{\small {sub}}}" if sub else "")
    return (rf"\node[anchor=north, align=center, text width={width}cm, inner sep=2pt]"
            rf" (cap-{name}) at ({name}-anchor |- capline) {{{body}}};" "\n")


def group_box(name, members, color, label):
    fit = " ".join(f"({m})" for m in members)
    return (
        r"\begin{scope}[on background layer]" "\n"
        rf"\node[draw={color}, line width=0.5mm, densely dashed, rounded corners=8pt,"
        rf" fill={color}!5, inner xsep=12pt, inner ysep=10pt, fit={fit}] ({name}) {{}};" "\n"
        r"\end{scope}" "\n"
        rf"\node[anchor=north, text={color}, font=\bfseries\large]"
        rf" at ({name} |- grpline) {{{label}}};" "\n"
    )


arch = [
    to_head(str(HERE / "plotneuralnet")),
    COLORS,
    to_begin(),
    rf"\coordinate (capline) at (0,{CAP_Y},0);" "\n"
    rf"\coordinate (grpline) at (0,{GRP_Y},0);" "\n",

    # ---- input: any video clip, drawn as a stack of frames -----------------
    to_Conv("clip", "", ", , ,", offset="(0,0,0)", to="(0,0,0)",
            height=26, depth=14, width="{1,1,1,1}"),
    caption("clip", "input clip", "", width=3.0),

    # ---- frozen encoder ----------------------------------------------------
    to_Conv("enc", "", "", offset="(3.0,0,0)", to="(clip-east)",
            height=30, depth=18, width=6),
    caption("enc", "encoder", "", width=3.2),
    to_connection("clip", "enc"),

    # ---- token sequence ----------------------------------------------------
    to_Conv("tok", "", "", offset="(2.6,0,0)", to="(enc-east)",
            height=30, depth=4, width=4),
    caption("tok", "tokens", r"$N\times D$", width=3.0),
    to_connection("enc", "tok"),

    # ---- 3 self-attention blocks ------------------------------------------
    # n_filer needs one (blank) entry per concatenated box or Box.sty indexes past
    # the end of its xlabel array
    to_Conv("sa", "", ", ,", offset="(2.4,0,0)", to="(tok-east)",
            height=30, depth=4, width="{4,4,4}"),
    caption("sa", r"self-attention $\times\,3$", r"$N\times D$", width=3.6),
    to_connection("tok", "sa"),

    # ---- cross-attention: N tokens -> 1 ------------------------------------
    to_Conv("xa", "", "", offset="(2.6,0,0)", to="(sa-east)",
            height=30, depth=4, width=4),
    caption("xa", "cross-attention", r"1 query $\leftarrow$ $N$ tokens", width=3.6),
    to_connection("sa", "xa"),

    # ---- the single learnable query, feeding the cross-attention from above -
    to_Conv("qry", "", "", offset="(0,4.4,0)", to="(xa-west)",
            height=4, depth=4, width=4),
    r"\node[anchor=south, align=center, text width=3.4cm] (cap-qry) at (qry-north)"
    r" {\textbf{learnable query}\\[1pt]{\small $1\times D$}};" "\n"
    r"\draw[-Stealth, line width=0.8mm, draw=\edgecolor, opacity=0.7]"
    r" (qry-south) -- (xa-north);" "\n",

    # ---- pooled vector -> scalar ------------------------------------------
    to_Conv("pool", "", "", offset="(2.4,0,0)", to="(xa-east)",
            height=4, depth=4, width=4),
    caption("pool", "pooled", r"$1\times D$", width=2.4),
    to_connection("xa", "pool"),

    to_Conv("lin", "", "", offset="(2.2,0,0)", to="(pool-east)",
            height=4, depth=4, width=2),
    caption("lin", "linear", r"$D\to1$", width=2.4),
    to_connection("pool", "lin"),

    r"\draw [connection] (lin-east) -- node {\midarrow} ($(lin-east)+(1.3,0,0)$);" "\n"
    r"\node[anchor=west, align=center, text=tumbluedark] (out) at ($(lin-east)+(1.5,0,0)$)"
    r" {\large$\hat{y}$};" "\n",

    # ---- what is frozen, what is trained ----------------------------------
    group_box("gfrozen", ["enc-nearnorthwest", "enc-nearsouthwest", "enc-farnortheast",
                          "enc-nearsoutheast", "cap-enc"],
              "tumbluedark", "frozen"),
    group_box("gtrain", ["sa-nearnorthwest", "sa-nearsouthwest", "lin-farnortheast",
                         "lin-nearsoutheast", "cap-sa", "cap-xa", "cap-pool", "cap-lin",
                         "cap-qry", "qry-nearnorthwest"],
              "tumorange", "attentive probe (trained)"),

    to_end(),
]

# Per-box fills. tikzeng hardcodes \ConvColor, so patch the emitted strings.
FILLS = {
    "clip": "tumbluelight", "enc": "tumbluedark", "tok": "tumbeige", "sa": "tumblue",
    "qry": "tumorange", "xa": "tumbluemid", "pool": "tumbeige",
    "lin": "tumgreen",
}


def recolor(chunk: str) -> str:
    for name, color in FILLS.items():
        if f"name={name}," in chunk:
            return chunk.replace(r"fill=\ConvColor", f"fill={color}")
    return chunk


def main() -> None:
    engine = shutil.which("tectonic") or shutil.which("pdflatex")
    if engine is None:
        sys.exit("no LaTeX engine found: install `tectonic` (or a texlive with pdflatex)")
    if shutil.which("pdftoppm") is None:
        sys.exit("pdftoppm not found: install poppler")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tex = tmp / "attentive_probe.tex"
        to_generate([recolor(c) for c in arch], str(tex))

        if Path(engine).name == "tectonic":
            cmd = [engine, "--keep-logs", "--outdir", str(tmp), str(tex)]
        else:
            cmd = [engine, "-interaction=nonstopmode", "-output-directory", str(tmp), str(tex)]
        r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        pdf = tmp / "attentive_probe.pdf"
        if not pdf.exists():
            sys.exit(f"LaTeX failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")

        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftoppm", "-png", "-r", str(DPI), "-singlefile",
                        str(pdf), str(OUT_PNG.with_suffix(""))], check=True)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
