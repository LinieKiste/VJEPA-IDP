# IDP final presentation

Slidev deck for the SS26 IDP final talk, on a local Slidev theme that reproduces
`HLU_Presentation_Template_Oct_2022_Modified_by_Tian.pptx` (the TUM corporate
design, chair variant).

```bash
pnpm install
pnpm dev      # http://localhost:3030
pnpm build    # static SPA into dist/
```

The interim deck lives in `../presentation/` and is untouched.

## How the theme was derived

Nothing here was eyeballed. The pptx was unzipped and every number was read out
of the OOXML, then checked against a LibreOffice render of the original:

| source | what came from it |
| --- | --- |
| `ppt/presentation.xml` | canvas `9144000 × 5143500` EMU = 720 × 405 pt |
| `ppt/theme/theme5.xml` | `<a:clrScheme name="TUM">`, `<a:fontScheme name="TUM Arial">` |
| `ppt/slideMasters/slideMaster5.xml` | body/title text styles, bullet ramp, footer + page-number boxes, wordmark box |
| `ppt/slideLayouts/slideLayout{11,14,17,19,21,23,25}.xml` | the content layouts |
| `ppt/slideMasters/slideMaster4.xml` + `slideLayout{6,7}.xml` | white title slide, affiliation block |
| `ppt/slideMasters/slideMaster3.xml` + `slideLayout4.xml` | photo title slide |
| `ppt/media/image1.wmf` | the TUM wordmark outline |

**The deck sets `canvasWidth: 720`, so 1 CSS px is exactly 1 PowerPoint point.**
Every `font-size` in `theme-tum/styles/` is therefore the literal `sz` value from
the XML (`sz="2500"` → `25px`), and every box position is the EMU offset
expressed as a percentage of the canvas. `theme-tum/styles/tokens.css` carries
the raw EMU numbers in comments next to each token.

Two details that needed measuring rather than reading:

- **First-baseline model.** PowerPoint puts the first baseline at
  `boxTop + lineHeight − descent`; a browser puts it at
  `boxTop + halfLeading + ascent`. The difference is exactly the half-leading,
  so the title/subtitle/body boxes carry a `padding-top` of that amount
  (`--tum-*-pad`). Verified against the reference render: predicted subtitle
  baseline 75.448 pt vs 75.473 pt measured.
- **Arial's effective single-line height** in PowerPoint/LibreOffice is
  1.19771 em, not the 1.15 em the font's `hhea` metrics imply. Body text at
  `lnSpc 114 %` therefore uses `line-height: 1.36539`.

### Verification

The built deck was screenshotted with headless Chromium and compared landmark by
landmark against the LibreOffice render of the original pptx. All five chrome
landmarks agree to within ~1 pt on a 720 × 405 canvas:

| landmark | Δleft | Δtop |
| --- | --- | --- |
| Title | +0.30 pt | −0.24 pt |
| Subtitle | −0.92 pt | −0.57 pt |
| Footer | −0.20 pt | −0.47 pt |
| Page number | +0.96 pt | −0.47 pt |
| Wordmark | −0.28 pt | −0.40 pt |

The horizontal deltas are glyph side-bearing differences between the compared
strings, not box offsets.

Fonts: the theme asks for Arial and falls back to Liberation Sans, which is
metric-compatible and is what the reference render used.

## Layouts

| layout | pptx original |
| --- | --- |
| `cover` | slideLayout7 "1_Start" — white, Uhrenturm sketch, affiliation block. `art: false` gives slideLayout6 |
| `cover-photo` | slideLayout4 "Start" on slideMaster3 — full-bleed flags photo, white text |
| `default` | slideLayout14 "1_Inhalt + Text"; adding `lead:` turns it into slideLayout11 "Inhalt + Text" |
| `two-cols` | slideLayout17 "1_zwei Inhalte" — use `::right::` |
| `content-image` | slideLayout19 "1_Zwei Inhalte + Text"; `band: true` gives slideLayout21 "(Hintergrund)" |
| `image` | slideLayout23 "1_große Bilder"; `size: full` gives slideLayout25 "1_Bilder formatfüllend" |
| `section` | **not in the pptx** — an addition, built from the TUM colour scheme |

Head fields work as frontmatter or as slots (`::title::`, `::subtitle::`,
`::lead::`) when you need rich content:

```yaml
---
layout: default
title: Slide title            # 25 pt black,     "Titel 1"
subtitle: Blue standfirst     # 18 pt TUM blue,  "Text Placeholder 18"
lead: Optional intro block    # 14 pt,           "Textplatzhalter 7"
---
```

Per-slide chrome: `footer: "…"` overrides the footer text, `noFooter: true`
hides it.

## Deck-level config

```yaml
themeConfig:
  chair: [line 1, line 2, line 3]   # affiliation block on the cover
  footer: …                         # footer line on every content slide
```

`themeConfig` is the only channel that works — Slidev filters unknown top-level
headmatter keys out of `configs` via its `HEADMATTER_FIELDS` whitelist.

## Assets

`public/attentive_probe.png` is generated, not hand-drawn — regenerate it with

```bash
../.venv/bin/python figs_src/attentive_probe.py
```

`figs_src/` vendors PlotNeuralNet (MIT: `layers/*.sty` + `pycore/tikzeng.py`) and
the generator writes TikZ, compiles it with `tectonic` (falling back to
`pdflatex`) and rasterises with `pdftoppm` at 300 dpi. Colours are the TUM
palette from `theme-tum/styles/tokens.css`. The diagram is deliberately generic —
no sample frame, no concrete dimensions — so it describes the probe head rather
than one experiment; the concrete numbers live in the slide's presenter notes.

`public/tum-uhrenturm.jpg` and `public/tum-flags.jpg` are lifted straight out of
the pptx (`ppt/media/image2.jpeg`, `image3.jpeg`). The flags photo has the
template's own `srcRect` crop (`l=398 t=14167 b=10833`, i.e. exactly 16:9) baked
in. The wordmark is inline SVG in `theme-tum/components/TumLogo.vue`, traced
from `ppt/media/image1.wmf`.

## Gotchas

- Do not put blank lines or `#` comments inside the headmatter block — the
  frontmatter parser stops at them and the rest leaks into slide 1 as content.
- Never type a layout prop as `string | boolean`. Vue casts an absent prop whose
  runtime type includes `Boolean` to `false`, which silently disabled the footer
  on every slide until it was split into `footer` + `noFooter`.
- Layout roots must carry the `slidev-layout` class or none of the theme's base
  typography applies.
- `slidev export` needs `pnpm add -D playwright-chromium`. Without it, build the
  SPA and screenshot it with system Chromium instead.
