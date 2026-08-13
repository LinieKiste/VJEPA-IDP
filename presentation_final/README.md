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
| `figure` | **not in the pptx** — an addition for diagrams, see below |
| `section` | **not in the pptx** — an addition, built from the TUM colour scheme |

`layout: figure` is for **diagrams**, `layout: image` for **photos**. The pptx
picture placeholders start at 34.4 % (`large`) or 23.8 % (`full`) and run to the
slide edge: right for a full-bleed photo, wrong for a figure, which ends up
pushed down the slide with its bottom labels behind the footer while the band
under the title goes unused. `figure` hands the picture everything between the
head text and 91.11 %, and letterboxes it there — the slide writes a bare
`<img>` (or `<video>`/`<svg>`) with no wrapper div and no inline sizing:

```md
---
layout: figure
title: V-JEPA
caption: optional grey line under the figure
---

<img src="/VJEPA_arch.png" alt="…" />
```

The top edge is **measured** from the rendered title/subtitle, not read off the
placeholder geometry, because a long subtitle wraps to a second line and
overruns its 6.25 % box (the Bland-Altman slide does exactly that). `pad: false`
removes the 6 px of air under the head text.

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

## Citations

Sources live in **`bib.ts`** — one entry per source, `kind: 'paper' | 'web'`. The
citation number is the entry's position in that object, so the markers on the
slides and the list on the "Quellen" slide cannot drift apart; inserting an entry
renumbers everything after it and nothing hardcodes a number.

| in a slide | renders |
| --- | --- |
| `<Cite id="sow" />` | superscript `[4]`, hover shows the full reference |
| `<Cite id="sow,psnn" />` | `[4,5]` |
| `<Cite id="sow" mode="short" />` | `Bagad et al. 2024`[4] |
| `<Cite id="sow" mode="full" />` | the whole reference inline |
| `<CiteFooter />` | source line for *this* slide, above the footer |
| `<CiteFooter mode="short" />` | `[4] Bagad et al. 2024  [5] J. Wilson et al. 2019` |
| `<References cols="2" />` | the full list (`id=` / `kind=` to filter) |

`<Cite>` registers itself in `cite-registry.ts` (page → ids) and `<CiteFooter>`
reads that back, so a slide's footnote line never repeats the ids. Pass
`:footnote="false"` on a `<Cite>` to keep it out of the line, or give
`<CiteFooter id="…">` an explicit list. An unknown id renders a red `[?]` instead
of breaking the build. To get the footnote line on every slide automatically,
drop `<CiteFooter />` into `theme-tum/layouts/default.vue` next to `<TumChrome>`.

Metadata comes from the Zotero library (`~/Zotero/zotero.sqlite`, read on
2026-08-13 through a copy — never open the live db while Zotero runs). Three
entries are **not in Zotero** and still carry identifiers from memory:
`jepa` (LeCun 2022), `dinov3`, `groundingdino` — verify before the talk. Zotero
itself holds only title + authors for `vjepa` (Bardes et al.) and `egoper`
(Lee et al.), so their venue/arXiv id is also unverified.

`<CiteFooter>` on a `layout: image` slide overlays the picture, which runs to
the slide edge — the strip has a translucent white backing so it stays readable,
but on a full-bleed figure prefer citing only in `<References>`.

## Gotchas

- Do not put blank lines or `#` comments inside the headmatter block — the
  frontmatter parser stops at them and the rest leaks into slide 1 as content.
- Never type a layout prop as `string | boolean`. Vue casts an absent prop whose
  runtime type includes `Boolean` to `false`, which silently disabled the footer
  on every slide until it was split into `footer` + `noFooter`.
- Layout roots must carry the `slidev-layout` class or none of the theme's base
  typography applies.
- **A wrapping title must push what follows, not print on top of it.** Every
  placeholder is pinned by a percentage from the pptx, so a two-line title keeps
  its origin and grows downwards into the next box. `cover`, `cover-photo` and
  `section` solve it in CSS — title and info flow inside one positioned
  `.tum-cover-head`, with the info's `margin-top: 6.29px` reproducing the
  template's gap for a one-line title. `figure` solves it in JS, because its
  picture box has to know where the head text actually ended. The content
  layouts (`default`, `two-cols`, `content-image`, `image`) are still pinned:
  a title long enough to wrap will run into the subtitle there.
- `slidev export` needs `pnpm add -D playwright-chromium`. Without it, build the
  SPA and screenshot it with system Chromium instead.
