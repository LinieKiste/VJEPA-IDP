/* ---------------------------------------------------------------------------
 * Bibliography for the deck.
 *
 * ONE place for every source. `<Cite id="..." />` numbers entries by their
 * order in this object, so the numbers on the slides and the list rendered by
 * `<References />` can never drift apart. Adding an entry in the middle
 * renumbers everything after it — that is fine, nothing hardcodes a number.
 *
 * kind: 'paper'  -> "Authors (Year): Title. Venue." + optional url
 *       'web'    -> "Authors/Site: Title. url (abgerufen am ...)"
 *
 * Metadata below was taken from the Zotero library (~/Zotero/zotero.sqlite) on
 * 2026-08-13 — authors, venue, year and DOI/arXiv id are the ones Zotero holds.
 * Entries marked `NICHT IN ZOTERO` were not in the library; their identifiers
 * are from memory and still need checking against the PDF.
 * ------------------------------------------------------------------------- */

export interface BibEntry {
  kind: 'paper' | 'web'
  /** "P. Bagad et al." / "Meta AI" — rendered verbatim. */
  authors: string
  title: string
  /** Conference / journal (paper) or site name (web). */
  venue?: string
  year?: string | number
  url?: string
  /** Web only: Abrufdatum, e.g. "13.08.2026". */
  accessed?: string
  /** Optional override for the short form, default "<authors> <year>". */
  short?: string
}

export const bib = {
  jepa: {
    // NICHT IN ZOTERO — Angaben aus dem Gedächtnis, vor dem Vortrag prüfen.
    kind: 'paper',
    authors: 'Y. LeCun',
    title: 'A Path Towards Autonomous Machine Intelligence',
    venue: 'OpenReview',
    year: 2022,
    url: 'https://openreview.net/forum?id=BZ5a1r-kVsf',
  },
  vjepa: {
    // Zotero hat nur Titel + Autoren (kein Venue, keine ID); TMLR/arXiv-Angabe
    // ergänzt und noch ungeprüft.
    kind: 'paper',
    authors: 'A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, N. Ballas',
    title: 'Revisiting Feature Prediction for Learning Visual Representations from Video',
    venue: 'TMLR 2024, arXiv:2404.08471',
    year: 2024,
    url: 'https://arxiv.org/abs/2404.08471',
    short: 'Bardes et al. 2024',
  },
  vjepa2: {
    kind: 'paper',
    authors: 'M. Assran, A. Bardes, D. Fan, Q. Garrido et al.',
    title:
      'V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning',
    venue: 'arXiv:2506.09985',
    year: 2025,
    url: 'https://arxiv.org/abs/2506.09985',
    short: 'Assran et al. 2025',
  },
  vjepa21: {
    kind: 'paper',
    authors: 'L. Mur-Labadia, M. Muckley, A. Bar, M. Assran et al.',
    title: 'V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning',
    venue: 'arXiv:2603.14482',
    year: 2026,
    url: 'https://arxiv.org/abs/2603.14482',
    short: 'Mur-Labadia et al. 2026',
  },
  dinov3: {
    // NICHT IN ZOTERO — arXiv-ID aus dem Gedächtnis, vor dem Vortrag prüfen.
    kind: 'paper',
    authors: 'O. Siméoni et al.',
    title: 'DINOv3',
    venue: 'arXiv:2508.10104',
    year: 2025,
    url: 'https://arxiv.org/abs/2508.10104',
  },
  sow: {
    kind: 'paper',
    authors: 'P. Bagad, M. Tapaswi, C. G. M. Snoek, A. Zisserman',
    title: 'The Sound of Water: Inferring Physical Properties from Pouring Liquids',
    venue: 'ICASSP 2025, DOI 10.1109/ICASSP49660.2025.10889950, arXiv:2411.11222',
    year: 2025,
    url: 'https://arxiv.org/abs/2411.11222',
    short: 'Bagad et al. 2025',
  },
  psnn: {
    kind: 'paper',
    authors: 'J. Wilson, A. Sterling, M. C. Lin',
    title: 'Analyzing Liquid Pouring Sequences via Audio-Visual Neural Networks',
    venue: 'IROS 2019, DOI 10.1109/IROS40897.2019.8968118',
    year: 2019,
    url: 'https://ieeexplore.ieee.org/document/8968118/',
  },
  uwlpd: {
    kind: 'paper',
    authors: 'C. Schenck, D. Fox',
    title: 'Visual Closed-Loop Control for Pouring Liquids',
    venue: 'ICRA 2017, arXiv:1610.02610',
    year: 2017,
    url: 'https://arxiv.org/abs/1610.02610',
  },
  schenckfcn: {
    kind: 'paper',
    authors: 'C. Schenck, D. Fox',
    title: 'Perceiving and Reasoning About Liquids Using Fully Convolutional Networks',
    venue: 'arXiv:1703.01564',
    year: 2017,
    url: 'https://arxiv.org/abs/1703.01564',
  },
  simliquid: {
    kind: 'paper',
    authors: 'Y. Huang, J. Zhang, R. Yu, S. Li, W. Ding',
    title: 'SimLiquid: A Simulation-Based Liquid Perception Pipeline for Robot Liquid Manipulation',
    venue: 'Authorea Preprint, DOI 10.22541/au.173321942.24618438/v1',
    year: 2024,
    url: 'https://www.authorea.com/doi/full/10.22541/au.173321942.24618438/v1',
    short: 'Huang et al. 2024',
  },
  egoper: {
    // Zotero hat nur Titel + Autoren; Venue (CVPR 2024) ergänzt, ungeprüft.
    kind: 'paper',
    authors: 'S.-P. Lee, Z. Lu, Z. Zhang, M. Hoai, E. Elhamifar',
    title: 'Error Detection in Egocentric Procedural Task Videos',
    venue: 'CVPR 2024',
    year: 2024,
    url: 'https://openaccess.thecvf.com/content/CVPR2024/html/Lee_Error_Detection_in_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.html',
    short: 'Lee et al. 2024',
  },
  slowfastllava: {
    kind: 'paper',
    authors: 'M. Xu, M. Gao, S. Li, J. Lu et al.',
    title:
      'SlowFast-LLaVA-1.5: A Family of Token-Efficient Video Large Language Models for Long-Form Video Understanding',
    venue: 'arXiv:2503.18943',
    year: 2025,
    url: 'https://arxiv.org/abs/2503.18943',
    short: 'Xu et al. 2025',
  },
  groundingdino: {
    // NICHT IN ZOTERO — arXiv-ID aus dem Gedächtnis, vor dem Vortrag prüfen.
    kind: 'paper',
    authors: 'S. Liu et al.',
    title:
      'Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection',
    venue: 'ECCV 2024, arXiv:2303.05499',
    year: 2023,
    url: 'https://arxiv.org/abs/2303.05499',
  },
  vjepa2repo: {
    kind: 'web',
    authors: 'Meta AI',
    title: 'facebookresearch/vjepa2 — Code und Checkpoints',
    venue: 'GitHub',
    url: 'https://github.com/facebookresearch/vjepa2',
    accessed: '13.08.2026',
  },
  blenderproc: {
    kind: 'web',
    authors: 'DLR-RM',
    title: 'BlenderProc',
    venue: 'GitHub',
    url: 'https://github.com/DLR-RM/BlenderProc',
    accessed: '13.08.2026',
  },
  manhattanwalk: {
    kind: 'web',
    authors: 'NYC Walk',
    title: 'Manhattan Evening Walk (Null-Schüttvorgang-Test)',
    venue: 'YouTube',
    url: 'https://www.youtube.com/watch?v=1aedKShR1rA',
    accessed: '09.08.2026',
  },
} satisfies Record<string, BibEntry>

export type BibKey = keyof typeof bib

const keys = Object.keys(bib) as BibKey[]

/** 1-based citation number, stable = declaration order above. */
export function bibNumber(id: string): number {
  return keys.indexOf(id as BibKey) + 1
}

export function bibEntry(id: string): BibEntry | undefined {
  return (bib as Record<string, BibEntry>)[id]
}

export function bibKeys(): BibKey[] {
  return keys
}

/** "Bagad et al. 2025" — for inline use in running text. */
export function bibShort(e: BibEntry): string {
  return e.short ?? [e.authors, e.year].filter(Boolean).join(' ')
}

/** One-line full reference, without the leading number. */
export function bibFull(e: BibEntry): string {
  if (e.kind === 'web') {
    const tail = e.accessed ? ` (abgerufen am ${e.accessed})` : ''
    return `${e.authors}: ${e.title}. ${e.venue ?? ''}${tail}`.replace(/\s+\./g, '.')
  }
  const year = e.year ? ` (${e.year})` : ''
  const venue = e.venue ? ` ${e.venue}.` : ''
  return `${e.authors}${year}: ${e.title}.${venue}`
}
