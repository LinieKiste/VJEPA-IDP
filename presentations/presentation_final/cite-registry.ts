/* ---------------------------------------------------------------------------
 * Which sources are cited on which slide.
 *
 * `<Cite>` registers itself here on mount and unregisters on unmount, so
 * `<CiteFooter>` can print the sources for its own slide without the slide
 * having to repeat the ids. Keyed by page number (not by "current page"),
 * because Slidev keeps neighbouring slides mounted.
 * ------------------------------------------------------------------------- */

import { reactive } from 'vue'

/** page -> id -> how many <Cite> instances on that page use it */
const registry = reactive(new Map<number, Map<string, number>>())

export function registerCite(page: number, ids: string[]) {
  let onPage = registry.get(page)
  if (!onPage) {
    onPage = reactive(new Map<string, number>())
    registry.set(page, onPage)
  }
  for (const id of ids) onPage.set(id, (onPage.get(id) ?? 0) + 1)
}

export function unregisterCite(page: number, ids: string[]) {
  const onPage = registry.get(page)
  if (!onPage) return
  for (const id of ids) {
    const n = (onPage.get(id) ?? 0) - 1
    if (n > 0) onPage.set(id, n)
    else onPage.delete(id)
  }
  if (onPage.size === 0) registry.delete(page)
}

/** Ids cited on `page`, in the order they were first registered. */
export function citedOn(page: number): string[] {
  return [...(registry.get(page)?.keys() ?? [])]
}
