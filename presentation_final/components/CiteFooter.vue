<script setup lang="ts">
/**
 * Source line for one slide, printed between the content box (ends at 91.1 %)
 * and the TUM footer (starts at 94.4 %). Drop `<CiteFooter />` anywhere on a
 * slide that uses <Cite> — it collects that slide's ids itself.
 *
 *   <CiteFooter />               -> "[4] Bagad et al. (2024): The Sound of ..."
 *   <CiteFooter mode="short" />  -> "[4] Bagad et al. 2024"
 *   <CiteFooter id="sow,psnn" /> -> explicit list instead of the collected one
 */
import { computed, unref } from 'vue'
import { useSlideContext } from '@slidev/client'
import { bibEntry, bibFull, bibNumber, bibShort } from '../bib'
import { citedOn } from '../cite-registry'

const props = withDefaults(
  defineProps<{
    /** Override the auto-collected ids (comma separated). */
    id?: string
    mode?: 'full' | 'short'
    /** Prefix, e.g. "Quellen: ". Empty by default. */
    label?: string
  }>(),
  { mode: 'full', label: '' },
)

const { $page } = useSlideContext()

const ids = computed(() =>
  props.id ? props.id.split(',').map(s => s.trim()).filter(Boolean) : citedOn(unref($page)),
)

const lines = computed(() =>
  ids.value
    .map((id) => {
      const e = bibEntry(id)
      if (!e) return { id, num: '?', text: `unbekannte Quelle: ${id}`, url: undefined }
      return {
        id,
        num: String(bibNumber(id)),
        text: props.mode === 'short' ? bibShort(e) : bibFull(e),
        url: e.url,
      }
    })
    // Keep the printed order stable: by citation number, not by mount order.
    .sort((a, b) => Number(a.num) - Number(b.num)),
)
</script>

<template>
  <div v-if="lines.length" class="tum-cite-footer">
    <span v-if="label" class="tum-cite-label">{{ label }}</span>
    <span v-for="l in lines" :key="l.id" class="tum-cite-item">
      <span class="tum-cite-num">[{{ l.num }}]</span>
      <a v-if="l.url" class="tum-cite-link" :href="l.url" target="_blank" rel="noopener">{{ l.text }}</a>
      <span v-else>{{ l.text }}</span>
    </span>
  </div>
</template>

<style scoped>
/* Same left margin as the content placeholder; sits just above the footer. */
.tum-cite-footer {
  position: absolute;
  left: var(--tum-margin-l2);
  /* Shrink-to-fit, so the translucent backing only covers its own text and not
   * a figure sharing the lower half of the slide. */
  max-width: var(--tum-col-w);
  /* Footer box starts at 94.39 % -> 5.61 % from the bottom; keep ~7 px clear. */
  bottom: 7.4%;
  font-size: var(--tum-chair-size);
  line-height: var(--tum-chair-line);
  color: #666;
  display: flex;
  flex-wrap: wrap;
  gap: 0 10px;
  /* Stays readable when an image layout runs a picture to the slide edge. */
  background: rgba(255, 255, 255, 0.82);
  padding: 1px 2px;
  border-radius: 2px;
}
.tum-cite-label {
  color: #666;
}
.tum-cite-num {
  color: var(--tum-blue);
}
.slidev-layout .tum-cite-link,
.slidev-layout .tum-cite-link:visited {
  color: inherit;
  text-decoration: none;
}
.slidev-layout .tum-cite-link:hover {
  text-decoration: underline;
}
</style>
