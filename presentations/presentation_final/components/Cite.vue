<script setup lang="ts">
/**
 * Inline citation marker.
 *
 *   <Cite id="sow" />                 -> superscript [4]
 *   <Cite id="sow,psnn" />            -> [4,5]
 *   <Cite id="sow" mode="short" />    -> Bagad et al. 2024 [4]
 *   <Cite id="sow" mode="full" />     -> the whole reference, inline
 *
 * The number is the entry's position in `bib.ts`. An unknown id renders a red
 * [?] instead of failing the build, so a typo is visible on the slide.
 * Every marker also registers with the per-slide registry that <CiteFooter>
 * reads, and links to the source's url when there is one.
 */
import { computed, onBeforeUnmount, onMounted, unref } from 'vue'
import { useSlideContext } from '@slidev/client'
import { bibEntry, bibFull, bibNumber, bibShort } from '../bib'
import { registerCite, unregisterCite } from '../cite-registry'

const props = withDefaults(
  defineProps<{
    /** Bib key, or several separated by commas. */
    id: string
    mode?: 'num' | 'short' | 'full'
    /** Set false to keep it out of the slide's <CiteFooter>. */
    footnote?: boolean
  }>(),
  { mode: 'num', footnote: true },
)

const { $page } = useSlideContext()

const ids = computed(() => props.id.split(',').map(s => s.trim()).filter(Boolean))

const items = computed(() =>
  ids.value.map((id) => {
    const entry = bibEntry(id)
    return {
      id,
      entry,
      num: entry ? bibNumber(id) : 0,
      short: entry ? bibShort(entry) : id,
      full: entry ? bibFull(entry) : `unbekannte Quelle: ${id}`,
      url: entry?.url,
    }
  }),
)

const nums = computed(() => items.value.map(i => (i.entry ? String(i.num) : '?')).join(','))
const tooltip = computed(() => items.value.map(i => i.full).join(' · '))

onMounted(() => {
  if (props.footnote) registerCite(unref($page), ids.value)
})
onBeforeUnmount(() => {
  if (props.footnote) unregisterCite(unref($page), ids.value)
})
</script>

<template>
  <sup v-if="mode === 'num'" class="cite" :title="tooltip">
    [<template v-for="(i, k) in items" :key="i.id">
      <span v-if="k">,</span>
      <a v-if="i.url" class="cite-link" :href="i.url" target="_blank" rel="noopener"
         :class="{ 'cite--missing': !i.entry }">{{ i.entry ? i.num : '?' }}</a>
      <span v-else :class="{ 'cite--missing': !i.entry }">{{ i.entry ? i.num : '?' }}</span>
    </template>]
  </sup>

  <span v-else-if="mode === 'short'" class="cite-inline" :title="tooltip">
    <template v-for="(i, k) in items" :key="i.id">
      <span v-if="k">, </span>
      <a v-if="i.url" class="cite-link" :href="i.url" target="_blank" rel="noopener">{{ i.short }}</a>
      <span v-else>{{ i.short }}</span>
      <sup class="cite">[{{ i.entry ? i.num : '?' }}]</sup>
    </template>
  </span>

  <span v-else class="cite-inline" :title="tooltip">
    <template v-for="(i, k) in items" :key="i.id">
      <span v-if="k"> · </span>
      <a v-if="i.url" class="cite-link" :href="i.url" target="_blank" rel="noopener">{{ i.full }}</a>
      <span v-else>{{ i.full }}</span>
    </template>
  </span>
</template>

<style scoped>
/* line-height: 0 keeps the raised marker from growing the line box (and with
 * it a table row); <sup> already supplies the vertical shift. */
.cite {
  font-size: 0.62em;
  line-height: 0;
  color: var(--tum-blue);
  white-space: nowrap;
  margin-left: 0.1em;
}
.cite--missing {
  color: #c00;
}
.slidev-layout .cite-link,
.slidev-layout .cite-link:visited {
  color: inherit;
  text-decoration: none;
}
.slidev-layout .cite-link:hover {
  text-decoration: underline;
}
.cite-inline {
  color: inherit;
}
</style>
