<script setup lang="ts">
/**
 * The full source list, for a "Quellen" slide at the end.
 *
 *   <References />                       -> everything in bib.ts, numbered
 *   <References cols="2" />              -> two columns
 *   <References id="sow,psnn,vjepa2" />  -> only these
 *   <References kind="web" />            -> only webpages (or "paper")
 */
import { computed } from 'vue'
import { bibEntry, bibFull, bibKeys, bibNumber, type BibEntry } from '../bib'

const props = withDefaults(
  defineProps<{
    id?: string
    kind?: 'paper' | 'web'
    cols?: number | string
  }>(),
  { cols: 1 },
)

const entries = computed(() => {
  const ids = props.id
    ? props.id.split(',').map(s => s.trim()).filter(Boolean)
    : (bibKeys() as string[])
  return ids
    .map(id => ({ id, num: bibNumber(id), e: bibEntry(id) as BibEntry }))
    .filter(r => r.e && (!props.kind || r.e.kind === props.kind))
    .sort((a, b) => a.num - b.num)
    .map(r => ({
      ...r,
      // Split the one-line reference so the authors can be bold without the
      // template re-assembling (and re-spacing) the rest by hand.
      rest: bibFull(r.e).slice(r.e.authors.length),
    }))
})

const styleCols = computed(() => ({ columns: String(props.cols) }))
</script>

<template>
  <!-- Plain divs, not <ol>: the theme styles list markers, and a numbered
       marker next to the [n] citation key would be two numbers per row. -->
  <div class="tum-refs" :style="styleCols">
    <div v-for="r in entries" :key="r.id" class="tum-ref">
      <span class="tum-ref-num">[{{ r.num }}]</span>
      <a v-if="r.e.url" class="tum-ref-body" :href="r.e.url" target="_blank" rel="noopener">
        <span class="tum-ref-authors">{{ r.e.authors }}</span>{{ r.rest }}
      </a>
      <span v-else class="tum-ref-body">
        <span class="tum-ref-authors">{{ r.e.authors }}</span>{{ r.rest }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.tum-refs {
  list-style: none;
  margin: 0;
  padding: 0;
  font-size: 9px;
  line-height: 12px;
  column-gap: 18px;
}
.tum-ref {
  display: flex;
  gap: 4px;
  break-inside: avoid;
  margin-bottom: 5px;
}
.tum-ref-num {
  color: var(--tum-blue);
  flex: none;
}
.tum-ref-authors {
  font-weight: 700;
}
.slidev-layout .tum-ref-body,
.slidev-layout .tum-ref-body:visited {
  color: inherit;
  text-decoration: none;
}
.slidev-layout .tum-ref-body:hover {
  text-decoration: underline;
}
</style>
