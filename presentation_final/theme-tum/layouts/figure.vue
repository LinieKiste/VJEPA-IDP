<script setup lang="ts">
/**
 * Not in the pptx — an addition for DIAGRAMS.
 *
 * The template's picture placeholders (`layout: image`) start at 34.4 % / 23.8 %
 * and run to the slide edge, which is right for a photo but wrong for a figure:
 * a tall diagram ends up pushed down the slide with its bottom labels behind
 * the footer, and the band under the title is wasted.
 *
 * This layout gives the figure everything between the head text and the top of
 * the content box's bottom edge (91.11 %), and letterboxes it there. Slides
 * just write a plain `<img>`; the box does the centring and scaling.
 *
 *   pad: false   — no breathing room under the head text (default 6 px)
 *   caption:     — small grey line under the figure
 *
 * The top edge is MEASURED from the rendered title/subtitle rather than taken
 * from the placeholder geometry, because both can wrap to a second line — a
 * two-line subtitle overruns its 6.25 % box and would sit on top of the figure.
 */
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import TumChrome from '../components/TumChrome.vue'
import { useTumSlide, type TumSlideProps } from '../composables/useTumSlide'

defineOptions({ inheritAttrs: false })
const props = withDefaults(
  defineProps<TumSlideProps & { caption?: string; pad?: boolean }>(),
  { pad: true },
)
const { title, subtitle, hasTitle, hasSubtitle } = useTumSlide(props)

const titleEl = ref<HTMLElement | null>(null)
const subtitleEl = ref<HTMLElement | null>(null)
/** Distance from the slide top to the bottom of the head text, in canvas px. */
const headBottom = ref<number | null>(null)

function measure() {
  const bottoms = [titleEl.value, subtitleEl.value]
    .filter((el): el is HTMLElement => !!el)
    .map(el => el.offsetTop + el.offsetHeight)
  headBottom.value = bottoms.length ? Math.max(...bottoms) : null
}

let ro: ResizeObserver | undefined
onMounted(() => {
  measure()
  // Fonts and a re-layout after the slide is scaled can both change the height.
  ro = new ResizeObserver(measure)
  for (const el of [titleEl.value, subtitleEl.value]) if (el) ro.observe(el)
})
watch([hasTitle, hasSubtitle], measure)
onBeforeUnmount(() => ro?.disconnect())
</script>

<template>
  <div class="slidev-layout tum-slide">
    <h1 v-if="hasTitle" ref="titleEl" class="tum-title">
      <slot name="title">{{ title }}</slot>
    </h1>

    <div v-if="hasSubtitle" ref="subtitleEl" class="tum-subtitle">
      <slot name="subtitle">{{ subtitle }}</slot>
    </div>

    <div
      class="tum-figure"
      :class="[hasSubtitle ? 'tum-figure--below-subtitle' : '', pad ? 'tum-figure--pad' : '']"
      :style="headBottom != null ? { top: `${headBottom}px` } : undefined"
    >
      <div class="tum-figure-box">
        <slot />
      </div>
      <div v-if="caption || $slots.caption" class="tum-figure-caption">
        <slot name="caption">{{ caption }}</slot>
      </div>
    </div>

    <TumChrome :footer="footer" :show-footer="!noFooter" />
  </div>
</template>
