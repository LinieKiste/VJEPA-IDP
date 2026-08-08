<script setup lang="ts">
/**
 * pptx title slides on slideMaster4 ("Titel 3"):
 *   slideLayout7 "1_Start" — chair block, wordmark, title, info, Uhrenturm art
 *   slideLayout6 "Start"   — the same without the art
 * Neither carries a footer or a slide number.
 *
 * `art: false` gives the layout6 variant. The chair block falls back to
 * `themeConfig.chair` in the headmatter.
 */
import { computed } from 'vue'
import { configs } from '@slidev/client'
import TumLogo from '../components/TumLogo.vue'
import { useTumSlide, type TumSlideProps } from '../composables/useTumSlide'

defineOptions({ inheritAttrs: false })
const props = withDefaults(
  defineProps<
    TumSlideProps & {
      /** Lines of the top-left affiliation block (8 pt, TUM blue). */
      chair?: string[]
      /** Uhrenturm sketch; `false` renders slideLayout6. */
      art?: string | false
    }
  >(),
  { art: '/tum-uhrenturm.jpg' },
)

const { title } = useTumSlide(props)
const chairLines = computed<string[]>(
  () => props.chair ?? props.frontmatter?.chair ?? (configs.themeConfig?.chair as string[]) ?? [],
)
</script>

<template>
  <div class="slidev-layout tum-slide">
    <div v-if="chairLines.length" class="tum-chair">
      <p v-for="line in chairLines" :key="line">{{ line }}</p>
    </div>

    <TumLogo />

    <h1 class="tum-cover-title"><slot name="title">{{ title }}</slot></h1>
    <div class="tum-cover-info tum-rich"><slot /></div>

    <img v-if="art" class="tum-cover-art" :src="art" alt="" />
  </div>
</template>
