<script setup lang="ts">
/**
 * Section divider — an ADDITION, not a template layout. The pptx ships four
 * title variants and no divider, so this reuses the title-slide geometry
 * (slideLayout6) on a TUM-blue field with the white wordmark.
 */
import TumChrome from '../components/TumChrome.vue'
import { useTumSlide, type TumSlideProps } from '../composables/useTumSlide'

defineOptions({ inheritAttrs: false })
const props = defineProps<TumSlideProps>()
const { title } = useTumSlide(props)
</script>

<template>
  <div class="slidev-layout tum-slide tum-section">
    <h1 class="tum-cover-title"><slot name="title">{{ title }}</slot></h1>
    <div class="tum-cover-info tum-rich"><slot /></div>

    <TumChrome logo-color="#ffffff" :footer="footer" :show-footer="!noFooter" />
  </div>
</template>
