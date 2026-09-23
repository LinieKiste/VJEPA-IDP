<script setup lang="ts">
/**
 * pptx layout "1_zwei Inhalte" (slideLayout17): two equal content boxes,
 * off (311162 | 4647179, 1222524) ext (4180910, 3475102) each.
 *
 * Slots: title, subtitle, default (left column), right (right column).
 */
import TumChrome from '../components/TumChrome.vue'
import { useTumSlide, type TumSlideProps } from '../composables/useTumSlide'

defineOptions({ inheritAttrs: false })
const props = defineProps<TumSlideProps>()
const { title, subtitle, hasTitle, hasSubtitle } = useTumSlide(props)
</script>

<template>
  <div class="slidev-layout tum-slide">
    <h1 v-if="hasTitle" class="tum-title">
      <slot name="title">{{ title }}</slot>
    </h1>

    <div v-if="hasSubtitle" class="tum-subtitle">
      <slot name="subtitle">{{ subtitle }}</slot>
    </div>

    <div class="tum-col-left tum-rich"><slot /></div>
    <div class="tum-col-right tum-rich"><slot name="right" /></div>

    <TumChrome :footer="footer" :show-footer="!noFooter" />
  </div>
</template>
