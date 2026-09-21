<script setup lang="ts">
/**
 * pptx layouts "1_große Bilder" (slideLayout23 — picture box at y=34.4333%,
 * leaving room for a lead line) and "1_Bilder formatfüllend" (slideLayout25 —
 * picture box at y=23.7683% running to the slide edge).
 *
 * `size: full` selects layout25; the default `large` selects layout23.
 * `crop: false` switches the full layout to `contain` — for wide diagrams
 * where the format-filling crop would cut off content.
 */
import TumChrome from '../components/TumChrome.vue'
import { useTumSlide, type TumSlideProps } from '../composables/useTumSlide'

defineOptions({ inheritAttrs: false })
const props = withDefaults(
  defineProps<TumSlideProps & { size?: 'large' | 'full'; crop?: boolean }>(),
  { size: 'large', crop: true },
)
const { title, subtitle, lead, hasTitle, hasSubtitle, hasLead } = useTumSlide(props)
</script>

<template>
  <div class="slidev-layout tum-slide">
    <h1 v-if="hasTitle" class="tum-title">
      <slot name="title">{{ title }}</slot>
    </h1>

    <div v-if="hasSubtitle" class="tum-subtitle">
      <slot name="subtitle">{{ subtitle }}</slot>
    </div>

    <div v-if="hasLead" class="tum-lead tum-rich">
      <slot name="lead">{{ lead }}</slot>
    </div>

    <div
      :class="[
        size === 'full' ? 'tum-image-full' : 'tum-image-large',
        crop === false ? 'tum-image-contain' : '',
      ]"
    >
      <slot />
    </div>

    <TumChrome :footer="footer" :show-footer="!noFooter" />
  </div>
</template>
