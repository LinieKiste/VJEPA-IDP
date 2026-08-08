<script setup lang="ts">
/**
 * pptx layouts "Inhalt + Text" / "1_Inhalt + Text" (slideLayout11/12/14/15).
 *
 * Head fields (frontmatter or slot)
 *   title     — "Titel 1"             25 pt black
 *   subtitle  — "Text Placeholder 18" 18 pt TUM blue
 *   lead      — "Textplatzhalter 7"   14 pt intro block, optional
 *   default   — "Inhaltsplatzhalter 2"
 *
 * Supplying a lead drops the content box from y=23.7683% to y=34.7193%,
 * exactly as the template does between layout14 and layout11.
 */
import TumChrome from '../components/TumChrome.vue'
import { useTumSlide, type TumSlideProps } from '../composables/useTumSlide'

defineOptions({ inheritAttrs: false })
const props = defineProps<TumSlideProps>()
const { slots, title, subtitle, lead, hasTitle, hasSubtitle, hasLead } = useTumSlide(props)
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

    <div class="tum-content tum-rich" :class="{ 'tum-content--below-lead': hasLead }">
      <slot />
    </div>

    <TumChrome :footer="footer" :show-footer="!noFooter" />
  </div>
</template>
