<script setup lang="ts">
/**
 * pptx layouts "1_Zwei Inhalte + Text" (slideLayout19) and its
 * "(Hintergrund)" sibling (slideLayout21), which lays the same two boxes over
 * a 5%-black panel — <a:schemeClr val="bg1"><a:lumMod val="95000"/>.
 *
 * Set `band: true` to get the panel. Slots: title, subtitle, lead,
 * default (left content), right (picture placeholder).
 */
import TumChrome from '../components/TumChrome.vue'
import { useTumSlide, type TumSlideProps } from '../composables/useTumSlide'

defineOptions({ inheritAttrs: false })
const props = withDefaults(
  defineProps<TumSlideProps & { band?: boolean }>(),
  { band: false },
)
const { title, subtitle, lead, hasTitle, hasSubtitle, hasLead } = useTumSlide(props)
</script>

<template>
  <div class="slidev-layout tum-slide">
    <div v-if="band" class="tum-band" />

    <h1 v-if="hasTitle" class="tum-title">
      <slot name="title">{{ title }}</slot>
    </h1>

    <div v-if="hasSubtitle" class="tum-subtitle">
      <slot name="subtitle">{{ subtitle }}</slot>
    </div>

    <div v-if="hasLead" class="tum-lead tum-rich">
      <slot name="lead">{{ lead }}</slot>
    </div>

    <div class="tum-col-left tum-col-left--band tum-rich"><slot /></div>
    <div class="tum-col-right tum-col-right--band tum-rich"><slot name="right" /></div>

    <TumChrome :footer="footer" :show-footer="!noFooter" />
  </div>
</template>
