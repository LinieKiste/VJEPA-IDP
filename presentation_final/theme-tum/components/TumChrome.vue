<script setup lang="ts">
import { computed } from 'vue'
import { configs, useNav } from '@slidev/client'
import TumLogo from './TumLogo.vue'

const props = withDefaults(
  defineProps<{
    /** Wordmark tint. slideMaster3 (photo cover) uses the white version. */
    logoColor?: string
    /** Footer text. Falls back to `themeConfig.footer`. */
    footer?: string
    /** Render the footer placeholder at all. */
    showFooter?: boolean
    /** Render the slide-number placeholder. */
    pagenum?: boolean
  }>(),
  { logoColor: 'var(--tum-blue)', showFooter: true, pagenum: true },
)

const { currentPage } = useNav()

// `themeConfig` is the sanctioned channel for theme options. Read it off the
// module-level `configs` rather than through `useSlideContext()`, which is only
// populated for components inside a slide's injection scope.
const footerText = computed(() =>
  props.showFooter ? (props.footer || (configs.themeConfig?.footer as string) || '') : '',
)
</script>

<template>
  <TumLogo :color="logoColor" />
  <div v-if="footerText" class="tum-footer">{{ footerText }}</div>
  <div v-if="pagenum" class="tum-pagenum">{{ currentPage }}</div>
</template>
