<script setup lang="ts">
/**
 * pptx title slides on slideMaster3: a full-bleed photo (Fahnen_HG.jpg,
 * off 0,0 ext = whole canvas, srcRect l=398 t=14167 b=10833) with white text
 * and the white wordmark (ppt/media/image4.emf).
 *   slideLayout4 "Start" / slideLayout5 "1_Start"
 */
import TumChrome from '../components/TumChrome.vue'
import { useTumSlide, type TumSlideProps } from '../composables/useTumSlide'

defineOptions({ inheritAttrs: false })
const props = withDefaults(
  defineProps<TumSlideProps & { image?: string }>(),
  { image: '/tum-flags.jpg' },
)
const { title } = useTumSlide(props)
</script>

<template>
  <div class="slidev-layout tum-slide tum-cover--photo tum-chrome--light">
    <img class="tum-cover-photo" :src="image" alt="" />

    <div class="tum-cover-head">
      <h1 class="tum-cover-title"><slot name="title">{{ title }}</slot></h1>
      <div class="tum-cover-info tum-rich"><slot /></div>
    </div>

    <TumChrome logo-color="#ffffff" :footer="footer" :show-footer="!!footer" :pagenum="false" />
  </div>
</template>
