<script setup lang="ts">
/**
 * The TUM wordmark, traced from the template's own vector art
 * (ppt/media/image1.wmf, "20150416 tum logo blau png final").
 *
 * The path below is that WMF's single outline, extracted from a PDF render of
 * the deck and translated to the origin. In slide points the mark measures
 * 47.476563 x 25 — exactly the aspect of its placeholder box
 * (cx=604774, cy=318516 EMU).
 */
withDefaults(defineProps<{ color?: string }>(), { color: 'var(--tum-blue)' })
</script>

<template>
  <svg
    class="tum-logo"
    viewBox="0 0 47.476563 25"
    xmlns="http://www.w3.org/2000/svg"
    role="img"
    aria-label="Technische Universität München"
  >
    <path
      :fill="color"
      fill-rule="evenodd"
      d="M 0 0 L 18.605469 0 L 18.605469 20.503906 L 23.722657 20.503906 L 23.722657 0 L 47.476563 0 L 47.476563 25 L 42.980469 25 L 42.980469 4.496093 L 37.832031 4.496093 L 37.832031 25 L 33.339844 25 L 33.339844 4.496093 L 28.191407 4.496093 L 28.191407 25 L 14.109375 25 L 14.109375 4.496093 L 8.960938 4.496093 L 8.960938 25 L 4.492188 25 L 4.492188 4.496093 L 0 4.496093 Z"
    />
  </svg>
</template>
