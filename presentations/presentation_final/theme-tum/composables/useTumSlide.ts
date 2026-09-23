import { computed, useSlots, type ComputedRef, type Slots } from 'vue'

export interface TumSlideProps {
  /** Slidev hands the whole slide frontmatter to the layout as this prop. */
  frontmatter?: Record<string, any>
  title?: string
  subtitle?: string
  lead?: string
  /** Per-slide footer override; defaults to `themeConfig.footer`. */
  footer?: string
  /**
   * Suppress the footer on this slide. Kept as a separate boolean because Vue
   * casts an absent prop typed `string | boolean` to `false`, which would
   * silently hide the footer on every slide.
   */
  noFooter?: boolean
}

/**
 * Slidev consumes the `title` frontmatter key for navigation and the TOC, so it
 * never reaches the layout as a plain prop — it only arrives inside
 * `frontmatter`. Read every head field through here so `title:` behaves like
 * `subtitle:` and `lead:` do.
 */
export function useTumSlide(props: TumSlideProps): {
  slots: Slots
  title: ComputedRef<string>
  subtitle: ComputedRef<string>
  lead: ComputedRef<string>
  hasTitle: ComputedRef<boolean>
  hasSubtitle: ComputedRef<boolean>
  hasLead: ComputedRef<boolean>
} {
  const slots = useSlots()

  const title = computed(() => props.title ?? props.frontmatter?.title ?? '')
  const subtitle = computed(() => props.subtitle ?? props.frontmatter?.subtitle ?? '')
  const lead = computed(() => props.lead ?? props.frontmatter?.lead ?? '')

  return {
    slots,
    title,
    subtitle,
    lead,
    hasTitle: computed(() => !!title.value || !!slots.title),
    hasSubtitle: computed(() => !!subtitle.value || !!slots.subtitle),
    hasLead: computed(() => !!lead.value || !!slots.lead),
  }
}
