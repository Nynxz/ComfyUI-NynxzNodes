<template>
  <section class="fsec" :class="{ collapsed: collapsible && !isOpen }">
    <header class="fsec-head">
      <button
        v-if="collapsible"
        type="button"
        class="fsec-toggle"
        :aria-expanded="isOpen"
        @click="isOpen = !isOpen"
      >
        <i class="mdi" :class="isOpen ? 'mdi-chevron-down' : 'mdi-chevron-right'" />
        <span class="fsec-label">{{ label }}</span>
      </button>
      <span v-else class="fsec-label">{{ label }}</span>
      <InfoHover v-if="help || $slots.help" :label="`About ${label}`">
        <slot name="help">
          <p class="fsec-tip-title">{{ label }}</p>
          <p class="fsec-tip-text">{{ help }}</p>
        </slot>
      </InfoHover>
      <span v-if="note" class="fsec-note">{{ note }}</span>
      <span class="fsec-acts" :class="{ hidden: collapsible && !isOpen }">
        <slot name="actions" />
      </span>
    </header>
    <template v-if="!collapsible || isOpen">
      <slot />
    </template>
  </section>
</template>

<script setup lang="ts">
/** One block of the inspector: a small-caps label, optional `?` info hover, optional collapse
 *  toggle, muted note, and right-aligned action icons. */
import { ref, watch } from "vue";

import InfoHover from "@/components/layout/InfoHover.vue";

const props = withDefaults(
  defineProps<{
    label: string;
    note?: string;
    help?: string;
    collapsible?: boolean;
    defaultOpen?: boolean;
  }>(),
  {
    note: "",
    help: "",
    collapsible: false,
    defaultOpen: true,
  },
);

const isOpen = ref(props.defaultOpen);
watch(
  () => props.defaultOpen,
  (v) => {
    if (v) isOpen.value = true;
  },
);
</script>

<style scoped>
.fsec {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-top: 10px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border, #34343c) 60%, transparent);
}
.fsec:first-child {
  padding-top: 0;
  border-top: 0;
}
.fsec-head {
  display: flex;
  align-items: center;
  gap: 5px;
  min-height: 24px;
}
.fsec-toggle {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  padding: 0;
  border: 0;
  background: transparent;
  color: inherit;
  font: inherit;
  cursor: pointer;
}
.fsec-toggle .mdi {
  font-size: 14px;
  line-height: 1;
  color: var(--zen-muted, #9aa0aa);
}
.fsec-toggle:hover .fsec-label,
.fsec-toggle:hover .mdi {
  color: var(--zen-text, #e5e5ea);
}
.fsec-label {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--zen-muted, #9aa0aa);
}
.fsec-note {
  overflow: hidden;
  min-width: 0;
  white-space: nowrap;
  text-overflow: ellipsis;
  font-size: 10px;
  color: color-mix(in srgb, var(--zen-muted, #9aa0aa) 75%, transparent);
}
.fsec-acts {
  display: flex;
  gap: 2px;
  margin-left: auto;
}
.fsec-acts.hidden {
  visibility: hidden;
  pointer-events: none;
}
.fsec-tip-title {
  margin: 0 0 4px;
  font-size: 11.5px;
  font-weight: 600;
  color: var(--zen-text, #e5e5ea);
}
.fsec-tip-text {
  margin: 0;
  color: var(--zen-muted, #9aa0aa);
}
</style>
