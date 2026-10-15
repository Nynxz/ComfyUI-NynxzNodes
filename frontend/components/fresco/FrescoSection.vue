<template>
  <section class="fsec">
    <header class="fsec-head">
      <span class="fsec-label">{{ label }}</span>
      <span v-if="note" class="fsec-note">{{ note }}</span>
      <span class="fsec-acts"><slot name="actions" /></span>
    </header>
    <slot />
  </section>
</template>

<script setup lang="ts">
/** One block of the inspector: a small caps label, a note, and its actions as icons at the end. */
defineProps<{ label: string; note?: string }>();
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
  gap: 6px;
  min-height: 22px;
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
</style>
