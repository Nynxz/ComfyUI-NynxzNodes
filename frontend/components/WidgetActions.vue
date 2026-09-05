<!-- The dashed action row that closes a list-shaped node body: one primary "add" button that
     fills the width, and whatever secondary icon buttons the widget passes in the slot, inside
     the same outline.
     Its own component because both stack widgets need exactly this and "keep them consistent" is
     a promise better kept by construction than by remembering to copy 40 lines of CSS. The
     outline belongs to the ROW rather than to the add button so the icon buttons read as part of
     the same control group — and so the whole row can be a drop target where a widget wants one,
     which is a far easier thing to aim at than a button. -->
<template>
  <div class="wa" :class="{ active }">
    <button class="wa-main" :title="title" @click="emit('primary')">
      <i v-if="icon" class="mdi" :class="icon" />
      <span class="wa-label">{{ label }}</span>
      <span v-if="count" class="wa-count">{{ count }}</span>
    </button>
    <slot />
  </div>
</template>

<script setup lang="ts">
withDefaults(
  defineProps<{
    /** Primary button text. */
    label: string;
    icon?: string;
    title?: string;
    /** Badge on the primary button. Hidden at 0 — a count of nothing is noise. */
    count?: number;
    /** Highlight the row, e.g. while something is being dragged onto it. */
    active?: boolean;
  }>(),
  { icon: "mdi mdi-plus", active: false },
);
const emit = defineEmits<{ primary: [] }>();
</script>

<style scoped>
.wa {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 3px;
  border: 1px dashed var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 7px);
  transition:
    border-color 0.12s ease,
    background 0.12s ease;
}
.wa.active {
  border-color: var(--zen-accent, #6366f1);
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 10%, transparent);
}
/* the slotted icon buttons keep their own size; only the primary button flexes */
.wa > :not(.wa-main) {
  flex: none;
}
.wa-main {
  flex: 1;
  min-width: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  height: 26px; /* matches ZenIconButton, so the row is one height whatever it contains */
  border: none;
  border-radius: var(--zen-radius, 5px);
  background: transparent;
  color: var(--zen-muted, #9aa0aa);
  font: inherit;
  font-size: 11px;
  font-weight: 600;
  /* A label that can change (a drag hint, a growing count) must never WRAP: in a widget whose
     drop zones were measured once, a second line moves everything below it. */
  white-space: nowrap;
  overflow: hidden;
  cursor: pointer;
  transition:
    color 0.12s ease,
    background 0.12s ease;
}
.wa-main:hover {
  color: var(--zen-text, #e5e5ea);
  background: color-mix(in srgb, var(--zen-text, #fff) 8%, transparent);
}
.wa-main .mdi {
  font-size: 15px;
}
.wa-label {
  overflow: hidden;
  text-overflow: ellipsis;
}
.wa-count {
  padding: 0 5px;
  border-radius: var(--zen-radius, 8px);
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 22%, transparent);
  color: var(--zen-text, #e5e5ea);
  font-size: 10px;
  font-weight: 700;
}
</style>
