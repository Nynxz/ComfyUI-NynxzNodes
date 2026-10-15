<template>
  <div class="fp" role="radiogroup">
    <button
      v-for="o in options"
      :key="o.value"
      type="button"
      role="radio"
      class="fp-pill"
      :class="{ on: o.value === modelValue }"
      :aria-checked="o.value === modelValue"
      :disabled="disabled"
      :title="o.title"
      @click="emit('update:modelValue', o.value)"
    >
      {{ o.label }}
    </button>
  </div>
</template>

<script setup lang="ts">
/** A choice between a few options as free-standing pills — no enclosing frame. */
defineProps<{
  modelValue: string;
  options: { value: string; label: string; title?: string }[];
  disabled?: boolean;
}>();
const emit = defineEmits<{ "update:modelValue": [string] }>();
</script>

<style scoped>
.fp {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.fp-pill {
  height: 24px;
  padding: 0 12px;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: 999px;
  background: transparent;
  color: var(--zen-muted, #9aa0aa);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
  transition:
    background 0.12s ease,
    color 0.12s ease,
    border-color 0.12s ease;
}
.fp-pill:hover:not(:disabled) {
  color: var(--zen-text, #e5e5ea);
  border-color: color-mix(in srgb, var(--zen-text, #fff) 25%, transparent);
}
.fp-pill.on {
  border-color: color-mix(in srgb, var(--zen-accent, #6366f1) 70%, transparent);
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 18%, transparent);
  color: var(--zen-text, #e5e5ea);
}
.fp-pill:disabled {
  opacity: 0.45;
  cursor: default;
}
</style>
