<template>
  <div class="mc" role="radiogroup">
    <button
      v-for="m in MODES"
      :key="m.id"
      type="button"
      role="radio"
      class="mc-card"
      :class="{ on: modelValue === m.id }"
      :aria-checked="modelValue === m.id"
      @click="emit('update:modelValue', m.id)"
    >
      <i class="mc-icon" :class="m.icon" />
      <span class="mc-label">{{ m.label }}</span>
      <span class="mc-model">{{ m.model }}</span>
      <span class="mc-what">{{ m.what }}</span>
    </button>
  </div>
</template>

<script setup lang="ts">
/** The prompt modes as cards to pick from: which model the layout is written for. */
import { PROMPT_MODES } from "@/layout/modes";
import type { PromptModeId } from "@/layout/promptScene";

defineProps<{ modelValue: PromptModeId | null }>();
const emit = defineEmits<{ "update:modelValue": [PromptModeId] }>();

const ABOUT: Record<PromptModeId, { model: string; what: string }> = {
  ideogram: {
    model: "Ideogram 4",
    what: "A JSON caption: description, style, background, then each box with its colours.",
  },
  klein: {
    model: "FLUX 3 Image Klein",
    what: "A caption that cites each box as <obj_N>, followed by the box table.",
  },
  moss: {
    model: "MOSS-Image",
    what: "A layout JSON in pixels: objects, characters and text, with tags and palettes.",
  },
};
const MODES = PROMPT_MODES.map((m) => ({
  id: m.id as PromptModeId,
  label: m.label,
  icon: m.icon,
  ...ABOUT[m.id as PromptModeId],
}));
</script>

<style scoped>
.mc {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.mc-card {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
  padding: 12px;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 7px);
  background: var(--zen-surface, #1e1e24);
  color: var(--zen-text, #e5e5ea);
  font: inherit;
  font-size: 12px;
  text-align: left;
  cursor: pointer;
}
.mc-card:hover {
  border-color: color-mix(in srgb, var(--zen-accent, #6366f1) 50%, var(--zen-border, #34343c));
}
.mc-card.on {
  border-color: var(--zen-accent, #6366f1);
  box-shadow: 0 0 0 1px var(--zen-accent, #6366f1);
}
.mc-icon {
  font-size: 22px;
  color: var(--zen-accent, #6366f1);
}
.mc-label {
  font-size: 14px;
  font-weight: 600;
}
.mc-model {
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.mc-what {
  margin-top: 4px;
  font-size: 11px;
  line-height: 1.4;
  color: var(--zen-muted, #9aa0aa);
}
</style>
