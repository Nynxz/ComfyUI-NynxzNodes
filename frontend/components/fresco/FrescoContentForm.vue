<template>
  <div class="fcf">
    <div class="fcf-label">Prompt</div>
    <ZenInput
      ref="promptInput"
      type="textarea"
      :rows="promptRows"
      sm
      :model-value="content.prompt"
      :placeholder="promptPlaceholder"
      @update:model-value="emit('patch', { prompt: String($event ?? '') })"
    />

    <div class="fcf-label">References</div>
    <FrescoRefs :model-value="content.refs" @update:model-value="emit('patch', { refs: $event })" />
    <div class="fcf-grid">
      <span :title="BLEND_TIP">blend</span>
      <div class="fcf-pair">
        <ZenSelect
          size="sm"
          :model-value="content.blend"
          :options="BLENDS"
          :disabled="activeRefs < 2"
          :title="BLEND_TIP"
          @update:model-value="emit('patch', { blend: $event as Blend })"
        />
        <ZenNumber
          size="sm"
          :model-value="content.softness"
          :min="0"
          :max="1"
          :step="0.05"
          :precision="2"
          :disabled="activeRefs < 2"
          title="Softness between references — 0 hard cells, 1 fully feathered"
          @update:model-value="emit('patch', { softness: $event })"
        />
      </div>
      <span :title="INFLUENCE_TIP">influence</span>
      <ZenNumber
        size="sm"
        :model-value="content.influence"
        :min="0"
        :max="4"
        :step="0.05"
        :precision="2"
        :title="INFLUENCE_TIP"
        @update:model-value="emit('patch', { influence: $event })"
      />
    </div>

    <div class="fcf-label">LoRAs</div>
    <FrescoLoras
      :model-value="content.loras"
      @update:model-value="emit('patch', { loras: $event })"
    />

    <slot />

    <div class="fcf-label">Edit</div>
    <div class="fcf-grid">
      <span :title="DENOISE_TIP">denoise</span>
      <ZenNumber
        size="sm"
        :model-value="content.denoise"
        :min="0"
        :max="1"
        :step="0.05"
        :precision="2"
        :title="DENOISE_TIP"
        @update:model-value="emit('patch', { denoise: $event })"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * What a layer — or the background — carries: its prompt, references and how they blend, how
 * strongly the area follows them, its LoRAs, and how much of a source image it may rewrite. The
 * default slot sits between the LoRAs and the edit fields, for the layer's shape.
 *
 * Every number is a ZenNumber: drag to scrub, the steppers, or type an exact value.
 */
import { computed, ref } from "vue";
import { ZenInput, ZenNumber, ZenSelect } from "@nynxz/zenkit-ui";

import FrescoLoras from "@/components/fresco/FrescoLoras.vue";
import FrescoRefs from "@/components/fresco/FrescoRefs.vue";
import { BLENDS, type Blend, type FrescoContent } from "@/lib/frescoScene";

const props = withDefaults(
  defineProps<{ content: FrescoContent; promptPlaceholder?: string; promptRows?: number }>(),
  { promptPlaceholder: "", promptRows: 3 },
);
const emit = defineEmits<{ patch: [Partial<FrescoContent>] }>();

const BLEND_TIP =
  "How several references combine: Mix interleaves them evenly, Weighted gives a stronger " +
  "reference more of the reading, Tiled gives each one part to itself, Salient lets each part " +
  "take from whichever reference shows something there. The number is the softness between them.";
const INFLUENCE_TIP =
  "How strongly this area follows its own prompt and references against the rest of the " +
  "picture. Raise it when they aren't coming through; too high and the area stops fitting in. " +
  "Below 1 also weakens the LoRAs — their text side reads the same prompt.";
const DENOISE_TIP =
  "How much of a source image this may rewrite. Only used when Fresco Apply has an image.";

const activeRefs = computed(() => props.content.refs.filter((r) => r.on).length);

const promptInput = ref<{ $el?: HTMLElement } | null>(null);
defineExpose({
  focusPrompt: () => promptInput.value?.$el?.querySelector?.("textarea")?.focus(),
});
</script>

<style scoped>
.fcf {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.fcf > * {
  flex: none;
}
.fcf-label {
  margin-top: 6px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--zen-muted, #9aa0aa);
}
.fcf-label:first-child {
  margin-top: 0;
}
.fcf-grid {
  display: grid;
  grid-template-columns: 62px minmax(0, 1fr);
  align-items: center;
  gap: 6px;
  color: var(--zen-muted, #9aa0aa);
}
.fcf-pair {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 72px;
  gap: 6px;
}
</style>
