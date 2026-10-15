<template>
  <div class="fcf">
    <FrescoSection label="Prompt">
      <ZenInput
        ref="promptInput"
        type="textarea"
        :rows="promptRows"
        sm
        :model-value="content.prompt"
        :placeholder="promptPlaceholder"
        @update:model-value="emit('patch', { prompt: String($event ?? '') })"
      />
    </FrescoSection>

    <FrescoSection
      label="References"
      :note="content.refs.length ? String(content.refs.length) : ''"
    >
      <template #actions>
        <ZenIconButton
          size="sm"
          icon="mdi mdi-upload-outline"
          title="Upload images"
          :disabled="refsPanel?.busy"
          @click="refsPanel?.upload()"
        />
        <ZenIconButton
          size="sm"
          icon="mdi mdi-view-grid-plus-outline"
          title="Browse images"
          @click="refsPanel?.openBrowse()"
        />
      </template>
      <FrescoRefs
        ref="refsPanel"
        :model-value="content.refs"
        @update:model-value="emit('patch', { refs: $event })"
      />
      <div v-if="activeRefs > 1" class="fcf-row">
        <span :title="BLEND_TIP">blend</span>
        <ZenSelect
          size="sm"
          :model-value="content.blend"
          :options="BLENDS"
          :title="BLEND_TIP"
          @update:model-value="emit('patch', { blend: $event as Blend })"
        />
        <ZenNumber
          class="fcf-soft"
          size="sm"
          :model-value="content.softness"
          :min="0"
          :max="1"
          :step="0.05"
          :precision="2"
          title="Softness between references — 0 hard cells, 1 fully feathered"
          @update:model-value="emit('patch', { softness: $event })"
        />
      </div>
    </FrescoSection>

    <FrescoSection label="LoRAs" :note="content.loras.length ? String(content.loras.length) : ''">
      <template #actions>
        <ZenIconButton size="sm" icon="mdi mdi-plus" title="Add a LoRA" @click="addLora" />
      </template>
      <FrescoLoras
        v-if="content.loras.length"
        :model-value="content.loras"
        @update:model-value="emit('patch', { loras: $event })"
      />
    </FrescoSection>

    <slot />

    <FrescoSection label="Strength">
      <div class="fcf-pairs">
        <label class="fcf-pair">
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
        </label>
        <label class="fcf-pair">
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
        </label>
      </div>
    </FrescoSection>
  </div>
</template>

<script setup lang="ts">
/**
 * What a layer — or the background — carries: its prompt, references and how they blend, how
 * strongly the area follows them, its LoRAs, and how much of a source image it may rewrite. The
 * default slot sits between the LoRAs and the strengths, for the layer's capture and shape.
 *
 * Every number is a ZenNumber: drag to scrub, the steppers, or type an exact value.
 */
import { computed, ref } from "vue";
import { ZenIconButton, ZenInput, ZenNumber, ZenSelect } from "@nynxz/zenkit-ui";

import FrescoLoras from "@/components/fresco/FrescoLoras.vue";
import FrescoRefs from "@/components/fresco/FrescoRefs.vue";
import FrescoSection from "@/components/fresco/FrescoSection.vue";
import { BLENDS, uid, type Blend, type FrescoContent } from "@/lib/frescoScene";

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
const refsPanel = ref<InstanceType<typeof FrescoRefs> | null>(null);
function addLora() {
  emit("patch", {
    loras: [...props.content.loras, { id: uid("o"), name: "", strength: 1, on: true }],
  });
}
defineExpose({
  focusPrompt: () => promptInput.value?.$el?.querySelector?.("textarea")?.focus(),
});
</script>

<style scoped>
.fcf {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.fcf > * {
  flex: none;
}
.fcf-row {
  display: grid;
  grid-template-columns: 52px minmax(0, 1fr) 92px;
  align-items: center;
  gap: 6px;
  color: var(--zen-muted, #9aa0aa);
}
.fcf-pairs {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 6px 10px;
}
.fcf-pair {
  display: grid;
  grid-template-columns: 52px minmax(0, 1fr);
  align-items: center;
  gap: 6px;
  color: var(--zen-muted, #9aa0aa);
}
</style>
