<template>
  <div class="lm">
    <div v-for="mixin in mixins" :key="mixin.id" class="lm-card">
      <div class="lm-head">
        <ZenInput
          sm
          class="lm-name"
          :model-value="mixin.name"
          placeholder="Name"
          @update:model-value="patch(mixin.id, { name: String($event ?? '') })"
        />
        <span class="lm-count" :title="`${boxCount(mixin.id)} box(es) name ${mixin.name}`">
          {{ boxCount(mixin.id) }}
        </span>
        <ZenIconButton
          size="sm"
          icon="mdi mdi-close"
          :title="`Remove ${mixin.name} (and from every box)`"
          @click="remove(mixin.id)"
        />
      </div>
      <ZenInput
        sm
        :model-value="mixin.trigger"
        placeholder="Trigger words (added to each box's prompt)"
        @update:model-value="patch(mixin.id, { trigger: String($event ?? '') })"
      />
      <FrescoSection label="Look" :note="mixin.refs.length ? String(mixin.refs.length) : ''">
        <template #actions>
          <ZenIconButton
            size="sm"
            icon="mdi mdi-upload-outline"
            title="Upload images"
            :disabled="panels[mixin.id]?.busy"
            @click="panels[mixin.id]?.upload()"
          />
          <ZenIconButton
            size="sm"
            icon="mdi mdi-view-grid-plus-outline"
            title="Browse images"
            @click="panels[mixin.id]?.openBrowse()"
          />
        </template>
        <FrescoRefs
          :ref="(el) => setPanel(mixin.id, el)"
          :model-value="mixin.refs"
          @update:model-value="patch(mixin.id, { refs: $event })"
        />
      </FrescoSection>
      <FrescoSection label="LoRAs" :note="mixin.loras.length ? String(mixin.loras.length) : ''">
        <template #actions>
          <ZenIconButton
            size="sm"
            icon="mdi mdi-plus"
            title="Add a LoRA"
            @click="patch(mixin.id, { loras: [...mixin.loras, newLora()] })"
          />
        </template>
        <FrescoLoras
          v-if="mixin.loras.length"
          :model-value="mixin.loras"
          @update:model-value="patch(mixin.id, { loras: $event })"
        />
      </FrescoSection>
    </div>
    <p v-if="!mixins.length" class="lm-empty">
      Something several boxes share — a character, an object, a style: give it reference images,
      LoRAs and trigger words once, then pick it in each box. Needs Regions Apply after Layout
      Extras.
    </p>
  </div>
</template>

<script setup lang="ts">
/**
 * A layout's mixins: what several boxes share, with its reference images, LoRAs and trigger words
 * given once. Boxes name mixins by id; Layout Extras gives each box its mixins' LoRAs and trigger
 * words and reads their images into its prompt.
 */
import { ZenIconButton, ZenInput } from "@nynxz/zenkit-ui";
import { reactive } from "vue";

import FrescoLoras from "@/components/fresco/FrescoLoras.vue";
import FrescoRefs from "@/components/fresco/FrescoRefs.vue";
import FrescoSection from "@/components/fresco/FrescoSection.vue";
import type { FrescoLora } from "@/lib/frescoScene";
import { uid } from "@/lib/frescoScene";
import type { Mixin, PromptBox } from "@/layout/promptScene";

const { boxes } = defineProps<{ boxes: PromptBox[] }>();
const mixins = defineModel<Mixin[]>({ required: true });

type RefsPanel = { upload: () => void; openBrowse: () => void; busy: boolean };
const panels = reactive<Record<string, RefsPanel | undefined>>({});
function setPanel(id: string, el: unknown) {
  panels[id] = (el as RefsPanel | null) ?? undefined;
}

function patch(id: string, change: Partial<Mixin>) {
  mixins.value = mixins.value.map((m) => (m.id === id ? { ...m, ...change } : m));
}
function remove(id: string) {
  mixins.value = mixins.value.filter((m) => m.id !== id);
}
function boxCount(id: string): number {
  return boxes.filter((b) => b.mixins.includes(id)).length;
}
function newLora(): FrescoLora {
  return { id: uid("o"), name: "", strength: 1, on: true };
}
</script>

<style scoped>
.lm {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.lm-card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
  border: 1px solid color-mix(in srgb, var(--zen-border, #34343c) 70%, transparent);
  border-radius: var(--zen-radius, 7px);
}
.lm-head {
  display: flex;
  align-items: center;
  gap: 6px;
}
.lm-name {
  flex: 1;
  min-width: 0;
}
.lm-count {
  min-width: 18px;
  font-size: 11px;
  text-align: center;
  color: var(--zen-muted, #9aa0aa);
}
.lm-empty {
  margin: 0;
  font-size: 11px;
  line-height: 1.45;
  color: var(--zen-muted, #9aa0aa);
}
</style>
