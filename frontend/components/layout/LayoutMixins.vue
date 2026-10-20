<template>
  <div class="lm">
    <div class="lm-grid">
      <button
        v-for="mixin in mixins"
        :key="mixin.id"
        type="button"
        :class="['lm-tile', { on: mixin.id === openId }]"
        :title="`${mixin.name} — in ${boxCount(mixin.id)} box(es)`"
        @click="open(mixin.id)"
      >
        <MixinAvatar :mixin="mixin" :size="40" />
        <span class="lm-tile-name">{{ mixin.name }}</span>
        <span class="lm-tile-meta">
          <span>{{ boxCount(mixin.id) }} box{{ boxCount(mixin.id) === 1 ? "" : "es" }}</span>
          <i
            v-if="loraCount(mixin)"
            class="mdi mdi-puzzle-outline"
            :title="`${loraCount(mixin)} LoRA(s)`"
          />
        </span>
      </button>
      <button type="button" class="lm-tile lm-new" title="New mixin" @click="emit('create')">
        <i class="mdi mdi-plus" />
        <span class="lm-tile-name">New</span>
      </button>
    </div>

    <p v-if="!mixins.length" class="lm-empty">
      A mixin is something several boxes share — a character, an object, a style: reference images,
      LoRAs and trigger words given once, then picked in each box (Box tab). Needs Regions Apply
      after Layout Extras.
    </p>

    <div v-if="current" class="lm-edit">
      <div class="lm-head">
        <MixinAvatar :mixin="current" :size="28" />
        <ZenInput
          sm
          class="lm-name"
          :model-value="current.name"
          placeholder="Name"
          @update:model-value="patch({ name: String($event ?? '') })"
        />
        <input
          class="lm-color"
          type="color"
          :value="current.color"
          title="Its colour"
          @input="patch({ color: ($event.target as HTMLInputElement).value })"
        />
        <ZenIconButton
          size="sm"
          icon="mdi mdi-select-group"
          :title="`Select the ${boxCount(current.id)} box(es) it's in`"
          :disabled="!boxCount(current.id)"
          @click="emit('selectBoxes', current.id)"
        />
        <ZenIconButton
          size="sm"
          danger
          icon="mdi mdi-delete-outline"
          :title="`Delete ${current.name} (and take it out of every box)`"
          @click="remove(current.id)"
        />
      </div>
      <FrescoSection label="Trigger words" note="added to each box's prompt">
        <ZenInput
          sm
          :model-value="current.trigger"
          placeholder="e.g. amb3r"
          @update:model-value="patch({ trigger: String($event ?? '') })"
        />
      </FrescoSection>
      <FrescoSection
        label="Look"
        :note="current.refs.length ? String(current.refs.length) : 'reference images'"
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
          :model-value="current.refs"
          @update:model-value="patch({ refs: $event })"
        />
      </FrescoSection>
      <FrescoSection label="LoRAs" :note="current.loras.length ? String(current.loras.length) : ''">
        <template #actions>
          <ZenIconButton
            size="sm"
            icon="mdi mdi-plus"
            title="Add a LoRA"
            @click="patch({ loras: [...current.loras, newLora()] })"
          />
        </template>
        <FrescoLoras
          v-if="current.loras.length"
          :model-value="current.loras"
          @update:model-value="patch({ loras: $event })"
        />
      </FrescoSection>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * The layout's Mixins tab: every mixin as a tile, and the open one's editor — what several boxes
 * share, with its reference images, LoRAs and trigger words given once. Boxes name mixins by id;
 * Layout Extras gives each box its mixins' LoRAs and trigger words and reads their images into its
 * prompt.
 */
import { ZenIconButton, ZenInput } from "@nynxz/zenkit-ui";
import { computed, ref } from "vue";

import FrescoLoras from "@/components/fresco/FrescoLoras.vue";
import FrescoRefs from "@/components/fresco/FrescoRefs.vue";
import FrescoSection from "@/components/fresco/FrescoSection.vue";
import MixinAvatar from "@/components/layout/MixinAvatar.vue";
import type { FrescoLora } from "@/lib/frescoScene";
import { uid } from "@/lib/frescoScene";
import type { Mixin, PromptBox } from "@/layout/promptScene";

const { boxes } = defineProps<{ boxes: PromptBox[] }>();
const mixins = defineModel<Mixin[]>({ required: true });
/** The mixin whose editor is open. */
const openId = defineModel<string | null>("open", { default: null });
const emit = defineEmits<{ create: []; selectBoxes: [id: string] }>();

const refsPanel = ref<{ upload: () => void; openBrowse: () => void; busy: boolean } | null>(null);
const current = computed(
  () => mixins.value.find((m) => m.id === openId.value) ?? mixins.value[0] ?? null,
);

function open(id: string) {
  openId.value = id;
}
function patch(change: Partial<Mixin>) {
  const id = current.value?.id;
  if (!id) return;
  mixins.value = mixins.value.map((m) => (m.id === id ? { ...m, ...change } : m));
}
function remove(id: string) {
  mixins.value = mixins.value.filter((m) => m.id !== id);
  if (openId.value === id) openId.value = null;
}
function boxCount(id: string): number {
  return boxes.filter((b) => b.mixins.includes(id)).length;
}
function loraCount(mixin: Mixin): number {
  return mixin.loras.filter((l) => l.on && l.name).length;
}
function newLora(): FrescoLora {
  return { id: uid("o"), name: "", strength: 1, on: true };
}
</script>

<style scoped>
.lm {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.lm-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(76px, 1fr));
  gap: 6px;
}
.lm-tile {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  min-width: 0;
  padding: 8px 4px 6px;
  border: 1px solid color-mix(in srgb, var(--zen-border, #34343c) 70%, transparent);
  border-radius: var(--zen-radius, 7px);
  background: transparent;
  color: var(--zen-text, #e5e5ea);
  cursor: pointer;
}
.lm-tile:hover {
  background: color-mix(in srgb, var(--zen-text, #fff) 4%, transparent);
}
.lm-tile.on {
  border-color: var(--zen-accent, #3b82f6);
  background: color-mix(in srgb, var(--zen-accent, #3b82f6) 12%, transparent);
}
.lm-tile-name {
  max-width: 100%;
  overflow: hidden;
  font-size: 11px;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.lm-tile-meta {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  font-size: 10px;
  color: var(--zen-muted, #9aa0aa);
}
.lm-new {
  justify-content: center;
  border-style: dashed;
  color: var(--zen-muted, #9aa0aa);
}
.lm-new .mdi {
  font-size: 22px;
}
.lm-empty {
  margin: 0;
  font-size: 11px;
  line-height: 1.45;
  color: var(--zen-muted, #9aa0aa);
}
.lm-edit {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-top: 10px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border, #34343c) 60%, transparent);
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
.lm-color {
  flex: none;
  width: 22px;
  height: 22px;
  padding: 0;
  border: 0;
  border-radius: 4px;
  background: transparent;
  cursor: pointer;
}
</style>
