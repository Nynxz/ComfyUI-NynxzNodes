<template>
  <div class="lm">
    <div class="lm-bar">
      <span class="lm-bar-note">Shared across boxes — refs, LoRAs &amp; triggers</span>
      <ZenPopover v-model:open="libOpen" placement="bottom-end" role="dialog">
        <template #trigger="{ toggle }">
          <ZenButton
            sm
            variant="ghost"
            icon="mdi mdi-bookshelf"
            title="Saved mixin presets — reuse characters, props or styles across layouts"
            @click="openLib(toggle)"
          >
            Library
          </ZenButton>
        </template>
        <div class="lm-lib">
          <div class="lm-lib-head">
            <span class="lm-lib-title">Mixin library</span>
            <ZenButton
              v-if="current"
              sm
              icon="mdi mdi-content-save-outline"
              :disabled="!current.name.trim() || saving"
              title="Save the open mixin to your library"
              @click="saveCurrent"
            >
              {{ savedNote || `Save "${current.name}"` }}
            </ZenButton>
          </div>
          <p v-if="!savedMixins.length" class="lm-empty">
            No saved mixins yet. Save a character, prop or style here to drop it into any layout.
          </p>
          <div v-else class="lm-lib-list">
            <div v-for="item in savedMixins" :key="item.name" class="lm-lib-row">
              <button
                type="button"
                class="lm-lib-pick"
                title="Add this mixin to the layout"
                @click="addFromLibrary(item)"
              >
                <MixinAvatar :mixin="readMixin(item.mixin, 0)" :size="24" />
                <span class="lm-lib-name">{{ item.name }}</span>
              </button>
              <ZenIconButton
                size="sm"
                danger
                icon="mdi mdi-delete-outline"
                title="Delete from library"
                @click="removeSaved(item.name)"
              />
            </div>
          </div>
        </div>
      </ZenPopover>
    </div>

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
            v-if="refCount(mixin)"
            class="mdi mdi-image-outline"
            :title="`${refCount(mixin)} reference(s)`"
          />
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
      A mixin is something several boxes share — a character, a prop, a style: reference images,
      LoRAs and trigger words given once, then ticked in any box.
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
          v-if="selectedBoxes.length"
          size="sm"
          icon="mdi mdi-checkbox-marked-circle-plus-outline"
          :title="`Assign ${current.name} to the ${selectedBoxes.length} selected box(es)`"
          @click="emit('assignSelected', current.id)"
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
      <FrescoSection label="Description" note="who or what it is — in every box that uses it">
        <ZenInput
          type="textarea"
          :rows="3"
          sm
          :model-value="current.desc"
          placeholder="e.g. Amber, a teenage girl with a short copper-red bob, green eyes and freckles"
          @update:model-value="patch({ desc: String($event ?? '') })"
        />
      </FrescoSection>
      <FrescoSection label="Trigger words" note="added to each box's prompt">
        <ZenInput
          sm
          :model-value="current.trigger"
          placeholder="e.g. amb3r"
          @update:model-value="patch({ trigger: String($event ?? '') })"
        />
      </FrescoSection>
      <FrescoSection
        label="References"
        :note="current.refs.length ? String(current.refs.length) : 'visual refs'"
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
 * The layout's Mixins tab: every mixin as a tile, a reusable preset library, and the open mixin's
 * editor — what several boxes share, with its reference images, LoRAs and trigger words given once.
 */
import { ZenButton, ZenIconButton, ZenInput, ZenPopover } from "@nynxz/zenkit-ui";
import { computed, ref } from "vue";

import FrescoLoras from "@/components/fresco/FrescoLoras.vue";
import FrescoRefs from "@/components/fresco/FrescoRefs.vue";
import FrescoSection from "@/components/fresco/FrescoSection.vue";
import MixinAvatar from "@/components/layout/MixinAvatar.vue";
import {
  deleteMixinPreset,
  listSavedMixins,
  saveMixinPreset,
  type SavedMixin,
} from "@/layout/library";
import { readMixin, type Mixin, type PromptBox } from "@/layout/promptScene";
import type { FrescoLora } from "@/lib/frescoScene";
import { uid } from "@/lib/sceneUtil";

const props = withDefaults(defineProps<{ boxes: PromptBox[]; selectedBoxes?: string[] }>(), {
  selectedBoxes: () => [],
});
const mixins = defineModel<Mixin[]>({ required: true });
/** The mixin whose editor is open. */
const openId = defineModel<string | null>("open", { default: null });
const emit = defineEmits<{
  create: [];
  selectBoxes: [id: string];
  assignSelected: [id: string];
}>();

const refsPanel = ref<{ upload: () => void; openBrowse: () => void; busy: boolean } | null>(null);
const current = computed(
  () => mixins.value.find((m) => m.id === openId.value) ?? mixins.value[0] ?? null,
);

const libOpen = ref(false);
const savedMixins = ref<SavedMixin[]>([]);
const saving = ref(false);
const savedNote = ref("");

async function refreshLibrary() {
  savedMixins.value = await listSavedMixins();
}
function openLib(toggle: () => void) {
  toggle();
  if (libOpen.value) void refreshLibrary();
}
async function saveCurrent() {
  const m = current.value;
  if (!m || !m.name.trim()) return;
  saving.value = true;
  try {
    await saveMixinPreset(m.name.trim(), JSON.parse(JSON.stringify(m)));
    savedNote.value = "Saved";
    setTimeout(() => (savedNote.value = ""), 1500);
    await refreshLibrary();
  } finally {
    saving.value = false;
  }
}
function addFromLibrary(item: SavedMixin) {
  const loaded = readMixin(item.mixin, mixins.value.length);
  const copy: Mixin = {
    ...loaded,
    id: uid("c"),
    refs: loaded.refs.map((r) => ({ ...r, id: uid("r") })),
    loras: loaded.loras.map((l) => ({ ...l, id: uid("o") })),
  };
  mixins.value = [...mixins.value, copy];
  openId.value = copy.id;
  libOpen.value = false;
}
async function removeSaved(name: string) {
  await deleteMixinPreset(name);
  await refreshLibrary();
}

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
  return props.boxes.filter((b) => b.mixins.includes(id)).length;
}
function refCount(mixin: Mixin): number {
  return mixin.refs.filter((r) => r.on).length;
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
.lm-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
}
.lm-bar-note {
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.lm-lib {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 280px;
  padding: 10px;
  color: var(--zen-text, #e5e5ea);
}
.lm-lib-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
}
.lm-lib-title {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--zen-muted, #9aa0aa);
}
.lm-lib-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 240px;
  overflow-y: auto;
}
.lm-lib-row {
  display: flex;
  align-items: center;
  gap: 4px;
}
.lm-lib-pick {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 6px;
  border: 1px solid transparent;
  border-radius: var(--zen-radius, 6px);
  background: transparent;
  color: var(--zen-text, #e5e5ea);
  font: inherit;
  font-size: 12px;
  text-align: left;
  cursor: pointer;
}
.lm-lib-pick:hover {
  border-color: var(--zen-border, #34343c);
  background: color-mix(in srgb, var(--zen-text, #fff) 5%, transparent);
}
.lm-lib-name {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
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
