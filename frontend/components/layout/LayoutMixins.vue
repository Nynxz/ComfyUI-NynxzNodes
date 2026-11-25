<template>
  <div class="lm">
    <div class="lm-bar">
      <ZenButton
        sm
        variant="ghost"
        icon="mdi mdi-bookshelf"
        title="Open Mixin Library — browse, inspect, create, and swap saved mixins"
        @click="emit('openLibrary')"
      >
        Library
      </ZenButton>
    </div>

    <div class="lm-grid">
      <button
        v-for="(mixin, idx) in mixins"
        :key="mixin.id"
        type="button"
        draggable="true"
        :class="[
          'lm-tile',
          {
            on: mixin.id === openId,
            dragging: dragIndex === idx,
            'drop-target': dropIndex === idx && dragIndex !== null && dragIndex !== idx,
          },
        ]"
        :title="tileTooltip(mixin)"
        @click="open(mixin.id)"
        @contextmenu.prevent.stop="openTileMenu($event, mixin)"
        @dragstart="onTileDragStart($event, idx)"
        @dragover="onTileDragOver($event, idx)"
        @dragleave="onTileDragLeave(idx)"
        @drop.prevent="onTileDrop(idx)"
        @dragend="onTileDragEnd"
      >
        <MixinAvatar :mixin="mixin" :size="40" />
        <span class="lm-tile-name">{{ mixin.name }}</span>
        <span class="lm-tile-meta">
          <span>{{ boxCount(mixin.id) }} box{{ boxCount(mixin.id) === 1 ? "" : "es" }}</span>
          <i
            v-if="mixin.libraryId"
            class="mdi mdi-link-variant lm-linked-ico"
            title="Linked to Mixin Library preset"
          />
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
          size="sm"
          icon="mdi mdi-swap-horizontal"
          :title="`Swap ${current.name} with another mixin from the library`"
          @click="emit('swapFromLibrary', current)"
        />
        <ZenIconButton
          size="sm"
          :icon="
            savedId === current.id
              ? 'mdi mdi-check'
              : current.libraryId
                ? 'mdi mdi-cloud-upload-outline'
                : 'mdi mdi-content-save-outline'
          "
          :disabled="!current.name.trim() || saving"
          :title="
            savedId === current.id
              ? 'Saved & synced with library'
              : current.libraryId
                ? `Update linked library preset (${current.name})`
                : `Save ${current.name} to library & link`
          "
          @click="saveToLibrary(current)"
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
        <MixinFusionBar
          :active-count="refCount(current)"
          :blend="current.blend"
          :softness="current.softness"
          :style-mode="current.styleMode"
          :style-strength="current.styleStrength"
          @update:blend="patch({ blend: $event })"
          @update:softness="patch({ softness: $event })"
          @update:style-mode="patch({ styleMode: $event })"
          @update:style-strength="patch({ styleStrength: $event })"
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

    <ZenContextMenu ref="tileMenu" :items="tileMenuItems" />
  </div>
</template>

<script setup lang="ts">
/**
 * The layout's Mixins tab: every mixin as a tile (right-click to swap, inspect, duplicate, save or
 * delete), a button for the full Mixin Library modal, and the open mixin's inline editor.
 */
import {
  ZenButton,
  ZenContextMenu,
  ZenIconButton,
  ZenInput,
  type ContextMenuItem,
} from "@nynxz/zenkit-ui";
import { computed, ref } from "vue";

import FrescoLoras from "@/components/fresco/FrescoLoras.vue";
import FrescoRefs from "@/components/fresco/FrescoRefs.vue";
import FrescoSection from "@/components/fresco/FrescoSection.vue";
import MixinAvatar from "@/components/layout/MixinAvatar.vue";
import MixinFusionBar from "@/components/layout/MixinFusionBar.vue";
import { saveMixinPreset } from "@/layout/library";
import type { Mixin, PromptBox } from "@/layout/promptScene";
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
  openLibrary: [initial?: Mixin];
  swapFromLibrary: [target: Mixin];
  swapBoxes: [idA: string, idB: string];
}>();

const refsPanel = ref<{ upload: () => void; openBrowse: () => void; busy: boolean } | null>(null);
const current = computed(
  () => mixins.value.find((m) => m.id === openId.value) ?? mixins.value[0] ?? null,
);

const saving = ref(false);
const savedId = ref<string | null>(null);

async function saveToLibrary(mixin: Mixin, asNew = false) {
  if (!mixin.name.trim()) return;
  saving.value = true;
  try {
    const clean = JSON.parse(JSON.stringify(mixin)) as Mixin;
    delete clean.libraryId;
    const saved = await saveMixinPreset({
      id: asNew ? undefined : mixin.libraryId,
      name: mixin.name.trim(),
      mixin: clean,
    });
    if (mixin.libraryId !== saved.id) {
      mixins.value = mixins.value.map((m) =>
        m.id === mixin.id ? { ...m, libraryId: saved.id } : m,
      );
    }
    savedId.value = mixin.id;
    setTimeout(() => {
      if (savedId.value === mixin.id) savedId.value = null;
    }, 1500);
  } finally {
    saving.value = false;
  }
}

function unlinkMixin(id: string) {
  mixins.value = mixins.value.map((m) => {
    if (m.id !== id) return m;
    const next = { ...m };
    delete next.libraryId;
    return next;
  });
}

function duplicateMixin(source: Mixin) {
  const copy: Mixin = {
    ...JSON.parse(JSON.stringify(source)),
    id: uid("c"),
    name: `${source.name || "Mixin"} copy`,
    refs: source.refs.map((r) => ({ ...r, id: uid("r") })),
    loras: source.loras.map((l) => ({ ...l, id: uid("o") })),
  };
  delete copy.libraryId;
  mixins.value = [...mixins.value, copy];
  openId.value = copy.id;
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
function tileTooltip(mixin: Mixin): string {
  const parts = [`${mixin.name || "Mixin"} (${boxCount(mixin.id)} box(es))`];
  if (mixin.desc.trim()) parts.push(mixin.desc.trim().slice(0, 110));
  if (mixin.trigger.trim()) parts.push(`Trigger: ${mixin.trigger.trim()}`);
  const refs = refCount(mixin);
  const loras = mixin.loras.filter((l) => l.on && l.name);
  if (refs) parts.push(`${refs} reference image(s)`);
  if (loras.length) {
    parts.push(
      `LoRAs: ${loras
        .map(
          (l) =>
            `${l.name
              .split(/[\\/]/)
              .pop()
              ?.replace(/\.[^.]+$/, "")} (${l.strength.toFixed(2)})`,
        )
        .join(", ")}`,
    );
  }
  parts.push("Drag to reorder · right-click to swap, inspect or delete");
  return parts.join("\n");
}
function newLora(): FrescoLora {
  return { id: uid("o"), name: "", strength: 1, on: true };
}

// --- drag-to-reorder mixin tiles ----------------------------------------------------------------

const dragIndex = ref<number | null>(null);
const dropIndex = ref<number | null>(null);

function onTileDragStart(e: DragEvent, index: number) {
  dragIndex.value = index;
  dropIndex.value = index;
  if (e.dataTransfer) {
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("application/x-nynxz-mixin", mixins.value[index]?.id ?? "");
  }
}
function onTileDragOver(e: DragEvent, index: number) {
  if (dragIndex.value === null) return;
  e.preventDefault();
  if (e.dataTransfer) e.dataTransfer.dropEffect = "move";
  dropIndex.value = index;
}
function onTileDragLeave(index: number) {
  if (dropIndex.value === index) dropIndex.value = null;
}
function onTileDrop(targetIndex: number) {
  const from = dragIndex.value;
  dragIndex.value = null;
  dropIndex.value = null;
  if (from === null || from === targetIndex) return;
  const next = [...mixins.value];
  const [moved] = next.splice(from, 1);
  if (!moved) return;
  next.splice(targetIndex, 0, moved);
  mixins.value = next;
}
function onTileDragEnd() {
  dragIndex.value = null;
  dropIndex.value = null;
}

// --- right-click menu on mixin tiles ------------------------------------------------------------

const tileMenu = ref<{ show: (e: MouseEvent) => void } | null>(null);
const menuMixin = ref<Mixin | null>(null);
function openTileMenu(e: MouseEvent, mixin: Mixin) {
  menuMixin.value = mixin;
  openId.value = mixin.id;
  tileMenu.value?.show(e);
}

const tileMenuItems = computed<ContextMenuItem[]>(() => {
  const m = menuMixin.value;
  if (!m) return [];
  const others = mixins.value.filter((x) => x.id !== m.id);
  const items: ContextMenuItem[] = [
    {
      label: "Inspect in Library dialog…",
      icon: "mdi mdi-eye-outline",
      run: () => emit("openLibrary", m),
    },
    {
      label: "Swap from Library…",
      icon: "mdi mdi-swap-horizontal",
      run: () => emit("swapFromLibrary", m),
    },
  ];
  if (others.length) {
    items.push("-");
    for (const other of others.slice(0, 6)) {
      items.push({
        label: `Swap places with "${other.name}"`,
        icon: "mdi mdi-swap-horizontal-variant",
        run: () => emit("swapBoxes", m.id, other.id),
      });
    }
  }
  items.push("-");
  if (props.selectedBoxes.length) {
    items.push({
      label: `Assign to ${props.selectedBoxes.length} selected box(es)`,
      icon: "mdi mdi-checkbox-marked-circle-plus-outline",
      run: () => emit("assignSelected", m.id),
    });
  }
  if (boxCount(m.id)) {
    items.push({
      label: `Select ${boxCount(m.id)} box(es) using this`,
      icon: "mdi mdi-select-group",
      run: () => emit("selectBoxes", m.id),
    });
  }
  items.push(
    {
      label: "Duplicate",
      icon: "mdi mdi-content-copy",
      run: () => duplicateMixin(m),
    },
    {
      label: m.libraryId ? "Update linked Library preset" : "Save to Library & link",
      icon: m.libraryId ? "mdi mdi-cloud-upload-outline" : "mdi mdi-content-save-outline",
      disabled: !m.name.trim(),
      run: () => void saveToLibrary(m, false),
    },
    ...(m.libraryId
      ? [
          {
            label: "Save as new Library preset",
            icon: "mdi mdi-content-save-plus-outline",
            disabled: !m.name.trim(),
            run: () => void saveToLibrary(m, true),
          },
          {
            label: "Unlink from Library",
            icon: "mdi mdi-link-variant-off",
            run: () => unlinkMixin(m.id),
          },
        ]
      : []),
    "-",
    {
      label: "Delete mixin",
      icon: "mdi mdi-delete-outline",
      danger: true,
      run: () => remove(m.id),
    },
  );
  return items;
});
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
  justify-content: flex-end;
  gap: 6px;
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
  cursor: grab;
  transition:
    border-color 0.12s ease,
    background 0.12s ease,
    opacity 0.12s ease;
}
.lm-tile:active {
  cursor: grabbing;
}
.lm-tile.dragging {
  opacity: 0.4;
}
.lm-tile.drop-target {
  border-color: var(--zen-accent, #6366f1);
  box-shadow: 0 0 0 1px var(--zen-accent, #6366f1);
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 14%, transparent);
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
.lm-linked-ico {
  color: var(--zen-accent, #6366f1);
}
.lm-new {
  justify-content: center;
  border-style: dashed;
  color: var(--zen-muted, #9aa0aa);
  cursor: pointer;
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
