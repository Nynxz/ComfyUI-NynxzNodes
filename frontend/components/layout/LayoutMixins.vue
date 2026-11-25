<template>
  <div class="lm">
    <div class="lm-grid">
      <button
        v-for="(mixin, idx) in mixins"
        :key="mixin.id"
        type="button"
        draggable="true"
        data-zen-drag
        :class="[
          'lm-tile',
          {
            on: mixin.id === openId,
            dragging: dragIndex === idx,
            'drop-target': dropIndex === idx && dragIndex !== null && dragIndex !== idx,
          },
        ]"
        :style="{ '--lm-c': mixin.color }"
        :title="tileTooltip(mixin)"
        @click="open(mixin.id)"
        @contextmenu.prevent.stop="openTileMenu($event, mixin)"
        @dragstart="onTileDragStart($event, idx)"
        @dragover="onTileDragOver($event, idx)"
        @dragleave="onTileDragLeave(idx)"
        @drop.prevent="onTileDrop(idx)"
        @dragend="onTileDragEnd"
      >
        <span class="lm-hero">
          <img
            v-if="firstRef(mixin)"
            class="lm-hero-img"
            :src="thumbUrl(firstRef(mixin)!.ref, firstRef(mixin)!.type)"
            alt=""
            data-peek
            draggable="false"
          />
          <span v-else class="lm-hero-mono">{{ initial(mixin) }}</span>
          <span class="lm-dot" />
          <span v-if="mixin.libraryId || refCount(mixin) > 1" class="lm-hero-badges">
            <i
              v-if="mixin.libraryId"
              class="mdi mdi-link-variant lm-linked-ico"
              title="Linked to Mixin Library preset"
            />
            <span v-if="refCount(mixin) > 1" class="lm-more">+{{ refCount(mixin) - 1 }}</span>
          </span>
        </span>
        <span class="lm-foot">
          <span class="lm-tile-name">{{ mixin.name || "Untitled" }}</span>
          <span class="lm-tile-meta">
            <span class="lm-stat" :class="{ active: boxCount(mixin.id) > 0 }">
              <i class="mdi mdi-vector-square" />{{ boxCount(mixin.id) }}
            </span>
            <span v-if="refCount(mixin)" class="lm-stat">
              <i class="mdi mdi-image-outline" />{{ refCount(mixin) }}
            </span>
            <span v-if="loraCount(mixin)" class="lm-stat">
              <i class="mdi mdi-puzzle-outline" />{{ loraCount(mixin) }}
            </span>
          </span>
        </span>
      </button>

      <button
        type="button"
        class="lm-tile lm-action"
        title="Create a new mixin"
        @click="emit('create')"
      >
        <i class="mdi mdi-plus" />
        <span class="lm-action-label">New</span>
      </button>
      <button
        type="button"
        class="lm-tile lm-action"
        title="Open Mixin Library — browse, inspect, create, and swap saved mixins"
        @click="emit('openLibrary')"
      >
        <i class="mdi mdi-bookshelf" />
        <span class="lm-action-label">Library</span>
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
 * The layout's Mixins tab: visual mixin cards (drag to reorder, right-click to swap, inspect,
 * duplicate, save or delete), matching New & Library utility cards, and the open mixin's editor.
 */
import { ZenContextMenu, ZenIconButton, ZenInput, type ContextMenuItem } from "@nynxz/zenkit-ui";
import { computed, ref } from "vue";

import FrescoLoras from "@/components/fresco/FrescoLoras.vue";
import FrescoRefs from "@/components/fresco/FrescoRefs.vue";
import FrescoSection from "@/components/fresco/FrescoSection.vue";
import MixinAvatar from "@/components/layout/MixinAvatar.vue";
import MixinFusionBar from "@/components/layout/MixinFusionBar.vue";
import { saveMixinPreset } from "@/layout/library";
import type { Mixin, PromptBox } from "@/layout/promptScene";
import type { FrescoLora, FrescoRef } from "@/lib/frescoScene";
import { thumbUrl } from "@/lib/mediaRefs";
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
function firstRef(mixin: Mixin): FrescoRef | null {
  return mixin.refs.find((r) => r.on) ?? null;
}
function initial(mixin: Mixin): string {
  return (mixin.name.trim()[0] ?? "?").toUpperCase();
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
.lm-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(84px, 1fr));
  grid-auto-rows: 1fr;
  gap: 7px;
}
.lm-tile {
  position: relative;
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 106px;
  padding: 0;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--zen-border, #34343c) 75%, transparent);
  border-radius: 8px;
  background: var(--zen-surface-2, #222229);
  color: var(--zen-text, #e5e5ea);
  font: inherit;
  text-align: left;
  cursor: grab;
  transition:
    border-color 0.14s ease,
    box-shadow 0.14s ease,
    background 0.14s ease,
    opacity 0.12s ease;
}
.lm-tile:hover {
  border-color: color-mix(in srgb, var(--zen-text, #fff) 28%, var(--zen-border, #34343c));
  background: color-mix(in srgb, var(--zen-text, #fff) 4%, var(--zen-surface-2, #222229));
}
.lm-tile:active {
  cursor: grabbing;
}
.lm-tile.on {
  border-color: var(--zen-accent, #6366f1);
  box-shadow: 0 0 0 1px var(--zen-accent, #6366f1);
}
.lm-tile.dragging {
  opacity: 0.4;
}
.lm-tile.drop-target {
  border-color: var(--zen-accent, #6366f1);
  box-shadow: 0 0 0 1.5px var(--zen-accent, #6366f1);
}
.lm-hero {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 64px;
  flex: none;
  overflow: hidden;
  background: linear-gradient(
    155deg,
    color-mix(in srgb, var(--lm-c, #6366f1) 30%, #15151b),
    color-mix(in srgb, var(--lm-c, #6366f1) 10%, #15151b)
  );
}
.lm-hero-img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.lm-hero-mono {
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: 1.5px solid color-mix(in srgb, var(--lm-c, #6366f1) 80%, #fff);
  background: color-mix(in srgb, var(--lm-c, #6366f1) 28%, rgb(0 0 0 / 35%));
  color: #fff;
  font-size: 14px;
  font-weight: 700;
  line-height: 1;
}
.lm-dot {
  position: absolute;
  top: 5px;
  left: 5px;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--lm-c, #6366f1);
  box-shadow: 0 0 0 1.5px rgb(0 0 0 / 55%);
}
.lm-hero-badges {
  position: absolute;
  top: 4px;
  right: 4px;
  display: inline-flex;
  align-items: center;
  gap: 3px;
  padding: 1px 4px;
  border-radius: 999px;
  background: rgb(0 0 0 / 62%);
  color: #fff;
  font-size: 9.5px;
  line-height: 1.3;
}
.lm-linked-ico {
  color: #a5b4fc;
  font-size: 10px;
}
.lm-more {
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.lm-foot {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex: 1;
  padding: 5px 6px 6px;
  min-width: 0;
}
.lm-tile-name {
  max-width: 100%;
  overflow: hidden;
  font-size: 11px;
  font-weight: 500;
  line-height: 1.25;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.lm-tile-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 10px;
  line-height: 1.2;
  color: var(--zen-muted, #9aa0aa);
}
.lm-stat {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  font-variant-numeric: tabular-nums;
}
.lm-stat.active {
  color: var(--zen-text, #e5e5ea);
}
.lm-stat .mdi {
  font-size: 10.5px;
}
.lm-action {
  align-items: center;
  justify-content: center;
  gap: 5px;
  padding: 8px 4px;
  border-style: dashed;
  background: transparent;
  color: var(--zen-muted, #9aa0aa);
  text-align: center;
  cursor: pointer;
}
.lm-action:hover {
  border-color: var(--zen-accent, #6366f1);
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 10%, transparent);
  color: var(--zen-text, #e5e5ea);
}
.lm-action .mdi {
  font-size: 20px;
  line-height: 1;
}
.lm-action-label {
  font-size: 11px;
  font-weight: 500;
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
