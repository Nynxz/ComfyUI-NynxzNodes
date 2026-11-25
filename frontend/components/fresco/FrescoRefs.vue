<template>
  <div
    class="fr"
    :class="{ dropping: dragOver }"
    @dragover="onDragOver"
    @dragleave="dragOver = false"
    @drop="onDrop"
  >
    <div
      v-if="refs.length"
      :ref="(el) => (reorder.el.value = el as HTMLElement | null)"
      class="fr-rows"
      :class="{ settling }"
      v-bind="reorder.handlers"
    >
      <div
        v-for="(r, i) in refs"
        :key="r.id"
        class="fr-row"
        :class="{ off: !r.on, dragging: dragIndex === i }"
        :style="rowStyle(i)"
      >
        <div class="fr-gutter">
          <div v-if="refs.length > 1" class="fr-grip" data-zen-drag title="Drag to reorder">
            <i class="mdi mdi-drag-vertical" />
          </div>
          <ZenDot
            :model-value="r.on"
            :title="r.on ? 'Enabled — click to mute' : 'Muted — click to enable'"
            @update:model-value="patch(i, { on: $event })"
          />
        </div>
        <div class="fr-thumb" :title="r.ref">
          <img
            :src="thumbUrl(r.ref, r.type)"
            :style="{ objectFit: r.fit === 'stretch' ? 'fill' : r.fit }"
            data-peek
            loading="lazy"
            draggable="false"
          />
        </div>
        <div class="fr-info">
          <span class="fr-name" :title="r.ref">{{ shortName(r.ref) }}</span>
          <button type="button" class="fr-fit" :title="FIT_LABEL[r.fit]" @click="cycleFit(i)">
            <i class="mdi" :class="FIT_ICON[r.fit]" />
            <span>{{ FIT_SHORT[r.fit] }}</span>
          </button>
        </div>
        <ZenNumber
          class="fr-str"
          size="sm"
          :model-value="r.strength"
          :min="0"
          :max="4"
          :step="0.05"
          title="Strength — this image's share when there are several"
          @update:model-value="patch(i, { strength: Math.max(0, $event) })"
        />
        <ZenIconButton
          size="sm"
          icon="mdi mdi-auto-fix"
          title="Cut out subject with SAM3"
          @click="openCutout(i)"
        />
        <ZenIconButton size="sm" icon="mdi mdi-close" title="Remove" @click="remove(i)" />
      </div>
    </div>

    <button
      v-else
      type="button"
      class="fr-empty"
      :disabled="busy"
      title="Browse images — or drop them here"
      @click="browse = true"
    >
      <i class="mdi" :class="busy ? 'mdi-loading mdi-spin' : 'mdi-image-plus-outline'" />
      Drop images, or browse
    </button>

    <input
      ref="fileInput"
      type="file"
      accept="image/*"
      multiple
      style="display: none"
      @change="onUpload"
    />
    <ZenMediaPicker
      v-model:open="browse"
      :library="mediaLibrary"
      :kinds="['image']"
      title="Add references"
      :subfolder="'fresco'"
      multiple
      @pick="addPicked"
    />
    <RefCutoutDialog v-model:open="cutoutOpen" :source="cutoutSource" @apply="applyCutout" />
  </div>
</template>

<script setup lang="ts">
/**
 * Reference images as a LoRA-style vertical list with drag-to-reorder, or a slim drop strip
 * while there are none. Each row exposes drag grip, enable/mute, a thumbnail with hover-peek,
 * filename + clickable fit pill, strength, always-visible SAM3 cutout wand, and remove.
 */
import { useDragSurface } from "@nynxz/zenkit-nodekit";
import {
  ZenDot,
  ZenIconButton,
  ZenMediaPicker,
  ZenNumber,
  insertionIndex,
  landingTop,
  rowStride,
  shiftStyle,
  sortShift,
} from "@nynxz/zenkit-ui";
import { computed, ref, type CSSProperties } from "vue";

import RefCutoutDialog from "@/components/fresco/RefCutoutDialog.vue";
import { hasMediaDrop, mediaLibrary, readMediaDrop } from "@/framework";
import type { CutoutResult } from "@/layout/cutout";
import { thumbUrl, uploadImage, type SourceType } from "@/lib/mediaRefs";
import {
  REF_FITS,
  isImage,
  newRef,
  refFromMedia,
  type FrescoRef,
  type RefFit,
} from "@/lib/frescoScene";

const refs = defineModel<FrescoRef[]>({ required: true });

const FIT_ICON: Record<RefFit, string> = {
  contain: "mdi-fit-to-page-outline",
  cover: "mdi-crop",
  stretch: "mdi-arrow-expand-all",
};
const FIT_SHORT: Record<RefFit, string> = {
  contain: "Contain",
  cover: "Cover",
  stretch: "Stretch",
};
const FIT_LABEL: Record<RefFit, string> = {
  contain: "Contain — the whole image (click to cycle fit)",
  cover: "Cover — centre-cropped to fill (click to cycle fit)",
  stretch: "Stretch — distorted to fill (click to cycle fit)",
};

const fileInput = ref<HTMLInputElement | null>(null);
const busy = ref(false);
const browse = ref(false);
const dragOver = ref(false);
const cutoutOpen = ref(false);
const cutoutIndex = ref(-1);
const cutoutSource = computed(() =>
  cutoutIndex.value >= 0 ? (refs.value[cutoutIndex.value] ?? null) : null,
);

// --- pointer drag-to-reorder (matches LoraStack's zoom-aware live-shift gesture) ----------------

const dragIndex = ref(-1);
const dropIndex = ref(-1);
const dragOffset = ref(0);
const stride = ref(0);
const settling = ref(false);
const settle = ref<{ index: number; y: number } | null>(null);
let rowMids: number[] = [];
let rowTops: number[] = [];
let rowHeights: number[] = [];

function resetDrag() {
  dragIndex.value = -1;
  dropIndex.value = -1;
  dragOffset.value = 0;
}

const reorder = useDragSurface({
  onStart(ctx) {
    const grip = (ctx.event.target as HTMLElement | null)?.closest?.(".fr-grip");
    const host = reorder.el.value;
    if (!grip || !host) return false;
    const kids = Array.from(host.children) as HTMLElement[];
    const i = kids.indexOf(grip.closest(".fr-row") as HTMLElement);
    if (i < 0) return false;
    rowMids = kids.map((k) => k.offsetTop + k.offsetHeight / 2);
    rowTops = kids.map((k) => k.offsetTop);
    rowHeights = kids.map((k) => k.offsetHeight);
    stride.value = rowStride(rowTops, rowHeights, i);
    dragIndex.value = i;
    dropIndex.value = i;
    dragOffset.value = 0;
  },
  onMove(ctx) {
    if (dragIndex.value < 0) return;
    dragOffset.value = ctx.delta.y;
    const center = (rowMids[dragIndex.value] ?? 0) + ctx.delta.y;
    dropIndex.value = insertionIndex(rowMids, dragIndex.value, center);
  },
  onEnd() {
    const from = dragIndex.value;
    const to = dropIndex.value;
    const offset = dragOffset.value;
    resetDrag();
    if (from < 0 || to < 0) return;
    settling.value = true;
    settle.value = {
      index: to,
      y: (rowTops[from] ?? 0) + offset - landingTop(rowTops, rowHeights, from, to),
    };
    if (from !== to) {
      const next = [...refs.value];
      const [moved] = next.splice(from, 1);
      if (moved) {
        next.splice(to, 0, moved);
        refs.value = next;
      }
    }
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        settling.value = false;
        settle.value = null;
      }),
    );
  },
});

function rowStyle(i: number): CSSProperties {
  if (settle.value && settle.value.index === i) {
    return { transform: `translate3d(0, ${settle.value.y}px, 0)`, transition: "none" };
  }
  if (dragIndex.value === i) {
    return { transform: `translate3d(0, ${dragOffset.value}px, 0)` };
  }
  if (dragIndex.value >= 0) {
    return shiftStyle(sortShift(i, dragIndex.value, dropIndex.value), stride.value) ?? {};
  }
  return {};
}

function shortName(refPath: string): string {
  const leaf = refPath.split(/[\\/]/).pop() ?? refPath;
  return leaf.replace(/\.[^.]+$/, "") || leaf;
}

function openCutout(i: number) {
  cutoutIndex.value = i;
  cutoutOpen.value = true;
}
function applyCutout(cutout: CutoutResult, how: "replace" | "add") {
  if (how === "replace" && cutoutIndex.value >= 0 && cutoutIndex.value < refs.value.length) {
    patch(cutoutIndex.value, { ref: cutout.ref, type: cutout.type });
  } else {
    add([{ ref: cutout.ref, type: cutout.type }]);
  }
}

function patch(i: number, fields: Partial<FrescoRef>) {
  refs.value = refs.value.map((r, j) => (j === i ? { ...r, ...fields } : r));
}
function remove(i: number) {
  refs.value = refs.value.filter((_, j) => j !== i);
}
function cycleFit(i: number) {
  const r = refs.value[i];
  if (r) patch(i, { fit: REF_FITS[(REF_FITS.indexOf(r.fit) + 1) % REF_FITS.length] });
}
function add(entries: { ref: string; type: SourceType }[]) {
  if (!entries.length) return;
  refs.value = [...refs.value, ...entries.map(newRef)];
}

async function addFiles(files: File[]) {
  const images = files.filter((f) => f.type.startsWith("image/"));
  if (!images.length) return;
  busy.value = true;
  try {
    const done: { ref: string; type: SourceType }[] = [];
    for (const file of images) {
      const up = await uploadImage(file);
      if (up) done.push(up);
    }
    add(done);
  } finally {
    busy.value = false;
  }
}
async function onUpload(e: Event) {
  const input = e.target as HTMLInputElement;
  await addFiles([...(input.files ?? [])]);
  input.value = "";
}
function onDragOver(e: DragEvent) {
  if (!hasMediaDrop(e)) return;
  e.preventDefault();
  e.stopPropagation();
  dragOver.value = true;
}
/** Anything nodekit knows how to read: the asset browser, the Media Viewer, ZenKit media, a URL,
 *  or files from the desktop (uploaded into input/fresco). */
async function onDrop(e: DragEvent) {
  if (!hasMediaDrop(e)) return;
  e.preventDefault();
  e.stopPropagation();
  dragOver.value = false;
  busy.value = true;
  try {
    const media = await readMediaDrop(e, "fresco");
    add(media.filter(isImage).map(refFromMedia));
  } finally {
    busy.value = false;
  }
}
defineExpose({
  upload: () => fileInput.value?.click(),
  openBrowse: () => (browse.value = true),
  busy,
});
function addPicked(picked: string[]) {
  add(
    picked.map((ref) => {
      const [type, ...rest] = ref.split("/");
      return {
        ref: rest.join("/"),
        type: (["input", "temp", "output"].includes(type!) ? type : "input") as SourceType,
      };
    }),
  );
}
</script>

<style scoped>
.fr {
  display: flex;
  flex-direction: column;
  gap: 5px;
  align-items: stretch;
  padding: 2px;
  border-radius: var(--zen-radius, 7px);
  outline: 1.5px dashed transparent;
  outline-offset: -1.5px;
  transition: outline-color 0.12s ease;
}
.fr.dropping {
  outline-color: var(--zen-accent, #6366f1);
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 8%, transparent);
}
.fr-rows {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.fr-rows.settling .fr-row {
  transition: none;
}
.fr-row {
  position: relative;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 5px;
  border: 1px solid color-mix(in srgb, var(--zen-border, #34343c) 55%, transparent);
  border-radius: var(--zen-radius, 7px);
  background: color-mix(in srgb, var(--zen-text, #fff) 3%, transparent);
  transition: transform 0.14s ease;
}
.fr-row.dragging {
  z-index: 2;
  border-color: var(--zen-accent, #6366f1);
  background: var(--zen-surface-2, #26262d);
  box-shadow: 0 6px 16px rgb(0 0 0 / 35%);
  transition: none;
}
.fr-row.off .fr-thumb,
.fr-row.off .fr-info {
  opacity: 0.42;
}
.fr-gutter {
  flex: none;
  display: inline-flex;
  align-items: center;
  gap: 2px;
}
.fr-grip {
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 12px;
  cursor: grab;
  color: var(--zen-muted, #9aa0aa);
  font-size: 14px;
  line-height: 1;
  touch-action: none;
  -webkit-user-select: none;
  user-select: none;
}
.fr-grip > * {
  pointer-events: none;
}
.fr-grip:hover {
  color: var(--zen-text, #e5e5ea);
}
.fr-grip:active {
  cursor: grabbing;
}
.fr-thumb {
  flex: none;
  width: 42px;
  height: 42px;
  border-radius: min(var(--zen-radius, 7px), 5px);
  overflow: hidden;
  background-color: #18181d;
  background-image:
    linear-gradient(45deg, #24242c 25%, transparent 25%),
    linear-gradient(-45deg, #24242c 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #24242c 75%),
    linear-gradient(-45deg, transparent 75%, #24242c 75%);
  background-size: 10px 10px;
  background-position:
    0 0,
    0 5px,
    5px -5px,
    -5px 0;
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--zen-border, #34343c) 75%, transparent);
}
.fr-thumb img {
  width: 100%;
  height: 100%;
  display: block;
}
.fr-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 3px;
}
.fr-name {
  max-width: 100%;
  overflow: hidden;
  font-size: 11px;
  line-height: 1.2;
  color: var(--zen-text, #e5e5ea);
  white-space: nowrap;
  text-overflow: ellipsis;
}
.fr-fit {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  padding: 1px 6px;
  border: 1px solid color-mix(in srgb, var(--zen-border, #34343c) 70%, transparent);
  border-radius: var(--zen-radius, 7px);
  background: transparent;
  color: var(--zen-muted, #9aa0aa);
  font: inherit;
  font-size: 10px;
  line-height: 1.3;
  cursor: pointer;
  transition:
    border-color 0.12s ease,
    color 0.12s ease;
}
.fr-fit:hover {
  border-color: var(--zen-accent, #6366f1);
  color: var(--zen-text, #e5e5ea);
}
.fr-fit .mdi {
  font-size: 11px;
}
.fr-str {
  width: 58px;
  flex: none;
}
.fr-empty {
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  height: 32px;
  padding: 0 8px;
  border: 1.5px dashed var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 7px);
  background: transparent;
  color: var(--zen-muted, #9aa0aa);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
  transition:
    border-color 0.12s ease,
    background 0.12s ease,
    color 0.12s ease;
}
.fr-empty:hover {
  border-color: var(--zen-accent, #6366f1);
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 10%, transparent);
  color: var(--zen-text, #e5e5ea);
}
.fr-empty .mdi {
  font-size: 15px;
}
</style>
