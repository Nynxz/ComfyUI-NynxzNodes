<template>
  <div
    class="fr"
    :class="{ dropping: dragOver }"
    @dragover.prevent="dragOver = true"
    @dragleave="dragOver = false"
    @drop.prevent="onDrop"
  >
    <div v-for="(r, i) in refs" :key="r.id" class="fr-tile" :class="{ off: !r.on }">
      <div class="fr-thumb" :title="r.ref">
        <img
          :src="thumbUrl(r.ref, r.type)"
          :style="{ objectFit: r.fit === 'stretch' ? 'fill' : r.fit }"
          data-peek
          loading="lazy"
          draggable="false"
        />
        <div class="fr-acts">
          <button type="button" class="fr-act" :title="FIT_LABEL[r.fit]" @click="cycleFit(i)">
            <i class="mdi" :class="FIT_ICON[r.fit]" />
          </button>
          <button
            type="button"
            class="fr-act"
            :title="r.on ? 'Mute' : 'Unmute'"
            @click="patch(i, { on: !r.on })"
          >
            <i class="mdi" :class="r.on ? 'mdi-eye-outline' : 'mdi-eye-off-outline'" />
          </button>
          <button type="button" class="fr-act danger" title="Remove" @click="remove(i)">
            <i class="mdi mdi-close" />
          </button>
        </div>
      </div>
      <ZenNumber
        class="fr-str"
        bare
        :model-value="r.strength"
        :min="0"
        :max="4"
        :step="0.05"
        title="Strength — this image's share when there are several"
        @update:model-value="patch(i, { strength: Math.max(0, $event) })"
      />
    </div>

    <div class="fr-tile">
      <button
        type="button"
        class="fr-add"
        :disabled="busy"
        title="Add reference images — or drop them here"
        @click="fileInput?.click()"
      >
        <i class="mdi" :class="busy ? 'mdi-loading mdi-spin' : 'mdi-image-plus-outline'" />
      </button>
      <button type="button" class="fr-browse" title="Browse images" @click="browse = true">
        browse
      </button>
    </div>

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
  </div>
</template>

<script setup lang="ts">
/**
 * A layer's reference images as a row of tiles, the add tile always last — so adding the first
 * image doesn't change the block's height. Each has a strength, a fit and a mute; several on one
 * layer blend into its conditioning the way Text Encode (Fusion) blends them.
 */
import { ref } from "vue";
import { ZenMediaPicker, ZenNumber } from "@nynxz/zenkit-ui";

import { mediaLibrary } from "@/framework";
import { thumbUrl, uploadImage, type SourceType } from "@/lib/fusionApi";
import { REF_FITS, uid, type FrescoRef, type RefFit } from "@/lib/frescoScene";

const refs = defineModel<FrescoRef[]>({ required: true });

const FIT_ICON: Record<RefFit, string> = {
  contain: "mdi-fit-to-page-outline",
  cover: "mdi-crop",
  stretch: "mdi-arrow-expand-all",
};
const FIT_LABEL: Record<RefFit, string> = {
  contain: "Contain — the whole image (click to change)",
  cover: "Cover — centre-cropped to fill (click to change)",
  stretch: "Stretch — distorted to fill (click to change)",
};

const fileInput = ref<HTMLInputElement | null>(null);
const busy = ref(false);
const browse = ref(false);
const dragOver = ref(false);

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
  refs.value = [
    ...refs.value,
    ...entries.map((e) => ({
      id: uid("r"),
      ...e,
      strength: 1,
      fit: "contain" as RefFit,
      on: true,
    })),
  ];
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
async function onDrop(e: DragEvent) {
  dragOver.value = false;
  await addFiles([...(e.dataTransfer?.files ?? [])]);
}
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
  display: grid;
  grid-template-columns: repeat(auto-fill, 64px);
  gap: 8px;
  padding: 2px;
  border-radius: var(--zen-radius, 7px);
  outline: 1px dashed transparent;
  transition: outline-color 0.12s ease;
}
.fr.dropping {
  outline-color: var(--zen-accent, #6366f1);
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 8%, transparent);
}
.fr-tile {
  display: flex;
  flex-direction: column;
  gap: 3px;
  width: 64px;
}
.fr-tile.off .fr-thumb {
  opacity: 0.4;
}
.fr-thumb {
  position: relative;
  width: 64px;
  height: 64px;
  border-radius: 6px;
  overflow: hidden;
  background: #000;
  box-shadow: inset 0 0 0 1px var(--zen-border, #34343c);
}
.fr-thumb img {
  width: 100%;
  height: 100%;
  display: block;
}
.fr-acts {
  position: absolute;
  inset: auto 2px 2px 2px;
  display: flex;
  justify-content: flex-end;
  gap: 2px;
  opacity: 0;
  transition: opacity 0.12s ease;
}
.fr-thumb:hover .fr-acts {
  opacity: 1;
}
.fr-act {
  display: grid;
  place-items: center;
  width: 18px;
  height: 18px;
  padding: 0;
  border: 0;
  border-radius: 4px;
  background: rgb(0 0 0 / 70%);
  color: #fff;
  cursor: pointer;
  font-size: 11px;
}
.fr-act.danger:hover {
  background: var(--zen-danger, #dc2626);
}
.fr-str {
  width: 64px;
}
.fr-add {
  display: grid;
  place-items: center;
  width: 64px;
  height: 64px;
  padding: 0;
  border: 1px dashed var(--zen-border, #34343c);
  border-radius: 6px;
  background: transparent;
  color: var(--zen-muted, #9aa0aa);
  font-size: 22px;
  cursor: pointer;
  transition:
    border-color 0.12s ease,
    color 0.12s ease;
}
.fr-add:hover {
  border-color: var(--zen-accent, #6366f1);
  color: var(--zen-text, #e5e5ea);
}
.fr-browse {
  height: 22px;
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--zen-muted, #9aa0aa);
  font-size: 10px;
  cursor: pointer;
}
.fr-browse:hover {
  color: var(--zen-text, #e5e5ea);
}
</style>
