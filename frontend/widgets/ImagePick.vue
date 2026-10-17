<template>
  <ZenWidget
    fill
    :gap="6"
    pad="4px 2px"
    class="li"
    :class="{ 'li-dropping': dragOver }"
    @dragover.prevent="dragOver = true"
    @dragleave="dragOver = false"
    @drop.prevent="onDrop"
  >
    <div v-if="dragOver" class="li-dropmask">
      <i class="mdi mdi-tray-arrow-down" /><span>Drop image to load</span>
    </div>
    <!-- picker row (above the image): searchable grid combo + upload -->
    <div class="li-row">
      <ZenCombo
        class="li-pick"
        :grid="viewMode === 'grid'"
        :grid-min="zoom"
        :model-value="selected"
        :items="items"
        placeholder="Select an image…"
        :empty-text="`No images in ${srcType}/`"
        @update:model-value="pick(String($event))"
        @open="ensureImages"
      >
        <template #search>
          <ZenToggleGroup
            class="li-src"
            :model-value="srcType"
            :options="SRC_OPTS"
            @update:model-value="setSrc"
          />
        </template>
        <template #selected>
          <span class="li-sel">
            <i class="mdi mdi-image-outline li-sel-ic" />
            <span class="li-sel-name">{{ selected ? short(selected) : "Select an image…" }}</span>
          </span>
        </template>
        <template #option="{ item }">
          <template v-if="viewMode === 'grid'">
            <span
              class="li-gthumb"
              :data-peek="thumbByName(String(item.value))"
              data-peek-kind="image"
              :data-peek-label="short(String(item.value))"
              ><img :src="thumbByName(String(item.value))" loading="lazy"
            /></span>
            <span class="li-gname" :title="String(item.value)">{{
              short(String(item.value))
            }}</span>
          </template>
          <template v-else>
            <img
              class="li-thumb"
              :src="thumbByName(String(item.value))"
              :data-peek="thumbByName(String(item.value))"
              data-peek-kind="image"
              :data-peek-label="short(String(item.value))"
              loading="lazy"
            />
            <span class="li-lname" :title="String(item.value)">{{
              short(String(item.value))
            }}</span>
          </template>
        </template>
        <template #footer="{ close }">
          <div class="li-foot">
            <ZenToggleGroup
              :model-value="viewMode"
              :options="VIEW_OPTS"
              @update:model-value="setView"
            />
            <template v-if="viewMode === 'grid'">
              <ZenIconButton
                icon="mdi mdi-magnify-minus-outline"
                title="Smaller"
                @click="zoomBy(-20)"
              />
              <ZenIconButton
                icon="mdi mdi-magnify-plus-outline"
                title="Larger"
                @click="zoomBy(20)"
              />
            </template>
            <ZenButton
              class="li-foot-browse"
              variant="ghost"
              sm
              icon="mdi mdi-view-grid-outline"
              @click="openBrowse(close)"
              >Browse</ZenButton
            >
          </div>
        </template>
      </ZenCombo>
      <ZenIconButton icon="mdi mdi-view-grid-outline" title="Browse images" @click="openBrowse()" />
      <ZenIconButton icon="mdi mdi-upload" title="Upload an image" @click="triggerUpload" />
    </div>

    <!-- selected preview: fills the node body; the image object-fit:contains inside it -->
    <div v-if="selected" class="li-preview">
      <img v-if="previewUrl" :src="previewUrl" @load="onPreviewLoad" />
      <span v-else class="li-loading">Loading…</span>
      <!-- hover overlay: info popover + open mask editor -->
      <div class="li-ov">
        <div class="li-ov-actions">
          <ZenIconButton
            class="li-ov-btn"
            icon="mdi mdi-draw"
            title="Edit mask"
            @click="openMaskEditor"
          />
          <div
            ref="infoBtn"
            class="li-ov-btn li-ov-info"
            title="Image details"
            @pointerenter="openInfo"
            @pointerleave="infoOpen = false"
          >
            <i class="mdi mdi-information-outline" />
          </div>
          <!-- teleported to body + fixed so it floats over the node instead of
                         being clipped by the preview's overflow:hidden -->
          <Teleport to="body">
            <div v-if="infoOpen" class="li-info-pop" :style="infoStyle">
              <div class="li-info-row" title="Dimensions">
                <i class="mdi mdi-resize" /><b>{{ infoDims }}</b>
              </div>
              <div class="li-info-row" title="File">
                <i class="mdi mdi-file-image-outline" /><b :title="infoName">{{ infoName }}</b>
              </div>
              <div class="li-info-row" title="File size">
                <i class="mdi mdi-harddisk" /><b>{{ infoSize }}</b>
              </div>
              <div class="li-info-row" title="Modified">
                <i class="mdi mdi-clock-outline" /><b>{{ infoDate }}</b>
              </div>
            </div>
          </Teleport>
        </div>
      </div>
    </div>
    <div v-else class="li-preview li-none">
      <i class="mdi mdi-image-outline" />
      <span>No image — pick, browse, or drop one here</span>
    </div>

    <input ref="fileInput" type="file" accept="image/*" style="display: none" @change="onUpload" />

    <!-- browse dialog -->
    <ZenMediaPicker
      :open="pickerOpen"
      :library="mediaLibrary"
      :kinds="['image']"
      title="Load image"
      confirm-label="Load"
      :selected="selected ? [mediaRef(selected)] : []"
      @update:open="pickerOpen = $event"
      @pick="onPicked"
    />
  </ZenWidget>
</template>

<script lang="ts">
// Read by nodekit's `discoverWidgets`: the io type comes from this file's NAME
// (ImagePick.vue -> NYNXZ_IMAGE_PICK, matching nodes/load_image/io_types.py), and these two
// exports carry what the type alone cannot.
import type { NodeDef, WidgetOptions } from "@/framework";

export const widgetOptions: WidgetOptions = {
  name: "image",
  minHeight: 220,
  fill: true, // node keeps its size; the preview fits (contain) instead of resizing it
  default: "",
};

export const nodeDef: Omit<NodeDef, "widgets"> = {
  is: "nynxz.IO.LoadImage",
  minSize: [300, 320],
  hideOutputImages: true, // only the picker's own preview, never ComfyUI's default strip
};
</script>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { ComfyApp } from "@comfy/app";
import {
  ZenButton,
  ZenCombo,
  ZenIconButton,
  ZenMediaPicker,
  ZenToggleGroup,
  ZenWidget,
  type ComboItem,
} from "@nynxz/zenkit-ui";
import {
  listImages,
  thumbUrl,
  uploadImage,
  fmtSize,
  type ImageItem,
  type SourceType,
} from "@/lib/imagePickApi";
import { annotatedValue, uploadSidecarAsClipspaceImage } from "@/lib/maskUpload";
import { mediaLibrary } from "@/framework";

interface DOMWidget {
  name?: string;
  value: unknown;
  callback?: (v: unknown) => void;
}
interface ImageRef {
  filename: string;
  subfolder?: string;
  type?: string;
}
interface NynxzNode {
  id?: string | number;
  previewMediaType?: string;
  images?: ImageRef[];
  imgs?: HTMLImageElement[];
  imageIndex?: number | null;
  hideOutputImages?: boolean;
  properties?: Record<string, unknown>;
  widgets?: DOMWidget[];
  widgets_values?: unknown[];
  graph?: { setDirtyCanvas?: (a: boolean, b?: boolean) => void };
}
const props = defineProps<{
  widget?: DOMWidget;
  node?: NynxzNode;
}>();

const images = ref<ImageItem[]>([]);
const selected = ref<string>("");
// Legacy Nynxz sidecar mask reference. New saves use ComfyUI's native clipspace alpha image model.
const maskRef = ref<string>("");
const srcType = ref<SourceType>("input");
const SRC_OPTS = [
  { value: "input", label: "input" },
  { value: "temp", label: "temp" },
  { value: "output", label: "output" },
];
const dragOver = ref(false);

// Value model: input images store a plain relative name (stock LoadImage behaviour);
// temp/output images store ComfyUI's annotated form `name [type]`, which the Python
// node's get_annotated_filepath() loads from the right dir.
function annot(it: Pick<ImageItem, "type" | "name">) {
  return it.type === "input" ? it.name : `${it.name} [${it.type}]`;
}
function parseVal(v: string): { name: string; type: SourceType } {
  const m = v.match(/^(.*)\s+\[(input|temp|output)\]$/);
  return m ? { name: m[1], type: m[2] as SourceType } : { name: v, type: "input" };
}
// Serialized widget value: a plain image path when there's no mask, or a compound
// { image, mask } JSON when a sidecar mask exists. `image` is always the pristine original.
function applyValue(v: unknown) {
  if (typeof v !== "string") return;
  if (v.startsWith("{")) {
    try {
      const o = JSON.parse(v);
      selected.value = String(o.image ?? "");
      maskRef.value = String(o.mask ?? "");
      if (selected.value) srcType.value = parseVal(selected.value).type;
      suppressDefaultPreviewSoon();
      return;
    } catch {
      /* not JSON — treat as a plain path */
    }
  }
  selected.value = v;
  maskRef.value = "";
  if (selected.value) srcType.value = parseVal(selected.value).type;
  suppressDefaultPreviewSoon();
}
if (typeof props.widget?.value === "string" && props.widget.value) applyValue(props.widget.value);
const fileInput = ref<HTMLInputElement | null>(null);
const viewMode = ref<"grid" | "list">("grid");
const zoom = ref(92); // grid cell min-width (px)
const VIEW_OPTS = [
  { value: "grid", icon: "mdi mdi-view-grid-outline", title: "Grid view" },
  { value: "list", icon: "mdi mdi-view-list", title: "List view" },
];
function setView(v: unknown) {
  viewMode.value = String(v) === "list" ? "list" : "grid";
}
function zoomBy(d: number) {
  zoom.value = Math.max(56, Math.min(220, zoom.value + d));
}

const pickerOpen = ref(false);

onMounted(async () => {
  // The host restores widget.value on graph load (often after setup) — pick it up and
  // keep tracking host writes, so the selection + mask survive a page reload.
  if (typeof props.widget?.value === "string" && props.widget.value) applyValue(props.widget.value);
  await loadSelectedImageDetails();
  const prev = props.widget?.callback;
  if (props.widget) {
    // eslint-disable-next-line vue/no-mutating-props -- `widget` is ComfyUI's host object, not Vue state; writing its value IS how a node body persists.
    props.widget.callback = function (this: DOMWidget, v: unknown) {
      if (prev) {
        try {
          prev.call(this, v);
        } catch {
          /* ignore */
        }
      }
      applyValue(v);
    };
  }
});

// Lazy: fetch the image list on first picker open (not on mount) — a Load-Image node the
// user never opens won't hit /nynxz/images on graph load. listImages() is cached + shared.
async function ensureImages() {
  images.value = await listImages(srcType.value);
}
async function setSrc(v: unknown) {
  const t = String(v) as SourceType;
  if (t !== "input" && t !== "temp" && t !== "output") return;
  srcType.value = t;
  images.value = await listImages(t);
}
async function loadSelectedImageDetails() {
  if (!selected.value) return;
  const { type } = parseVal(selected.value);
  srcType.value = type;
  images.value = await listImages(type);
}

function isClipspaceItem(it: ImageItem) {
  const name = it.name.replace(/\\/g, "/").toLowerCase();
  const subfolder = it.subfolder.replace(/\\/g, "/").toLowerCase();
  return (
    name === "clipspace" ||
    name.startsWith("clipspace/") ||
    subfolder === "clipspace" ||
    subfolder.startsWith("clipspace/")
  );
}
const pickerImages = computed(() => images.value.filter((it) => !isClipspaceItem(it)));
const byName = computed(() => {
  const m = new Map<string, ImageItem>();
  for (const it of images.value) m.set(annot(it), it);
  return m;
});
const items = computed<ComboItem[]>(() =>
  pickerImages.value.map((it) => ({
    value: annot(it),
    label: short(it.name),
    keywords: it.name,
  })),
);

function short(name: string) {
  return (parseVal(name).name.split("/").pop() || name).replace(/\.(png|jpe?g|webp|gif|bmp)$/i, "");
}
function thumbByName(value: string) {
  if (!value) return "";
  const it = byName.value.get(value);
  if (it) return thumbUrl(it);
  // List not loaded yet — derive filename/subfolder/type from the (possibly annotated)
  // value so an already-selected image still previews without fetching the whole list.
  const { name, type } = parseVal(value);
  const i = name.lastIndexOf("/");
  return thumbUrl({
    filename: i >= 0 ? name.slice(i + 1) : name,
    subfolder: i >= 0 ? name.slice(0, i) : "",
    type,
  });
}
const previewUrl = computed(() => thumbByName(selected.value));
const imgW = ref(0);
const imgH = ref(0);
function onPreviewLoad(e: Event) {
  const img = e.target as HTMLImageElement;
  if (img.naturalWidth > 0 && img.naturalHeight > 0) {
    imgW.value = img.naturalWidth;
    imgH.value = img.naturalHeight;
  }
}

// --- preview info popover (dimensions / filename / size / date) ---
const infoOpen = ref(false);
const infoBtn = ref<HTMLElement | null>(null);
const infoStyle = ref<Record<string, string>>({});
// Position the (teleported, fixed) info card just under its button, clamped to the viewport.
function openInfo() {
  const el = infoBtn.value;
  if (el) {
    const r = el.getBoundingClientRect();
    const W = 210;
    infoStyle.value = {
      position: "fixed",
      top: `${r.bottom + 6}px`,
      left: `${Math.max(8, Math.min(r.right - W, window.innerWidth - W - 8))}px`,
      zIndex: "100001",
    };
  }
  infoOpen.value = true;
}
const selectedItem = computed(() =>
  selected.value ? byName.value.get(selected.value) : undefined,
);
const infoName = computed(() => parseVal(selected.value).name.split("/").pop() || selected.value);
const infoDims = computed(() => (imgW.value && imgH.value ? `${imgW.value} × ${imgH.value}` : "—"));
const infoSize = computed(() => fmtSize(selectedItem.value?.size));
const infoDate = computed(() => friendlyDate(selectedItem.value?.mtime));
function friendlyDate(mtime?: number) {
  if (!mtime) return "—";
  const date = new Date(mtime * 1000);
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startOfDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const daysAgo = Math.round((startOfToday.getTime() - startOfDate.getTime()) / 86_400_000);
  const time = date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
  if (daysAgo === 0) return `Today, ${time}`;
  if (daysAgo === 1) return `Yesterday, ${time}`;
  return date.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: date.getFullYear() === now.getFullYear() ? undefined : "numeric",
  });
}

function refOf(value: string) {
  const { name, type } = parseVal(value);
  const i = name.lastIndexOf("/");
  return {
    filename: i >= 0 ? name.slice(i + 1) : name,
    subfolder: i >= 0 ? name.slice(0, i) : "",
    type,
  };
}

function suppressDefaultPreview() {
  const node = props.node;
  if (!node) return;
  node.hideOutputImages = true;
  node.images = undefined;
  node.imgs = undefined;
  node.imageIndex = null;
  node.previewMediaType = undefined;
  node.graph?.setDirtyCanvas?.(true, true);
}

function suppressDefaultPreviewSoon() {
  window.setTimeout(suppressDefaultPreview, 0);
  window.setTimeout(suppressDefaultPreview, 100);
}

function prepareNativeMaskEditorNode() {
  const node = props.node;
  if (!node || !selected.value) return;

  const ref = refOf(selected.value);
  node.hideOutputImages = true;
  node.previewMediaType = "image";
  node.images = [{ filename: ref.filename, subfolder: ref.subfolder, type: ref.type }];
  node.imageIndex = 0;
  // Do not set node.imgs here: that makes ComfyUI draw its default image preview over/around
  // the custom Nynxz picker. The native mask editor can load from node.images + widget value.
  node.imgs = undefined;

  if (node.properties) node.properties["image"] = selected.value;
  if (node.widgets_values && node.widgets && props.widget) {
    const widgetIndex = node.widgets.indexOf(props.widget);
    if (widgetIndex >= 0) node.widgets_values[widgetIndex] = selected.value;
  }
}

async function openMaskEditor() {
  if (!selected.value || !props.node) return;

  if (maskRef.value) {
    const migrated = await uploadSidecarAsClipspaceImage(
      refOf(maskRef.value),
      refOf(selected.value),
    );
    if (!migrated) {
      console.error("[Nynxz] Failed to migrate sidecar mask to native clipspace image");
      return;
    }
    selected.value = annotatedValue(migrated);
    maskRef.value = "";
    commit({ keepNativePreview: true });
  } else {
    prepareNativeMaskEditorNode();
  }

  prepareNativeMaskEditorNode();
  ComfyApp.clipspace_return_node = props.node;
  if (typeof ComfyApp.open_maskeditor === "function") {
    ComfyApp.open_maskeditor();
  } else {
    console.error("[Nynxz] ComfyUI native mask editor is not available");
    suppressDefaultPreviewSoon();
  }
}

function commit(opts: { keepNativePreview?: boolean } = {}) {
  if (props.widget) {
    const v = selected.value;
    // eslint-disable-next-line vue/no-mutating-props -- `widget` is ComfyUI's host object, not Vue state; writing its value IS how a node body persists.
    props.widget.value = v;
    try {
      props.widget.callback?.(v);
    } catch {
      /* no callback */
    }
  }
  if (!opts.keepNativePreview) suppressDefaultPreviewSoon();
  props.node?.graph?.setDirtyCanvas?.(true, true);
}
function pick(name: string) {
  selected.value = name;
  maskRef.value = "";
  commit();
} // new image → no mask

function openBrowse(close?: () => void) {
  close?.();
  pickerOpen.value = true;
}
// The shared media picker speaks refs ("output/sub/a.png"); the widget stores ComfyUI's
// annotated value ("sub/a.png [output]").
function mediaRef(value: string) {
  const { name, type } = parseVal(value);
  return `${type}/${name}`;
}
async function onPicked(refs: string[]) {
  const ref = refs[0];
  if (!ref) return;
  const slash = ref.indexOf("/");
  const type = ref.slice(0, slash) as SourceType;
  pick(annot({ type, name: ref.slice(slash + 1) }));
  await loadSelectedImageDetails();
}

// --- upload ---
function triggerUpload() {
  fileInput.value?.click();
}
async function uploadAndSelect(file: File) {
  if (!file || !file.type.startsWith("image/")) return;
  const it = await uploadImage(file);
  if (!it) return;
  srcType.value = "input"; // uploads always land in the input dir
  images.value = await listImages("input", true);
  pick(it.name);
}
async function onUpload(e: Event) {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (file) await uploadAndSelect(file);
}
async function onDrop(e: DragEvent) {
  dragOver.value = false;
  const file = e.dataTransfer?.files?.[0];
  if (file) await uploadAndSelect(file);
}
</script>

<style scoped>
/* fill mode: the panel fills the node body so the preview fits the current size (it doesn't
   resize the node when the image changes); the picker row is fixed, the preview takes the rest */
.li {
  position: relative;
}

.li-row {
  display: flex;
  align-items: center;
  gap: 6px;
  flex: none;
}

.li-preview {
  position: relative;
  flex: 1 1 0;
  min-height: 0;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--zen-radius, 8px);
}
.li-preview img {
  display: block;
  border-radius: var(--zen-radius, 8px);
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  object-fit: contain;
}
.li-none {
  flex-direction: column;
  gap: 6px;
  color: var(--zen-muted, #9aa0aa);
  border: 1px dashed var(--zen-border, #34343c);
  text-align: center;
  padding: 8px;
}
.li-none .mdi {
  font-size: 26px;
  opacity: 0.8;
}
.li-loading {
  color: var(--zen-muted, #9aa0aa);
  font-size: 11px;
}

/* drag-and-drop overlay */
.li-dropmask {
  position: absolute;
  inset: 0;
  z-index: 5;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 16%, var(--zen-bg, #1a1a1f) 78%);
  border: 2px dashed var(--zen-accent, #6366f1);
  border-radius: var(--zen-radius, 8px);
  color: var(--zen-text, #e5e5ea);
  pointer-events: none;
}
.li-dropmask .mdi {
  font-size: 32px;
  color: var(--zen-accent, #6366f1);
}
.li-pick {
  flex: 1;
  min-width: 0;
}

.li-sel {
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
}
.li-sel-ic {
  flex: none;
  font-size: 15px;
  color: var(--zen-muted, #9aa0aa);
}

.li-sel-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* preview hover tools: lightweight corner actions; no full-image darkening */
.li-ov {
  position: absolute;
  inset: 0;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.12s ease;
}
.li-preview:hover .li-ov {
  opacity: 1;
  pointer-events: auto;
}
.li-ov-actions {
  position: absolute;
  top: 6px;
  right: 6px;
  z-index: 2;
  display: flex;
  align-items: center;
  gap: 4px;
  max-width: calc(100% - 12px);
}
.li-ov-btn {
  position: relative;
  flex: 0 0 24px;
  width: 24px;
  height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: var(--zen-radius, 6px);
  background: rgba(0, 0, 0, 0.42);
  color: #fff;
  backdrop-filter: blur(4px);
}
.li-ov-btn:hover {
  background: var(--zen-accent, #6366f1) !important;
  border-color: var(--zen-accent, #6366f1) !important;
  color: var(--zen-accent-text, #fff) !important;
}
.li-ov-info {
  cursor: help;
}
.li-ov-info > .mdi {
  font-size: 16px;
}
.li-info-pop {
  /* position/top/left/z-index come from the inline fixed style (teleported to body) */
  box-sizing: border-box;
  width: 210px;
  max-width: calc(100vw - 24px);
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px;
  background: color-mix(in srgb, var(--zen-surface, #202026) 94%, #000 6%);
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 8px);
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45);
  cursor: default;
}
.li-info-row {
  display: grid;
  grid-template-columns: 18px minmax(0, 1fr);
  align-items: center;
  min-width: 0;
  gap: 7px;
  min-height: 19px;
  font-size: 11px;
}
.li-info-row > .mdi {
  color: var(--zen-muted, #9aa0aa);
  font-size: 14px;
  text-align: center;
}
.li-info-row b {
  display: block;
  min-width: 0;
  color: var(--zen-text, #e5e5ea);
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.li-thumb {
  flex: none;
  box-sizing: border-box;
  width: 40px;
  height: 40px;
  object-fit: cover;
  border-radius: var(--zen-radius, 6px);
  background: var(--zen-input, #1b1b20);
  border: 1px solid var(--zen-border, #34343c);
}
.li-thumb.sm {
  width: 18px;
  height: 18px;
  border-radius: var(--zen-radius, 5px);
}
.li-thumb.ph {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--zen-muted, #9aa0aa);
}
/* grid-combo cell (contain square + name) */
.li-gthumb {
  width: 100%;
  aspect-ratio: 1;
  border-radius: var(--zen-radius, 6px);
  overflow: hidden;
  background: var(--zen-surface, #202026);
  display: flex;
  align-items: center;
  justify-content: center;
}
.li-gthumb img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.li-gname {
  font-size: 10px;
  color: var(--zen-text, #e5e5ea);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-align: center;
}
.li-lname {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.li-foot {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
}
.li-foot-browse {
  margin-left: auto;
}
/* compact source switcher inside the combo search row */
.li-src :deep(.zen-tg-b) {
  padding: 2px 7px;
  font-size: 10px;
}
</style>
