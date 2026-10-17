<template>
  <ZenWidget fill :gap="0" pad="4px 2px" class="sp">
    <div
      v-if="current"
      ref="previewEl"
      class="sp-preview"
      :class="{ 'sp-none': isBroken, 'sp-hot': pointerOver }"
    >
      <img
        v-if="!isBroken"
        :key="currentUrl"
        :src="currentUrl"
        draggable="false"
        @error="onError(current.filename)"
      />
      <template v-else>
        <i class="mdi mdi-image-off-outline" title="Saved image unavailable" />
      </template>

      <!-- batch pager: only when more than one image was saved -->
      <div v-if="items.length > 1" class="sp-pager">
        <ZenIconButton icon="mdi mdi-chevron-left" title="Previous" @click="step(-1)" />
        <span class="sp-count">{{ index + 1 }} / {{ items.length }}</span>
        <ZenIconButton icon="mdi mdi-chevron-right" title="Next" @click="step(1)" />
      </div>

      <!-- hover actions (top-right): download + open in lightbox -->
      <div v-if="!isBroken" class="sp-actions">
        <button class="sp-act" title="Download" @click="download">
          <i class="mdi mdi-download" />
        </button>
        <button class="sp-act" title="Open in viewer" @click="lightbox = true">
          <i class="mdi mdi-fullscreen" />
        </button>
      </div>
    </div>

    <div v-else class="sp-preview sp-none">
      <i class="mdi mdi-image-outline" title="Nothing saved yet" />
    </div>

    <!-- Fullscreen viewer. ZenLightbox straight from zenkit-ui rather than the ZenKit
         runtime's shared viewer, so the pack works with or without ZenKit installed. -->
    <ZenLightbox
      v-if="lightbox"
      :items="lbItems"
      :index="index"
      @update:index="(i: number) => (index = i)"
      @close="lightbox = false"
    />
  </ZenWidget>
</template>

<script lang="ts">
// Read by nodekit's `discoverWidgets`: the io type comes from this file's NAME
// (SavePreview.vue -> NYNXZ_SAVE_PREVIEW, matching nodes/save/io_types.py), and these two
// exports carry what the type alone cannot.
//
// Preview Image reuses this same widget type; its node def lives in frontend/nodes/ because
// discovery binds one node def per widget file.
import { settingId, type NodeDef, type WidgetOptions } from "@/framework";
import { previewRecords } from "@/lib/saveApi";

export const widgetOptions: WidgetOptions = {
  name: "save_preview",
  minHeight: 200,
  fill: true, // node keeps its size; the saved image fits (contain) inside it
  dragThrough: true, // visual-only body; a press falls through to drag the node
  serialize: false, // preview URLs go stale on restart
  default: [],
};

export const nodeDef: Omit<NodeDef, "widgets"> = {
  is: "nynxz.Save.Image",
  minSize: [300, 320],
  hideOutputImages: true, // show only our preview, never ComfyUI's default strip
  output: { widget: "save_preview", from: previewRecords("saved", "output") },
  // Middle-click the save_info socket -> spawn + wire a Save Info.
  slotLinks: [{ input: "save_info", spawn: "nynxz.IO.SaveInfo" }],
  // Persisted to comfy.settings.json; read server-side by nodes/save/paths.py.
  settings: [
    {
      id: settingId("SaveImage.BaseDirectory"),
      name: "Save Image — base directory",
      category: ["Nynxz", "Save Image", "Base directory"],
      type: "text",
      defaultValue: "",
      tooltip:
        "Base folder the Nynxz Save Image node writes into (project folders live under " +
        "it). Empty = the ComfyUI output directory.",
    },
  ],
};
</script>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { ZenIconButton, ZenLightbox, ZenWidget, type LightboxItem } from "@nynxz/zenkit-ui";

interface SavedItem {
  filename: string;
  subfolder: string;
  type?: string;
}
interface DOMWidget {
  name?: string;
  value: unknown;
  callback?: (v: unknown) => void;
}
const props = defineProps<{ widget?: DOMWidget; node?: unknown }>();

const items = ref<SavedItem[]>([]);
const index = ref(0);
const broken = ref<Set<string>>(new Set());
const bust = ref(0);
const lightbox = ref(false);

// The body is drag-through (pointer-events: none), so CSS :hover never fires on it. Track the
// pointer against the preview's box instead, on both renderers.
const previewEl = ref<HTMLElement | null>(null);
const pointerOver = ref(false);
function trackPointer(e: PointerEvent) {
  const r = previewEl.value?.getBoundingClientRect();
  pointerOver.value =
    !!r &&
    e.clientX >= r.left &&
    e.clientX <= r.right &&
    e.clientY >= r.top &&
    e.clientY <= r.bottom;
}
function dropPointer() {
  pointerOver.value = false;
}

function sig(list: SavedItem[]): string {
  return list.map((it) => `${it.type}/${it.subfolder}/${it.filename}`).join("|");
}
function applyValue(v: unknown) {
  const next = (Array.isArray(v) ? v : []).filter(
    (it): it is SavedItem => !!it && typeof it.filename === "string",
  );
  if (sig(next) === sig(items.value) && next.length) return;
  items.value = next;
  index.value = Math.min(index.value, Math.max(0, next.length - 1));
  broken.value = new Set();
  bust.value++;
}
applyValue(props.widget?.value);

function urlOf(it: SavedItem): string {
  const p = new URLSearchParams({
    filename: it.filename,
    type: it.type || "output",
    subfolder: it.subfolder || "",
  }).toString();
  return `/view?${p}&t=${bust.value}`;
}

const current = computed<SavedItem | undefined>(() => items.value[index.value]);
const isBroken = computed(() => !!current.value && broken.value.has(current.value.filename));
const currentUrl = computed(() => (current.value ? urlOf(current.value) : ""));
const lbItems = computed<LightboxItem[]>(() =>
  items.value.map((it) => ({
    src: urlOf(it),
    kind: "image",
    label: it.filename,
    meta: it.subfolder || "output",
  })),
);

function step(d: number) {
  const n = items.value.length;
  if (!n) return;
  index.value = (index.value + d + n) % n;
}
function onError(filename: string) {
  broken.value = new Set(broken.value).add(filename);
}
function download() {
  const it = current.value;
  if (!it) return;
  const a = document.createElement("a");
  a.href = urlOf(it);
  a.download = it.filename.split("/").pop() || "image.png"; // same-origin /view -> downloads
  document.body.appendChild(a);
  a.click();
  a.remove();
}

onMounted(() => {
  document.addEventListener("pointermove", trackPointer, { passive: true });
  document.documentElement.addEventListener("pointerleave", dropPointer);
  applyValue(props.widget?.value);
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
onUnmounted(() => {
  document.removeEventListener("pointermove", trackPointer);
  document.documentElement.removeEventListener("pointerleave", dropPointer);
});
</script>
<style scoped>
.sp {
  position: relative;
}
.sp-preview {
  position: relative;
  flex: 1 1 0;
  min-height: 0;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--zen-radius, 8px);
}
.sp-preview img {
  display: block;
  border-radius: var(--zen-radius, 8px);
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  object-fit: contain;
  /* no native image-drag ghost; the drag falls through to litegraph (moves the node) */
  -webkit-user-drag: none;
  user-select: none;
}
.sp-none {
  flex-direction: column;
  gap: 6px;
  color: var(--zen-muted, #9aa0aa);
  border: 1px dashed var(--zen-border, #34343c);
  text-align: center;
  padding: 8px;
}
.sp-none .mdi {
  font-size: 26px;
  opacity: 0.8;
}

/* batch pager (bottom-center) */
.sp-pager {
  position: absolute;
  bottom: 6px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 3;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 2px 4px;
  border-radius: var(--zen-radius, 7px);
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.12);
  backdrop-filter: blur(4px);
  /* the body is pointer-events:none (drag-through), so controls opt back in + stay visible */
  pointer-events: auto;
}
.sp-count {
  font-size: 11px;
  color: #fff;
  min-width: 38px;
  text-align: center;
}

/* hover actions (top-right): download + lightbox */
.sp-actions {
  position: absolute;
  top: 6px;
  right: 6px;
  z-index: 3;
  display: flex;
  gap: 4px;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.12s ease;
}
.sp-preview.sp-hot .sp-actions {
  opacity: 1;
  pointer-events: auto;
}
.sp-act {
  width: 26px;
  height: 26px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: var(--zen-radius, 6px);
  background: rgba(0, 0, 0, 0.46);
  color: #fff;
  cursor: pointer;
  backdrop-filter: blur(4px);
  transition:
    background 0.12s ease,
    border-color 0.12s ease;
}
.sp-act:hover {
  background: var(--zen-accent, #6366f1);
  border-color: var(--zen-accent, #6366f1);
}
.sp-act .mdi {
  font-size: 16px;
}
</style>
