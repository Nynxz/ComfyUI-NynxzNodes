<template>
  <div
    :ref="(el) => (wheel.el.value = el as HTMLElement | null)"
    class="fs"
    v-bind="wheel.attrs"
    @keydown="onKey"
  >
    <nav class="fs-rail" aria-label="Tools">
      <ZenIconButton
        v-for="t in TOOLS"
        :key="t.value"
        size="sm"
        :icon="t.icon"
        :active="tool === t.value"
        :title="t.title"
        @click="tool = t.value"
      />
      <span class="fs-rail-sep" />
      <ZenIconButton
        size="sm"
        :icon="showFaces ? 'mdi mdi-card-text-outline' : 'mdi mdi-card-off-outline'"
        :active="showFaces"
        title="Show each layer's references and prompt on the canvas"
        @click="setShowFaces(!showFaces)"
      />
      <ZenIconButton
        size="sm"
        icon="mdi mdi-plus-box-outline"
        title="Add a layer (or drag on the canvas)"
        @click="createLayer({ x: 0.3, y: 0.3, w: 0.4, h: 0.4 })"
      />
      <ZenPopover placement="right-start" role="dialog">
        <template #trigger="{ toggle }">
          <ZenIconButton
            size="sm"
            icon="mdi mdi-image-outline"
            :active="!!backdropUrl"
            title="Trace image — something to draw over. Fresco Apply shows its source image here after a run."
            @click="toggle"
          />
        </template>
        <div class="fs-menu">
          <ZenButton sm icon="mdi mdi-view-grid-outline" @click="backdropBrowse = true">
            Pick an image…
          </ZenButton>
          <ZenButton v-if="backdropUrl" sm icon="mdi mdi-close" @click="setBackdrop(null)">
            Remove
          </ZenButton>
        </div>
      </ZenPopover>
    </nav>

    <header class="fs-status">
      <ZenPopover placement="bottom-start" role="dialog">
        <template #trigger="{ toggle }">
          <ZenIconButton
            size="sm"
            icon="mdi mdi-help-circle-outline"
            title="Controls"
            @click="toggle"
          />
        </template>
        <dl class="fs-help">
          <template v-for="row in HELP" :key="row[0]">
            <dt>{{ row[0] }}</dt>
            <dd>{{ row[1] }}</dd>
          </template>
        </dl>
      </ZenPopover>
      <span class="fs-tip">
        <template v-if="mismatch">
          Source is {{ sourceSize![0] }}×{{ sourceSize![1] }}
          <button type="button" class="fs-link" @click="matchSource">match canvas</button>
        </template>
      </span>
      <label
        class="fs-ctl"
        :class="{ dim: tool === 'select' }"
        title="Brush size in canvas pixels — [ and ]"
      >
        <i class="mdi mdi-brush" />
        <ZenNumber
          size="sm"
          :model-value="Math.round(brushSize * scene.width)"
          :min="4"
          :max="Math.round(scene.width * 0.4)"
          :step="2"
          :precision="0"
          @update:model-value="brushSize = Math.max(4, $event) / scene.width"
        />
      </label>
      <label class="fs-ctl" :class="{ dim: !backdropUrl }" title="Trace image opacity, %">
        <i class="mdi mdi-image-outline" />
        <ZenNumber
          size="sm"
          :model-value="Math.round(backdropOpacity * 100)"
          :min="10"
          :max="100"
          :step="5"
          :precision="0"
          :disabled="!backdropUrl"
          @update:model-value="backdropOpacity = $event / 100"
        />
      </label>
      <ZenPopover placement="bottom-end" role="dialog">
        <template #trigger="{ toggle }">
          <button type="button" class="fs-chip" title="Canvas size" @click="toggle">
            {{ scene.width }}×{{ scene.height }}
          </button>
        </template>
        <ZenResolution
          :width="scene.width"
          :height="scene.height"
          :snap="32"
          :presets="[0.5, 1, 1.5, 2]"
          :max-side="2048"
          :max-mp="4"
          @update:width="setSize($event, scene.height)"
          @update:height="setSize(scene.width, $event)"
        />
      </ZenPopover>
    </header>

    <div class="fs-stage" @wheel="onStageWheel">
      <ZenBoxCanvas
        v-model:boxes="boxes"
        v-model:selected="selected"
        v-model:tool="tool"
        :aspect="scene.width / scene.height"
        :backdrop="backdropUrl"
        :backdrop-opacity="backdropOpacity"
        :brush-size="brushSize"
        :snap="1 / 64"
        @create="createLayer"
        @commit="commit"
        @update:mask="setMask"
        @delete="removeLayers"
        @open="openLayer"
      >
        <template v-if="showFaces" #box="{ box, maskStyle }">
          <FrescoBoxFace
            v-if="layerById.get(box.id)"
            :layer="layerById.get(box.id)!"
            :mask-style="maskStyle"
          />
        </template>
        <template #empty>
          <div class="fs-hint-empty">
            <i class="mdi mdi-gesture-tap-box" />
            <span>Drag to draw a layer</span>
          </div>
        </template>
      </ZenBoxCanvas>
    </div>

    <aside class="fs-side">
      <section class="fs-pane fs-layers">
        <header class="fs-head">
          <span class="fs-title">Layers</span>
          <span class="fs-count">{{ scene.layers.length }}</span>
        </header>
        <ZenScroll class="fs-body">
          <ZenLayerList
            :selected="listSelection"
            :items="layerItems"
            @update:selected="selectFromList"
            @reorder="reorder"
            @toggle-hidden="toggleOn"
            @toggle-locked="toggleLock"
          />
        </ZenScroll>
      </section>

      <section class="fs-pane fs-inspect">
        <header class="fs-head">
          <template v-if="currentLayer">
            <span class="fs-swatch" :style="{ background: currentLayer.color }" />
            <ZenInput
              class="fs-name"
              sm
              :model-value="currentLayer.name"
              @update:model-value="patchCurrent({ name: String($event ?? '') })"
            />
            <ZenIconButton
              size="sm"
              danger
              icon="mdi mdi-delete-outline"
              title="Delete this layer"
              @click="removeLayers([currentLayer.id])"
            />
          </template>
          <template v-else-if="selected.length > 1">
            <span class="fs-title">{{ selected.length }} layers</span>
            <ZenIconButton
              size="sm"
              danger
              icon="mdi mdi-delete-outline"
              title="Delete these layers"
              @click="removeLayers([...selected])"
            />
          </template>
          <span v-else class="fs-title">Scene</span>
        </header>

        <ZenScroll class="fs-body">
          <FrescoContentForm
            v-if="currentLayer"
            ref="layerForm"
            :content="currentLayer"
            prompt-placeholder="What goes here — write it out in full. Empty: the scene prompt, read against the references"
            @patch="patchCurrent"
          >
            <div class="fs-label">Shape</div>
            <div class="fs-grid">
              <span title="Soften the edge, in canvas pixels">feather</span>
              <ZenNumber
                size="sm"
                :model-value="currentLayer.feather"
                :min="0"
                :max="256"
                :step="1"
                :precision="0"
                title="Soften the edge, in canvas pixels"
                @update:model-value="patchCurrent({ feather: Math.round($event) })"
              />
            </div>
            <div class="fs-row">
              <span class="fs-note">
                {{ currentLayer.mask ? "Painted shape" : "The whole box" }} · <kbd>B</kbd> paint ·
                <kbd>E</kbd> erase
              </span>
              <ZenButton
                sm
                icon="mdi mdi-crop-square"
                :disabled="!currentLayer.mask"
                title="Clear the painted shape — the layer becomes its whole box"
                @click="setMask(currentLayer.id, null)"
              >
                Box
              </ZenButton>
            </div>
          </FrescoContentForm>

          <p v-else-if="selected.length > 1" class="fs-note fs-pad">
            Drag any of them on the canvas to move them together; the frame's handles scale them.
          </p>

          <div v-else class="fs-form">
            <div class="fs-label">Scene prompt</div>
            <ZenInput
              type="textarea"
              :rows="3"
              sm
              :model-value="scene.prompt"
              placeholder="The whole picture — Fresco Apply's positive"
              @update:model-value="patchScene({ prompt: String($event ?? '') })"
            />
            <div class="fs-label fs-section">
              Background
              <span class="fs-label-note">wherever no layer is</span>
            </div>
            <FrescoContentForm
              :content="scene.background"
              :prompt-rows="2"
              prompt-placeholder="Optional — what fills the rest of the frame"
              @patch="patchBackground"
            />
            <div class="fs-label">Settings</div>
            <label
              class="fs-field wide"
              title="How overlapping layers share the pixels they both cover"
            >
              <span>overlap</span>
              <ZenToggleGroup
                :model-value="scene.settings.overlap"
                :options="OVERLAPS"
                @update:model-value="patchSettings({ overlap: $event as Overlap })"
              />
            </label>
            <label
              class="fs-field wide"
              title="Resolution reference images are read at. Higher reads more detail and costs more tokens in every layer's copy of the prompt."
            >
              <span>ref detail</span>
              <ZenNumber
                :model-value="scene.settings.visual_size"
                :min="128"
                :max="1024"
                :step="32"
                @update:model-value="patchSettings({ visual_size: $event })"
              />
            </label>
          </div>
        </ZenScroll>
      </section>
    </aside>

    <ZenMediaPicker
      v-model:open="backdropBrowse"
      :library="mediaLibrary"
      :kinds="['image']"
      title="Trace image"
      @pick="pickBackdrop"
    />
  </div>
</template>

<script lang="ts">
import { mediaLibrary, nodeId, type NodeDef, type WidgetOptions } from "@/framework";

/** The body is the node's, and nothing in it changes size as you work — see the layout below. */
export const widgetOptions: WidgetOptions = { minHeight: 520, fill: true, default: {} };

export const nodeDef: Omit<NodeDef, "widgets"> = {
  is: nodeId("Fresco"),
  minSize: [980, 640],
  slotLinks: [{ output: "fresco", spawn: nodeId("Fresco.Apply") }],
};
</script>

<script setup lang="ts">
/**
 * Fresco — compose a scene as layers on a canvas.
 *
 * Layout, fixed so nothing moves as you work: a tool rail, the canvas, a status bar under it,
 * and a side column split between the layer list and an inspector that always shows something —
 * the selected layer, or the Scene (prompt, background, settings) when nothing is selected. Both
 * side panes scroll inside themselves.
 *
 * The scene is the widget value (lib/frescoScene.ts), committed once per finished edit. The trace
 * image is a node property, so drawing over one never re-runs anything; Fresco Apply sends its
 * source image back after a run and it lands here.
 */
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import { api } from "@comfy/api";
import {
  ZenBoxCanvas,
  ZenButton,
  ZenIconButton,
  ZenInput,
  ZenLayerList,
  ZenMediaPicker,
  ZenNumber,
  ZenPopover,
  ZenResolution,
  ZenScroll,
  ZenToggleGroup,
  useWheelClaim,
  type BoxTool,
  type LayerItem,
  type ZenBox,
} from "@nynxz/zenkit-ui";

import FrescoBoxFace from "@/components/fresco/FrescoBoxFace.vue";
import FrescoContentForm from "@/components/fresco/FrescoContentForm.vue";
import { thumbUrl, type SourceType } from "@/lib/fusionApi";
import {
  BACKGROUND_ID,
  newLayer,
  parseScene,
  summarize,
  type FrescoContent,
  type FrescoLayer,
  type FrescoScene,
  type FrescoSettings,
  type Overlap,
} from "@/lib/frescoScene";

const props = defineProps<{
  widget?: { value: unknown; callback?: (v: unknown) => void };
  node?: {
    id?: string | number;
    graph?: { setDirtyCanvas?: (a: boolean, b: boolean) => void };
    properties?: Record<string, unknown>;
  };
}>();

const TOOLS: { value: BoxTool; icon: string; title: string }[] = [
  { value: "select", icon: "mdi mdi-cursor-default-outline", title: "Select, move, draw — V" },
  { value: "brush", icon: "mdi mdi-brush", title: "Paint the layer's shape — B" },
  { value: "erase", icon: "mdi mdi-eraser", title: "Erase from the layer's shape — E" },
];
const OVERLAPS = [
  { value: "top", label: "Cover", title: "A layer covers the layers under it, like paint" },
  { value: "share", label: "Share", title: "Overlapping layers split the overlap" },
];

const scene = ref<FrescoScene>(parseScene(props.widget?.value));
const selected = ref<string[]>([]);
const tool = ref<BoxTool>("select");
const brushSize = ref(0.06);
const layerForm = ref<{ focusPrompt: () => void } | null>(null);

function commit() {
  const snapshot = JSON.parse(JSON.stringify(scene.value)) as FrescoScene;
  if (props.widget) {
    // eslint-disable-next-line vue/no-mutating-props -- the widget value is how a node body persists into the saved graph.
    props.widget.value = snapshot;
    try {
      props.widget.callback?.(snapshot);
    } catch {
      /* no callback */
    }
  }
  props.node?.graph?.setDirtyCanvas?.(true, true);
}

function patchScene(fields: Partial<FrescoScene>) {
  scene.value = { ...scene.value, ...fields };
  commit();
}
function patchBackground(fields: Partial<FrescoContent>) {
  patchScene({ background: { ...scene.value.background, ...fields } });
}
function patchSettings(fields: Partial<FrescoSettings>) {
  patchScene({ settings: { ...scene.value.settings, ...fields } });
}
function setSize(width: number, height: number) {
  patchScene({ width: Math.round(width), height: Math.round(height) });
}
function updateLayers(fn: (l: FrescoLayer) => FrescoLayer) {
  scene.value = { ...scene.value, layers: scene.value.layers.map(fn) };
  commit();
}

// --- the canvas -------------------------------------------------------------------------------

const boxes = computed<ZenBox[]>({
  get: () =>
    scene.value.layers.map((l) => ({
      id: l.id,
      x: l.x,
      y: l.y,
      w: l.w,
      h: l.h,
      color: l.color,
      label: l.name,
      hidden: !l.on,
      locked: l.locked,
      mask: l.mask,
    })),
  // Live while dragging; the canvas's `commit` (at the end of the gesture) persists.
  set: (next) => {
    const rects = new Map(next.map((b) => [b.id, b]));
    scene.value = {
      ...scene.value,
      layers: scene.value.layers.map((l) => {
        const b = rects.get(l.id);
        return b ? { ...l, x: b.x, y: b.y, w: b.w, h: b.h } : l;
      }),
    };
  },
});

const layerById = computed(() => new Map(scene.value.layers.map((l) => [l.id, l])));

/** Whether boxes show their references and prompt. A display preference, so a node property. */
const FACES_KEY = "nynxzFrescoFaces";
const showFaces = ref(props.node?.properties?.[FACES_KEY] !== false);
function setShowFaces(on: boolean) {
  showFaces.value = on;
  if (props.node?.properties) {
    // eslint-disable-next-line vue/no-mutating-props -- node.properties is ComfyUI's own store for what the backend must not see.
    props.node.properties[FACES_KEY] = on;
  }
}

function createLayer(rect: { x: number; y: number; w: number; h: number }) {
  const layer = newLayer(rect, scene.value.layers);
  scene.value = { ...scene.value, layers: [...scene.value.layers, layer] };
  selected.value = [layer.id];
  tool.value = "select";
  commit();
}
function setMask(id: string, mask: string | null) {
  updateLayers((l) => (l.id === id ? { ...l, mask } : l));
}
function removeLayers(ids: string[]) {
  const gone = new Set(ids);
  scene.value = { ...scene.value, layers: scene.value.layers.filter((l) => !gone.has(l.id)) };
  selected.value = selected.value.filter((id) => !gone.has(id));
  commit();
}
async function openLayer(id: string) {
  selected.value = [id];
  await nextTick();
  layerForm.value?.focusPrompt();
}

const HELP: [string, string][] = [
  ["Drag on empty canvas", "draw a layer"],
  ["Drag a layer", "move it — Shift keeps it on one axis"],
  ["Drag a handle", "resize — Shift keeps the shape"],
  ["Shift / Ctrl + click", "add to the selection"],
  ["Shift + drag on empty", "select several"],
  ["Double-click a layer", "edit its prompt"],
  ["Arrows", "nudge — Shift for more"],
  ["Delete", "remove the selected layers"],
  ["V · B · E", "select · paint the shape · erase it"],
  ["[  ]  or wheel", "brush size, while painting"],
  ["Alt while dragging", "no snapping"],
];

/** Over the canvas while painting, the wheel sizes the brush; anywhere else it scrolls a panel
 *  or goes to the graph (see `wheel`). */
function onStageWheel(e: WheelEvent) {
  if (tool.value === "select") return;
  e.preventDefault();
  e.stopPropagation();
  const step = e.deltaY < 0 ? 1.12 : 1 / 1.12;
  brushSize.value = Math.min(0.3, Math.max(0.01, brushSize.value * step));
}

const wheel = useWheelClaim({
  ours: (_e, target) => tool.value !== "select" && !!target.closest(".fs-stage"),
  focusTarget: () => wheel.el.value?.querySelector<HTMLElement>(".zbc") ?? null,
});

function onKey(e: KeyboardEvent) {
  if (tool.value === "select" || (e.key !== "[" && e.key !== "]")) return;
  const step = e.key === "]" ? 1.2 : 1 / 1.2;
  brushSize.value = Math.min(0.3, Math.max(0.01, brushSize.value * step));
  e.preventDefault();
  e.stopPropagation();
}

// --- the layer list ---------------------------------------------------------------------------

const layerItems = computed<LayerItem[]>(() => [
  ...[...scene.value.layers].reverse().map((l) => ({
    id: l.id,
    label: l.name,
    color: l.color,
    sub: summarize(l),
    thumb: l.refs[0] ? thumbUrl(l.refs[0].ref, l.refs[0].type) : undefined,
    hidden: !l.on,
    locked: l.locked,
  })),
  {
    id: BACKGROUND_ID,
    label: "Scene",
    sub: scene.value.prompt.trim().slice(0, 48) || "prompt, background, settings",
    color: "var(--zen-surface-2, #26262d)",
    hidden: !scene.value.background.on,
    pinned: true,
  },
]);

/** The Scene row stands for "nothing selected" — the inspector shows the scene then. */
const listSelection = computed(() => (selected.value.length ? selected.value : [BACKGROUND_ID]));
function selectFromList(ids: string[]) {
  selected.value = ids.filter((id) => id !== BACKGROUND_ID);
}

/** `index` is a display position, top-first; layers are stored bottom-first. */
function reorder(id: string, index: number) {
  const layers = [...scene.value.layers];
  const from = layers.findIndex((l) => l.id === id);
  if (from < 0) return;
  const [moved] = layers.splice(from, 1);
  layers.splice(Math.max(0, layers.length - index), 0, moved!);
  scene.value = { ...scene.value, layers };
  commit();
}
function toggleOn(id: string) {
  if (id === BACKGROUND_ID) {
    patchBackground({ on: !scene.value.background.on });
    return;
  }
  updateLayers((l) => (l.id === id ? { ...l, on: !l.on } : l));
}
function toggleLock(id: string) {
  updateLayers((l) => (l.id === id ? { ...l, locked: !l.locked } : l));
}

// --- the inspector ----------------------------------------------------------------------------

const currentLayer = computed<FrescoLayer | null>(() =>
  selected.value.length === 1
    ? (scene.value.layers.find((l) => l.id === selected.value[0]) ?? null)
    : null,
);
function patchCurrent(fields: Partial<FrescoLayer>) {
  const id = currentLayer.value?.id;
  if (id) updateLayers((l) => (l.id === id ? { ...l, ...fields } : l));
}

// --- the trace image (a node property: drawing over it must not re-run anything) ---------------

const BACKDROP_KEY = "nynxzFrescoBackdrop";
interface Backdrop {
  ref: string;
  type: SourceType;
  opacity: number;
  /** Width and height of a source image Fresco Apply sent back. */
  size?: [number, number];
}
function readBackdrop(): Backdrop | null {
  const raw = props.node?.properties?.[BACKDROP_KEY] as Partial<Backdrop> | undefined;
  return raw?.ref
    ? {
        ref: raw.ref,
        type: (raw.type ?? "input") as SourceType,
        opacity: raw.opacity ?? 0.6,
        size: raw.size,
      }
    : null;
}
const backdrop = ref<Backdrop | null>(readBackdrop());
const backdropBrowse = ref(false);
const backdropUrl = computed(() =>
  backdrop.value ? thumbUrl(backdrop.value.ref, backdrop.value.type) : null,
);
const backdropOpacity = computed({
  get: () => backdrop.value?.opacity ?? 0.6,
  set: (opacity: number) => {
    if (backdrop.value) setBackdrop({ ...backdrop.value, opacity });
  },
});
function setBackdrop(next: Backdrop | null) {
  backdrop.value = next;
  if (props.node?.properties) {
    // eslint-disable-next-line vue/no-mutating-props -- node.properties is ComfyUI's own store for what the backend must not see.
    props.node.properties[BACKDROP_KEY] = next;
  }
  props.node?.graph?.setDirtyCanvas?.(true, true);
}
function pickBackdrop(refs: string[]) {
  const first = refs[0];
  if (!first) return;
  const [type, ...rest] = first.split("/");
  setBackdrop({
    ref: rest.join("/"),
    type: (["input", "temp", "output"].includes(type!) ? type : "input") as SourceType,
    opacity: backdrop.value?.opacity ?? 0.6,
  });
}

const sourceSize = computed(() => backdrop.value?.size ?? null);
const mismatch = computed(() => {
  const size = sourceSize.value;
  if (!size) return false;
  return Math.abs(size[0] / size[1] - scene.value.width / scene.value.height) > 0.01;
});
function matchSource() {
  const size = sourceSize.value;
  if (size) setSize(size[0], size[1]);
}

/** Fresco Apply's run, carrying this node's id: its source image becomes the trace image. */
function isMine(id: unknown): boolean {
  const mine = String(props.node?.id ?? "");
  const theirs = String(id ?? "");
  return !!mine && (theirs === mine || theirs.endsWith(`:${mine}`));
}
function onExecuted(e: Event) {
  const output = (e as CustomEvent).detail?.output as Record<string, unknown[]> | undefined;
  if (!output?.fresco_node?.some(isMine)) return;
  const image = output.fresco_source?.[0] as
    { filename: string; subfolder?: string; type?: string } | undefined;
  if (!image) return;
  const size = output.fresco_size as [number, number] | undefined;
  setBackdrop({
    ref: image.subfolder ? `${image.subfolder}/${image.filename}` : image.filename,
    type: (image.type ?? "temp") as SourceType,
    opacity: backdrop.value?.opacity ?? 0.6,
    size: size && size.length === 2 ? [Number(size[0]), Number(size[1])] : undefined,
  });
}
onMounted(() => api.addEventListener("executed", onExecuted));
onBeforeUnmount(() => api.removeEventListener("executed", onExecuted));
</script>

<style scoped>
/* rail | canvas | side, with the status bar over the canvas so all three end on one line. Every track is a fixed size or a
   share of the node, never its content, so selecting or typing can't move anything. */
.fs {
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr) 300px;
  grid-template-rows: 30px minmax(0, 1fr);
  gap: 6px 8px;
  width: 100%;
  height: 100%;
  min-height: 0;
  color: var(--zen-text, #e5e5ea);
  font-size: 12px;
}

.fs-rail {
  grid-row: 1 / 3;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 4px 0;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 7px);
  background: color-mix(in srgb, var(--zen-text, #fff) 2%, transparent);
}
.fs-rail-sep {
  width: 18px;
  height: 1px;
  margin: 2px 0;
  background: var(--zen-border, #34343c);
}
.fs-menu {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
}

.fs-stage {
  grid-column: 2;
  grid-row: 2;
  min-width: 0;
  min-height: 0;
}
.fs-hint-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}
.fs-hint-empty i {
  font-size: 28px;
  opacity: 0.6;
}

.fs-status {
  grid-column: 2;
  grid-row: 1;
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
  padding: 0 4px;
  color: var(--zen-muted, #9aa0aa);
  font-size: 11px;
}
.fs-help {
  display: grid;
  grid-template-columns: auto auto;
  gap: 4px 14px;
  margin: 0;
  padding: 10px 12px;
  font-size: 11px;
}
.fs-help dt {
  color: var(--zen-text, #e5e5ea);
  white-space: nowrap;
}
.fs-help dd {
  margin: 0;
  color: var(--zen-muted, #9aa0aa);
}
.fs-tip {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.fs-link {
  padding: 0;
  border: 0;
  background: none;
  color: var(--zen-accent, #6366f1);
  font: inherit;
  cursor: pointer;
}
.fs-ctl {
  flex: none;
  display: flex;
  align-items: center;
  gap: 6px;
  width: 104px;
  transition: opacity 0.12s ease;
}
.fs-ctl.dim {
  opacity: 0.4;
}
.fs-chip {
  flex: none;
  height: 22px;
  padding: 0 8px;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: 999px;
  background: var(--zen-input, #1b1b20);
  color: var(--zen-text, #e5e5ea);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  cursor: pointer;
}

.fs-side {
  grid-column: 3;
  grid-row: 1 / 3;
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 0;
}
.fs-pane {
  display: flex;
  flex-direction: column;
  min-height: 0;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 7px);
  background: color-mix(in srgb, var(--zen-text, #fff) 2%, transparent);
  overflow: hidden;
}
.fs-layers {
  flex: 0 0 34%;
}
.fs-inspect {
  flex: 1 1 0;
}
.fs-head {
  flex: none;
  display: flex;
  align-items: center;
  gap: 6px;
  height: 34px;
  padding: 0 8px;
  border-bottom: 1px solid var(--zen-border, #34343c);
}
.fs-title {
  flex: 1;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--zen-muted, #9aa0aa);
}
.fs-count {
  padding: 0 6px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--zen-text, #fff) 8%, transparent);
  color: var(--zen-muted, #9aa0aa);
  font-size: 10px;
}
.fs-swatch {
  flex: none;
  width: 12px;
  height: 12px;
  border-radius: 3px;
}
.fs-name {
  flex: 1;
  min-width: 0;
}
.fs-body {
  flex: 1 1 0;
  min-height: 0;
  padding: 6px;
}
.fs-form {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.fs-form > * {
  flex: none;
}
.fs-label {
  display: flex;
  align-items: baseline;
  gap: 6px;
  margin-top: 6px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--zen-muted, #9aa0aa);
}
.fs-label:first-child {
  margin-top: 0;
}
.fs-label-note {
  font-weight: 400;
  letter-spacing: 0;
  text-transform: none;
}
.fs-grid {
  display: grid;
  grid-template-columns: 62px minmax(0, 1fr);
  align-items: center;
  gap: 6px;
  color: var(--zen-muted, #9aa0aa);
}
.fs-section {
  margin-top: 12px;
  padding-top: 8px;
  border-top: 1px solid var(--zen-border, #34343c);
}
.fs-field {
  display: grid;
  grid-template-columns: 58px 1fr 42px;
  align-items: center;
  gap: 6px;
  color: var(--zen-muted, #9aa0aa);
}
.fs-field.wide {
  grid-template-columns: 58px 1fr;
}
.fs-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
}
.fs-note {
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.fs-note kbd {
  padding: 0 4px;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: 3px;
  font-family: var(--zen-mono, monospace);
  font-size: 10px;
}
.fs-pad {
  margin: 0;
  padding: 4px 2px;
}
</style>
