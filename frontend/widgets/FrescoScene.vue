<template>
  <div class="fs">
    <div class="fs-bar">
      <ZenToggleGroup
        :model-value="tool"
        :options="TOOLS"
        @update:model-value="tool = $event as BoxTool"
      />
      <label v-if="tool !== 'select'" class="fs-brush" title="Brush size">
        <i class="mdi mdi-circle-medium" />
        <ZenSlider v-model="brushSize" :min="0.01" :max="0.3" :step="0.005" />
      </label>
      <span class="fs-sep" />
      <ZenPopover placement="bottom-start" role="dialog">
        <template #trigger="{ toggle }">
          <ZenIconButton
            size="sm"
            icon="mdi mdi-image-outline"
            :active="!!backdropUrl"
            title="Trace image — something to draw over. Not sent anywhere."
            @click="toggle"
          />
        </template>
        <div class="fs-menu">
          <ZenButton sm icon="mdi mdi-view-grid-outline" @click="backdropBrowse = true">
            Pick an image…
          </ZenButton>
          <label class="fs-menu-row">
            opacity
            <ZenSlider v-model="backdropOpacity" :min="0.1" :max="1" :step="0.05" />
          </label>
          <ZenButton v-if="backdropUrl" sm icon="mdi mdi-close" @click="setBackdrop(null)">
            Remove
          </ZenButton>
        </div>
      </ZenPopover>
      <ZenIconButton
        size="sm"
        icon="mdi mdi-plus-box-outline"
        title="Add a layer in the middle of the canvas (or drag on the canvas to draw one)"
        @click="createLayer({ x: 0.3, y: 0.3, w: 0.4, h: 0.4 })"
      />
      <span class="fs-grow" />
      <ZenPopover placement="bottom-end" role="dialog">
        <template #trigger="{ toggle }">
          <button type="button" class="fs-chip" title="Canvas size" @click="toggle">
            <i class="mdi mdi-aspect-ratio" />{{ scene.width }}×{{ scene.height }}
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
    </div>

    <div class="fs-main">
      <div class="fs-stage">
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
          <template #empty>
            <div class="fs-hint">
              <i class="mdi mdi-gesture-tap-box" />
              <span>Drag to draw a layer</span>
              <small>Give it a prompt, references and LoRAs on the right</small>
            </div>
          </template>
        </ZenBoxCanvas>
      </div>

      <ZenScroll class="fs-panel" data-capture-wheel="true">
        <section class="fs-sec">
          <div class="fs-h">Scene</div>
          <ZenInput
            type="textarea"
            :rows="2"
            autosize
            sm
            :model-value="scene.prompt"
            placeholder="The whole picture — the conditioning output"
            @update:model-value="patchScene({ prompt: String($event ?? '') })"
          />
        </section>

        <section class="fs-sec">
          <div class="fs-h">
            Layers
            <span class="fs-count">{{ scene.layers.length }}</span>
          </div>
          <ZenLayerList
            v-model:selected="selected"
            :items="layerItems"
            @reorder="reorder"
            @toggle-hidden="toggleOn"
            @toggle-locked="toggleLock"
          />
        </section>

        <section v-if="current" class="fs-sec fs-inspect">
          <div class="fs-h">
            <span
              class="fs-swatch"
              :style="{ background: currentLayer?.color ?? 'var(--zen-muted)' }"
            />
            <ZenInput
              v-if="currentLayer"
              class="fs-name"
              sm
              :model-value="currentLayer.name"
              @update:model-value="patchCurrent({ name: String($event ?? '') })"
            />
            <span v-else class="fs-name">Background</span>
            <ZenIconButton
              v-if="currentLayer"
              size="sm"
              danger
              icon="mdi mdi-delete-outline"
              title="Delete this layer"
              @click="removeLayers([currentLayer.id])"
            />
          </div>

          <ZenInput
            ref="promptInput"
            type="textarea"
            :rows="2"
            autosize
            sm
            :model-value="current.prompt"
            :placeholder="
              currentLayer
                ? 'What goes here. Empty: the scene prompt, read against the references'
                : 'What fills the rest of the frame'
            "
            @update:model-value="patchCurrent({ prompt: String($event ?? '') })"
          />

          <div class="fs-label">References</div>
          <FrescoRefs
            :model-value="current.refs"
            @update:model-value="patchCurrent({ refs: $event })"
          />

          <div class="fs-label">LoRAs</div>
          <FrescoLoras
            :model-value="current.loras"
            @update:model-value="patchCurrent({ loras: $event })"
          />

          <template v-if="currentLayer">
            <div class="fs-label">Shape</div>
            <label class="fs-field" title="Soften the edge, in canvas pixels">
              <span>feather</span>
              <ZenSlider
                :model-value="currentLayer.feather"
                :min="0"
                :max="128"
                :step="1"
                @update:model-value="patchCurrent({ feather: $event })"
              />
              <span class="fs-val">{{ currentLayer.feather }}px</span>
            </label>
            <div class="fs-maskrow">
              <span class="fs-note">
                {{ currentLayer.mask ? "Painted shape" : "Whole box" }} — paint with <kbd>B</kbd>,
                erase with <kbd>E</kbd>
              </span>
              <ZenButton
                v-if="currentLayer.mask"
                sm
                icon="mdi mdi-eraser-variant"
                @click="setMask(currentLayer.id, null)"
              >
                Whole box
              </ZenButton>
            </div>
          </template>

          <div class="fs-label">Edit</div>
          <label
            class="fs-field"
            title="How much of a source image this may rewrite — read by Region Latent and the SoftPaint Sampler. Ignored when generating from scratch."
          >
            <span>denoise</span>
            <ZenSlider
              :model-value="current.denoise"
              :min="0"
              :max="1"
              :step="0.01"
              @update:model-value="patchCurrent({ denoise: $event })"
            />
            <span class="fs-val">{{ current.denoise.toFixed(2) }}</span>
          </label>
        </section>
        <p v-else-if="selected.length > 1" class="fs-note fs-multi">
          {{ selected.length }} layers selected — drag to move them together, Delete to remove.
        </p>
      </ZenScroll>
    </div>

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

/** The body is the node's: canvas on the left, layers on the right, toolbar on top. */
export const widgetOptions: WidgetOptions = { minHeight: 480, fill: true, default: {} };

export const nodeDef: Omit<NodeDef, "widgets"> = {
  is: nodeId("Fresco"),
  minSize: [920, 640],
};
</script>

<script setup lang="ts">
/**
 * Fresco — compose a scene as layers on a canvas.
 *
 * Every layer is a box with what belongs in it: a prompt, reference images, LoRAs, a feather and
 * a denoise. Paint inside a box to give it a shape. The background, pinned under every layer,
 * carries the same for the rest of the frame.
 *
 * The whole scene is the widget value (see lib/frescoScene.ts), committed once per finished edit
 * — a drag writes it when it ends, not on every move. The trace image is a node property: it is
 * there to draw over and must not re-run the graph when it changes.
 */
import { computed, nextTick, ref } from "vue";
import {
  ZenBoxCanvas,
  ZenButton,
  ZenIconButton,
  ZenInput,
  ZenLayerList,
  ZenMediaPicker,
  ZenPopover,
  ZenResolution,
  ZenScroll,
  ZenSlider,
  ZenToggleGroup,
  type BoxTool,
  type LayerItem,
  type ZenBox,
} from "@nynxz/zenkit-ui";

import FrescoLoras from "@/components/fresco/FrescoLoras.vue";
import FrescoRefs from "@/components/fresco/FrescoRefs.vue";
import { thumbUrl, type SourceType } from "@/lib/fusionApi";
import {
  BACKGROUND_ID,
  newLayer,
  parseScene,
  summarize,
  type FrescoContent,
  type FrescoLayer,
  type FrescoScene,
} from "@/lib/frescoScene";

const props = defineProps<{
  widget?: { value: unknown; callback?: (v: unknown) => void };
  node?: {
    graph?: { setDirtyCanvas?: (a: boolean, b: boolean) => void };
    properties?: Record<string, unknown>;
  };
}>();

const TOOLS = [
  { value: "select", icon: "mdi mdi-cursor-default-outline", title: "Select, move, draw (V)" },
  { value: "brush", icon: "mdi mdi-brush", title: "Paint the layer's shape (B)" },
  { value: "erase", icon: "mdi mdi-eraser", title: "Erase from the layer's shape (E)" },
];

const scene = ref<FrescoScene>(parseScene(props.widget?.value));
const selected = ref<string[]>([]);
const tool = ref<BoxTool>("select");
const brushSize = ref(0.06);
const promptInput = ref<{ $el?: HTMLElement } | null>(null);

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
function setSize(width: number, height: number) {
  patchScene({ width: Math.round(width), height: Math.round(height) });
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
  // Live while dragging; `commit` (from the canvas, at the end of the gesture) persists.
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

function createLayer(rect: { x: number; y: number; w: number; h: number }) {
  const layer = newLayer(rect, scene.value.layers);
  scene.value = { ...scene.value, layers: [...scene.value.layers, layer] };
  selected.value = [layer.id];
  commit();
}

function setMask(id: string, mask: string | null) {
  scene.value = {
    ...scene.value,
    layers: scene.value.layers.map((l) => (l.id === id ? { ...l, mask } : l)),
  };
  commit();
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
  promptInput.value?.$el?.querySelector?.("textarea")?.focus();
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
    label: "Background",
    sub: summarize(scene.value.background),
    hidden: !scene.value.background.on,
    pinned: true,
  },
]);

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
    const bg = scene.value.background;
    patchScene({ background: { ...bg, on: !bg.on } });
    return;
  }
  scene.value = {
    ...scene.value,
    layers: scene.value.layers.map((l) => (l.id === id ? { ...l, on: !l.on } : l)),
  };
  commit();
}
function toggleLock(id: string) {
  scene.value = {
    ...scene.value,
    layers: scene.value.layers.map((l) => (l.id === id ? { ...l, locked: !l.locked } : l)),
  };
  commit();
}

// --- the inspector ----------------------------------------------------------------------------

const currentLayer = computed<FrescoLayer | null>(() =>
  selected.value.length === 1
    ? (scene.value.layers.find((l) => l.id === selected.value[0]) ?? null)
    : null,
);
const current = computed<FrescoContent | null>(() =>
  selected.value.length === 1 && selected.value[0] === BACKGROUND_ID
    ? scene.value.background
    : currentLayer.value,
);

function patchCurrent(fields: Partial<FrescoLayer>) {
  if (selected.value[0] === BACKGROUND_ID) {
    patchScene({ background: { ...scene.value.background, ...fields } });
    return;
  }
  const id = currentLayer.value?.id;
  if (!id) return;
  scene.value = {
    ...scene.value,
    layers: scene.value.layers.map((l) => (l.id === id ? { ...l, ...fields } : l)),
  };
  commit();
}

// --- the trace image (a node property: drawing over it must not re-run anything) ---------------

const BACKDROP_KEY = "nynxzFrescoBackdrop";
interface Backdrop {
  ref: string;
  type: SourceType;
  opacity: number;
}
function readBackdrop(): Backdrop | null {
  const raw = props.node?.properties?.[BACKDROP_KEY] as Partial<Backdrop> | undefined;
  return raw?.ref
    ? { ref: raw.ref, type: (raw.type ?? "input") as SourceType, opacity: raw.opacity ?? 0.6 }
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
</script>

<style scoped>
.fs {
  display: flex;
  flex-direction: column;
  gap: 6px;
  width: 100%;
  height: 100%;
  min-height: 0;
  color: var(--zen-text, #e5e5ea);
  font-size: 12px;
}
.fs-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  flex: none;
}
.fs-brush {
  display: flex;
  align-items: center;
  gap: 4px;
  width: 130px;
  color: var(--zen-muted, #9aa0aa);
}
.fs-sep {
  width: 1px;
  height: 18px;
  background: var(--zen-border, #34343c);
}
.fs-grow {
  flex: 1;
}
.fs-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 24px;
  padding: 0 8px;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: 999px;
  background: var(--zen-input, #1b1b20);
  color: var(--zen-text, #e5e5ea);
  font-size: 11px;
  cursor: pointer;
}
.fs-menu {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px;
  min-width: 200px;
}
.fs-menu-row {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--zen-muted, #9aa0aa);
}

.fs-main {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  grid-template-rows: minmax(0, 1fr);
  gap: 8px;
}
.fs-stage {
  min-width: 0;
  min-height: 0;
  height: 100%;
}
.fs-hint {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  text-align: center;
}
.fs-hint i {
  font-size: 28px;
  opacity: 0.6;
}
.fs-hint small {
  opacity: 0.7;
}
.fs-panel {
  min-height: 0;
  padding-right: 4px;
}
.fs-sec {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
  margin-bottom: 8px;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 7px);
  background: color-mix(in srgb, var(--zen-text, #fff) 2%, transparent);
}
.fs-h {
  display: flex;
  align-items: center;
  gap: 6px;
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
  font-size: 10px;
  letter-spacing: 0;
}
.fs-inspect .fs-h {
  text-transform: none;
  letter-spacing: 0;
  font-size: 12px;
  color: var(--zen-text, #e5e5ea);
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
.fs-label {
  margin-top: 4px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--zen-muted, #9aa0aa);
}
.fs-field {
  display: grid;
  grid-template-columns: 54px 1fr 40px;
  align-items: center;
  gap: 6px;
  color: var(--zen-muted, #9aa0aa);
}
.fs-val {
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--zen-text, #e5e5ea);
}
.fs-maskrow {
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
.fs-multi {
  padding: 8px;
}
</style>
