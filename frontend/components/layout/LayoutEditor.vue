<template>
  <div
    :ref="(el) => (wheel.el.value = el as HTMLElement | null)"
    class="le"
    :class="{ 'in-panel': inPanel }"
    v-bind="wheel.attrs"
    @keydown="onKey"
    @contextmenu="keepContextMenu"
  >
    <ZenSplit
      class="le-main"
      :panes="COLUMNS"
      :sizes="splits.columns"
      gutter="gap"
      @update:sizes="saveSplit('columns', $event)"
    >
      <template #pane-0>
        <div class="le-canvascol">
          <header class="le-toolbar" aria-label="Canvas tools">
            <template v-if="tools.length > 1">
              <ZenIconButton
                v-for="t in tools"
                :key="t.value"
                size="sm"
                :icon="t.icon"
                :active="!railActive && tool === t.value"
                :title="t.title"
                @click="pickTool(t.value)"
              />
              <span class="le-sep" />
            </template>
            <ZenButton
              sm
              variant="ghost"
              icon="mdi mdi-plus-box-outline"
              title="Add a box — or drag on the canvas"
              @click="addBox({ x: 0.3, y: 0.3, w: 0.4, h: 0.4 })"
            >
              Box
            </ZenButton>
            <slot name="add" />
            <span class="le-sep" />
            <ZenPopover placement="bottom-start" role="dialog">
              <template #trigger="{ toggle }">
                <ZenIconButton
                  size="sm"
                  icon="mdi mdi-image-outline"
                  :active="!!backdropUrl"
                  :title="backdropTitle"
                  @click="toggle"
                />
              </template>
              <div class="le-menu">
                <span class="le-menu-title">Trace image</span>
                <ZenButton sm icon="mdi mdi-view-grid-outline" @click="backdropBrowse = true">
                  {{ backdropUrl ? "Pick another…" : "Pick an image…" }}
                </ZenButton>
                <template v-if="backdropUrl">
                  <label class="le-opacity">
                    Opacity
                    <ZenSlider
                      :model-value="Math.round(backdropOpacity * 100)"
                      :min="10"
                      :max="100"
                      :step="5"
                      @update:model-value="backdropOpacity = $event / 100"
                    />
                    <span>{{ Math.round(backdropOpacity * 100) }}%</span>
                  </label>
                  <ZenButton
                    sm
                    :icon="backdrop?.pinned ? 'mdi mdi-pin' : 'mdi mdi-pin-outline'"
                    :title="
                      backdrop?.pinned
                        ? 'Pinned — runs won\'t replace it'
                        : 'Pin it, so the next run doesn\'t replace it'
                    "
                    @click="setBackdrop({ ...backdrop!, pinned: !backdrop!.pinned })"
                  >
                    {{ backdrop?.pinned ? "Unpin" : "Pin" }}
                  </ZenButton>
                  <ZenButton sm icon="mdi mdi-close" @click="setBackdrop(null)">Remove</ZenButton>
                </template>
              </div>
            </ZenPopover>
            <slot name="rail" />
            <label
              v-if="mode.paints && tool !== 'select'"
              class="le-ctl"
              title="Brush size in canvas pixels — [ and ]"
            >
              <i class="mdi mdi-brush" />
              <ZenNumber
                size="sm"
                :model-value="Math.round(brushSize * size.width)"
                :min="4"
                :max="Math.round(size.width * 0.4)"
                :step="2"
                :precision="0"
                @update:model-value="brushSize = Math.max(4, $event) / size.width"
              />
            </label>
            <span class="le-grow" />
            <button
              type="button"
              class="le-chip"
              :class="{ warn: mismatch }"
              :title="
                mismatch
                  ? `The trace image is ${sourceSize![0]}×${sourceSize![1]} — change the canvas in the Scene panel`
                  : 'Canvas size — set in the Scene panel'
              "
              @click="selected = []"
            >
              <i :class="mismatch ? 'mdi mdi-alert-outline' : 'mdi mdi-aspect-ratio'" />
              {{ size.width }}×{{ size.height }}
            </button>
          </header>

          <div
            class="le-stage"
            :class="{ dropping: dropOver }"
            @wheel="onStageWheel"
            @dragover="onStageDragOver"
            @dragleave="dropOver = false"
            @drop="onStageDrop"
          >
            <ZenBoxCanvas
              v-model:boxes="boxes"
              v-model:selected="selected"
              v-model:tool="tool"
              :aspect="size.width / size.height"
              :backdrop="backdropUrl"
              :backdrop-opacity="backdropOpacity"
              :brush-size="brushSize"
              :snap="1 / 64"
              :history="history"
              @create="addBox"
              @commit="commit"
              @update:mask="setMask"
              @delete="removeBoxes"
              @open="(id: string) => emit('open', id)"
              @menu="openMenu"
              @copy="copySelection"
              @paste="pasteAt"
              @duplicate="duplicate"
            >
              <template v-if="$slots.box" #box="face">
                <slot name="box" v-bind="face" />
              </template>
              <template #overlay>
                <slot name="overlay" />
              </template>
              <template #empty>
                <div class="le-hint-empty">
                  <i class="mdi mdi-gesture-tap-box" />
                  <span>Drag to draw a box</span>
                </div>
              </template>
            </ZenBoxCanvas>
          </div>
        </div>
      </template>
      <template #pane-1>
        <ZenSplit
          class="le-side"
          direction="vertical"
          :panes="ROWS"
          :sizes="splits.rows"
          gutter="gap"
          @update:sizes="saveSplit('rows', $event)"
        >
          <template #pane-0>
            <section class="le-pane">
              <header class="le-head">
                <span class="le-title">Layers</span>
                <span class="le-count">{{ layoutBoxes.length }}</span>
              </header>
              <ZenScroll class="le-body">
                <ZenLayerList
                  :selected="listSelection"
                  :items="layerItems"
                  @update:selected="selectFromList"
                  @reorder="reorder"
                  @toggle-hidden="toggleHidden"
                  @toggle-locked="(id: string) => patchBoxes([id], (b) => ({ locked: !b.locked }))"
                />
              </ZenScroll>
            </section>
          </template>
          <template #pane-1>
            <section class="le-pane">
              <header class="le-head">
                <template v-if="current">
                  <span class="le-swatch" :style="{ background: current.color }" />
                  <ZenInput
                    class="le-grow"
                    sm
                    :model-value="current.name"
                    @update:model-value="
                      patchBoxes([current.id], () => ({ name: String($event ?? '') }))
                    "
                  />
                  <slot name="element-actions" :id="current.id" />
                  <ZenIconButton
                    size="sm"
                    danger
                    icon="mdi mdi-delete-outline"
                    title="Delete this box"
                    @click="removeBoxes([current.id])"
                  />
                </template>
                <template v-else-if="selected.length > 1">
                  <span class="le-title">{{ selected.length }} boxes</span>
                  <ZenIconButton
                    size="sm"
                    danger
                    icon="mdi mdi-delete-outline"
                    title="Delete these boxes"
                    @click="removeBoxes([...selected])"
                  />
                </template>
                <span v-else class="le-title">Scene</span>
              </header>

              <ZenScroll class="le-body">
                <slot v-if="current" name="element" :id="current.id" />
                <slot v-else-if="selected.length > 1" name="multi" :ids="selected">
                  <p class="le-note">
                    Drag any of them on the canvas to move them together; the frame's handles scale
                    them.
                  </p>
                </slot>
                <div v-else class="le-scene">
                  <slot name="scene" />
                  <FrescoSection label="Canvas" :note="`${size.width}×${size.height}`">
                    <ZenResolution
                      :width="size.width"
                      :height="size.height"
                      :snap="sizeSnap"
                      :presets="[0.5, 1, 1.5, 2]"
                      :max-side="maxSide"
                      :max-mp="4"
                      @update:width="setSize($event, size.height)"
                      @update:height="setSize(size.width, $event)"
                    />
                    <p v-if="mismatch" class="le-note">
                      The trace image is {{ sourceSize![0] }}×{{ sourceSize![1] }}.
                      <button type="button" class="le-link" @click="matchSource">Match it</button>
                    </p>
                  </FrescoSection>
                </div>
              </ZenScroll>
            </section>
          </template>
        </ZenSplit>
      </template>
    </ZenSplit>

    <footer class="le-foot">
      <ZenIconButton
        size="sm"
        icon="mdi mdi-undo"
        :disabled="!layout.history.canUndo.value"
        title="Undo — Ctrl+Z"
        @click="undo"
      />
      <ZenIconButton
        size="sm"
        icon="mdi mdi-redo"
        :disabled="!layout.history.canRedo.value"
        title="Redo — Ctrl+Shift+Z"
        @click="redo"
      />
      <span class="le-sep" />
      <slot name="foot" />
      <ZenPopover placement="top-start" role="dialog">
        <template #trigger="{ toggle }">
          <ZenIconButton
            size="sm"
            icon="mdi mdi-help-circle-outline"
            title="Controls and shortcuts"
            @click="toggle"
          />
        </template>
        <dl class="le-help">
          <template v-for="row in helpRows" :key="row[0]">
            <dt>{{ row[0] }}</dt>
            <dd>{{ row[1] }}</dd>
          </template>
        </dl>
      </ZenPopover>
      <span class="le-grow" />
      <slot name="status" />
      <ZenButton
        sm
        variant="ghost"
        icon="mdi mdi-bookshelf"
        title="Save this layout, or load a saved one"
        @click="libraryOpen = true"
      >
        Layouts
      </ZenButton>
      <ZenPopover v-model:open="jsonOpen" placement="top-end" role="dialog">
        <template #trigger="{ toggle }">
          <ZenButton
            sm
            variant="ghost"
            icon="mdi mdi-code-json"
            title="Copy the layout or its prompt out, or paste one in"
            @click="toggle"
          >
            JSON
          </ZenButton>
        </template>
        <div class="le-json">
          <span class="le-menu-title">JSON</span>
          <div class="le-row">
            <ZenButton sm icon="mdi mdi-content-copy" @click="copyText(exportLayout(mode, scene))">
              Copy layout
            </ZenButton>
            <ZenButton
              v-if="mode.exportText"
              sm
              icon="mdi mdi-text-box-outline"
              @click="copyPrompt"
            >
              Copy prompt
            </ZenButton>
          </div>
          <ZenInput
            v-model="jsonText"
            type="textarea"
            :rows="7"
            sm
            :placeholder="`Paste a layout${mode.importText ? ' or the model\'s own JSON' : ''} here`"
          />
          <div class="le-row">
            <span class="le-note le-grow">{{ jsonNote }}</span>
            <ZenButton sm icon="mdi mdi-import" :disabled="!jsonText.trim()" @click="importText">
              Import
            </ZenButton>
          </div>
        </div>
      </ZenPopover>
      <ZenIconButton
        v-if="canPopOut"
        size="sm"
        icon="mdi mdi-open-in-new"
        title="Open in a ZenKit panel — bigger, dockable, and the node keeps a summary"
        @click="emit('popOut')"
      />
    </footer>

    <ZenContextMenu ref="menu" :items="menuItems" />
    <LayoutBrowser
      v-model:open="libraryOpen"
      :mode="mode"
      :scene="scene"
      :modes="hostModes"
      @load="loadSaved"
    />
    <ZenMediaPicker
      v-model:open="backdropBrowse"
      :library="mediaLibrary"
      :kinds="['image']"
      title="Trace image"
      @pick="pickBackdrop"
    />
  </div>
</template>

<script setup lang="ts">
/**
 * The layout editor every mode shares: a tool rail, the canvas with a status bar over it, and a
 * side column of layers over an inspector. What the boxes mean is the mode's (layout/mode.ts) and
 * what the inspector shows is the host's, through slots; everything about handling boxes is here —
 * draw, move, paint, undo, the right-click menu, copy and paste, saved layouts, JSON, a trace image.
 *
 * Slots: `rail` (extra tools), `box` and `overlay` (passed to the canvas), `element` and
 * `element-actions` (the selected box, by id), `multi` (several selected), `scene` (nothing selected).
 */
import { computed, ref, watch } from "vue";
import {
  ZenBoxCanvas,
  ZenButton,
  ZenContextMenu,
  ZenIconButton,
  ZenInput,
  ZenLayerList,
  ZenMediaPicker,
  ZenNumber,
  ZenPopover,
  ZenResolution,
  ZenScroll,
  ZenSlider,
  ZenSplit,
  boxAt,
  placeRects,
  useWheelClaim,
  type BoxRect,
  type BoxTool,
  type ContextMenuItem,
  type LayerItem,
  type ZenBox,
} from "@nynxz/zenkit-ui";

import { hasMediaDrop, hasZenPanels, mediaLibrary } from "@/framework";
import { keepContextMenu } from "@/lib/ownContextMenu";
import { thumbUrl, type SourceType } from "@/lib/fusionApi";
import { copiedFor, copyBoxes } from "@/layout/clipboard";
import { exportLayout, importLayout } from "@/layout/jsonIo";
import FrescoSection from "@/components/fresco/FrescoSection.vue";
import LayoutBrowser from "@/components/layout/LayoutBrowser.vue";
import { layoutMode, type AnyLayoutMode, type LayoutBox } from "@/layout/mode";
import type { SharedLayout } from "@/layout/store";

interface Backdrop {
  ref: string;
  type: SourceType;
  opacity: number;
  /** Width and height of a source image a run sent back. */
  size?: [number, number];
  /** Kept when a run sends its image back. */
  pinned?: boolean;
}

const props = withDefaults(
  defineProps<{
    mode: AnyLayoutMode;
    /** Every mode this node can be in; a saved layout of any of them loads as it is. */
    modes?: AnyLayoutMode[];
    layout: SharedLayout<unknown>;
    node?: {
      id?: string | number;
      graph?: { setDirtyCanvas?: (a: boolean, b: boolean) => void };
      properties?: Record<string, unknown>;
    };
    /** Shown in a ZenKit panel rather than on the node. */
    inPanel?: boolean;
    /** Node property names, so each node type keeps its own (and old workflows keep theirs). */
    propertyPrefix?: string;
    /** More rows for the controls popover. */
    help?: [string, string][];
    /** A rail tool of the host's is active, so none of the canvas tools shows as picked. */
    railActive?: boolean;
    /** What the canvas shows for a box's mask, when it differs from the mask itself. */
    maskView?: (id: string) => string | null;
    /** Whether a box is drawn clipped to its mask. */
    maskClips?: (id: string) => boolean;
    /** Media can be dropped on the canvas (handled by the host's `drop`). */
    acceptsDrops?: boolean;
    backdropTitle?: string;
    sizeSnap?: number;
    maxSide?: number;
  }>(),
  {
    node: undefined,
    modes: undefined,
    inPanel: false,
    propertyPrefix: "nynxzLayout",
    help: () => [],
    railActive: false,
    maskView: undefined,
    maskClips: undefined,
    acceptsDrops: false,
    backdropTitle: "Trace image — something to draw over",
    sizeSnap: 16,
    maxSide: 2048,
  },
);
const emit = defineEmits<{
  popOut: [];
  open: [id: string];
  /** A canvas tool was picked on the rail (even the one already picked). */
  pickTool: [tool: BoxTool];
  /** Media dropped at `x`,`y` (0..1), over box `hit` if any. */
  drop: [event: DragEvent, at: { x: number; y: number }, hit: string | null];
}>();

const selected = defineModel<string[]>("selected", { default: () => [] });
const tool = defineModel<BoxTool>("tool", { default: "select" });

const canPopOut = !props.inPanel && hasZenPanels();
const hostModes = computed(() => props.modes ?? [props.mode]);
const scene = props.layout.scene;
const size = computed(() => props.mode.size(scene.value));
const layoutBoxes = computed<LayoutBox[]>(() => props.mode.boxes(scene.value));
const brushSize = ref(0.06);

const tools = computed(() =>
  props.mode.paints
    ? [
        {
          value: "select" as BoxTool,
          icon: "mdi mdi-cursor-default-outline",
          title: "Select, move, draw — V",
        },
        { value: "brush" as BoxTool, icon: "mdi mdi-brush", title: "Paint the box's shape — B" },
        {
          value: "erase" as BoxTool,
          icon: "mdi mdi-eraser",
          title: "Erase from the box's shape — E",
        },
      ]
    : [
        {
          value: "select" as BoxTool,
          icon: "mdi mdi-cursor-default-outline",
          title: "Select, move, draw — V",
        },
      ],
);

const HELP: [string, string][] = [
  ["Drag on empty canvas", "draw a box"],
  ["Drag a box", "move it — Shift keeps it on one axis"],
  ["Drag a handle", "resize — Shift keeps the shape"],
  ["Shift / Ctrl + click", "add to the selection"],
  ["Shift + drag on empty", "select several"],
  ["Double-click a box", "edit it"],
  ["Right-click", "copy, paste, order, hide, lock, delete"],
  ["Ctrl+Z · Ctrl+Shift+Z", "undo · redo"],
  ["Ctrl+C · X · V · D", "copy · cut · paste at the pointer · duplicate"],
  ["Arrows", "nudge — Shift for more"],
  ["Delete", "remove the selected boxes"],
  ["Alt while dragging", "no snapping"],
];
const helpRows = computed<[string, string][]>(() => [
  ...HELP,
  ...(props.mode.paints
    ? ([
        ["V · B · E", "select · paint the shape · erase it"],
        ["[  ]  or wheel", "brush size, while painting"],
      ] as [string, string][])
    : []),
  ...props.help,
]);

// --- editing ------------------------------------------------------------------------------------

function commit() {
  props.layout.commit();
  props.node?.graph?.setDirtyCanvas?.(true, true);
}
function update(next: unknown) {
  scene.value = next;
  commit();
}
function undo(): boolean {
  const stepped = props.layout.history.undo();
  props.node?.graph?.setDirtyCanvas?.(true, true);
  return stepped;
}
function redo(): boolean {
  const stepped = props.layout.history.redo();
  props.node?.graph?.setDirtyCanvas?.(true, true);
  return stepped;
}
const history = { undo, redo };
function pickTool(t: BoxTool) {
  tool.value = t;
  emit("pickTool", t);
}

function setSize(width: number, height: number) {
  update(props.mode.resize(scene.value, width, height));
}

/** Change boxes `ids` by `fn`, then persist. */
function patchBoxes(ids: string[], fn: (b: LayoutBox) => Partial<LayoutBox>) {
  const want = new Set(ids);
  update(
    props.mode.setBoxes(
      scene.value,
      layoutBoxes.value.map((b) => (want.has(b.id) ? { ...b, ...fn(b) } : b)),
    ),
  );
}

const boxes = computed<ZenBox[]>({
  get: () =>
    layoutBoxes.value.map((b) => ({
      id: b.id,
      x: b.x,
      y: b.y,
      w: b.w,
      h: b.h,
      color: b.color,
      label: b.name,
      hidden: b.hidden,
      locked: b.locked,
      mask: b.mask,
      maskView: tool.value === "select" ? (props.maskView?.(b.id) ?? null) : null,
      maskClips: props.maskClips?.(b.id) ?? true,
    })),
  set: (next) => {
    const rects = new Map(next.map((b) => [b.id, b]));
    scene.value = props.mode.setBoxes(
      scene.value,
      layoutBoxes.value.map((b) => {
        const r = rects.get(b.id);
        return r ? { ...b, x: r.x, y: r.y, w: r.w, h: r.h } : b;
      }),
    );
  },
});

/** Add a box over `rect`, select it and return its id. */
function addBox(rect: BoxRect): string {
  const { scene: next, id } = props.mode.add(scene.value, rect);
  update(next);
  selected.value = [id];
  tool.value = "select";
  return id;
}
function setMask(id: string, mask: string | null) {
  patchBoxes([id], () => ({ mask }));
}
function removeBoxes(ids: string[]) {
  const gone = new Set(ids);
  update(
    props.mode.setBoxes(
      scene.value,
      layoutBoxes.value.filter((b) => !gone.has(b.id)),
    ),
  );
  selected.value = selected.value.filter((id) => !gone.has(id));
}
function toggleHidden(id: string) {
  if (id === SCENE_ROW) {
    if (props.mode.toggleScene) update(props.mode.toggleScene(scene.value));
    return;
  }
  patchBoxes([id], (b) => ({ hidden: !b.hidden }));
}

const rectsOf = (ids: string[]) =>
  layoutBoxes.value.filter((b) => ids.includes(b.id)).map(({ x, y, w, h }) => ({ x, y, w, h }));

function copySelection(ids: string[]) {
  const ordered = layoutBoxes.value.filter((b) => ids.includes(b.id)).map((b) => b.id);
  copyBoxes(props.mode.id, props.mode.copy(scene.value, ordered), rectsOf(ordered));
}
function landCopies(items: unknown[], rects: BoxRect[]) {
  const { scene: next, ids } = props.mode.paste(scene.value, items, rects);
  update(next);
  selected.value = ids;
}
function pasteAt(at: [number, number] | null) {
  const copied = copiedFor(props.mode.id);
  if (copied) landCopies(copied.items, placeRects(copied.rects, at));
}
function duplicate(ids: string[]) {
  const ordered = layoutBoxes.value.filter((b) => ids.includes(b.id)).map((b) => b.id);
  landCopies(props.mode.copy(scene.value, ordered), placeRects(rectsOf(ordered), null));
}

/** Move boxes `ids` in the stack: +1 up, -1 down, Infinity to the top, -Infinity to the bottom. */
function restack(ids: string[], by: number) {
  const moving = new Set(ids);
  const order = [...layoutBoxes.value];
  if (!Number.isFinite(by)) {
    const kept = order.filter((b) => !moving.has(b.id));
    const moved = order.filter((b) => moving.has(b.id));
    update(props.mode.setBoxes(scene.value, by > 0 ? [...kept, ...moved] : [...moved, ...kept]));
    return;
  }
  const indices = order.map((b, i) => (moving.has(b.id) ? i : -1)).filter((i) => i >= 0);
  for (const i of by > 0 ? indices.reverse() : indices) {
    const j = i + by;
    if (j < 0 || j >= order.length || moving.has(order[j]!.id)) continue;
    [order[i], order[j]] = [order[j]!, order[i]!];
  }
  update(props.mode.setBoxes(scene.value, order));
}

// --- right-click --------------------------------------------------------------------------------

const menu = ref<{ show: (e: MouseEvent) => void } | null>(null);
const menuTarget = ref<{ ids: string[]; at: [number, number] | null }>({ ids: [], at: null });
function openMenu(e: MouseEvent, ids: string[]) {
  menuTarget.value = { ids, at: stagePoint(e) };
  menu.value?.show(e);
}
const menuItems = computed<ContextMenuItem[]>(() => {
  const { ids, at } = menuTarget.value;
  const canPaste = !!copiedFor(props.mode.id);
  if (!ids.length) {
    return [
      {
        label: "Paste",
        icon: "mdi mdi-content-paste",
        hint: "Ctrl+V",
        disabled: !canPaste,
        run: () => pasteAt(at),
      },
      {
        label: "Add a box here",
        icon: "mdi mdi-plus-box-outline",
        run: () => {
          const [x, y] = at ?? [0.5, 0.5];
          addBox(placeRects([{ x: 0, y: 0, w: 0.3, h: 0.3 }], [x, y])[0]!);
        },
      },
      {
        label: "Select all",
        icon: "mdi mdi-select-all",
        hint: "Ctrl+A",
        run: () => (selected.value = layoutBoxes.value.filter((b) => !b.hidden).map((b) => b.id)),
      },
      "-",
      {
        label: "Undo",
        icon: "mdi mdi-undo",
        hint: "Ctrl+Z",
        disabled: !props.layout.history.canUndo.value,
        run: undo,
      },
      {
        label: "Redo",
        icon: "mdi mdi-redo",
        hint: "Ctrl+Shift+Z",
        disabled: !props.layout.history.canRedo.value,
        run: redo,
      },
    ];
  }
  const chosen = layoutBoxes.value.filter((b) => ids.includes(b.id));
  const allHidden = chosen.every((b) => b.hidden);
  const allLocked = chosen.every((b) => b.locked);
  return [
    {
      label: "Duplicate",
      icon: "mdi mdi-content-duplicate",
      hint: "Ctrl+D",
      run: () => duplicate(ids),
    },
    { label: "Copy", icon: "mdi mdi-content-copy", hint: "Ctrl+C", run: () => copySelection(ids) },
    {
      label: "Cut",
      icon: "mdi mdi-content-cut",
      hint: "Ctrl+X",
      run: () => {
        copySelection(ids);
        removeBoxes(ids);
      },
    },
    {
      label: "Paste",
      icon: "mdi mdi-content-paste",
      hint: "Ctrl+V",
      disabled: !canPaste,
      run: () => pasteAt(at),
    },
    "-",
    {
      label: "Bring to front",
      icon: "mdi mdi-arrange-bring-to-front",
      run: () => restack(ids, Infinity),
    },
    { label: "Bring forward", icon: "mdi mdi-arrange-bring-forward", run: () => restack(ids, 1) },
    { label: "Send backward", icon: "mdi mdi-arrange-send-backward", run: () => restack(ids, -1) },
    {
      label: "Send to back",
      icon: "mdi mdi-arrange-send-to-back",
      run: () => restack(ids, -Infinity),
    },
    "-",
    {
      label: allHidden ? "Show" : "Hide",
      icon: allHidden ? "mdi mdi-eye-outline" : "mdi mdi-eye-off-outline",
      run: () => patchBoxes(ids, () => ({ hidden: !allHidden })),
    },
    {
      label: allLocked ? "Unlock" : "Lock",
      icon: allLocked ? "mdi mdi-lock-open-outline" : "mdi mdi-lock-outline",
      run: () => patchBoxes(ids, () => ({ locked: !allLocked })),
    },
    "-",
    {
      label: ids.length > 1 ? `Delete ${ids.length} boxes` : "Delete",
      icon: "mdi mdi-delete-outline",
      hint: "Del",
      danger: true,
      run: () => removeBoxes(ids),
    },
  ];
});

// --- saved layouts ---------------------------------------------------------------------------

const libraryOpen = ref(false);
/** A saved layout, as the node can show it: as saved when one of `modes` is its own, else
 *  carried over through the common layout. */
function loadSaved(modeId: string, saved: unknown) {
  const native = hostModes.value.find((m) => m.id === modeId);
  const from = layoutMode(modeId);
  const next = native
    ? native.read(saved)
    : from
      ? props.mode.fromCommon(from.toCommon(from.read(saved)))
      : null;
  if (next == null) return;
  props.layout.replace(next);
  selected.value = [];
  props.node?.graph?.setDirtyCanvas?.(true, true);
}

// --- JSON ---------------------------------------------------------------------------------------

const jsonOpen = ref(false);
const jsonText = ref("");
const jsonNote = ref("");
watch(jsonOpen, () => (jsonNote.value = ""));
async function copyPrompt() {
  try {
    await copyText(await props.mode.exportText!(scene.value));
  } catch (e) {
    jsonNote.value = `Couldn't write the prompt: ${(e as Error).message}`;
  }
}
async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    jsonNote.value = "Copied.";
  } catch {
    jsonText.value = text;
    jsonNote.value = "Couldn't reach the clipboard — it's in the box to copy by hand.";
  }
}
function importText() {
  const next = importLayout(props.mode, jsonText.value, scene.value);
  if (next == null) {
    jsonNote.value = `Not a layout this ${props.mode.label} editor can read.`;
    return;
  }
  props.layout.replace(next);
  selected.value = [];
  props.node?.graph?.setDirtyCanvas?.(true, true);
  jsonText.value = "";
  jsonNote.value = "Imported — Ctrl+Z on the canvas brings back what was there.";
}

// --- panes --------------------------------------------------------------------------------------

const COLUMNS = [{ min: 320 }, { size: 300, min: 240, max: 640 }];
const ROWS = [{ size: 0.34, min: 90 }, { min: 160 }];
const splitKey = `${props.propertyPrefix}Split`;
const splits = ref<{ columns?: number[]; rows?: number[] }>(
  (props.node?.properties?.[splitKey] as { columns?: number[]; rows?: number[] }) ?? {},
);
function saveSplit(which: "columns" | "rows", sizes: number[] | undefined) {
  if (!sizes) return;
  splits.value = { ...splits.value, [which]: sizes };
  if (props.node?.properties) {
    // eslint-disable-next-line vue/no-mutating-props -- node.properties is ComfyUI's own store for what the backend must not see.
    props.node.properties[splitKey] = splits.value;
  }
}

// --- layer list and inspector -------------------------------------------------------------------

const SCENE_ROW = "background";
const layerItems = computed<LayerItem[]>(() => [
  ...[...layoutBoxes.value].reverse().map((b) => ({
    id: b.id,
    label: b.name,
    color: b.color,
    ...props.mode.describe?.(scene.value, b.id),
    hidden: b.hidden,
    locked: b.locked,
  })),
  {
    id: SCENE_ROW,
    label: "Scene",
    sub: props.mode.sceneLine?.(scene.value) ?? "the whole picture",
    color: "var(--zen-surface-2, #26262d)",
    hidden: props.mode.sceneHidden?.(scene.value) ?? false,
    pinned: true,
  },
]);
const listSelection = computed(() => (selected.value.length ? selected.value : [SCENE_ROW]));
function selectFromList(ids: string[]) {
  selected.value = ids.filter((id) => id !== SCENE_ROW);
}
/** `index` is a display position, top-first; boxes are stored bottom-first. */
function reorder(id: string, index: number) {
  const order = [...layoutBoxes.value];
  const from = order.findIndex((b) => b.id === id);
  if (from < 0) return;
  const [moved] = order.splice(from, 1);
  order.splice(Math.max(0, order.length - index), 0, moved!);
  update(props.mode.setBoxes(scene.value, order));
}
const current = computed<LayoutBox | null>(() =>
  selected.value.length === 1
    ? (layoutBoxes.value.find((b) => b.id === selected.value[0]) ?? null)
    : null,
);

// --- the stage: points, drops, the wheel and keys -----------------------------------------------

/** Where `e` is on the canvas, 0..1, or null off it. */
function stagePoint(e: MouseEvent): [number, number] | null {
  const rect = wheel.el.value?.querySelector<HTMLElement>(".zbc")?.getBoundingClientRect();
  if (!rect?.width) return null;
  const x = (e.clientX - rect.left) / rect.width;
  const y = (e.clientY - rect.top) / rect.height;
  return x < 0 || x > 1 || y < 0 || y > 1 ? null : [x, y];
}

const dropOver = ref(false);
function onStageDragOver(e: DragEvent) {
  if (!props.acceptsDrops || !hasMediaDrop(e)) return;
  e.preventDefault();
  dropOver.value = true;
}
function onStageDrop(e: DragEvent) {
  dropOver.value = false;
  if (!props.acceptsDrops || !hasMediaDrop(e)) return;
  const at = stagePoint(e);
  if (!at) return;
  e.preventDefault();
  e.stopPropagation();
  emit("drop", e, { x: at[0], y: at[1] }, boxAt(boxes.value, at[0], at[1])?.id ?? null);
}

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
  ours: (_e, target) => tool.value !== "select" && !!target.closest(".le-stage"),
  focusTarget: () => wheel.el.value?.querySelector<HTMLElement>(".zbc-keys") ?? null,
});

function onKey(e: KeyboardEvent) {
  if (!(e.target as HTMLElement | null)?.closest?.(".le-stage")) return;
  if (tool.value === "select" || (e.key !== "[" && e.key !== "]")) return;
  const step = e.key === "]" ? 1.2 : 1 / 1.2;
  brushSize.value = Math.min(0.3, Math.max(0.01, brushSize.value * step));
  e.preventDefault();
  e.stopPropagation();
}

// --- the trace image (a node property: drawing over it must not re-run anything) ---------------

const backdropKey = `${props.propertyPrefix}Backdrop`;
function readBackdrop(): Backdrop | null {
  const raw = props.node?.properties?.[backdropKey] as Partial<Backdrop> | undefined;
  return raw?.ref
    ? {
        ref: raw.ref,
        type: (raw.type ?? "input") as SourceType,
        opacity: raw.opacity ?? 0.6,
        size: raw.size,
        pinned: raw.pinned === true,
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
    props.node.properties[backdropKey] = next;
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
    pinned: true,
  });
}
/** A run sent its source image back: it becomes the trace image, unless one is pinned. */
function offerBackdrop(image: { ref: string; type: SourceType; size?: [number, number] }) {
  if (backdrop.value?.pinned) return;
  setBackdrop({ ...image, opacity: backdrop.value?.opacity ?? 0.6 });
}
const sourceSize = computed(() => backdrop.value?.size ?? null);
const mismatch = computed(() => {
  const s = sourceSize.value;
  return !!s && Math.abs(s[0] / s[1] - size.value.width / size.value.height) > 0.01;
});
function matchSource() {
  const s = sourceSize.value;
  if (s) setSize(s[0], s[1]);
}

defineExpose({ addBox, update, patchBoxes, stagePoint, offerBackdrop, backdrop });
</script>

<style scoped>
/* rail | canvas | side, with the status bar over the canvas so all three end on one line. Every
   track is a fixed size or a share of the node, never its content, so selecting or typing can't
   move anything. */
.le {
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto;
  gap: 6px;
  width: 100%;
  height: 100%;
  min-height: 0;
  color: var(--zen-text, #e5e5ea);
  font-size: 12px;
}
/* In a ZenKit panel the window draws its own frame; inset so the rail and the panes don't sit on
   it and double its border. On the node, the node body already pads. */
.le.in-panel {
  box-sizing: border-box;
  padding: 8px;
}
.le-main {
  min-width: 0;
  min-height: 0;
}
.le-canvascol {
  display: grid;
  grid-template-rows: 28px minmax(0, 1fr);
  gap: 6px;
  height: 100%;
  min-width: 0;
  min-height: 0;
}
.le-toolbar,
.le-foot {
  display: flex;
  align-items: center;
  gap: 4px;
  min-width: 0;
}
.le-foot {
  padding: 4px 6px;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 7px);
  background: color-mix(in srgb, var(--zen-text, #fff) 2%, transparent);
}
.le-sep {
  width: 1px;
  height: 16px;
  margin: 0 4px;
  background: var(--zen-border, #34343c);
}
.le-opacity {
  display: grid;
  grid-template-columns: auto 1fr 34px;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.le-menu {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
}
.le-menu-title {
  color: var(--zen-muted, #9aa0aa);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.le-menu .zen-btn {
  justify-content: flex-start;
}
.le-json {
  color: var(--zen-text, #e5e5ea);
  font-size: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 300px;
  padding: 10px;
}
.le-row {
  display: flex;
  align-items: center;
  gap: 6px;
}
.le-grow {
  flex: 1;
  min-width: 0;
}
.le-stage.dropping {
  outline: 2px dashed var(--zen-accent, #6366f1);
  outline-offset: -2px;
  border-radius: var(--zen-radius, 7px);
}
.le-stage {
  min-width: 0;
  min-height: 0;
}
.le-hint-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}
.le-hint-empty i {
  font-size: 28px;
  opacity: 0.6;
}
.le-help {
  display: grid;
  grid-template-columns: auto auto;
  gap: 4px 14px;
  margin: 0;
  padding: 10px 12px;
  font-size: 11px;
}
.le-help dt {
  color: var(--zen-text, #e5e5ea);
  white-space: nowrap;
}
.le-help dd {
  margin: 0;
  color: var(--zen-muted, #9aa0aa);
}
.le-tip {
  flex: 0 1 auto;
  min-width: 0;
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.le-link {
  padding: 0;
  border: 0;
  background: none;
  color: var(--zen-accent, #6366f1);
  font: inherit;
  cursor: pointer;
}
.le-ctl {
  flex: none;
  display: flex;
  align-items: center;
  gap: 6px;
  width: 104px;
  transition: opacity 0.12s ease;
}
.le-ctl.dim {
  opacity: 0.4;
}
.le-chip {
  flex: none;
  display: inline-flex;
  align-items: center;
  gap: 4px;
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
.le-chip.warn {
  border-color: color-mix(in srgb, var(--zen-warning, #f5b041) 60%, transparent);
  color: var(--zen-warning, #f5b041);
}
.le-scene {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.le-side {
  height: 100%;
  min-height: 0;
}
.le-pane {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 7px);
  background: color-mix(in srgb, var(--zen-text, #fff) 2%, transparent);
  overflow: hidden;
}
.le-head {
  flex: none;
  display: flex;
  align-items: center;
  gap: 6px;
  height: 34px;
  padding: 0 8px;
  border-bottom: 1px solid var(--zen-border, #34343c);
}
.le-title {
  flex: 1;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--zen-muted, #9aa0aa);
}
.le-count {
  padding: 0 6px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--zen-text, #fff) 8%, transparent);
  color: var(--zen-muted, #9aa0aa);
  font-size: 10px;
}
.le-swatch {
  flex: none;
  width: 12px;
  height: 12px;
  border-radius: 3px;
}
.le-body {
  flex: 1 1 0;
  min-height: 0;
  padding: 6px;
}
.le-note {
  margin: 0;
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
</style>
