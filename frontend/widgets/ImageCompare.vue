<template>
  <ZenWidget fill :gap="6" pad="4px 2px" class="ic" ref="rootEl">
    <!-- header row: only shown for batch nav, or as the cog fallback when the node-header
         injection isn't available (legacy litegraph). On Nodes 2.0 the cog lives in the
         actual node header (see headerButton.ts) and this row stays out of the way. -->
    <div v-if="showBatchNav" class="ic-head">
      <div class="ic-head-nav">
        <ZenIconButton
          icon="mdi mdi-chevron-left"
          title="Previous A"
          :disabled="beforeCount <= 1"
          @click="bumpBefore(-1)"
        />
        <span class="ic-nav-lbl">A {{ beforeCount ? beforeIndex + 1 : 0 }}/{{ beforeCount }}</span>
        <ZenIconButton
          icon="mdi mdi-chevron-right"
          title="Next A"
          :disabled="beforeCount <= 1"
          @click="bumpBefore(1)"
        />
        <span class="ic-nav-gap" />
        <ZenIconButton
          icon="mdi mdi-chevron-left"
          title="Previous B"
          :disabled="afterCount <= 1"
          @click="bumpAfter(-1)"
        />
        <span class="ic-nav-lbl">B {{ afterCount ? afterIndex + 1 : 0 }}/{{ afterCount }}</span>
        <ZenIconButton
          icon="mdi mdi-chevron-right"
          title="Next B"
          :disabled="afterCount <= 1"
          @click="bumpAfter(1)"
        />
      </div>
    </div>

    <div
      v-if="hasAny"
      ref="viewport"
      class="ic-view"
      :class="{ dragging: isDragging && mode !== 'box' }"
      :style="{ cursor: viewportCursor }"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="endDrag"
      @pointercancel="endDrag"
      @lostpointercapture="endDrag"
      @dblclick.stop="toggleSwap"
    >
      <img
        v-if="afterImage"
        ref="baseImg"
        :src="afterImage"
        alt="B"
        draggable="false"
        class="ic-img"
        @load="readNatural"
      />
      <img
        v-if="beforeImage"
        :src="beforeImage"
        alt="A"
        draggable="false"
        class="ic-img"
        :style="beforeStyle"
      />
      <div v-if="hasBoth && mode === 'slider'" class="ic-handle" :style="handleStyle" />
      <div v-if="hasBoth && mode === 'box'" class="ic-box" :style="boxStyle">
        <span class="ic-box-tag">{{ swapped ? "B" : "A" }}</span>
      </div>
    </div>
    <div v-else class="ic-empty">
      <i class="mdi mdi-image-multiple-outline" />
    </div>

    <!-- controls appear/disappear at the BOTTOM (below the image) so toggling never
         shifts the image — revealed by the node-header cog. -->
    <ZenPopover v-model:open="optsOpen" :anchor="optsAnchor" placement="bottom-end" role="dialog">
      <div class="ic-pop-head">Compare</div>
      <ZenMenuItem
        v-for="m in MODES"
        :key="m.value"
        :icon="m.icon"
        @select="setOpt('mode', m.value)"
      >
        {{ m.label }}
        <template #lead><ZenDot :model-value="mode === m.value" /></template>
      </ZenMenuItem>
      <ZenMenuItem icon="mdi mdi-swap-horizontal" @select="toggleSwap">
        Swap A / B
        <template #lead><ZenDot :model-value="swapped" /></template>
      </ZenMenuItem>
      <ZenMenuItem
        v-for="t in visibleToggles"
        :key="t.key"
        :icon="t.icon"
        @select="setOpt(t.key, isOn(t) ? t.off : t.on)"
      >
        {{ t.label }}
        <template #lead><ZenDot :model-value="isOn(t)" /></template>
      </ZenMenuItem>
    </ZenPopover>
  </ZenWidget>
</template>

<script lang="ts">
// Read by nodekit's `discoverWidgets`: the io type comes from this file's NAME
// (ImageCompare.vue -> NYNXZ_IMAGE_COMPARE, matching `io_types.py`), and these two exports carry
// what the type alone cannot. This replaces the old per-node `widget.ts` manifest.
import { viewUrl, type NodeDef, type WidgetOptions } from "@/framework";

export const widgetOptions: WidgetOptions = {
  name: "compare_view",
  minHeight: 200,
  fill: true, // node keeps its size; the image fits (contain) instead of resizing it
  serialize: false, // preview URLs go stale on restart
  default: { beforeImages: [], afterImages: [] },
};

export const nodeDef: Omit<NodeDef, "widgets"> = {
  is: "nynxz.Image.Compare",
  minSize: [440, 360],
  // On execute, turn the saved temp-preview records into /view URLs for the compare widget.
  output: {
    widget: "compare_view",
    from: (o) => ({
      beforeImages: ((o.a_images as Record<string, string>[]) ?? []).map(viewUrl),
      afterImages: ((o.b_images as Record<string, string>[]) ?? []).map(viewUrl),
    }),
  },
};
</script>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { ZenDot, ZenIconButton, ZenMenuItem, ZenPopover, ZenWidget } from "@nynxz/zenkit-ui";
import {
  addNodeHeaderButton,
  type HeaderButtonAnchor,
  type NodeHeaderButtonHandle,
} from "@/framework";
import {
  boxRect,
  centreBox,
  clampBox,
  containFit,
  cursorFor,
  DEFAULT_BOX,
  hitBox,
  isBox,
  MIN_PX,
  moveBox,
  resizeBox,
  spanBox,
  toImage,
  type Box,
  type BoxHit,
} from "@/lib/compareBox";

interface DOMWidget {
  name: string;
  value: unknown;
  callback?: (v: unknown) => void;
}
interface CompareValue {
  beforeImages: string[];
  afterImages: string[];
}

const props = defineProps<{
  widget?: DOMWidget;
  node?: {
    widgets?: DOMWidget[];
    properties?: Record<string, unknown>;
    graph?: { setDirtyCanvas?: (a: boolean, b: boolean) => void };
  };
}>();

// Reveal options live on the node (persisted in node.properties) — no separate input.
const OPTS_KEY = "nynxz.compareOpts";
/** Each setting has two values, so each is one toggle row: on is the value it isn't by default. */
const TOGGLES = [
  {
    key: "interaction",
    label: "Drag to reveal",
    icon: "mdi mdi-drag-variant",
    on: "drag",
    off: "hover",
  },
  {
    key: "orientation",
    label: "Vertical split",
    icon: "mdi mdi-arrow-up-down",
    on: "vertical",
    off: "horizontal",
  },
] as const;
type Mode = "slider" | "fade" | "box";
const MODES: { value: Mode; label: string; icon: string }[] = [
  { value: "slider", label: "Swipe", icon: "mdi mdi-compare-horizontal" },
  { value: "fade", label: "Cross-fade", icon: "mdi mdi-gradient-horizontal" },
  { value: "box", label: "Box", icon: "mdi mdi-selection-drag" },
];
const isMode = (v: unknown): v is Mode => MODES.some((m) => m.value === v);

// ZenWidget is a component, so this ref holds its instance; `.el` is the DOM node.
const rootEl = ref<{ el: HTMLElement | null } | null>(null);
let headerCog: NodeHeaderButtonHandle | null = null;
const viewport = ref<HTMLDivElement | null>(null);
const percent = ref(50);
const beforeIndex = ref(0);
const afterIndex = ref(0);
const isDragging = ref(false);
// The viewport fills the node body (fill mode); the images object-fit:contain inside it, so
// changing the image fits within the current node size rather than resizing the node.

// widget.value is a host getter Vue can't track; mirror it into a ref and refresh
// it whenever the host calls the widget callback (onExecuted pushes new URLs).
const data = ref<CompareValue>({ beforeImages: [], afterImages: [] });
const interaction = ref<"hover" | "drag">("hover");
const orientation = ref<"horizontal" | "vertical">("horizontal");
const mode = ref<Mode>("slider");
const swapped = ref(false);
const box = ref<Box>({ ...DEFAULT_BOX });
const baseImg = ref<HTMLImageElement | null>(null);
// The viewport's inner size and the base (B) image's natural size: together they say where the
// contained image sits, which box mode needs because its box is kept in image fractions.
const viewSize = ref({ w: 0, h: 0 });
const natural = ref({ w: 0, h: 0 });
const fit = computed(() =>
  containFit(viewSize.value.w, viewSize.value.h, natural.value.w, natural.value.h),
);
const boxHover = ref<BoxHit>({ kind: "draw" });
let boxDrag: {
  hit: BoxHit;
  start: Box;
  sx: number;
  sy: number;
  anchor: { x: number; y: number };
  moved: boolean;
} | null = null;
const optsOpen = ref(false);
const optsAnchor = ref<HeaderButtonAnchor | null>(null);

function isCompare(v: unknown): v is CompareValue {
  return (
    !!v &&
    typeof v === "object" &&
    (Array.isArray((v as CompareValue).beforeImages) ||
      Array.isArray((v as CompareValue).afterImages))
  );
}

// A = 'before', B = 'after'; swap flips which saved set feeds each side.
const aImages = computed(() => (swapped.value ? data.value.afterImages : data.value.beforeImages));
const bImages = computed(() => (swapped.value ? data.value.beforeImages : data.value.afterImages));
const beforeImage = computed(() => aImages.value[beforeIndex.value] ?? null);
const afterImage = computed(() => bImages.value[afterIndex.value] ?? null);
const hasBoth = computed(() => !!beforeImage.value && !!afterImage.value);
const hasAny = computed(() => !!beforeImage.value || !!afterImage.value);
const beforeCount = computed(() => aImages.value.length);
const afterCount = computed(() => bImages.value.length);
const showBatchNav = computed(() => beforeCount.value > 1 || afterCount.value > 1);
// Split direction and hover/drag only mean something for the swipe and fade.
const visibleToggles = computed(() => (mode.value === "box" ? [] : TOGGLES));

const boxPx = computed(() => boxRect(box.value, fit.value));
const boxStyle = computed(() => {
  const r = boxPx.value;
  return {
    left: `${r.left}px`,
    top: `${r.top}px`,
    width: `${r.right - r.left}px`,
    height: `${r.bottom - r.top}px`,
  };
});

const beforeStyle = computed(() => {
  if (!hasBoth.value) return undefined;
  if (mode.value === "fade") return { opacity: 1 - percent.value / 100 };
  if (mode.value === "box") {
    const r = boxPx.value;
    const { w, h } = viewSize.value;
    return { clipPath: `inset(${r.top}px ${w - r.right}px ${h - r.bottom}px ${r.left}px)` };
  }
  return orientation.value === "horizontal"
    ? { clipPath: `inset(0 ${100 - percent.value}% 0 0)` }
    : { clipPath: `inset(0 0 ${100 - percent.value}% 0)` };
});
const handleStyle = computed(() =>
  orientation.value === "horizontal"
    ? { left: `${percent.value}%`, top: "50%" }
    : { top: `${percent.value}%`, left: "50%" },
);
const viewportCursor = computed(() => {
  if (!hasBoth.value) return "default";
  if (mode.value === "box") return cursorFor(boxHover.value);
  if (mode.value === "fade" || interaction.value === "drag") return "grab";
  return orientation.value === "horizontal" ? "ew-resize" : "ns-resize";
});

function updateFromPointer(e: PointerEvent) {
  const el = viewport.value;
  if (!el) return;
  const r = el.getBoundingClientRect();
  if (orientation.value === "horizontal") {
    if (r.width <= 0) return;
    percent.value = Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100));
  } else {
    if (r.height <= 0) return;
    percent.value = Math.max(0, Math.min(100, ((e.clientY - r.top) / r.height) * 100));
  }
}
/** The pointer in viewport pixels, inside its border (where the images and box are laid out). */
function local(e: PointerEvent) {
  const el = viewport.value!;
  const r = el.getBoundingClientRect();
  return { x: e.clientX - r.left - el.clientLeft, y: e.clientY - r.top - el.clientTop };
}
const minFrac = () => ({
  w: fit.value.width > 0 ? Math.min(1, MIN_PX / fit.value.width) : 0,
  h: fit.value.height > 0 ? Math.min(1, MIN_PX / fit.value.height) : 0,
});

// Box mode: press inside to move the box, on an edge or corner to resize it, outside to draw a
// new one (a click without a drag moves the box there instead).
function boxDown(e: PointerEvent) {
  const p = local(e);
  const hit = hitBox(p.x, p.y, box.value, fit.value);
  boxHover.value = hit;
  boxDrag = {
    hit,
    start: { ...box.value },
    sx: p.x,
    sy: p.y,
    anchor: toImage(p.x, p.y, fit.value),
    moved: false,
  };
}
function boxMove(e: PointerEvent) {
  const p = local(e);
  if (!boxDrag) {
    boxHover.value = hitBox(p.x, p.y, box.value, fit.value);
    return;
  }
  const dx = p.x - boxDrag.sx;
  const dy = p.y - boxDrag.sy;
  if (!boxDrag.moved && Math.hypot(dx, dy) < 3) return;
  boxDrag.moved = true;
  const min = minFrac();
  const { hit, start } = boxDrag;
  const fx = fit.value.width > 0 ? dx / fit.value.width : 0;
  const fy = fit.value.height > 0 ? dy / fit.value.height : 0;
  if (hit.kind === "draw")
    box.value = spanBox(boxDrag.anchor, toImage(p.x, p.y, fit.value), min.w, min.h);
  else if (hit.kind === "move") box.value = moveBox(start, fx, fy);
  else box.value = resizeBox(start, hit.edges, fx, fy, min.w, min.h);
}
function boxUp(e: PointerEvent) {
  const drag = boxDrag;
  boxDrag = null;
  if (!drag) return;
  const p = local(e);
  if (!drag.moved && drag.hit.kind === "draw" && e.type === "pointerup")
    box.value = centreBox(box.value, toImage(p.x, p.y, fit.value));
  boxHover.value = hitBox(p.x, p.y, box.value, fit.value);
  if (drag.moved || drag.hit.kind === "draw") persist();
}

function onPointerDown(e: PointerEvent) {
  if (!hasBoth.value || e.button !== 0) return;
  isDragging.value = true;
  if (interaction.value === "drag" || mode.value === "box") {
    try {
      (e.target as Element).setPointerCapture?.(e.pointerId);
    } catch {
      /* fine */
    }
  }
  if (mode.value === "box") boxDown(e);
  else updateFromPointer(e);
}
function onPointerMove(e: PointerEvent) {
  if (mode.value === "box") {
    if (hasBoth.value) boxMove(e);
    return;
  }
  if (interaction.value === "hover" || isDragging.value) updateFromPointer(e);
}
function endDrag(e: PointerEvent) {
  if (!isDragging.value) return;
  isDragging.value = false;
  if (mode.value === "box") boxUp(e);
  try {
    (e.target as Element).releasePointerCapture?.(e.pointerId);
  } catch {
    /* fine */
  }
}
function bumpBefore(d: number) {
  if (beforeCount.value)
    beforeIndex.value = (beforeIndex.value + d + beforeCount.value) % beforeCount.value;
}
function bumpAfter(d: number) {
  if (afterCount.value)
    afterIndex.value = (afterIndex.value + d + afterCount.value) % afterCount.value;
}

function applyOpts(v: unknown) {
  if (!v || typeof v !== "object") return;
  const o = v as Record<string, unknown>;
  if (o.interaction === "hover" || o.interaction === "drag") interaction.value = o.interaction;
  if (o.orientation === "horizontal" || o.orientation === "vertical")
    orientation.value = o.orientation;
  if (isMode(o.mode)) mode.value = o.mode;
  if (typeof o.swapped === "boolean") swapped.value = o.swapped;
  if (isBox(o.box)) box.value = clampBox(o.box, 0.01, 0.01);
}
function persist() {
  const props_ = props.node?.properties;
  if (props_)
    props_[OPTS_KEY] = {
      interaction: interaction.value,
      orientation: orientation.value,
      mode: mode.value,
      swapped: swapped.value,
      box: { ...box.value },
    };
  props.node?.graph?.setDirtyCanvas?.(true, true);
}
function isOn(t: (typeof TOGGLES)[number]) {
  return { interaction, orientation }[t.key].value === t.on;
}
function setOpt(key: "interaction" | "orientation" | "mode", value: string) {
  applyOpts({ [key]: value });
  persist();
}
// Swapping trades the batch positions too, so the two pictures on screen change sides rather
// than each side jumping to whatever sits at the other side's index.
function toggleSwap() {
  swapped.value = !swapped.value;
  [beforeIndex.value, afterIndex.value] = [afterIndex.value, beforeIndex.value];
  persist();
}
function readNatural() {
  const img = baseImg.value;
  if (img?.complete && img.naturalWidth > 0)
    natural.value = { w: img.naturalWidth, h: img.naturalHeight };
}
watch(afterImage, () => void nextTick(readNatural));
watch(
  viewport,
  (el, _old, onCleanup) => {
    if (!el) return;
    const ro = new ResizeObserver(() => {
      viewSize.value = { w: el.clientWidth, h: el.clientHeight };
    });
    ro.observe(el);
    viewSize.value = { w: el.clientWidth, h: el.clientHeight };
    onCleanup(() => ro.disconnect());
  },
  { flush: "post" },
);
function toggleOpts(anchor: HeaderButtonAnchor) {
  optsAnchor.value = anchor;
  optsOpen.value = !optsOpen.value;
}

onMounted(() => {
  if (isCompare(props.widget?.value)) data.value = props.widget!.value as CompareValue;
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
      if (isCompare(v)) {
        data.value = v;
        beforeIndex.value = 0;
        afterIndex.value = 0;
      }
    };
  }
  // Restore saved reveal options from node.properties.
  applyOpts(props.node?.properties?.[OPTS_KEY]);

  // Controls cog in the node header — works on both renderers (DOM header on 2.0,
  // canvas title button on 1.0) and self-heals across switches.
  headerCog = addNodeHeaderButton(props.node, rootEl.value?.el ?? null, {
    icon: "mdi mdi-cog",
    text: "⚙",
    title: "Compare controls",
    onClick: toggleOpts,
  });
});
watch(optsOpen, (v) => headerCog?.setActive(v));
onBeforeUnmount(() => headerCog?.destroy());
</script>

<style scoped>
/* content-driven: the viewport sizes to the image aspect; the node grows to fit it */
.ic-head {
  flex: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  min-height: 26px;
}
.ic-head-nav {
  display: flex;
  align-items: center;
  gap: 2px;
}
.ic-nav-lbl {
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
  font-variant-numeric: tabular-nums;
  min-width: 30px;
  text-align: center;
}
.ic-nav-gap {
  width: 8px;
}
/* fills the node body; the image is contained inside, so the box never drives the node size */
.ic-view {
  position: relative;
  flex: 1 1 0;
  min-height: 0;
  width: 100%;
  overflow: hidden;
  border-radius: var(--zen-radius, 8px);
  background: var(--zen-input, #1b1b20);
  border: 1px solid var(--zen-border, #34343c);
  user-select: none;
  touch-action: none;
}
.ic-view.dragging {
  cursor: grabbing !important;
}
.ic-img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  pointer-events: none;
  transition: opacity 80ms ease;
}
.ic-handle {
  position: absolute;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 35%, transparent);
  border: 2px solid var(--zen-accent, #6366f1);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
  pointer-events: none;
  transform: translate(-50%, -50%);
  z-index: 2;
}
/* box mode: A shows through this frame over B; the frame itself only draws the outline */
.ic-box {
  position: absolute;
  box-sizing: border-box;
  border: 2px solid var(--zen-accent, #6366f1);
  border-radius: 2px;
  box-shadow:
    0 0 0 1px rgba(0, 0, 0, 0.45),
    0 2px 10px rgba(0, 0, 0, 0.35);
  pointer-events: none;
  z-index: 2;
}
.ic-box-tag {
  position: absolute;
  top: 4px;
  left: 4px;
  padding: 0 5px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 700;
  line-height: 16px;
  color: #fff;
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 80%, transparent);
}
/* fills the same body space the viewport will — so swapping empty↔image doesn't resize the node */
.ic-empty {
  flex: 1 1 0;
  min-height: 0;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--zen-muted, #9aa0aa);
  border: 1px dashed var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 8px);
}
.ic-empty .mdi {
  font-size: 30px;
  opacity: 0.5;
}
.ic-pop-head {
  padding: 4px 8px 6px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--zen-muted, #9aa0aa);
}
</style>
