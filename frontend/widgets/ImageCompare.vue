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
      :class="{ dragging: isDragging }"
      :style="{ cursor: viewportCursor }"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="endDrag"
      @pointercancel="endDrag"
      @lostpointercapture="endDrag"
    >
      <img v-if="afterImage" :src="afterImage" alt="B" draggable="false" class="ic-img" />
      <img
        v-if="beforeImage"
        :src="beforeImage"
        alt="A"
        draggable="false"
        class="ic-img"
        :style="beforeStyle"
      />
      <div v-if="hasBoth && mode === 'slider'" class="ic-handle" :style="handleStyle" />
    </div>
    <div v-else class="ic-empty">
      <i class="mdi mdi-image-multiple-outline" />
    </div>

    <!-- controls appear/disappear at the BOTTOM (below the image) so toggling never
         shifts the image — revealed by the node-header cog. -->
    <ZenPopover v-model:open="optsOpen" :anchor="optsAnchor" placement="bottom-end" role="dialog">
      <div class="ic-pop-head">Compare</div>
      <ZenMenuItem icon="mdi mdi-swap-horizontal" @select="toggleSwap">
        Swap A / B
        <template #lead><ZenDot :model-value="swapped" /></template>
      </ZenMenuItem>
      <ZenMenuItem
        v-for="t in TOGGLES"
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
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { ZenDot, ZenIconButton, ZenMenuItem, ZenPopover, ZenWidget } from "@nynxz/zenkit-ui";
import {
  addNodeHeaderButton,
  type HeaderButtonAnchor,
  type NodeHeaderButtonHandle,
} from "@/framework";

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
  {
    key: "mode",
    label: "Cross-fade",
    icon: "mdi mdi-gradient-horizontal",
    on: "fade",
    off: "slider",
  },
] as const;

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
const mode = ref<"slider" | "fade">("slider");
const swapped = ref(false);
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

const beforeStyle = computed(() => {
  if (!hasBoth.value) return undefined;
  if (mode.value === "fade") return { opacity: 1 - percent.value / 100 };
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
function onPointerDown(e: PointerEvent) {
  if (!hasBoth.value) return;
  isDragging.value = true;
  if (interaction.value === "drag") {
    try {
      (e.target as Element).setPointerCapture?.(e.pointerId);
    } catch {
      /* fine */
    }
  }
  updateFromPointer(e);
}
function onPointerMove(e: PointerEvent) {
  if (interaction.value === "hover" || isDragging.value) updateFromPointer(e);
}
function endDrag(e: PointerEvent) {
  if (!isDragging.value) return;
  isDragging.value = false;
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
  if (o.mode === "slider" || o.mode === "fade") mode.value = o.mode;
  if (typeof o.swapped === "boolean") swapped.value = o.swapped;
}
function persist() {
  const props_ = props.node?.properties;
  if (props_)
    props_[OPTS_KEY] = {
      interaction: interaction.value,
      orientation: orientation.value,
      mode: mode.value,
      swapped: swapped.value,
    };
  props.node?.graph?.setDirtyCanvas?.(true, true);
}
function isOn(t: (typeof TOGGLES)[number]) {
  return { interaction, orientation, mode }[t.key].value === t.on;
}
function setOpt(key: "interaction" | "orientation" | "mode", value: string) {
  applyOpts({ [key]: value });
  persist();
}
function toggleSwap() {
  swapped.value = !swapped.value;
  persist();
}
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
