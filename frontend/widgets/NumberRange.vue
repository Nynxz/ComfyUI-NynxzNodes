<template>
  <ZenWidget ref="rootEl" :gap="0" pad="2px 6px" class="nr" @pointerenter="readValue">
    <div class="nr-row">
      <span class="nr-end">{{ show(range.min) }}</span>
      <ZenSlider
        ref="sliderEl"
        class="nr-slider"
        @pointerenter="tipOn('hover', true)"
        @pointerleave="tipOn('hover', false)"
        @pointerdown="tipOn('drag', true)"
        :model-value="value"
        :min="range.min"
        :max="range.max"
        :step="range.step"
        @update:model-value="setValue"
      />
      <span class="nr-end">{{ show(range.max) }}</span>
    </div>

    <!-- What it is set to, over the thumb while the slider is pointed at or dragged. The number
         widget above shows it on the node itself, but a slider promoted out of a subgraph on its
         own has nothing else to read it from. On <body>, so the node doesn't clip it. -->
    <Teleport to="body">
      <div v-if="tip" class="nr-tip" :style="{ left: `${tip.x}px`, top: `${tip.y}px` }">
        {{ value.toFixed(decimals) }}
      </div>
    </Teleport>

    <ZenPopover v-model:open="menuOpen" :anchor="menuAnchor" placement="bottom-end" role="dialog">
      <div class="nr-menu">
        <div class="nr-head">Range</div>
        <label class="nr-field">
          <span>Minimum</span>
          <ZenNumber
            :model-value="range.min"
            :step="range.step"
            :precision="decimals"
            @update:model-value="setRange({ min: $event })"
          />
        </label>
        <label class="nr-field">
          <span>Maximum</span>
          <ZenNumber
            :model-value="range.max"
            :step="range.step"
            :precision="decimals"
            @update:model-value="setRange({ max: $event })"
          />
        </label>
        <label class="nr-field">
          <span>Step</span>
          <ZenNumber
            :model-value="range.step"
            :step="whole ? 1 : 0.01"
            :min="whole ? 1 : 0"
            :precision="whole ? 0 : 4"
            @update:model-value="setRange({ step: $event })"
          />
        </label>
        <div class="nr-presets">
          <button v-for="p in presets" :key="p.label" class="nr-preset" @click="setRange(p.range)">
            {{ p.label }}
          </button>
        </div>
      </div>
    </ZenPopover>
  </ZenWidget>
</template>

<script lang="ts">
import type { WidgetOptions } from "@/framework";

/** Per-widget overrides picked up by discovery (see nodes/primitive/number.py). */
export const widgetOptions: WidgetOptions = {
  minHeight: 22,
  minWidth: 200,
};
</script>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { ZenNumber, ZenPopover, ZenSlider, ZenWidget } from "@nynxz/zenkit-ui";

import { isOwnMount } from "@/lib/ownMount";
import {
  addNodeHeaderButton,
  type HeaderButtonAnchor,
  type NodeHeaderButtonHandle,
} from "@/framework";

interface Range {
  min: number;
  max: number;
  step: number;
}
/** ComfyUI's number widget, as much of it as this drives. */
interface NumberLike {
  name: string;
  type?: string;
  value: unknown;
  options?: Record<string, unknown>;
  callback?: (value: unknown) => void;
}

const props = defineProps<{
  widget?: { value: unknown; callback?: (v: unknown) => void };
  node?: {
    graph?: { setDirtyCanvas?: (a: boolean, b: boolean) => void };
    properties?: Record<string, unknown>;
    widgets?: NumberLike[];
    constructor?: { comfyClass?: string };
    onConfigure?: (...args: unknown[]) => void;
  };
}>();

// The number is ComfyUI's own widget on the node (`value`): it is promoted into a subgraph,
// linked and scripted like any other. This is the range it is set within: a slider across it,
// and the limits themselves, which are written onto that widget so its own arrows, drag and
// typed values keep to them too.
//
// The range is how the value is set, not part of what the node outputs, so it lives in
// node.properties (saved with the graph, not hashed) and this widget's value stays `{}`.
const RANGE_KEY = "nynxzRange";
// ComfyUI's host objects, not Vue state: writing to them is how a node body persists.
const hostNode = props.node;
const saved = hostNode?.properties;
const number = hostNode?.widgets?.find((w) => w.name === "value");
const whole = /\.Int$/.test(hostNode?.constructor?.comfyClass ?? "");

const DEFAULT: Range = whole ? { min: 0, max: 100, step: 1 } : { min: 0, max: 1, step: 0.01 };
const presets: { label: string; range: Range }[] = whole
  ? [
      { label: "0 – 100", range: { min: 0, max: 100, step: 1 } },
      { label: "1 – 50", range: { min: 1, max: 50, step: 1 } },
      { label: "0 – 4096", range: { min: 0, max: 4096, step: 8 } },
    ]
  : [
      { label: "0 – 1", range: { min: 0, max: 1, step: 0.01 } },
      { label: "-1 – 1", range: { min: -1, max: 1, step: 0.01 } },
      { label: "0 – 10", range: { min: 0, max: 10, step: 0.1 } },
    ];

function tidy(raw: unknown): Range {
  const r = (raw && typeof raw === "object" ? raw : {}) as Partial<Record<keyof Range, unknown>>;
  const num = (v: unknown, fallback: number) =>
    Number.isFinite(+(v as number)) && v !== null && v !== "" ? +(v as number) : fallback;
  let min = num(r.min, DEFAULT.min);
  let max = num(r.max, DEFAULT.max);
  if (min > max) [min, max] = [max, min];
  let step = Math.abs(num(r.step, DEFAULT.step)) || DEFAULT.step;
  if (whole) {
    min = Math.round(min);
    max = Math.round(max);
    step = Math.max(1, Math.round(step));
  }
  return { min, max, step };
}
const range = ref<Range>(tidy(saved?.[RANGE_KEY]));

/** Decimal places the step asks for: what the value is rounded and shown to. */
const decimals = computed(() =>
  whole ? 0 : Math.min(6, (String(range.value.step).split(".")[1] ?? "").length),
);
const show = (n: number) => String(+n.toFixed(decimals.value));
// What the number widget holds, mirrored so the slider can show it: it is not Vue state.
const value = ref(Number(number?.value) || 0);
function readValue() {
  const now = Number(number?.value);
  if (Number.isFinite(now)) value.value = now;
}

// --- the value bubble ------------------------------------------------------------------
const sliderEl = ref<{ $el?: HTMLElement } | null>(null);
const tip = ref<{ x: number; y: number } | null>(null);
const tipWhy = { hover: false, drag: false };
const THUMB = 14; // ZenSlider's thumb, which stops half its width short of each end
function placeTip() {
  const el = sliderEl.value?.$el;
  if (!el || !(tipWhy.hover || tipWhy.drag)) {
    tip.value = null;
    return;
  }
  const box = el.getBoundingClientRect();
  const { min, max } = range.value;
  const along = max > min ? Math.min(1, Math.max(0, (value.value - min) / (max - min))) : 0;
  // The node is scaled with the canvas, and the thumb with it.
  const thumb = el.offsetWidth ? (THUMB * box.width) / el.offsetWidth : THUMB;
  tip.value = { x: box.left + thumb / 2 + along * (box.width - thumb), y: box.top - 6 };
}
const endDrag = () => tipOn("drag", false);
function tipOn(why: "hover" | "drag", on: boolean) {
  tipWhy[why] = on;
  if (why === "drag") {
    const listen = on ? window.addEventListener : window.removeEventListener;
    listen("pointerup", endDrag, true);
    listen("pointercancel", endDrag, true);
  }
  if (on) readValue();
  placeTip();
}

function fit(n: number) {
  const { min, max, step } = range.value;
  const stepped = min + Math.round((n - min) / step) * step;
  return +Math.min(max, Math.max(min, stepped)).toFixed(decimals.value);
}
function setValue(next: number) {
  const fitted = fit(next);
  value.value = fitted;
  placeTip();
  if (!number || number.value === fitted) return;
  number.value = fitted;
  try {
    number.callback?.(fitted);
  } catch {
    /* no callback registered */
  }
  hostNode?.graph?.setDirtyCanvas?.(true, true);
}

/** Put the range on ComfyUI's widget, so its own controls keep to it. `step` there is ten
 *  times the real one for historical reasons; `step2` is the real one. */
function apply() {
  if (!number) return;
  const { min, max, step } = range.value;
  number.options = {
    ...number.options,
    min,
    max,
    step: step * 10,
    step2: step,
    round: whole ? 1 : step,
    precision: decimals.value,
  };
}
function setRange(change: Partial<Range>) {
  range.value = tidy({ ...range.value, ...change });
  if (saved) saved[RANGE_KEY] = { ...range.value };
  apply();
  readValue();
  setValue(value.value); // a value now outside the range comes back inside it
  hostNode?.graph?.setDirtyCanvas?.(true, true);
}

// --- the header cog -----------------------------------------------------------------
const rootEl = ref<{ el: HTMLElement | null } | null>(null);
const menuOpen = ref(false);
const menuAnchor = ref<HeaderButtonAnchor | null>(null);
let cog: NodeHeaderButtonHandle | null = null;
let unhook = () => {};
onMounted(() => {
  apply();
  readValue();
  // Not on a copy promoted onto a subgraph's node: the cog would sit in that node's header.
  if (isOwnMount(props)) {
    cog = addNodeHeaderButton(props.node, rootEl.value?.el ?? null, {
      icon: "mdi mdi-cog",
      text: "⚙",
      title: "Range",
      onClick: (anchor) => {
        menuAnchor.value = anchor;
        menuOpen.value = !menuOpen.value;
      },
    });
  }
  if (!number) return;
  // The number is changed from its own control, and restored by a loaded graph after this
  // component is set up: the slider follows both.
  const prev = number.callback;
  const onChange = function (this: unknown, next: unknown) {
    prev?.call(this, next);
    readValue();
  };
  number.callback = onChange;
  const prevConfigure = hostNode?.onConfigure;
  const onConfigure = function (this: unknown, ...args: unknown[]) {
    prevConfigure?.apply(this, args);
    range.value = tidy(hostNode?.properties?.[RANGE_KEY]);
    apply();
    readValue();
  };
  if (hostNode) hostNode.onConfigure = onConfigure;
  unhook = () => {
    if (number.callback === onChange) number.callback = prev;
    if (hostNode?.onConfigure === onConfigure) hostNode.onConfigure = prevConfigure;
  };
});
onBeforeUnmount(() => {
  cog?.destroy();
  unhook();
  endDrag();
});
</script>

<style scoped>
.nr-row {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 18px;
}
.nr-slider {
  flex: 1;
  min-width: 0;
}
.nr-tip {
  position: fixed;
  z-index: 10000;
  transform: translate(-50%, -100%);
  padding: 2px 6px;
  font-size: 11px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  line-height: 1.3;
  white-space: nowrap;
  color: var(--zen-text, #e5e5ea);
  background: var(--zen-surface, #202026);
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 6px);
  box-shadow: 0 4px 12px rgb(0 0 0 / 45%);
  pointer-events: none;
}
.nr-end {
  flex: none;
  font-size: 10.5px;
  font-variant-numeric: tabular-nums;
  color: var(--zen-muted, #9aa0aa);
}
.nr-menu {
  display: flex;
  flex-direction: column;
  gap: 6px;
  width: 210px;
  padding: 6px 8px 8px;
}
.nr-head {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--zen-muted, #9aa0aa);
}
.nr-field {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  font-size: 12px;
  color: var(--zen-text, #e5e5ea);
}
.nr-presets {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  padding-top: 2px;
}
.nr-preset {
  padding: 2px 7px;
  font: inherit;
  font-size: 11px;
  color: var(--zen-muted, #cfd2d8);
  background: color-mix(in srgb, var(--zen-text, #fff) 6%, transparent);
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 6px);
  cursor: pointer;
}
.nr-preset:hover {
  border-color: var(--zen-accent, #6366f1);
  color: var(--zen-text, #e5e5ea);
}
</style>
