<template>
  <div class="cf">
    <span class="cf-label">{{ label }}</span>
    <div v-if="!linked" class="cf-field">
      <ZenColorPicker compact :model-value="swatch" @update:model-value="set" />
      <input
        class="cf-text"
        :class="{ invalid: !valid }"
        :value="text"
        spellcheck="false"
        title="Any CSS colour: #hex, rgb(…), hsl(…) or a name"
        @change="set(($event.target as HTMLInputElement).value.trim())"
        @keydown.enter="($event.target as HTMLInputElement).blur()"
      />
    </div>
    <span v-else class="cf-field cf-linked">linked</span>
  </div>
</template>

<script lang="ts">
import type { WidgetOptions } from "@/framework";

/** Drawn over a plain STRING input (see `color_input()` in nodes/_widgets.py), so the socket
 *  still takes any string — wire a colour in from elsewhere and this steps aside. */
export const widgetOptions: WidgetOptions = {
  minHeight: 28,
};
</script>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { ZenColorPicker } from "@nynxz/zenkit-ui";

interface InputSlot {
  link?: number | null;
  widget?: { name?: string };
}

const props = defineProps<{
  widget?: {
    name?: string;
    label?: string;
    value: unknown;
    callback?: (v: unknown) => void;
    options?: Record<string, unknown>;
  };
  node?: {
    inputs?: InputSlot[];
    onConnectionsChange?: (...args: unknown[]) => void;
    graph?: { setDirtyCanvas?: (a: boolean, b: boolean) => void };
  };
}>();

const label = computed(() => props.widget?.label ?? props.widget?.name ?? "colour");
const text = ref(String(props.widget?.value ?? ""));

// The canvas normalises any CSS colour to #rrggbb, and leaves fillStyle untouched on junk.
const probe = document.createElement("canvas").getContext("2d");
function toHex(value: string): string | null {
  if (!probe || !value) return null;
  probe.fillStyle = "#000001";
  probe.fillStyle = value;
  const out = String(probe.fillStyle);
  return out === "#000001" && value.toLowerCase() !== "#000001" ? null : out;
}
const valid = computed(() => toHex(text.value) !== null);
const swatch = computed(() => toHex(text.value) ?? "#000000");

function set(value: string) {
  text.value = value;
  const widget = props.widget;
  if (!widget) return;
  widget.value = value;
  try {
    widget.callback?.(value);
  } catch {
    /* no callback registered */
  }
  props.node?.graph?.setDirtyCanvas?.(true, true);
}

// A wired input carries its own colour, so the control would only show a stale one.
const linked = ref(false);
function syncLinked() {
  const slot = props.node?.inputs?.find((s) => s.widget?.name === props.widget?.name);
  linked.value = slot?.link != null;
}
onMounted(() => {
  const node = props.node;
  if (!node) return;
  const prev = node.onConnectionsChange;
  node.onConnectionsChange = function (this: unknown, ...args: unknown[]) {
    prev?.apply(this, args);
    syncLinked();
  };
  // Workflow load writes the saved value after the mount, and restores links after that.
  queueMicrotask(() => {
    text.value = String(props.widget?.value ?? "");
    syncLinked();
  });
});
</script>

<style scoped>
.cf {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 28px;
  padding: 0 2px;
  font-size: 12px;
}
.cf-label {
  flex: 0 1 40%;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--zen-muted, #9aa0aa);
}
.cf-field {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 6px;
  height: 26px;
  padding: 0 6px;
  box-sizing: border-box;
  border: 1px solid var(--zen-control-border, var(--zen-border, #34343c));
  border-radius: var(--zen-radius, 6px);
  background: var(--zen-field-bg, var(--zen-input, #15151a));
}
.cf-field:focus-within {
  border-color: var(--zen-accent, #6366f1);
}
.cf-text {
  flex: 1;
  min-width: 0;
  padding: 0;
  border: none;
  outline: none;
  background: none;
  color: var(--zen-text, #e5e5ea);
  font: inherit;
  font-family: var(--zen-mono, ui-monospace, monospace);
  font-size: 11px;
}
.cf-text.invalid {
  color: var(--zen-danger, #f87171);
}
.cf-linked {
  font-size: 11px;
  font-style: italic;
  color: var(--zen-muted, #9aa0aa);
}
</style>
