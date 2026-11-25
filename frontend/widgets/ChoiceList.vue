<template>
  <ZenWidget ref="rootEl" :gap="6" pad="2px" class="cl" @pointerenter="readChoice">
    <div
      v-if="editing && rows.length"
      :ref="(el) => (reorder.el.value = el as HTMLElement | null)"
      class="cl-rows"
      :class="{ settling }"
      v-bind="reorder.handlers"
    >
      <div
        v-for="(row, i) in rows"
        :key="row.id"
        class="cl-row"
        :class="{ on: row.id === chosenRow?.id, dragging: dragIndex === i }"
        :style="rowStyle(i)"
      >
        <div class="cl-grip" data-zen-drag title="Drag to reorder">
          <i class="mdi mdi-drag-vertical" />
        </div>
        <ZenDot
          :model-value="row.id === chosenRow?.id"
          :disabled="!row.text.trim()"
          title="Choose this one"
          @update:model-value="choose(row.id)"
        />
        <input
          :ref="(el) => inputRef(el, row.id)"
          class="cl-text"
          :value="row.text"
          placeholder="Option"
          spellcheck="false"
          @input="setText(row, ($event.target as HTMLInputElement).value)"
          @keydown.enter.prevent="addAfter(row)"
        />
        <ZenIconButton size="sm" icon="mdi mdi-close" title="Remove" @click="remove(row)" />
      </div>
    </div>

    <ZenPopover
      v-model:open="outputsOpen"
      :anchor="outputsAnchor"
      placement="bottom-end"
      role="dialog"
    >
      <div class="cl-pop-head">Outputs</div>
      <ZenMenuItem
        v-for="o in OPTIONAL_OUTPUTS"
        :key="o.name"
        :icon="o.icon"
        :title="o.hint"
        :disabled="shown.length === 1 && shown[0] === o.name"
        @select="toggleShown(o.name)"
      >
        {{ o.name }}
        <template #lead><ZenDot :model-value="shown.includes(o.name)" /></template>
      </ZenMenuItem>
      <template v-if="shown.includes('index')">
        <ZenMenuSeparator />
        <div class="cl-pop-head">Index</div>
        <ZenMenuItem
          v-for="f in INDEX_FLAGS"
          :key="f.name"
          :icon="f.icon"
          @select="toggleIndex(f.name)"
        >
          {{ f.label }}
          <template #lead><ZenDot :model-value="flags.includes(f.name)" /></template>
        </ZenMenuItem>
      </template>
    </ZenPopover>

    <!-- The dropdown is ComfyUI's own combo, above; this is the list it offers. Closed, it is one
         button. -->
    <template #footer>
      <WidgetActions v-if="editing" label="Add option" :count="live.length" @primary="add()">
        <ZenIconButton icon="mdi mdi-check" title="Done editing" @click="setEditing(false)" />
      </WidgetActions>
      <WidgetActions
        v-else
        label="Edit options"
        icon="mdi mdi-pencil-outline"
        :count="live.length"
        @primary="setEditing(true)"
      />
    </template>
  </ZenWidget>
</template>

<script lang="ts">
import type { WidgetOptions } from "@/framework";

/** Per-widget overrides picked up by discovery; the value default comes from the Python schema
 *  (see nodes/text/choice.py). */
export const widgetOptions: WidgetOptions = {
  minHeight: 30,
  minWidth: 220,
};
</script>

<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  type ComponentPublicInstance,
} from "vue";
import { useDragSurface } from "@nynxz/zenkit-nodekit";
import {
  insertionIndex,
  landingTop,
  rowStride,
  shiftStyle,
  sortShift,
  ZenDot,
  ZenIconButton,
  ZenMenuItem,
  ZenMenuSeparator,
  ZenPopover,
  ZenWidget,
} from "@nynxz/zenkit-ui";

import WidgetActions from "@/components/WidgetActions.vue";
import { isOwnMount } from "@/lib/ownMount";
import {
  addNodeHeaderButton,
  type HeaderButtonAnchor,
  type NodeHeaderButtonHandle,
} from "@/framework";
import {
  INDEX_FLAGS,
  offerOptions,
  OPTIONAL_OUTPUTS,
  shownFlags,
  shownOutputs,
  STATE_KEY,
  toggleFlag,
  toggleOutput,
  type ChoiceOutput,
  type IndexFlag,
} from "@/lib/choiceOutputs";

interface Option {
  id: number;
  text: string;
}

const props = defineProps<{
  widget?: { value: unknown; callback?: (v: unknown) => void };
  node?: {
    graph?: { setDirtyCanvas?: (a: boolean, b: boolean) => void };
    properties?: Record<string, unknown>;
    widgets?: ComboLike[];
    onConfigure?: (...args: unknown[]) => void;
  };
}>();

/** ComfyUI's combo widget, as much of it as this list drives. */
interface ComboLike {
  name: string;
  value: unknown;
  options?: { values?: unknown };
  callback?: (value: unknown) => void;
}

// The choice is ComfyUI's own combo widget on the node (`choice`), so it is promoted into a
// subgraph, linked and scripted like any other widget. This list is what that combo offers: it
// writes the combo's options, and moves its value when the chosen option is renamed or removed.
//
// The options and whether the list is open are the node's own state, in node.properties: they
// serialise with the graph but aren't what ComfyUI hashes. This widget's VALUE carries the list
// only while an output that needs it (index, count, options) is on, so rewording or reordering
// options no output reflects re-runs nothing.
// ComfyUI's host objects, not Vue state: writing to them is how a node body persists.
const host = props.widget;
const hostNode = props.node;
const saved = hostNode?.properties;
const state = (saved?.[STATE_KEY] ?? {}) as { options?: unknown; editing?: unknown };
const combo = hostNode?.widgets?.find((w) => w.name === "choice");

let seq = 0;
const rows = ref<Option[]>(
  (Array.isArray(state.options) ? state.options : [])
    .filter((o): o is Record<string, unknown> => !!o && typeof o === "object")
    .map((o) => {
      const id = Number.isFinite(+(o.id as number)) ? +(o.id as number) : ++seq;
      if (id > seq) seq = id;
      return { id, text: String(o.text ?? "") };
    }),
);
// A node that only has values (an API workflow, a script) still shows what it holds.
if (!rows.value.length) {
  const held = (host?.value && typeof host.value === "object" ? host.value : {}) as {
    options?: unknown;
  };
  const texts = Array.isArray(held.options) ? held.options.map(String) : [];
  if (!texts.length && combo?.value) texts.push(String(combo.value));
  rows.value = texts.map((text) => ({ id: ++seq, text }));
}
const editing = ref(state.editing === true || !rows.value.length);

/** The options that count: a row with nothing typed in it isn't one yet. */
const live = computed(() => rows.value.filter((r) => r.text.trim()));

// What the combo holds, mirrored so the list can show it: the combo is not Vue state.
const choice = ref(String(combo?.value ?? ""));
const chosenRow = computed(() => live.value.find((r) => r.text === choice.value));
function readChoice() {
  choice.value = String(combo?.value ?? "");
}
function setChoice(text: string) {
  choice.value = text;
  if (!combo || combo.value === text) return;
  combo.value = text;
  try {
    combo.callback?.(text);
  } catch {
    /* no callback registered */
  }
}

const shown = ref<ChoiceOutput[]>(shownOutputs(props.node));

/** Give the combo the options typed here (see offerOptions for why the list is the node's own). */
function offer() {
  offerOptions(
    hostNode,
    live.value.map((r) => r.text),
  );
}

function commit() {
  if (saved) {
    saved[STATE_KEY] = { options: rows.value.map((r) => ({ ...r })), editing: editing.value };
  }
  offer();
  if (host) {
    const listed = shown.value.some((name) => name !== "text");
    const value: Record<string, unknown> = listed ? { options: live.value.map((r) => r.text) } : {};
    host.value = value;
    try {
      host.callback?.(value);
    } catch {
      /* no callback registered */
    }
  }
  hostNode?.graph?.setDirtyCanvas?.(true, true);
}

function choose(id: number) {
  const row = live.value.find((r) => r.id === id);
  if (row) setChoice(row.text);
}
function setText(row: Option, text: string) {
  const wasChosen = row === chosenRow.value;
  row.text = text;
  commit();
  // The chosen option keeps being the chosen one under its new name. The first option typed
  // into an empty list is chosen, as a dropdown with one entry would be.
  if (wasChosen || !choice.value) setChoice(text.trim() ? text : (live.value[0]?.text ?? ""));
}
function setEditing(next: boolean) {
  editing.value = next;
  // Rows left empty were scaffolding: closing the list clears them away.
  if (!next) rows.value = live.value;
  else if (!rows.value.length) add();
  commit();
}

const inputs = new Map<number, HTMLInputElement>();
function inputRef(el: Element | ComponentPublicInstance | null, id: number) {
  if (el instanceof HTMLInputElement) inputs.set(id, el);
  else inputs.delete(id);
}
function add(at = rows.value.length) {
  const row = { id: ++seq, text: "" };
  rows.value.splice(at, 0, row);
  commit();
  void nextTick(() => inputs.get(row.id)?.focus());
}
/** Enter in an option starts the next one beneath it, as in any list you type into. */
function addAfter(row: Option) {
  add(rows.value.indexOf(row) + 1);
}
function remove(row: Option) {
  const wasChosen = row === chosenRow.value;
  rows.value = rows.value.filter((r) => r !== row);
  commit();
  if (wasChosen) setChoice(live.value[0]?.text ?? "");
}

// --- the header cog: which outputs the node shows -------------------------------------
const rootEl = ref<{ el: HTMLElement | null } | null>(null);
const outputsOpen = ref(false);
const outputsAnchor = ref<HeaderButtonAnchor | null>(null);
const flags = ref<IndexFlag[]>(shownFlags(props.node));
function toggleIndex(flag: IndexFlag) {
  toggleFlag(props.node, flag);
  flags.value = shownFlags(props.node);
}
function toggleShown(output: ChoiceOutput) {
  toggleOutput(props.node, output);
  shown.value = shownOutputs(props.node);
  commit(); // what the value carries follows what is shown
}
let cog: NodeHeaderButtonHandle | null = null;
let unhook = () => {};
onMounted(() => {
  // A loaded graph restores the outputs setting and the combo's value after this component is
  // set up; and the combo is changed from its own dropdown, which this list has to follow.
  shown.value = shownOutputs(props.node);
  offer();
  readChoice();
  if (combo) {
    const prev = combo.callback;
    const onPick = function (this: unknown, value: unknown) {
      prev?.call(this, value);
      readChoice();
    };
    combo.callback = onPick;
    const prevConfigure = hostNode?.onConfigure;
    const onConfigure = function (this: unknown, ...args: unknown[]) {
      prevConfigure?.apply(this, args);
      shown.value = shownOutputs(props.node);
      readChoice();
    };
    if (hostNode) hostNode.onConfigure = onConfigure;
    unhook = () => {
      if (combo.callback === onPick) combo.callback = prev;
      if (hostNode?.onConfigure === onConfigure) hostNode.onConfigure = prevConfigure;
    };
  }
  // Not on a copy promoted onto a subgraph's node: the cog would sit in that node's header.
  if (isOwnMount(props)) {
    cog = addNodeHeaderButton(props.node, rootEl.value?.el ?? null, {
      icon: "mdi mdi-cog",
      text: "⚙",
      title: "Outputs",
      onClick: (anchor) => {
        shown.value = shownOutputs(props.node);
        flags.value = shownFlags(props.node);
        outputsAnchor.value = anchor;
        outputsOpen.value = !outputsOpen.value;
      },
    });
  }
});
onBeforeUnmount(() => {
  cog?.destroy();
  unhook();
});

// --- reorder: the row itself moves under the pointer (see LoraStack.vue for the reasoning) ---
const dragIndex = ref(-1);
const dropIndex = ref(-1);
const dragOffset = ref(0);
const stride = ref(0);
const settling = ref(false);
const settle = ref<{ index: number; y: number } | null>(null);
let rowMids: number[] = [];
let rowTops: number[] = [];
let rowHeights: number[] = [];

function rowStyle(i: number) {
  if (dragIndex.value === i) return { transform: `translateY(${dragOffset.value}px)` };
  if (settle.value?.index === i) return { transform: `translateY(${settle.value.y}px)` };
  return shiftStyle(sortShift(i, dragIndex.value, dropIndex.value), stride.value);
}
const reorder = useDragSurface({
  onStart(ctx) {
    const grip = (ctx.event.target as HTMLElement | null)?.closest?.(".cl-grip");
    const surface = reorder.el.value;
    if (!grip || !surface) return false;
    const kids = Array.from(surface.children) as HTMLElement[];
    const i = kids.indexOf(grip.closest(".cl-row") as HTMLElement);
    if (i < 0) return false;
    rowMids = kids.map((k) => k.offsetTop + k.offsetHeight / 2);
    rowTops = kids.map((k) => k.offsetTop);
    rowHeights = kids.map((k) => k.offsetHeight);
    stride.value = rowStride(rowTops, rowHeights, i);
    dragIndex.value = i;
    dropIndex.value = i;
    dragOffset.value = 0;
  },
  onMove(ctx) {
    if (dragIndex.value < 0) return;
    dragOffset.value = ctx.delta.y;
    dropIndex.value = insertionIndex(
      rowMids,
      dragIndex.value,
      (rowMids[dragIndex.value] ?? 0) + ctx.delta.y,
    );
  },
  onEnd() {
    const from = dragIndex.value;
    const to = dropIndex.value;
    const offset = dragOffset.value;
    dragIndex.value = -1;
    dropIndex.value = -1;
    dragOffset.value = 0;
    if (from < 0 || to < 0) return;
    settling.value = true;
    settle.value = {
      index: to,
      y: (rowTops[from] ?? 0) + offset - landingTop(rowTops, rowHeights, from, to),
    };
    if (from !== to) {
      rows.value.splice(to, 0, rows.value.splice(from, 1)[0]!);
      commit();
    }
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        settling.value = false;
        settle.value = null;
      }),
    );
  },
});
</script>

<style scoped>
.cl-rows {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.cl-row {
  position: relative;
  display: flex;
  align-items: center;
  gap: 5px;
  transition: transform 0.15s ease;
}
.cl-rows.settling .cl-row {
  transition: none;
}
.cl-row.dragging {
  z-index: 5;
  transition: none;
  box-shadow: 0 6px 16px rgb(0 0 0 / 45%);
  border-radius: var(--zen-radius, 6px);
  cursor: grabbing;
}
.cl-grip {
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 12px;
  cursor: grab;
  color: var(--zen-muted, #9aa0aa);
  font-size: 14px;
  line-height: 1;
  touch-action: none;
  -webkit-user-select: none;
  user-select: none;
}
.cl-grip > * {
  pointer-events: none;
}
.cl-grip:hover {
  color: var(--zen-text, #e5e5ea);
}
.cl-text {
  flex: 1;
  min-width: 0;
  height: 24px;
  padding: 0 7px;
  font: inherit;
  font-size: 12px;
  color: var(--zen-text, #e5e5ea);
  background: var(--zen-surface, #202026);
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 6px);
  outline: none;
}
.cl-text:focus {
  border-color: var(--zen-accent, #6366f1);
}
.cl-row.on .cl-text {
  border-color: color-mix(in srgb, var(--zen-accent, #6366f1) 55%, transparent);
}
.cl-pop-head {
  padding: 4px 8px 6px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--zen-muted, #9aa0aa);
}
</style>
