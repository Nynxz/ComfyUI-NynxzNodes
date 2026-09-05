<template>
  <ZenWidget :gap="6" pad="2px" class="ls">
    <div class="ls-mid">
      <!-- The rows container is the drag surface: pointer capture has to live on an element that
           survives the whole gesture, and the grips are just the part of it you may grab. -->
      <div
        v-if="rows.length"
        :ref="(el) => (reorder.el.value = el as HTMLElement | null)"
        class="ls-rows"
        v-bind="reorder.handlers"
      >
        <div
          v-for="(row, i) in rows"
          :key="row.id"
          class="ls-row"
          :class="{
            off: !row.on,
            dragging: dragIndex === i,
            over: dropIndex === i && dragIndex !== i,
          }"
          :style="dragIndex === i ? { transform: `translateY(${dragOffset}px)` } : undefined"
        >
          <!-- Reorder and enable stay SEPARATE controls — one grabbed, one clicked, both always
               visible. They're paired into a tight gutter instead: the grip is 12px (the glyph
               is all the affordance it needs) and sits 2px from the dot, which reclaims the row
               width without either control changing what it does or hiding what it says.
               `data-zen-drag` is what stops the press reaching the node: the grip isn't a
               control, so mountWidget's INTERACTIVE guard needs telling — otherwise grabbing it
               drags the NODE under Nodes 2.0. Only the grip starts a reorder; making the whole
               row grabbable would swallow the slider and combo drags. -->
          <div class="ls-gutter">
            <div class="ls-grip" data-zen-drag title="Drag to reorder">
              <i class="mdi mdi-drag-vertical" />
            </div>
            <!-- ZenDot, not ZenSwitch: a per-row enable only needs to read as on/off, and the
                 switch spent 42px of the row on saying it. -->
            <ZenDot :model-value="row.on" title="Enable" @update:model-value="setOn(i, $event)" />
          </div>

          <LoraPicker
            class="ls-pick"
            :model-value="row.name"
            @update:model-value="setName(i, $event)"
            @browse="openBrowse(i)"
          />

          <ZenNumber
            class="ls-str"
            :model-value="row.strength"
            :step="0.05"
            :min="-10"
            :max="10"
            @update:model-value="setStrength(i, $event)"
          />
          <ZenIconButton icon="mdi mdi-close" title="Remove" @click="removeRow(i)" />
        </div>
      </div>
      <div v-else class="ls-empty">
        <i class="mdi mdi-layers-triple-outline" />
        <span>No LoRAs in this stack yet</span>
      </div>
    </div>

    <WidgetActions label="Add LoRA" :count="rows.length" @primary="addRow()">
      <ZenIconButton
        icon="mdi mdi-view-grid-outline"
        title="Browse library"
        @click="openBrowse(-1)"
      />
      <ZenIconButton
        v-if="rows.length"
        icon="mdi mdi-broom"
        danger
        title="Clear all"
        @click="clearAll"
      />
    </WidgetActions>

    <LoraBrowser
      v-model:open="browse"
      :selected="rows[browseTarget]?.name"
      @pick="pickFromBrowse"
    />
  </ZenWidget>
</template>

<script lang="ts">
import type { WidgetOptions } from "@/framework";

/**
 * Per-widget overrides picked up by lib/autoRegister.ts. The upstream pack carried these
 * in a `widget.ts` beside the component; here the component declares them itself, and the
 * value default comes from the Python schema (see nodes/lora/loader.py).
 */
export const widgetOptions: WidgetOptions = {
  minHeight: 60,
};
</script>

<script setup lang="ts">
import { ref } from "vue";
import { useDragSurface } from "@nynxz/zenkit-nodekit";
import { ZenDot, ZenIconButton, ZenNumber, ZenWidget } from "@nynxz/zenkit-ui";

import LoraBrowser from "@/components/LoraBrowser.vue";
import LoraPicker from "@/components/LoraPicker.vue";
import WidgetActions from "@/components/WidgetActions.vue";
import * as lib from "@/lib/loraLibrary";
import { rowsFrom } from "@/lib/widgetRows";

interface LoraRow {
  id: number;
  on: boolean;
  name: string;
  strength: number;
}

const props = defineProps<{
  widget?: { value: unknown; callback?: (v: unknown) => void };
  node?: {
    graph?: { setDirtyCanvas?: (a: boolean, b: boolean) => void };
    properties?: Record<string, unknown>;
  };
}>();

// Reordering must not bust the graph cache. The widget VALUE (what ComfyUI hashes) is the rows
// sorted by name — order-independent — so dragging never changes the node signature. The manual
// drag order lives in node.properties instead: it serialises with the graph but is NOT a hashed
// input. Each row carries a stable id so the two stay linked across reloads. LoRA merges are
// additive, so the backend applying them name-sorted gives the same result regardless of order.
const store = rowsFrom<LoraRow>(props, {
  orderKey: "nynxzLoraOrder",
  make: (raw, id) => ({
    id,
    on: raw.on !== false,
    name: String(raw.name ?? ""),
    strength: Number.isFinite(+(raw.strength as number)) ? +(raw.strength as number) : 1,
  }),
  project: (r) => ({ on: r.on, name: r.name, strength: r.strength }),
  sort: (a, b) => a.name.localeCompare(b.name) || a.strength - b.strength || +a.on - +b.on,
});
const { rows, commit } = store;

const dragIndex = ref(-1); // row being dragged, or -1
const dropIndex = ref(-1); // index the dragged row would land at, or -1
const dragOffset = ref(0); // how far the grabbed row has moved, element-space px
const browse = ref(false);
const browseTarget = ref(-1); // row index to pick into, or -1 = append a new row

// A node that opens with saved LoRAs fetches the list once (cached across nodes) so a deleted or
// renamed one is flagged without opening the picker; empty nodes stay lazy.
if (rows.value.some((r) => r.name)) void lib.ensure();

function setOn(i: number, on: boolean) {
  store.patch(i, { on });
}
function setName(i: number, name: string) {
  store.patch(i, { name });
}
function setStrength(i: number, strength: number) {
  store.patch(i, { strength });
}
function addRow(name = "") {
  rows.value.push({ id: store.nextId(), on: true, name, strength: 1 });
  commit();
}
function removeRow(i: number) {
  rows.value.splice(i, 1);
  commit();
}
function clearAll() {
  rows.value = [];
  commit();
}

// --- reorder (stack order = apply order, so dragging beats delete-and-re-add) ---------
//
// Pointer-driven, NOT HTML5 drag-and-drop. The native ghost is a snapshot the browser renders
// itself: `setDragImage` positions it but has no say over its scale, and the node body sits in a
// CSS-transformed container under Nodes 2.0, so the ghost came out at a size that matched neither
// the row nor the canvas zoom. Here the ROW ITSELF moves — the same element, inside the same
// transform — so it is exactly the row you grabbed, at any zoom, with nothing to keep in sync.
let rowMids: number[] = []; // each row's midpoint at grab time, layout space

const reorder = useDragSurface({
  onStart(ctx) {
    const grip = (ctx.event.target as HTMLElement | null)?.closest?.(".ls-grip");
    const host = reorder.el.value;
    if (!grip || !host) return false; // press wasn't on a handle — leave it to the controls
    const kids = Array.from(host.children) as HTMLElement[];
    const i = kids.indexOf(grip.closest(".ls-row") as HTMLElement);
    if (i < 0) return false;
    // offsetTop/offsetHeight are pre-transform, which is the space ctx.delta already lands in
    // (the helper divides the canvas zoom out), so the two add up without a second correction.
    rowMids = kids.map((k) => k.offsetTop + k.offsetHeight / 2);
    dragIndex.value = i;
    dropIndex.value = i;
    dragOffset.value = 0;
  },
  onMove(ctx) {
    if (dragIndex.value < 0) return;
    dragOffset.value = ctx.delta.y;
    const center = (rowMids[dragIndex.value] ?? 0) + ctx.delta.y;
    // Insertion index = how many OTHER rows the grabbed one has been dragged past. Counting
    // midpoints reads the same in both directions and needs no case for the row being dragged.
    dropIndex.value = rowMids.filter((m, k) => k !== dragIndex.value && m < center).length;
  },
  // Commit once, at the end: reordering on every move would rewrite the stack hundreds of times.
  onEnd() {
    const from = dragIndex.value;
    const to = dropIndex.value;
    endReorder();
    if (from < 0 || to < 0 || from === to) return;
    const moved = rows.value.splice(from, 1)[0];
    if (!moved) return;
    rows.value.splice(to, 0, moved);
    commit();
  },
});
function endReorder() {
  dragIndex.value = -1;
  dropIndex.value = -1;
  dragOffset.value = 0;
  rowMids = [];
}

function openBrowse(target: number) {
  browseTarget.value = target;
  browse.value = true;
}
function pickFromBrowse(name: string) {
  if (browseTarget.value < 0) addRow(name);
  else setName(browseTarget.value, name);
  browse.value = false;
}
</script>

<style scoped>
/* natural content flow — the node auto-grows to fit (see mountWidget), so rows are never
   scrolled and the Add button simply follows the last row. */
.ls-mid {
  display: flex;
  flex-direction: column;
}
.ls-rows {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.ls-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 5px;
  padding: 14px 8px;
  text-align: center;
  color: var(--zen-muted, #9aa0aa);
  border: 1px dashed var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 8px);
  background: color-mix(in srgb, var(--zen-text, #fff) 3%, transparent);
}
.ls-empty > .mdi {
  font-size: 24px;
  opacity: 0.8;
}
.ls-empty > span {
  font-size: 11px;
}
.ls-row {
  position: relative; /* anchors the drop ring */
  display: flex;
  align-items: center;
  gap: 6px;
}
.ls-row.off {
  opacity: 0.5;
}
/* The grabbed row is lifted, not ghosted — it's the real row, translated, so it stays exactly
   its own size at any canvas zoom. z-index carries it over its neighbours as it passes them. */
.ls-row.dragging {
  z-index: 5;
  box-shadow: 0 6px 16px rgb(0 0 0 / 45%);
  border-radius: var(--zen-radius, 6px);
  cursor: grabbing;
}
/* The drop-target ring is an overlay, not an `outline`. Descendants paint ABOVE their parent's
   outline, so the combo, the strength field and the remove button were each punching a hole in
   it wherever they overlapped — the ring only survived in the gaps. A positioned ::after with a
   z-index paints above every child instead, and sits just outside the row box so it rings the
   row rather than crowding its controls. pointer-events:none is load-bearing: a ring that ate
   dragover would block the drop it's advertising. */
.ls-row.over::after {
  content: "";
  position: absolute;
  inset: -2px;
  z-index: 3;
  border: 2px solid var(--zen-accent, #6366f1);
  border-radius: var(--zen-radius, 6px);
  pointer-events: none;
}

/* grip + dot travel together, so they pair at 2px instead of the row's 6px */
.ls-gutter {
  flex: none;
  display: inline-flex;
  align-items: center;
  gap: 2px;
}
.ls-grip {
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 12px;
  cursor: grab;
  color: var(--zen-muted, #9aa0aa);
  font-size: 14px;
  line-height: 1;
  /* The gesture-layer rules a drag handle needs (see useDragSurface): without touch-action the
     browser reserves the gesture for scroll and fires pointercancel on the first pixel of
     movement, and without user-select the drag turns into a text selection. */
  touch-action: none;
  -webkit-user-select: none;
  user-select: none;
}
/* the glyph must never be the pointer target — the grip itself is what the press resolves to */
.ls-grip > * {
  pointer-events: none;
}
.ls-grip:active {
  cursor: grabbing;
}
.ls-grip:hover {
  color: var(--zen-text, #e5e5ea);
}
.ls-pick {
  flex: 1;
  min-width: 0;
}
/* No width: ZenNumber's intrinsic width is derived from the widest value its own min/max can
   produce, so every row lines up at exactly the size the digits need. The 84px this used to
   hardcode was two pixels short of "-10.00". */
.ls-str {
  flex: none;
}
</style>
