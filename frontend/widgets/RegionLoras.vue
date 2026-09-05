<template>
  <ZenWidget :gap="6" pad="2px" class="rg">
    <!-- The groups container is the drag surface: pointer capture has to live on an element that
         survives the whole gesture, and it is also the box every drop zone is measured against.
         The "+ Region" button is inside it on purpose — it doubles as the zone that says "put this
         one somewhere new", and keeping it in the flow means the zones measured at grab time stay
         valid for the whole drag. -->
    <div
      :ref="(el) => (drag.el.value = el as HTMLElement | null)"
      class="rg-groups"
      v-bind="drag.handlers"
    >
      <div
        v-for="group in groups"
        :key="group.region"
        class="rg-group"
        :class="{ drop: dropRegion === group.region, source: dragRegion === group.region }"
        :data-region="group.region"
        :style="{ '--rg-hue': hue(group.region) }"
      >
        <div class="rg-head">
          <span class="rg-badge">{{ group.region }}</span>
          <span class="rg-title">Region {{ group.region }}</span>
          <span class="rg-note">{{ group.entries.length ? "" : "holds territory only" }}</span>
          <ZenIconButton
            icon="mdi mdi-plus"
            title="Add a LoRA to this region"
            @click="addRow(group.region)"
          />
        </div>

        <div v-if="group.entries.length" class="rg-rows">
          <div
            v-for="entry in group.entries"
            :key="entry.row.id"
            class="rg-row"
            :class="{ off: !entry.row.on, dragging: dragId === entry.row.id }"
            :data-row-id="entry.row.id"
            :style="
              dragId === entry.row.id ? { transform: `translateY(${dragOffset}px)` } : undefined
            "
          >
            <!-- Only the grip starts a drag; making the whole row grabbable would swallow the
                 picker's and the strength field's own drags. `data-zen-drag` is what stops the
                 press reaching the node: the grip is not a control, so mountWidget's INTERACTIVE
                 guard needs telling, or grabbing it drags the NODE under Nodes 2.0. -->
            <div class="rg-grip" data-zen-drag title="Drag onto another region">
              <i class="mdi mdi-drag-vertical" />
            </div>

            <ZenDot
              :model-value="entry.row.on"
              title="Enable"
              @update:model-value="store.patch(entry.index, { on: $event })"
            />

            <LoraPicker
              class="rg-pick"
              :model-value="entry.row.name"
              @update:model-value="store.patch(entry.index, { name: $event })"
              @browse="openBrowse(entry.index)"
            />

            <ZenNumber
              class="rg-str"
              :model-value="entry.row.strength"
              :step="0.05"
              :min="-4"
              :max="4"
              @update:model-value="store.patch(entry.index, { strength: $event })"
            />

            <ZenIconButton icon="mdi mdi-close" title="Remove" @click="removeRow(entry.index)" />
          </div>
        </div>
        <p v-else class="rg-slot">{{ dragging ? "drop here" : "no LoRA yet" }}</p>
      </div>

      <!-- The footer IS the new-region drop zone, which is why it lives inside the drag surface
           and why its contents may never change height: the zones are measured once, at grab
           time. `data-region` falls through onto its root, which is what the zone query finds.
           No "browse library" button here, deliberately — unlike the plain LoRA stack, a row in
           this table belongs to a REGION, and a browser opened from the footer would have no
           honest answer for which one its pick lands in. The per-row picker's own "Browse all"
           does, so that is the only way in. -->
      <WidgetActions
        data-region="new"
        :label="dragging ? `Move to region ${regionCount + 1}` : 'Region'"
        :title="dragging ? 'Drop to move this LoRA into a new region' : 'Show another region slot'"
        :active="dropRegion === regionCount + 1"
        @primary="addRegion"
      >
        <ZenIconButton
          v-if="canRemoveRegion"
          icon="mdi mdi-minus"
          title="Remove the last region slot"
          @click="removeRegion"
        />
        <ZenIconButton
          v-if="rows.length"
          icon="mdi mdi-broom"
          danger
          title="Clear all"
          @click="clearAll"
        />
      </WidgetActions>
    </div>

    <p v-if="!rows.length" class="rg-hint">
      Regions are numbered by their mask's place in the batch — region 1 is the first mask.
    </p>

    <LoraBrowser
      v-model:open="browse"
      :selected="rows[browseTarget]?.name"
      @pick="pickFromBrowse"
    />
  </ZenWidget>
</template>

<script lang="ts">
import type { WidgetOptions } from "@/framework";

/** Per-widget overrides picked up by lib/autoRegister.ts. */
export const widgetOptions: WidgetOptions = {
  minHeight: 90,
};
</script>

<script setup lang="ts">
/**
 * Region LoRAs — which LoRA runs on which region, as one table instead of a chain of nodes.
 *
 * Rows are grouped under the region they land on, and a region is just a NUMBER: the position of
 * its mask in the batch that built it. The widget cannot know how many regions will actually
 * arrive (masks are produced at execution time, the graph is edited long before), so the number of
 * groups on show is a display setting, kept in `node.properties` like the row order. The backend
 * is what checks a row against the regions that really turned up, and errors rather than dropping
 * it — a LoRA that was configured and then quietly did not run is the hardest failure to see.
 *
 * Several rows may name one region. They share its gate and their deltas add, which is stacking in
 * weight space, confined to that region.
 *
 * Moving a LoRA between regions is a DRAG, because that is the edit this node exists to make and
 * it is the one thing a per-row control could not express: this started as a "move to region"
 * menu, and swapping two LoRAs meant finding a row, opening its menu, finding the region, and
 * doing the whole thing again for the LoRA coming the other way. Dragging a row onto a region's
 * box moves it there; dropping on "+ Region" moves it into a new one.
 */
import { computed, ref } from "vue";
import { useDragSurface } from "@nynxz/zenkit-nodekit";
import { ZenDot, ZenIconButton, ZenNumber, ZenWidget } from "@nynxz/zenkit-ui";

import LoraBrowser from "@/components/LoraBrowser.vue";
import LoraPicker from "@/components/LoraPicker.vue";
import WidgetActions from "@/components/WidgetActions.vue";
import * as lib from "@/lib/loraLibrary";
import { rowsFrom } from "@/lib/widgetRows";

interface RegionRow {
  id: number;
  on: boolean;
  region: number;
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

function int(value: unknown, fallback: number): number {
  const n = Math.round(+(value as number));
  return Number.isFinite(n) ? n : fallback;
}

// The emitted value is region-major and sorted, so rearranging the table cannot change the node's
// signature — the deltas are additive, so the order genuinely does not affect the result.
const store = rowsFrom<RegionRow>(props, {
  orderKey: "nynxzRegionLoraOrder",
  make: (raw, id) => ({
    id,
    on: raw.on !== false,
    region: Math.max(1, int(raw.region, 1)),
    name: String(raw.name ?? ""),
    strength: Number.isFinite(+(raw.strength as number)) ? +(raw.strength as number) : 1,
  }),
  project: (r) => ({ on: r.on, region: r.region, name: r.name, strength: r.strength }),
  sort: (a, b) => a.region - b.region || a.name.localeCompare(b.name) || a.strength - b.strength,
});
const { rows } = store;

// How many region slots to SHOW. Presentational, so it lives in node.properties and never reaches
// the widget value — and it can only ever grow to cover the rows that exist, so a saved graph
// never opens with a row hidden under a group that isn't drawn.
const SLOTS_KEY = "nynxzRegionSlots";
const slots = ref(Math.max(1, int(props.node?.properties?.[SLOTS_KEY], 2)));
const regionCount = computed(() => Math.max(slots.value, ...rows.value.map((r) => r.region), 1));

function setSlots(n: number) {
  slots.value = Math.max(1, n);
  if (props.node?.properties) {
    // eslint-disable-next-line vue/no-mutating-props -- node.properties is ComfyUI's own store; writing it is how a node body persists something the backend must not see.
    props.node.properties[SLOTS_KEY] = slots.value;
  }
  props.node?.graph?.setDirtyCanvas?.(true, true);
}

const groups = computed(() =>
  Array.from({ length: regionCount.value }, (_, i) => ({
    region: i + 1,
    entries: rows.value.map((row, index) => ({ row, index })).filter((e) => e.row.region === i + 1),
  })),
);

// Only the LAST slot, and only when it is empty: removing one with rows in it would leave them
// pointing at a region with nothing to draw them under.
const canRemoveRegion = computed(
  () => regionCount.value > 1 && !rows.value.some((r) => r.region === regionCount.value),
);

/** A stable colour per region, so a row's group is readable at a glance. */
function hue(region: number): string {
  return `${(region * 67 + 200) % 360}`;
}

const browse = ref(false);
const browseTarget = ref(-1); // the row the browser will fill — always a real one, see the footer
if (rows.value.some((r) => r.name)) void lib.ensure();

function addRow(region: number) {
  rows.value.push({ id: store.nextId(), on: true, region, name: "", strength: 1 });
  store.commit();
}
function removeRow(index: number) {
  rows.value.splice(index, 1);
  store.commit();
}
function clearAll() {
  rows.value = [];
  store.commit();
}
function addRegion() {
  setSlots(regionCount.value + 1);
}
function removeRegion() {
  setSlots(regionCount.value - 1);
}

function openBrowse(index: number) {
  browseTarget.value = index;
  browse.value = true;
}
function pickFromBrowse(name: string) {
  store.patch(browseTarget.value, { name });
  browse.value = false;
}

// --- drag a LoRA between regions -----------------------------------------------------------
//
// Pointer-driven, NOT HTML5 drag-and-drop: the native ghost is a browser-rendered snapshot with no
// say over its scale, and a node body sits inside a CSS-transformed container, so the ghost comes
// out at a size matching neither the row nor the canvas zoom. Here the ROW ITSELF is translated —
// the same element, inside the same transform — so it is exactly the row you grabbed, at any zoom.
//
// Only the target REGION is resolved, not an insertion point, and that is not a shortcut: several
// LoRAs on one region are additive and share its gate, so their order within it has no effect on
// the result. Asking the drop for an index would be inventing a decision that does not exist.
interface Zone {
  region: number;
  top: number;
  bottom: number;
}
let zones: Zone[] = [];
let grabCenter = 0;

const dragId = ref(-1); // row id being dragged, or -1
const dropRegion = ref(-1); // region the row would land in, or -1
const dragOffset = ref(0); // how far the grabbed row has moved, element-space px
const dragging = computed(() => dragId.value >= 0);
const dragRegion = computed(() => rows.value.find((r) => r.id === dragId.value)?.region ?? -1);

const drag = useDragSurface({
  onStart(ctx) {
    const grip = (ctx.event.target as HTMLElement | null)?.closest?.(".rg-grip");
    const host = drag.el.value;
    if (!grip || !host) return false; // press wasn't on a handle — leave it to the controls
    const rowEl = grip.closest(".rg-row") as HTMLElement | null;
    const id = Number(rowEl?.dataset.rowId);
    if (!rowEl || !Number.isFinite(id)) return false;

    // offsetTop/offsetHeight are pre-transform, which is the space ctx.delta already lands in (the
    // helper divides the canvas zoom out), so the two add up without a second correction. They are
    // measured against `.rg-groups` because it is the only positioned ancestor — see its CSS.
    zones = [...host.querySelectorAll<HTMLElement>("[data-region]")].map((el) => ({
      region: el.dataset.region === "new" ? regionCount.value + 1 : Number(el.dataset.region),
      top: el.offsetTop,
      bottom: el.offsetTop + el.offsetHeight,
    }));
    grabCenter = rowEl.offsetTop + rowEl.offsetHeight / 2;
    dragId.value = id;
    dropRegion.value = dragRegion.value;
    dragOffset.value = 0;
  },
  onMove(ctx) {
    if (dragId.value < 0 || !zones.length) return;
    dragOffset.value = ctx.delta.y;
    const y = grabCenter + ctx.delta.y;
    // Clamped rather than nulled outside the zones: a drag that overshoots the top or the bottom
    // means the nearest end, which is what the pointer is heading for. There is no "cancel by
    // dragging away" here, so every position has to name a region.
    const hit = zones.find((z) => y >= z.top && y < z.bottom);
    dropRegion.value = hit
      ? hit.region
      : y < zones[0]!.top
        ? zones[0]!.region
        : zones.at(-1)!.region;
  },
  // Commit once, at the end: re-pointing on every move would rewrite the table hundreds of times.
  onEnd() {
    const id = dragId.value;
    const target = dropRegion.value;
    dragId.value = -1;
    dropRegion.value = -1;
    dragOffset.value = 0;
    zones = [];
    if (id < 0 || target < 1) return;
    const index = rows.value.findIndex((r) => r.id === id);
    if (index < 0 || rows.value[index]!.region === target) return;
    // Dropped past the last group: keep the new slot, so dragging the row back out later leaves
    // the empty region on screen instead of collapsing it.
    if (target > regionCount.value) setSlots(target);
    store.patch(index, { region: target });
  },
});
</script>

<style scoped>
/* The only positioned ancestor inside the widget, so every zone's offsetTop is measured against
   it. Nothing between it and a row may be positioned or the drag maths silently shifts. */
.rg-groups {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.rg-group {
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 5px 6px 6px;
  border: 1px solid var(--zen-border, #34343c);
  border-left: 3px solid hsl(var(--rg-hue) 62% 58%);
  border-radius: var(--zen-radius, 8px);
  background: color-mix(in srgb, var(--zen-text, #fff) 2%, transparent);
  transition:
    border-color 0.1s ease,
    background 0.1s ease;
}
/* the group the pointer is over, and the one the row came from */
.rg-group.drop {
  border-color: var(--zen-accent, #6366f1);
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 10%, transparent);
}
.rg-group.source:not(.drop) {
  border-style: dashed;
}

.rg-head {
  display: flex;
  align-items: center;
  gap: 6px;
}
.rg-badge {
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 17px;
  height: 17px;
  border-radius: 50%;
  background: hsl(var(--rg-hue) 62% 58%);
  color: #101014;
  font-size: 10px;
  font-weight: 700;
}
.rg-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--zen-text, #e5e5ea);
}
.rg-note {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 10px;
  font-style: italic;
  color: var(--zen-muted, #9aa0aa);
}
/* An empty group still has to be a target big enough to aim at, so it keeps a row's worth of
   height rather than collapsing to its header. */
.rg-slot {
  margin: 0;
  padding: 6px;
  text-align: center;
  font-size: 10px;
  color: var(--zen-muted, #9aa0aa);
  border: 1px dashed var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 6px);
}

.rg-rows {
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.rg-row {
  display: flex;
  align-items: center;
  gap: 6px;
}
.rg-row.off {
  opacity: 0.5;
}
/* The grabbed row is lifted, not ghosted — it's the real row, translated, so it stays exactly its
   own size at any canvas zoom. The z-index carries it over the groups it passes; no ancestor may
   set one, or it would be trapped inside its own group's stacking context. */
.rg-row.dragging {
  position: relative;
  z-index: 5;
  box-shadow: 0 6px 16px rgb(0 0 0 / 45%);
  border-radius: var(--zen-radius, 6px);
  cursor: grabbing;
}

.rg-grip {
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
.rg-grip > * {
  pointer-events: none;
}
.rg-grip:active {
  cursor: grabbing;
}
.rg-grip:hover {
  color: var(--zen-text, #e5e5ea);
}

.rg-pick {
  flex: 1;
  min-width: 0;
}
/* No width: ZenNumber's intrinsic width is now derived from the widest value its own min/max
   can produce, so every row lines up at exactly the size the digits need and nothing clips. */
.rg-str {
  flex: none;
}
.rg-hint {
  margin: 0;
  text-align: center;
  font-size: 10px;
  line-height: 1.4;
  color: var(--zen-muted, #9aa0aa);
}
</style>
