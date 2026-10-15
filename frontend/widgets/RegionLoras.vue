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
      :class="{ settling }"
      v-bind="drag.handlers"
    >
      <div
        v-for="group in groups"
        :key="group.region"
        class="rg-group"
        :class="{
          drop: dropRegion === group.region,
          source: dragRegion === group.region,
          lifted: dragHead === group.region,
        }"
        :data-region="group.region"
        :style="
          dragHead === group.region
            ? { transform: `translateY(${dragOffset}px)` }
            : swapStyle(group.region)
        "
      >
        <div class="rg-head">
          <div
            class="rg-grip"
            data-zen-drag
            title="Drag onto another region to swap them, prompt and LoRAs together"
          >
            <i class="mdi mdi-drag-vertical" />
          </div>
          <span class="rg-label" :title="`The ${ordinal(group.region)} mask in the regions batch`">
            <span class="rg-title">Region {{ group.region }}</span>
            <span v-if="group.entries.length" class="rg-count">{{ group.entries.length }}</span>
          </span>
          <ZenIconButton
            v-if="group.region === regionCount && canRemoveRegion"
            icon="mdi mdi-close"
            title="Remove this region slot"
            @click="removeRegion"
          />
          <ZenIconButton
            icon="mdi mdi-plus"
            title="Add a LoRA to this region"
            @click="addRow(group.region)"
          />
        </div>

        <ZenInput
          class="rg-prompt"
          type="textarea"
          :rows="1"
          autosize
          sm
          :model-value="prompts[group.region] ?? ''"
          placeholder="Prompt for this region (optional)"
          title="Replaces the main prompt in this region. Needs Regions Apply's clip wired."
          @update:model-value="setPrompt(group.region, String($event ?? ''))"
        />

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
        <p v-else-if="dragging || !prompts[group.region]" class="rg-slot">
          {{ dragging ? "drop here" : "no LoRA yet" }}
        </p>
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
        :title="
          dragHead > 0
            ? 'Drop to move this region into a new slot'
            : dragging
              ? 'Drop to move this LoRA into a new region'
              : 'Show another region slot'
        "
        :active="dropRegion === regionCount + 1"
        @primary="addRegion"
      >
        <ZenIconButton
          v-if="rows.length"
          icon="mdi mdi-broom"
          danger
          title="Clear all"
          @click="clearAll"
        />
      </WidgetActions>
    </div>

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
  minWidth: 260,
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
import { ZenDot, ZenIconButton, ZenInput, ZenNumber, ZenWidget } from "@nynxz/zenkit-ui";

import WidgetActions from "@/components/WidgetActions.vue";
import { loraLibrary as lib, LoraBrowser, LoraPicker } from "@nynxz/zenkit-ui";
import { entriesOf, rowsFrom } from "@/lib/widgetRows";

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

// A region's own prompt rides in the same value as a `{ region, prompt }` entry with no `name`, so
// the backend gets it with the LoRAs and older graphs, which have none, load unchanged.
const isPrompt = (raw: Record<string, unknown>) => "prompt" in raw && !raw.name;
const prompts = ref<Record<number, string>>(
  Object.fromEntries(
    entriesOf(props.widget?.value)
      .filter(isPrompt)
      .map((raw) => [Math.max(1, int(raw.region, 1)), String(raw.prompt ?? "")]),
  ),
);

// The emitted value is region-major and sorted, so rearranging the table cannot change the node's
// signature — the deltas are additive, so the order genuinely does not affect the result.
const store = rowsFrom<RegionRow>(props, {
  isRow: (raw) => !isPrompt(raw),
  extra: () =>
    Object.entries(prompts.value)
      .filter(([, prompt]) => prompt.trim())
      .map(([region, prompt]) => ({ region: +region, prompt }))
      .sort((a, b) => a.region - b.region),
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
const regionCount = computed(() =>
  Math.max(
    slots.value,
    ...rows.value.map((r) => r.region),
    ...Object.keys(prompts.value)
      .filter((r) => prompts.value[+r]?.trim())
      .map(Number),
    1,
  ),
);

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
  () =>
    regionCount.value > 1 &&
    !rows.value.some((r) => r.region === regionCount.value) &&
    !prompts.value[regionCount.value]?.trim(),
);

function setPrompt(region: number, prompt: string) {
  prompts.value = { ...prompts.value, [region]: prompt };
  store.commit();
}

function ordinal(n: number): string {
  const suffix =
    n % 100 >= 11 && n % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[n % 10] ?? "th");
  return `${n}${suffix}`;
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
  prompts.value = {};
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
// The swap lands with transitions off: the regions it previewed are already where they go.
const settling = ref(false);

/** While a region is dragged over another, that one slides into its place — the swap, shown
 *  before the drop. Below the dragged region it takes its top; above it, its bottom. */
function swapStyle(region: number) {
  if (dragHead.value <= 0 || region !== dropRegion.value || region === dragHead.value)
    return undefined;
  const from = zones.find((z) => z.region === dragHead.value);
  const to = zones.find((z) => z.region === region);
  if (!from || !to) return undefined;
  const top = to.top > from.top ? from.top : from.bottom - (to.bottom - to.top);
  return { transform: `translateY(${top - to.top}px)` };
}

const dragId = ref(-1); // row id being dragged, or -1
const dragHead = ref(-1); // region being dragged by its header, or -1
const dropRegion = ref(-1); // region the row would land in, or -1
const dragOffset = ref(0); // how far the grabbed row has moved, element-space px
const dragging = computed(() => dragId.value >= 0 || dragHead.value > 0);
const dragRegion = computed(() =>
  dragHead.value > 0
    ? dragHead.value
    : (rows.value.find((r) => r.id === dragId.value)?.region ?? -1),
);

/** Swap two regions' contents — their LoRAs and their prompts — or move one into an empty slot. */
function swapRegions(a: number, b: number) {
  for (const row of rows.value) {
    if (row.region === a) row.region = b;
    else if (row.region === b) row.region = a;
  }
  const { [a]: fromA = "", [b]: fromB = "" } = prompts.value;
  prompts.value = { ...prompts.value, [a]: fromB, [b]: fromA };
  if (b > regionCount.value) setSlots(b);
  store.commit();
}

const drag = useDragSurface({
  onStart(ctx) {
    const grip = (ctx.event.target as HTMLElement | null)?.closest?.(".rg-grip");
    const host = drag.el.value;
    if (!grip || !host) return false; // press wasn't on a handle — leave it to the controls
    const headEl = grip.closest(".rg-head")
      ? (grip.closest(".rg-group") as HTMLElement | null)
      : null;
    const rowEl = headEl ?? (grip.closest(".rg-row") as HTMLElement | null);
    const id = Number(rowEl?.dataset.rowId);
    const head = Number(headEl?.dataset.region);
    if (!rowEl || !(headEl ? head > 0 : Number.isFinite(id))) return false;

    // offsetTop/offsetHeight are pre-transform, which is the space ctx.delta already lands in (the
    // helper divides the canvas zoom out), so the two add up without a second correction. They are
    // measured against `.rg-groups` because it is the only positioned ancestor — see its CSS.
    zones = [...host.querySelectorAll<HTMLElement>("[data-region]")].map((el) => ({
      region: el.dataset.region === "new" ? regionCount.value + 1 : Number(el.dataset.region),
      top: el.offsetTop,
      bottom: el.offsetTop + el.offsetHeight,
    }));
    grabCenter = rowEl.offsetTop + rowEl.offsetHeight / 2;
    if (headEl) dragHead.value = head;
    else dragId.value = id;
    dropRegion.value = dragRegion.value;
    dragOffset.value = 0;
  },
  onMove(ctx) {
    if (!dragging.value || !zones.length) return;
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
    const head = dragHead.value;
    const target = dropRegion.value;
    settling.value = true;
    requestAnimationFrame(() => requestAnimationFrame(() => (settling.value = false)));
    dragId.value = -1;
    dragHead.value = -1;
    dropRegion.value = -1;
    dragOffset.value = 0;
    zones = [];
    if (target < 1) return;
    if (head > 0) {
      if (head !== target) swapRegions(head, target);
      return;
    }
    if (id < 0) return;
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
  padding: 4px 6px 6px;
  border: 1px solid var(--zen-surface-border, var(--zen-border, #34343c));
  border-radius: var(--zen-radius, 8px);
  background: color-mix(in srgb, var(--zen-text, #fff) 3%, transparent);
  transition:
    border-color 0.12s ease,
    background 0.12s ease,
    transform 0.15s ease;
}
.rg-group:hover {
  background: color-mix(in srgb, var(--zen-text, #fff) 5%, transparent);
}
/* the group the pointer is over, and the one the row came from */
.rg-group.drop {
  border-color: var(--zen-accent, #6366f1);
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 10%, transparent);
}
.rg-group.source:not(.drop) {
  border-style: dashed;
}

.rg-prompt {
  width: 100%;
}
.rg-groups.settling .rg-group,
.rg-groups.settling .rg-row {
  transition: none;
}
.rg-group.lifted {
  position: relative;
  z-index: 1;
  transition: none;
  background: color-mix(in srgb, var(--zen-text, #fff) 5%, var(--zen-surface, #1f1f24));
  box-shadow: 0 6px 18px rgb(0 0 0 / 35%);
}
.rg-head {
  display: flex;
  align-items: center;
  gap: 4px;
  min-height: 24px;
  padding-left: 2px;
}
.rg-label {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: help;
}
.rg-title {
  font-size: 12px;
  color: var(--zen-text, #e5e5ea);
  white-space: nowrap;
}
.rg-count {
  flex: none;
  padding: 0 5px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--zen-text, #fff) 7%, transparent);
  color: var(--zen-muted, #9aa0aa);
  font-size: 10px;
  line-height: 15px;
}
/* An empty group still has to be a target big enough to aim at, so it keeps a row's worth of
   height rather than collapsing to its header. */
.rg-slot {
  margin: 0;
  padding: 6px;
  text-align: center;
  font-size: 10px;
  color: var(--zen-muted, #9aa0aa);
  border: 1px dashed var(--zen-surface-border, var(--zen-border, #34343c));
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
</style>
