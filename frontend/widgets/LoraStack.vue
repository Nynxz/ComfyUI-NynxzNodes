<template>
  <ZenWidget ref="rootEl" :gap="6" pad="2px" class="ls">
    <div class="ls-mid">
      <!-- The rows container is the drag surface: pointer capture has to live on an element that
           survives the whole gesture, and the grips are just the part of it you may grab. -->
      <div
        v-if="items.length"
        :ref="(el) => (reorder.el.value = el as HTMLElement | null)"
        class="ls-rows"
        :class="{ settling }"
        v-bind="reorder.handlers"
      >
        <!-- Group headers and rows are siblings in ONE list, so a row dragged past a header
             changes group: a row belongs to the header above it, and to no group above the
             first one. -->
        <template v-for="(item, i) in items" :key="item.key">
          <div
            v-if="item.group"
            class="ls-group"
            :class="{ muted: item.group.muted, closed: !item.group.open }"
            :style="itemStyle(i)"
          >
            <button
              class="ls-fold"
              :title="item.group.open ? 'Fold' : 'Unfold'"
              @click="toggleOpen(item.group)"
            >
              <i class="mdi" :class="item.group.open ? 'mdi-chevron-down' : 'mdi-chevron-right'" />
            </button>
            <ZenDot
              :model-value="!item.group.muted"
              :title="item.group.muted ? 'Muted: none of these apply' : 'Mute the whole group'"
              @update:model-value="setMuted(item.group, !$event)"
            />
            <input
              v-if="renaming === item.group.id"
              v-model="renameText"
              v-focus
              class="ls-gname-edit"
              spellcheck="false"
              @keydown.enter.prevent="endRename(true)"
              @keydown.esc.prevent.stop="endRename(false)"
              @blur="endRename(true)"
            />
            <button
              v-else
              class="ls-gname"
              title="Click to fold · double-click to rename"
              @click="toggleOpen(item.group)"
              @dblclick.stop="startRename(item.group)"
            >
              {{ item.group.name }}
            </button>
            <i
              v-if="driven"
              class="mdi mdi-connection ls-driven"
              :title="`The select input can choose from this group: '${item.group.name}: 0' uses its first LoRA, '${item.group.name}: off' none`"
            />
            <i
              v-if="item.group.solo"
              class="mdi mdi-radiobox-marked ls-solo"
              title="Solo: only one of these is on at a time"
            />
            <span class="ls-gcount" :title="`${item.on} of ${item.count} on`">
              {{ summary(item) }}
            </span>
            <ZenIconButton
              size="sm"
              icon="mdi mdi-plus"
              title="Add a LoRA to this group"
              @click="addRow('', item.group.id)"
            />
            <ZenIconButton
              size="sm"
              icon="mdi mdi-dots-horizontal"
              title="Group options"
              :active="menuGroup?.id === item.group.id"
              @click="openGroupMenu($event, item.group)"
            />
          </div>

          <div
            v-else
            class="ls-row"
            :class="{
              off: !item.row.on || mutedBy(item.row),
              grouped: !!item.row.group,
              dragging: dragIndex === i,
            }"
            :style="itemStyle(i)"
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
              <div class="ls-grip" data-zen-drag title="Drag to reorder, or into a group">
                <i class="mdi mdi-drag-vertical" />
              </div>
              <!-- ZenDot, not ZenSwitch: a per-row enable only needs to read as on/off, and the
                   switch spent 42px of the row on saying it. -->
              <ZenDot
                :model-value="item.row.on"
                title="Enable"
                @update:model-value="edit(item.row, { on: $event })"
              />
            </div>

            <LoraPicker
              class="ls-pick"
              :model-value="item.row.name"
              @update:model-value="edit(item.row, { name: $event })"
              @browse="openBrowse(item.row)"
            />

            <ZenNumber
              class="ls-str"
              :model-value="item.row.strength"
              :step="0.05"
              :min="-10"
              :max="10"
              @update:model-value="edit(item.row, { strength: $event })"
            />
            <ZenIconButton icon="mdi mdi-close" title="Remove" @click="removeRow(item.row)" />
          </div>
        </template>
      </div>
      <div v-else class="ls-empty">
        <i class="mdi mdi-layers-triple-outline" />
        <span>No LoRAs in this stack yet</span>
      </div>
    </div>

    <ZenPopover v-model:open="menuOpen" :anchor="menuAnchor" placement="bottom-end">
      <template v-if="menuGroup">
        <ZenMenuItem icon="mdi mdi-pencil-outline" @select="menuDo(startRename)">
          Rename
        </ZenMenuItem>
        <ZenMenuItem
          :icon="menuGroup.muted ? 'mdi mdi-volume-high' : 'mdi mdi-volume-off'"
          @select="menuDo((g) => setMuted(g, !g.muted))"
        >
          {{ menuGroup.muted ? "Unmute" : "Mute" }}
        </ZenMenuItem>
        <ZenMenuItem icon="mdi mdi-radiobox-marked" @select="menuDo((g) => setSolo(g, !g.solo))">
          Solo: one at a time
          <template #lead><ZenDot :model-value="menuGroup.solo" /></template>
        </ZenMenuItem>
        <ZenMenuItem icon="mdi mdi-arrow-up" :disabled="!canMove(-1)" @select="menuDo(moveUp)">
          Move up
        </ZenMenuItem>
        <ZenMenuItem icon="mdi mdi-arrow-down" :disabled="!canMove(1)" @select="menuDo(moveDown)">
          Move down
        </ZenMenuItem>
        <ZenMenuSeparator />
        <ZenMenuItem icon="mdi mdi-folder-open-outline" @select="menuDo(ungroup)">
          Ungroup, keep the LoRAs
        </ZenMenuItem>
        <ZenMenuItem icon="mdi mdi-delete-outline" danger @select="menuDo(deleteGroup)">
          Delete group and its LoRAs
        </ZenMenuItem>
      </template>
    </ZenPopover>

    <ZenPopover
      v-if="mode"
      v-model:open="modeOpen"
      :anchor="modeAnchor"
      placement="bottom-end"
      role="dialog"
    >
      <div class="ls-pop-head">Applies to</div>
      <ZenMenuItem
        v-for="m in LORA_MODES"
        :key="m.value"
        :icon="m.icon"
        @select="pickMode(m.value)"
      >
        {{ m.label }}
        <template #lead><ZenDot :model-value="m.value === mode" /></template>
      </ZenMenuItem>
      <ZenMenuSeparator />
      <div class="ls-pop-head">Inputs</div>
      <ZenMenuItem icon="mdi mdi-connection" @select="toggleSelect">
        Select: choose a group's LoRAs from the graph
        <template #lead><ZenDot :model-value="selectOn" /></template>
      </ZenMenuItem>
    </ZenPopover>

    <!-- Clearing takes every row and group with it and can't be undone from the node, so it asks. -->
    <ZenPopover v-model:open="clearOpen" :anchor="clearAnchor" placement="top-end" role="dialog">
      <div class="ls-confirm">
        <div class="ls-confirm-text">{{ clearQuestion }}</div>
        <div class="ls-confirm-actions">
          <ZenButton size="sm" variant="ghost" @click="clearOpen = false">Cancel</ZenButton>
          <ZenButton size="sm" variant="danger" icon="mdi mdi-broom" @click="clearAll">
            Clear
          </ZenButton>
        </div>
      </div>
    </ZenPopover>

    <template #footer>
      <WidgetActions label="Add LoRA" :count="rows.length" @primary="addRow()">
        <ZenIconButton
          icon="mdi mdi-folder-plus-outline"
          title="New group: LoRAs you can fold away or mute together"
          @click="addGroup"
        />
        <ZenIconButton
          icon="mdi mdi-view-grid-outline"
          title="Browse library"
          @click="openBrowse(null)"
        />
        <ZenIconButton
          v-if="items.length"
          icon="mdi mdi-broom"
          danger
          title="Clear all"
          :active="clearOpen"
          @click="askClear"
        />
      </WidgetActions>
    </template>

    <LoraBrowser v-model:open="browse" :selected="browseRow?.name" @pick="pickFromBrowse" />
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
  minWidth: 260,
};
</script>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch, type Directive } from "vue";
import { useDragSurface } from "@nynxz/zenkit-nodekit";
import {
  insertionIndex,
  landingTop,
  rowStride,
  shiftStyle,
  sortShift,
  ZenButton,
  ZenDot,
  ZenIconButton,
  ZenMenuItem,
  ZenMenuSeparator,
  ZenNumber,
  ZenPopover,
  ZenWidget,
} from "@nynxz/zenkit-ui";

import WidgetActions from "@/components/WidgetActions.vue";
import { loraLibrary as lib, LoraBrowser, LoraPicker } from "@nynxz/zenkit-ui";
import { rowsFrom, type RowStore } from "@/lib/widgetRows";
import {
  addNodeHeaderButton,
  type HeaderButtonAnchor,
  type NodeHeaderButtonHandle,
} from "@/framework";
import {
  LORA_MODES,
  loraMode,
  selectShown,
  setLoraMode,
  setSelectShown,
  type LoraMode,
  type LoraNodeLike,
} from "@/lib/loraMode";

interface LoraRow {
  id: number;
  on: boolean;
  name: string;
  strength: number;
  /** The group it sits in, or 0 for none. */
  group: number;
}
/** A named set of rows that folds away and mutes as one: speed-up LoRAs, a style pack. */
interface LoraGroup {
  id: number;
  name: string;
  open: boolean;
  muted: boolean;
  /** Only one of its rows is on at a time: alternatives to pick between, not a set to combine. */
  solo: boolean;
}
type Item =
  | { key: string; group: LoraGroup; row?: undefined; count: number; on: number }
  | { key: string; row: LoraRow; group?: undefined };

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
//
// Groups are display too, and live beside the order in node.properties: folding or renaming one
// changes nothing the backend sees. Muting one does, and reaches the value as its rows being off,
// so the backend needs to know nothing about groups at all.
const GROUPS_KEY = "nynxzLoraGroups";
// ComfyUI's node, not Vue state: its properties are where a node body keeps what it draws.
const saved = props.node?.properties;
const savedGroups = saved?.[GROUPS_KEY];
const groups = ref<LoraGroup[]>(
  (Array.isArray(savedGroups) ? (savedGroups as Record<string, unknown>[]) : [])
    .filter((g) => g && Number.isFinite(+(g.id as number)))
    .map((g) => ({
      id: +(g.id as number),
      name: String(g.name ?? "Group"),
      open: g.open !== false,
      muted: g.muted === true,
      solo: g.solo === true,
    })),
);
const groupById = (id: number) => groups.value.find((g) => g.id === id);

// The node's `select` input chooses a group's LoRAs from the graph (nodes/lora/_select.py), by
// group name and place. Only while it is connected do rows carry those to the backend: the rest
// of the time renaming a group or reordering inside one still changes nothing that is hashed.
interface DrivableNode {
  inputs?: { name: string; link?: number | null }[];
  onConnectionsChange?: (...args: unknown[]) => void;
  onConfigure?: (...args: unknown[]) => void;
}
const hostNode = props.node as DrivableNode | undefined;
const selectLinked = () => hostNode?.inputs?.find((slot) => slot.name === "select")?.link != null;
const driven = ref(selectLinked());
const mutedBy = (row: LoraRow) => !!groupById(row.group)?.muted;

const store: RowStore<LoraRow> = rowsFrom<LoraRow>(props, {
  orderKey: "nynxzLoraOrder",
  make: (raw, id) => ({
    id,
    on: raw.on !== false,
    name: String(raw.name ?? ""),
    strength: Number.isFinite(+(raw.strength as number)) ? +(raw.strength as number) : 1,
    group: Number.isFinite(+(raw.group as number)) ? +(raw.group as number) : 0,
  }),
  project: (r) => {
    const row = { on: r.on && !mutedBy(r), name: r.name, strength: r.strength };
    const group = driven.value ? groupById(r.group) : undefined;
    if (!group) return row;
    // Its place in the group, from 0: what the select text counts in.
    const slot = rows.value.filter((other) => other.group === r.group).indexOf(r);
    return { ...row, group: group.name, slot };
  },
  sort: (a, b) => a.name.localeCompare(b.name) || a.strength - b.strength || +a.on - +b.on,
});
const { rows } = store;
// A row whose group is gone (an older graph, a hand-edited one) is simply ungrouped.
for (const row of rows.value) if (row.group && !groupById(row.group)) row.group = 0;

/** Write the groups, then the rows and the value. */
function commit() {
  if (saved) saved[GROUPS_KEY] = groups.value.map((g) => ({ ...g }));
  store.commit();
}

// What is drawn, top to bottom: the ungrouped rows, then each group's header and, unfolded, its
// rows. The drag below works on this list.
const items = computed<Item[]>(() => {
  const out: Item[] = rows.value
    .filter((row) => !row.group)
    .map((row) => ({ key: `r${row.id}`, row }));
  for (const group of groups.value) {
    const mine = rows.value.filter((row) => row.group === group.id);
    out.push({
      key: `g${group.id}`,
      group,
      count: mine.length,
      on: mine.filter((row) => row.on).length,
    });
    if (group.open) for (const row of mine) out.push({ key: `r${row.id}`, row });
  }
  return out;
});

// Only LoRA Loader has an "apply to" setting; the deprecated CLIP loader shares this widget.
const loraNode = props.node as LoraNodeLike | undefined;
const mode = ref<LoraMode | null>(loraNode ? loraMode(loraNode) : null);
const modeOpen = ref(false);
const modeAnchor = ref<HeaderButtonAnchor | null>(null);
function pickMode(next: LoraMode) {
  modeOpen.value = false;
  if (!loraNode) return;
  setLoraMode(loraNode, next);
  mode.value = next;
}
// Connecting or disconnecting `select` changes what the rows must carry, so the value is
// written again whenever the node's links change (and once a loaded graph has restored them).
function syncDriven() {
  if (loraNode) selectOn.value = selectShown(loraNode);
  const now = selectLinked();
  if (now === driven.value) return;
  driven.value = now;
  commit();
}
let unhook = () => {};
onMounted(() => {
  if (!hostNode) return;
  const prevLinks = hostNode.onConnectionsChange;
  const prevConfigure = hostNode.onConfigure;
  const onLinks = function (this: unknown, ...args: unknown[]) {
    prevLinks?.apply(this, args);
    syncDriven();
  };
  const onConfigure = function (this: unknown, ...args: unknown[]) {
    prevConfigure?.apply(this, args);
    syncDriven();
  };
  hostNode.onConnectionsChange = onLinks;
  hostNode.onConfigure = onConfigure;
  syncDriven();
  unhook = () => {
    if (hostNode.onConnectionsChange === onLinks) hostNode.onConnectionsChange = prevLinks;
    if (hostNode.onConfigure === onConfigure) hostNode.onConfigure = prevConfigure;
  };
});
onBeforeUnmount(() => unhook());

const selectOn = ref(loraNode ? selectShown(loraNode) : false);
function toggleSelect() {
  modeOpen.value = false;
  if (!loraNode) return;
  setSelectShown(loraNode, !selectOn.value);
  selectOn.value = selectShown(loraNode);
  syncDriven();
}

const rootEl = ref<{ el: HTMLElement | null } | null>(null);
let modeCog: NodeHeaderButtonHandle | null = null;
onMounted(() => {
  if (mode.value === null) return;
  modeCog = addNodeHeaderButton(props.node, rootEl.value?.el ?? null, {
    icon: "mdi mdi-cog",
    text: "⚙",
    title: "Applies to",
    onClick: (anchor) => {
      modeAnchor.value = anchor;
      modeOpen.value = !modeOpen.value;
    },
  });
});
watch(modeOpen, (open) => modeCog?.setActive(open));
onBeforeUnmount(() => modeCog?.destroy());

const dragIndex = ref(-1); // row being dragged, or -1
const dropIndex = ref(-1); // index the dragged row would land at, or -1
const dragOffset = ref(0); // how far the grabbed row has moved, element-space px
const browse = ref(false);
const browseRow = ref<LoraRow | null>(null); // the row to pick into, or null = add a new one

// A node that opens with saved LoRAs fetches the list once (cached across nodes) so a deleted or
// renamed one is flagged without opening the picker; empty nodes stay lazy.
if (rows.value.some((r) => r.name)) void lib.ensure();

function edit(row: LoraRow, fields: Partial<LoraRow>) {
  Object.assign(row, fields);
  if (fields.on) onlyOne(groupById(row.group), row);
  commit();
}
/** In a solo group, leave on just `keep` (or the first row that is on). */
function onlyOne(group: LoraGroup | undefined, keep?: LoraRow) {
  if (!group?.solo) return;
  const mine = rows.value.filter((row) => row.group === group.id);
  const chosen = keep?.on ? keep : mine.find((row) => row.on);
  for (const row of mine) if (row !== chosen) row.on = false;
}
function addRow(name = "", group = 0) {
  const target = groupById(group);
  // A solo group that already has its one keeps it: the new row waits its turn, switched off.
  const taken = !!target?.solo && rows.value.some((r) => r.group === group && r.on);
  const row: LoraRow = { id: store.nextId(), on: !taken, name, strength: 1, group };
  // After the last row of its group, so the list keeps reading group by group.
  const last = rows.value.map((r) => r.group).lastIndexOf(group);
  if (last < 0) rows.value.push(row);
  else rows.value.splice(last + 1, 0, row);
  if (target) target.open = true;
  commit();
}
function removeRow(row: LoraRow) {
  rows.value = rows.value.filter((r) => r !== row);
  commit();
}
const clearOpen = ref(false);
const clearAnchor = ref<HTMLElement | null>(null);
const clearQuestion = computed(() => {
  const count = (n: number, one: string) => `${n} ${one}${n === 1 ? "" : "s"}`;
  const parts = [count(rows.value.length, "LoRA")];
  if (groups.value.length) parts.push(count(groups.value.length, "group"));
  return `Remove ${parts.join(" and ")} from this node?`;
});
function askClear(e: MouseEvent) {
  clearAnchor.value = e.currentTarget as HTMLElement;
  clearOpen.value = true;
}
function clearAll() {
  clearOpen.value = false;
  rows.value = [];
  groups.value = [];
  commit();
}

// --- groups ---------------------------------------------------------------------------
const nextGroupId = () => Math.max(0, ...groups.value.map((g) => g.id)) + 1;
function addGroup() {
  const group: LoraGroup = {
    id: nextGroupId(),
    name: `Group ${groups.value.length + 1}`,
    open: true,
    muted: false,
    solo: false,
  };
  groups.value.push(group);
  commit();
  startRename(group);
}
function toggleOpen(group: LoraGroup) {
  group.open = !group.open;
  commit();
}
function setMuted(group: LoraGroup, muted: boolean) {
  group.muted = muted;
  commit();
}
function setSolo(group: LoraGroup, solo: boolean) {
  group.solo = solo;
  onlyOne(group);
  commit();
}
/** What a group's header says about its rows: the one in use for a solo group, else a tally. */
function summary(item: { group: LoraGroup; count: number; on: number }) {
  if (item.group.muted) return "muted";
  if (!item.group.solo) return `${item.on}/${item.count}`;
  const active = rows.value.find((row) => row.group === item.group.id && row.on);
  return active ? lib.short(active.name) || "unnamed" : "none on";
}
function ungroup(group: LoraGroup) {
  for (const row of rows.value) if (row.group === group.id) row.group = 0;
  groups.value = groups.value.filter((g) => g !== group);
  commit();
}
function deleteGroup(group: LoraGroup) {
  rows.value = rows.value.filter((row) => row.group !== group.id);
  groups.value = groups.value.filter((g) => g !== group);
  commit();
}
function move(group: LoraGroup, by: number) {
  const from = groups.value.indexOf(group);
  const to = from + by;
  if (from < 0 || to < 0 || to >= groups.value.length) return;
  groups.value.splice(to, 0, groups.value.splice(from, 1)[0]!);
  commit();
}
const moveUp = (group: LoraGroup) => move(group, -1);
const moveDown = (group: LoraGroup) => move(group, 1);

const renaming = ref(0); // id of the group whose name is being typed, or 0
const renameText = ref("");
const vFocus: Directive<HTMLInputElement> = {
  mounted(el) {
    el.focus();
    el.select();
  },
};
function startRename(group: LoraGroup) {
  renameText.value = group.name;
  renaming.value = group.id;
}
function endRename(keep: boolean) {
  const group = groupById(renaming.value);
  renaming.value = 0;
  const name = renameText.value.trim();
  if (!keep || !group || !name || name === group.name) return;
  group.name = name;
  commit();
}

const menuOpen = ref(false);
const menuAnchor = ref<HTMLElement | null>(null);
const menuGroup = ref<LoraGroup | null>(null);
function openGroupMenu(e: MouseEvent, group: LoraGroup) {
  menuAnchor.value = e.currentTarget as HTMLElement;
  menuGroup.value = group;
  menuOpen.value = true;
}
watch(menuOpen, (open) => {
  if (!open) menuGroup.value = null;
});
function canMove(by: number) {
  const at = menuGroup.value ? groups.value.indexOf(menuGroup.value) : -1;
  return at >= 0 && at + by >= 0 && at + by < groups.value.length;
}
function menuDo(action: (group: LoraGroup) => void) {
  const group = menuGroup.value;
  menuOpen.value = false;
  if (group) action(group);
}

// --- reorder (stack order = apply order, so dragging beats delete-and-re-add) ---------
//
// Pointer-driven, NOT HTML5 drag-and-drop. The native ghost is a snapshot the browser renders
// itself: `setDragImage` positions it but has no say over its scale, and the node body sits in a
// CSS-transformed container under Nodes 2.0, so the ghost came out at a size that matched neither
// the row nor the canvas zoom. Here the ROW ITSELF moves — the same element, inside the same
// transform — so it is exactly the row you grabbed, at any zoom, with nothing to keep in sync.
let rowMids: number[] = []; // each row's midpoint at grab time, layout space
const stride = ref(0); // how far the other rows move aside: the grabbed row and its gap
let rowTops: number[] = [];
let rowHeights: number[] = [];
// On the drop the list takes its new order with transitions off — the rows that moved aside are
// already where they belong — and only the dropped row glides the little way left to its slot.
const settling = ref(false);
const settle = ref<{ index: number; y: number } | null>(null);

const reorder = useDragSurface({
  onStart(ctx) {
    const grip = (ctx.event.target as HTMLElement | null)?.closest?.(".ls-grip");
    const host = reorder.el.value;
    if (!grip || !host) return false; // press wasn't on a handle — leave it to the controls
    const kids = Array.from(host.children) as HTMLElement[];
    const i = kids.indexOf(grip.closest(".ls-row") as HTMLElement);
    if (i < 0) return false;
    renaming.value = 0;
    // offsetTop/offsetHeight are pre-transform, which is the space ctx.delta already lands in
    // (the helper divides the canvas zoom out), so the two add up without a second correction.
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
    const center = (rowMids[dragIndex.value] ?? 0) + ctx.delta.y;
    // The rows it has passed move aside as it goes, so where it lands shows before the drop.
    dropIndex.value = insertionIndex(rowMids, dragIndex.value, center);
  },
  // Commit once, at the end: reordering on every move would rewrite the stack hundreds of times.
  onEnd() {
    const from = dragIndex.value;
    const to = dropIndex.value;
    const offset = dragOffset.value;
    endReorder();
    if (from < 0 || to < 0) return;
    settling.value = true;
    settle.value = {
      index: to,
      y: (rowTops[from] ?? 0) + offset - landingTop(rowTops, rowHeights, from, to),
    };
    if (from !== to) dropAt(from, to);
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        settling.value = false;
        settle.value = null;
      }),
    );
  },
});
/** Move the item at `from` to `to`, then read each row's group off the header now above it. */
function dropAt(from: number, to: number) {
  const list = [...items.value];
  const moved = list.splice(from, 1)[0];
  if (!moved?.row) return;
  list.splice(to, 0, moved);
  const shown = new Map<number, LoraRow[]>([[0, []]]);
  let current = 0;
  for (const item of list) {
    if (item.group) {
      current = item.group.id;
      shown.set(current, []);
    } else {
      item.row.group = current;
      shown.get(current)!.push(item.row);
    }
  }
  // A folded group's rows weren't in the list: they keep their place, ahead of anything just
  // dropped under the folded header.
  const next = [...shown.get(0)!];
  for (const group of groups.value) {
    const dropped = shown.get(group.id) ?? [];
    const hidden = group.open
      ? []
      : rows.value.filter((row) => row.group === group.id && !dropped.includes(row));
    next.push(...hidden, ...dropped);
  }
  rows.value = next;
  // Dropped into a solo group that already has its one: the newcomer is the one switched off.
  const joined = groupById(moved.row.group);
  if (joined?.solo) {
    const holder = next.find((row) => row.group === joined.id && row.on && row !== moved.row);
    onlyOne(joined, holder ?? moved.row);
  }
  commit();
}
/** Where item `i` is drawn: with the pointer if it is the grabbed one, aside if it is passed. */
function itemStyle(i: number) {
  if (dragIndex.value === i) return { transform: `translateY(${dragOffset.value}px)` };
  if (settle.value?.index === i) return { transform: `translateY(${settle.value.y}px)` };
  return shiftStyle(sortShift(i, dragIndex.value, dropIndex.value), stride.value);
}
function endReorder() {
  dragIndex.value = -1;
  dropIndex.value = -1;
  dragOffset.value = 0;
  rowMids = [];
}

function openBrowse(row: LoraRow | null) {
  browseRow.value = row;
  browse.value = true;
}
function pickFromBrowse(name: string) {
  if (browseRow.value) edit(browseRow.value, { name });
  else addRow(name);
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
  position: relative;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: transform 0.15s ease;
}
.ls-row.off {
  opacity: 0.5;
}
/* A group's rows step in under its header, joined by a rule down the side. */
.ls-row.grouped {
  margin-left: 6px;
  padding-left: 6px;
  border-left: 2px solid color-mix(in srgb, var(--zen-accent, #6366f1) 45%, transparent);
}
.ls-group {
  position: relative;
  display: flex;
  align-items: center;
  gap: 4px;
  min-height: 26px;
  padding: 0 2px 0 0;
  border-radius: var(--zen-radius, 6px);
  background: color-mix(in srgb, var(--zen-text, #fff) 6%, transparent);
  transition: transform 0.15s ease;
}
.ls-rows.settling .ls-group {
  transition: none;
}
.ls-group.muted .ls-gname {
  text-decoration: line-through;
  color: var(--zen-muted, #9aa0aa);
}
.ls-fold,
.ls-gname {
  border: 0;
  background: none;
  color: inherit;
  font: inherit;
  cursor: pointer;
  padding: 0;
}
.ls-fold {
  flex: none;
  width: 20px;
  height: 24px;
  color: var(--zen-muted, #9aa0aa);
  font-size: 16px;
  line-height: 1;
}
.ls-fold:hover {
  color: var(--zen-text, #e5e5ea);
}
.ls-gname {
  flex: 1;
  min-width: 0;
  height: 24px;
  overflow: hidden;
  text-align: left;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12px;
  font-weight: 600;
}
.ls-gname-edit {
  flex: 1;
  min-width: 0;
  height: 22px;
  padding: 0 5px;
  font: inherit;
  font-size: 12px;
  font-weight: 600;
  color: var(--zen-text, #e5e5ea);
  background: var(--zen-bg, #1a1a1f);
  border: 1px solid var(--zen-accent, #6366f1);
  border-radius: var(--zen-radius, 6px);
  outline: none;
}
.ls-driven,
.ls-solo {
  flex: none;
  font-size: 13px;
  color: var(--zen-accent, #6366f1);
}
.ls-gcount {
  flex: 0 1 auto;
  max-width: 45%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  padding: 0 4px;
  font-size: 10.5px;
  font-variant-numeric: tabular-nums;
  color: var(--zen-muted, #9aa0aa);
}
/* The grabbed row is lifted, not ghosted — it's the real row, translated, so it stays exactly
   its own size at any canvas zoom. z-index carries it over its neighbours as it passes them. */
.ls-rows.settling .ls-row {
  transition: none;
}
.ls-row.dragging {
  z-index: 5;
  transition: none;
  box-shadow: 0 6px 16px rgb(0 0 0 / 45%);
  border-radius: var(--zen-radius, 6px);
  cursor: grabbing;
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
  /* the icon font draws this glyph high in its box: measured 0.75px above the dot's centre at
     14px, so it comes down that much (in em, to hold at any size) */
  transform: translateY(0.054em);
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
.ls-confirm {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-width: 230px;
  padding: 6px 6px 4px;
}
.ls-confirm-text {
  font-size: 12px;
  line-height: 1.4;
  color: var(--zen-text, #e5e5ea);
}
.ls-confirm-actions {
  display: flex;
  justify-content: flex-end;
  gap: 6px;
}
.ls-pop-head {
  padding: 4px 8px 6px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--zen-muted, #9aa0aa);
}
</style>
