<template>
  <ZenWidget :gap="6" pad="2px" class="gs">
    <div class="gs-head">
      <!-- Which group colours this switcher controls. A group whose colour isn't selected is left
           alone entirely — the way to keep some groups out of reach. -->
      <div class="gs-colors" role="group" aria-label="Group colours">
        <button
          class="gs-all"
          :class="{ on: !settings.colors.length }"
          title="Every group colour"
          @click="setColors([])"
        >
          All
        </button>
        <button
          v-for="c in palette"
          :key="c.color"
          class="gs-color"
          :class="{ on: isColorOn(c.color), none: c.color === NO_COLOR }"
          :style="c.color === NO_COLOR ? undefined : { '--gs-c': c.color }"
          :title="`${c.color === NO_COLOR ? 'No colour' : c.color} — ${c.count} group${c.count === 1 ? '' : 's'}`"
          @click="toggleColor(c.color)"
        />
      </div>
      <ZenToggleGroup
        :model-value="settings.off"
        :options="[
          { value: 'bypass', icon: 'mdi mdi-debug-step-over', title: 'Off bypasses the group' },
          { value: 'mute', icon: 'mdi mdi-volume-off', title: 'Off mutes the group' },
        ]"
        @update:model-value="setOff($event as OffMode)"
      />
      <ZenSwitch
        :model-value="allOn"
        :disabled="!switchable.length"
        :title="allOn ? 'Turn every listed group off' : 'Turn every listed group on'"
        @update:model-value="setAll($event)"
      />
    </div>

    <div v-if="visible.length" class="gs-rows">
      <div
        v-for="g in visible"
        :key="g.key"
        class="gs-card"
        :class="{ off: g.state === 'off', empty: !g.nodes.length }"
        :style="g.color === NO_COLOR ? undefined : { '--gs-c': g.color }"
      >
        <div class="gs-card-head">
          <button
            class="gs-name"
            :disabled="!g.nodes.length"
            :title="g.state === 'on' ? 'Turn group off' : 'Turn group on'"
            @click="toggleGroup(g)"
          >
            <span class="gs-title">{{ g.title }}</span>
            <span
              class="gs-count"
              :class="{ mixed: g.state === 'mixed' }"
              :title="`${onCount(g)} of ${g.nodes.length} nodes running`"
            >
              {{ onCount(g) }}/{{ g.nodes.length }}
            </span>
          </button>
          <ZenIconButton
            v-if="g.nodes.length"
            :icon="isExpanded(g) ? 'mdi mdi-chevron-up' : 'mdi mdi-chevron-down'"
            :title="isExpanded(g) ? 'Hide nodes' : 'Show nodes'"
            @click="toggleExpanded(g)"
          />
          <ZenSwitch
            :model-value="g.state !== 'off'"
            :disabled="!g.nodes.length"
            :title="g.state === 'on' ? 'Turn group off' : 'Turn group on'"
            @update:model-value="toggleGroup(g)"
          />
        </div>
        <div v-if="isExpanded(g) && g.nodes.length" class="gs-nodes">
          <div v-for="n in g.nodes" :key="n.id" class="gs-node" :class="{ off: !n.on }">
            <button class="gs-name" @click="toggleNode(n)">
              <span class="gs-title">{{ n.title }}</span>
            </button>
            <ZenSwitch
              class="gs-node-switch"
              :model-value="n.on"
              :title="n.on ? 'Turn node off' : 'Turn node on'"
              @update:model-value="toggleNode(n)"
            />
          </div>
        </div>
      </div>
    </div>
    <div v-else class="gs-empty">
      <i class="mdi mdi-group" />
      <span>{{
        groups.length ? "No groups with the selected colours" : "No groups in this graph"
      }}</span>
    </div>
  </ZenWidget>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, shallowRef } from "vue";
import { ZenIconButton, ZenSwitch, ZenToggleGroup, ZenWidget } from "@nynxz/zenkit-ui";

import {
  NO_COLOR,
  readGroups,
  readSettings,
  setNodesOn,
  signature,
  type GroupEntry,
  type GroupNodeEntry,
  type LGGraph,
  type OffMode,
  type SwitcherSettings,
  writeSettings,
} from "./groups";

const props = defineProps<{
  widget?: { value: unknown; callback?: (v: unknown) => void };
  node?: { graph?: LGGraph | null };
}>();

const REFRESH_MS = 500;

const settings = ref<SwitcherSettings>(readSettings(props.widget?.value));
const groups = shallowRef<GroupEntry[]>([]);

const graph = () => props.node?.graph ?? null;

// Groups aren't observable, so the switcher re-reads them on a timer. It only re-renders when
// something it shows changed, and skips the read entirely while the tab is hidden.
let shown = "";
let lastValue: unknown = props.widget?.value;
function refresh() {
  const next = readGroups(graph(), props.node);
  const sig = signature(next);
  if (sig !== shown) {
    shown = sig;
    groups.value = next;
  }
  // A graph reload (or undo) replaces the widget value underneath us — follow it.
  const value = props.widget?.value;
  if (value !== lastValue) {
    lastValue = value;
    settings.value = readSettings(value);
  }
}

let timer: ReturnType<typeof setInterval> | undefined;
onMounted(() => {
  refresh();
  timer = setInterval(() => {
    if (!document.hidden) refresh();
  }, REFRESH_MS);
});
onBeforeUnmount(() => clearInterval(timer));

function commit(next: Partial<SwitcherSettings>) {
  settings.value = { ...settings.value, ...next };
  const value = { ...settings.value };
  lastValue = value;
  writeSettings(props.widget, value);
}

const palette = computed(() => {
  const counts = new Map<string, number>();
  for (const g of groups.value) counts.set(g.color, (counts.get(g.color) ?? 0) + 1);
  return [...counts].map(([color, count]) => ({ color, count }));
});

const isColorOn = (color: string) => settings.value.colors.includes(color);
const visible = computed(() =>
  settings.value.colors.length
    ? groups.value.filter((g) => settings.value.colors.includes(g.color))
    : groups.value,
);

function setColors(colors: string[]) {
  commit({ colors });
}
function toggleColor(color: string) {
  const colors = settings.value.colors;
  setColors(colors.includes(color) ? colors.filter((c) => c !== color) : [...colors, color]);
}

// Switching Bypass <-> Mute moves groups that are already off over to the new mode, so the
// switch changes what "off" means now rather than only for the next toggle.
function setOff(off: OffMode) {
  commit({ off });
  const offGroups = visible.value.filter((g) => g.state === "off");
  if (offGroups.length) {
    setNodesOn(graph(), offGroups.flatMap(nodesOf), false, off);
    refresh();
  }
}

const nodesOf = (g: GroupEntry) => g.nodes.map((n) => n.node);
const onCount = (g: GroupEntry) => g.nodes.filter((n) => n.on).length;

function toggleGroup(g: GroupEntry) {
  setNodesOn(graph(), nodesOf(g), g.state !== "on", settings.value.off);
  refresh();
}
function toggleNode(n: GroupNodeEntry) {
  setNodesOn(graph(), [n.node], !n.on, settings.value.off);
  refresh();
}
const switchable = computed(() => visible.value.filter((g) => g.nodes.length));
const allOn = computed(
  () => switchable.value.length > 0 && switchable.value.every((g) => g.state === "on"),
);
function setAll(on: boolean) {
  setNodesOn(graph(), visible.value.flatMap(nodesOf), on, settings.value.off);
  refresh();
}

const isExpanded = (g: GroupEntry) => settings.value.expanded.includes(g.key);
function toggleExpanded(g: GroupEntry) {
  const expanded = settings.value.expanded;
  commit({
    expanded: expanded.includes(g.key) ? expanded.filter((k) => k !== g.key) : [...expanded, g.key],
  });
}
</script>

<style scoped>
.gs-head {
  display: flex;
  align-items: center;
  gap: 6px;
  /* lines the master switch up with the cards' switches below it */
  padding-right: 7px;
}
.gs-colors {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
}
.gs-all {
  height: 18px;
  padding: 0 7px;
  border: 1px solid var(--zen-control-border, var(--zen-border, #34343c));
  border-radius: 999px;
  background: var(--zen-control-bg, transparent);
  color: var(--zen-muted, #9aa0aa);
  font: inherit;
  font-size: 10px;
  cursor: pointer;
}
.gs-all.on {
  color: var(--zen-text, #e5e5ea);
  border-color: var(--zen-accent, #6366f1);
}
/* A colour chip: the group colour itself, ringed in the accent while selected. */
.gs-color {
  width: 16px;
  height: 16px;
  padding: 0;
  border: 2px solid transparent;
  border-radius: 50%;
  background: var(--gs-c) padding-box;
  box-shadow: inset 0 0 0 1px rgb(255 255 255 / 12%);
  cursor: pointer;
  opacity: 0.55;
}
.gs-color.on {
  opacity: 1;
  border-color: var(--zen-accent, #6366f1);
}
.gs-color.none {
  background: transparent;
  box-shadow: inset 0 0 0 1px var(--zen-muted, #9aa0aa);
}

.gs-rows {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

/* One card per group: its own outline, a lift on hover, and the group's colour as a stripe down
   the left edge so the card still reads as that group without a separate swatch. */
.gs-card {
  display: flex;
  flex-direction: column;
  border: 1px solid var(--zen-surface-border, var(--zen-border, #34343c));
  border-left: 3px solid var(--gs-c, var(--zen-surface-border, var(--zen-border, #34343c)));
  border-radius: var(--zen-radius, 8px);
  background: color-mix(in srgb, var(--zen-text, #fff) 3%, transparent);
  transition:
    background 0.12s ease,
    border-color 0.12s ease;
}
.gs-card:hover {
  background: color-mix(in srgb, var(--zen-text, #fff) 5%, transparent);
}
.gs-card-head {
  display: flex;
  align-items: center;
  gap: 4px;
  min-height: 32px;
  padding: 3px 6px 3px 8px;
}
.gs-card.off .gs-name,
.gs-node.off .gs-name {
  opacity: 0.5;
}
.gs-card.empty .gs-name {
  opacity: 0.45;
}

/* The name is a button so the whole label toggles — a far bigger target than the switch. */
.gs-name {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 2px 0;
  border: none;
  background: none;
  color: var(--zen-text, #e5e5ea);
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.gs-name:disabled {
  cursor: default;
}
.gs-title {
  min-width: 0;
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.gs-count {
  flex: none;
  padding: 0 5px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--zen-text, #fff) 7%, transparent);
  color: var(--zen-muted, #9aa0aa);
  font-size: 10px;
  line-height: 16px;
  font-variant-numeric: tabular-nums;
}
/* Some nodes running, some not: the badge carries it, in the accent. */
.gs-count.mixed {
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 20%, transparent);
  color: var(--zen-text, #e5e5ea);
}

.gs-nodes {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 4px 6px 6px 8px;
  border-top: 1px solid var(--zen-surface-border, var(--zen-border, #34343c));
}
.gs-node {
  display: flex;
  align-items: center;
  gap: 6px;
  min-height: 24px;
}
.gs-node .gs-title {
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.gs-node-switch {
  zoom: 0.75;
}

.gs-empty {
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
.gs-empty > .mdi {
  font-size: 24px;
  opacity: 0.8;
}
.gs-empty > span {
  font-size: 11px;
}
</style>
