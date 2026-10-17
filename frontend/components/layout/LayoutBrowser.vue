<!-- Saved layouts, browsed like the LoRA browser: modes down the left, a grid of drawn previews in
     the middle, the picked layout on the right. Save the node's layout from the header; double-click
     a card (or Load) to bring one in. Loading is the host's: this emits what was picked. -->
<template>
  <ZenModal :open="open" title="Layouts" width="1180px" height="80vh" @update:open="setOpen">
    <template #header>
      <span class="lb-search">
        <ZenInput v-model="query" sm placeholder="Search names and prompts…" />
      </span>
      <span class="lb-grow" />
      <span class="lb-save">
        <ZenInput
          v-model="saveName"
          sm
          :placeholder="`Save this ${currentMode.label} layout as…`"
          @keydown.enter="save"
        />
        <ZenButton
          sm
          icon="mdi mdi-content-save-outline"
          :disabled="!saveName.trim()"
          @click="save"
        >
          Save
        </ZenButton>
      </span>
      <ZenIconButton icon="mdi mdi-refresh" title="Refresh" @click="refresh" />
    </template>

    <div class="lb">
      <nav class="lb-modes">
        <button
          v-for="s in scopes"
          :key="s.id"
          type="button"
          class="lb-mode"
          :class="{ on: scope === s.id }"
          @click="scope = s.id"
        >
          <i :class="s.icon" />
          <span class="lb-mname">{{ s.label }}</span>
          <span class="lb-count">{{ s.count }}</span>
        </button>
      </nav>

      <section class="lb-main">
        <p v-if="note" class="lb-note">{{ note }}</p>
        <ZenScroll class="lb-gridwrap">
          <div v-if="!visible.length" class="lb-empty">
            <i class="mdi mdi-bookshelf" />
            <span>{{
              all.length ? "Nothing matches." : "Nothing saved yet — name this layout above."
            }}</span>
          </div>
          <div class="lb-grid">
            <button
              v-for="item in visible"
              :key="keyOf(item)"
              type="button"
              class="lb-card"
              :class="{ on: picked && keyOf(picked) === keyOf(item) }"
              :title="`${item.name} — double-click to load`"
              @click="picked = item"
              @dblclick="load(item)"
            >
              <span class="lb-thumb">
                <LayoutThumb :preview="item.preview" />
              </span>
              <span class="lb-cname">{{ item.name }}</span>
              <span class="lb-cmeta">
                <i :class="modeOf(item.mode)?.icon ?? 'mdi mdi-shape-outline'" />
                {{ modeOf(item.mode)?.label ?? item.mode }} ·
                {{ item.preview?.boxes.length ?? 0 }} boxes
              </span>
            </button>
          </div>
        </ZenScroll>
      </section>

      <aside class="lb-detail">
        <template v-if="picked">
          <div class="lb-big">
            <LayoutThumb :preview="picked.preview" labels />
          </div>
          <div class="lb-row">
            <ZenInput
              v-model="renameTo"
              sm
              class="lb-grow"
              title="Rename — Enter to keep"
              @keydown.enter="rename"
              @blur="rename"
            />
          </div>
          <dl class="lb-facts">
            <dt>Mode</dt>
            <dd>{{ modeOf(picked.mode)?.label ?? picked.mode }}</dd>
            <template v-if="picked.preview?.width">
              <dt>Canvas</dt>
              <dd>{{ picked.preview.width }}×{{ picked.preview.height }}</dd>
            </template>
            <dt>Saved</dt>
            <dd>{{ when(picked.saved) }}</dd>
          </dl>
          <p v-if="picked.preview?.prompt" class="lb-prompt">{{ picked.preview.prompt }}</p>
          <ul class="lb-names">
            <li v-for="(b, i) in picked.preview?.boxes ?? []" :key="i">
              <span class="lb-dot" :style="{ background: b.color }" />{{ b.name || `Box ${i + 1}` }}
            </li>
          </ul>
          <span class="lb-grow" />
          <p v-if="!loadsNatively(picked.mode)" class="lb-note">
            Saved in {{ modeOf(picked.mode)?.label ?? picked.mode }} — loading keeps its boxes,
            names and prompts, and leaves the rest.
          </p>
          <div class="lb-row">
            <ZenButton
              sm
              :variant="confirmDelete ? 'danger' : 'default'"
              :icon="confirmDelete ? 'mdi mdi-check' : 'mdi mdi-delete-outline'"
              @click="remove"
            >
              {{ confirmDelete ? "Delete it" : "Delete" }}
            </ZenButton>
            <span class="lb-grow" />
            <ZenButton sm variant="primary" icon="mdi mdi-download" @click="load(picked)">
              Load into node
            </ZenButton>
          </div>
        </template>
        <div v-else class="lb-empty">
          <span>Pick a layout to see it.</span>
        </div>
      </aside>
    </div>
  </ZenModal>
</template>

<script setup lang="ts">
import { ZenButton, ZenIconButton, ZenInput, ZenModal, ZenScroll } from "@nynxz/zenkit-ui";
import { computed, ref, watch } from "vue";

import LayoutThumb from "@/components/layout/LayoutThumb.vue";
import {
  deleteLayout,
  listLayouts,
  loadLayout,
  renameLayout,
  saveLayout,
  type SavedLayout,
} from "@/layout/library";
import { layoutMode, layoutModes, previewOf, type AnyLayoutMode } from "@/layout/mode";

const props = defineProps<{
  open: boolean;
  /** The mode the node is in now: what Save saves as. */
  mode: AnyLayoutMode;
  /** The node's layout now, for Save. */
  scene: unknown;
  /** Every mode the node can show as it is; anything else is converted on load. */
  modes: AnyLayoutMode[];
}>();
const emit = defineEmits<{
  "update:open": [boolean];
  /** A saved layout to put on the node: its mode and its value, as saved. */
  load: [mode: string, layout: unknown];
}>();

const ALL = "all";
const all = ref<SavedLayout[]>([]);
const scope = ref<string>(ALL);
const query = ref("");
const picked = ref<SavedLayout | null>(null);
const renameTo = ref("");
const saveName = ref("");
const note = ref("");
const confirmDelete = ref(false);

const currentMode = computed(() => props.mode);
const keyOf = (s: SavedLayout) => `${s.mode}/${s.name}`;
const modeOf = (id: string) => layoutMode(id);
const loadsNatively = (id: string) => props.modes.some((m) => m.id === id);

function setOpen(v: boolean) {
  emit("update:open", v);
}
async function refresh() {
  all.value = await listLayouts(ALL);
  if (picked.value) {
    const key = keyOf(picked.value);
    picked.value = all.value.find((s) => keyOf(s) === key) ?? null;
  }
}
watch(
  () => props.open,
  (open) => {
    if (!open) return;
    note.value = "";
    scope.value = all.value.some((s) => s.mode === props.mode.id) ? props.mode.id : ALL;
    void refresh().then(() => {
      if (!all.value.some((s) => s.mode === props.mode.id)) scope.value = ALL;
    });
  },
  { immediate: true },
);
watch(picked, (p) => {
  renameTo.value = p?.name ?? "";
  confirmDelete.value = false;
});

const scopes = computed(() => {
  const counts = new Map<string, number>();
  for (const s of all.value) counts.set(s.mode, (counts.get(s.mode) ?? 0) + 1);
  const ids = new Set([...layoutModes().map((m) => m.id), ...counts.keys()]);
  return [
    { id: ALL, label: "All layouts", icon: "mdi mdi-bookshelf", count: all.value.length },
    ...[...ids].map((id) => ({
      id,
      label: layoutMode(id)?.label ?? id,
      icon: layoutMode(id)?.icon ?? "mdi mdi-shape-outline",
      count: counts.get(id) ?? 0,
    })),
  ];
});
const visible = computed(() => {
  const q = query.value.trim().toLowerCase();
  return all.value.filter(
    (s) =>
      (scope.value === ALL || s.mode === scope.value) &&
      (!q ||
        s.name.toLowerCase().includes(q) ||
        (s.preview?.prompt ?? "").toLowerCase().includes(q) ||
        (s.preview?.boxes ?? []).some((b) => (b.name ?? "").toLowerCase().includes(q))),
  );
});

async function save() {
  const name = saveName.value.trim();
  if (!name) return;
  try {
    const saved = await saveLayout(
      props.mode.id,
      name,
      JSON.parse(JSON.stringify(props.scene)),
      previewOf(props.mode, props.scene),
    );
    saveName.value = "";
    note.value = `Saved as ${saved}.`;
    await refresh();
    picked.value = all.value.find((s) => s.mode === props.mode.id && s.name === saved) ?? null;
  } catch (e) {
    note.value = `Couldn't save: ${(e as Error).message}`;
  }
}
async function load(item: SavedLayout) {
  try {
    emit("load", item.mode, await loadLayout(item.mode, item.name));
    setOpen(false);
  } catch (e) {
    note.value = `Couldn't load: ${(e as Error).message}`;
  }
}
async function rename() {
  const p = picked.value;
  const to = renameTo.value.trim();
  if (!p || !to || to === p.name) return;
  try {
    const name = await renameLayout(p.mode, p.name, to);
    await refresh();
    picked.value = all.value.find((s) => s.mode === p.mode && s.name === name) ?? null;
  } catch (e) {
    note.value = `Couldn't rename: ${(e as Error).message}`;
    renameTo.value = p.name;
  }
}
async function remove() {
  const p = picked.value;
  if (!p) return;
  if (!confirmDelete.value) {
    confirmDelete.value = true;
    return;
  }
  await deleteLayout(p.mode, p.name).catch(() => undefined);
  picked.value = null;
  await refresh();
}

function when(t: number) {
  return new Date(t * 1000).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}
</script>

<style scoped>
.lb {
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr) 280px;
  gap: 12px;
  height: 100%;
  min-height: 0;
  font-size: 12px;
  color: var(--zen-text, #e5e5ea);
}
.lb-search {
  width: 260px;
}
.lb-save {
  display: flex;
  gap: 6px;
  width: 340px;
}
.lb-save > :first-child {
  flex: 1;
}
.lb-grow {
  flex: 1;
}
.lb-modes {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.lb-mode {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 9px;
  border: 0;
  border-radius: var(--zen-radius, 7px);
  background: none;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.lb-mode:hover {
  background: color-mix(in srgb, var(--zen-text, #e5e5ea) 8%, transparent);
}
.lb-mode.on {
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 22%, transparent);
}
.lb-mode i {
  font-size: 15px;
  color: var(--zen-muted, #9aa0aa);
}
.lb-mname {
  flex: 1;
}
.lb-count {
  font-size: 10px;
  color: var(--zen-muted, #9aa0aa);
  font-variant-numeric: tabular-nums;
}
.lb-main {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 0;
}
.lb-gridwrap {
  flex: 1;
  min-height: 0;
}
.lb-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 10px;
  padding: 2px;
}
.lb-card {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 6px;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 7px);
  background: var(--zen-surface, #1e1e24);
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.lb-card:hover {
  border-color: color-mix(in srgb, var(--zen-accent, #6366f1) 50%, var(--zen-border, #34343c));
}
.lb-card.on {
  border-color: var(--zen-accent, #6366f1);
  box-shadow: 0 0 0 1px var(--zen-accent, #6366f1);
}
.lb-thumb {
  display: block;
  height: 120px;
}
.lb-cname {
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.lb-cmeta {
  font-size: 10.5px;
  color: var(--zen-muted, #9aa0aa);
}
.lb-detail {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 0;
  overflow: auto;
  padding-left: 12px;
  border-left: 1px solid var(--zen-border, #34343c);
}
.lb-big {
  height: 200px;
  flex: none;
}
.lb-row {
  display: flex;
  align-items: center;
  gap: 6px;
}
.lb-facts {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 2px 10px;
  margin: 0;
}
.lb-facts dt {
  color: var(--zen-muted, #9aa0aa);
}
.lb-facts dd {
  margin: 0;
}
.lb-prompt {
  margin: 0;
  padding: 6px 8px;
  border-radius: var(--zen-radius, 7px);
  background: var(--zen-surface-2, #26262d);
  font-size: 11px;
  line-height: 1.45;
  max-height: 110px;
  overflow: auto;
  white-space: pre-wrap;
}
.lb-names {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.lb-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  margin-right: 6px;
  border-radius: 2px;
}
.lb-note {
  margin: 0;
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.lb-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  height: 100%;
  min-height: 160px;
  color: var(--zen-muted, #9aa0aa);
}
.lb-empty i {
  font-size: 28px;
}
</style>
