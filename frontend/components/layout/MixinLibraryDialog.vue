<!-- Mixin Library modal: browse, inspect, create, edit and swap reusable mixins in the same
     3-column browser layout as LayoutBrowser and ZenMediaPicker (scopes on the left, visual card
     grid in the middle, full inspector + live editor sidebar on the right). -->
<template>
  <ZenModal :open="open" :title="modalTitle" width="1180px" height="82vh" @update:open="setOpen">
    <template #header>
      <span class="ml-search">
        <ZenInput v-model="query" sm placeholder="Search names, descriptions, triggers, LoRAs…" />
      </span>
      <span class="ml-grow" />
      <ZenButton sm icon="mdi mdi-plus" title="Create a new mixin preset" @click="startNew()">
        New mixin
      </ZenButton>
      <ZenIconButton icon="mdi mdi-refresh" title="Refresh library" @click="refresh" />
    </template>

    <div class="ml">
      <nav class="ml-nav">
        <button
          v-for="s in scopes"
          :key="s.id"
          type="button"
          class="ml-scope"
          :class="{ on: scope === s.id, sep: s.sep }"
          @click="scope = s.id"
        >
          <i :class="s.icon" />
          <span class="ml-sname">{{ s.label }}</span>
          <span class="ml-count">{{ s.count }}</span>
        </button>
      </nav>

      <section class="ml-main">
        <p v-if="note" class="ml-note">{{ note }}</p>
        <ZenScroll class="ml-gridwrap">
          <div v-if="!visible.length" class="ml-empty">
            <i class="mdi mdi-account-box-multiple-outline" />
            <span>{{ emptyText }}</span>
            <ZenButton
              v-if="scope !== 'layout' && !query.trim()"
              sm
              icon="mdi mdi-plus"
              @click="startNew()"
            >
              Create first mixin
            </ZenButton>
          </div>
          <div v-else class="ml-grid">
            <button
              v-for="card in visible"
              :key="card.key"
              type="button"
              class="ml-card"
              :class="{
                on: activeKey === card.key,
                target: swapTarget && card.source === 'layout' && card.mixin.id === swapTarget.id,
              }"
              :title="cardTitle(card)"
              @click="pickCard(card)"
              @dblclick="applyCard(card)"
              @contextmenu.prevent.stop="openCardMenu($event, card)"
            >
              <span class="ml-hero" :style="heroStyle(card.mixin)">
                <img
                  v-if="firstRef(card.mixin)"
                  class="ml-hero-img"
                  :src="thumbUrl(firstRef(card.mixin)!.ref, firstRef(card.mixin)!.type)"
                  alt=""
                />
                <MixinAvatar v-else :mixin="card.mixin" :size="48" />
                <span class="ml-dot" :style="{ background: card.mixin.color }" />
                <span v-if="activeRefs(card.mixin).length > 1" class="ml-badge">
                  +{{ activeRefs(card.mixin).length - 1 }}
                </span>
                <span v-if="card.source === 'layout'" class="ml-origin">Layout</span>
              </span>
              <span class="ml-cname">{{ card.mixin.name || "Untitled" }}</span>
              <span class="ml-cmeta">
                <span v-if="activeRefs(card.mixin).length" class="ml-chip">
                  <i class="mdi mdi-image-outline" />{{ activeRefs(card.mixin).length }}
                </span>
                <span v-if="activeLoras(card.mixin).length" class="ml-chip">
                  <i class="mdi mdi-puzzle-outline" />{{ activeLoras(card.mixin).length }}
                </span>
                <span v-if="card.mixin.trigger" class="ml-trigger">{{ card.mixin.trigger }}</span>
                <span
                  v-if="
                    !activeRefs(card.mixin).length &&
                    !activeLoras(card.mixin).length &&
                    !card.mixin.trigger
                  "
                  class="ml-muted"
                >
                  {{ card.mixin.desc ? "Description only" : "Empty" }}
                </span>
              </span>
            </button>
          </div>
        </ZenScroll>
      </section>

      <aside class="ml-detail">
        <template v-if="draft">
          <div class="ml-dhead">
            <MixinAvatar :mixin="draft" :size="36" />
            <ZenInput
              sm
              class="ml-grow"
              :model-value="draft.name"
              placeholder="Mixin name"
              @update:model-value="patchDraft({ name: String($event ?? '') })"
            />
            <input
              class="ml-color"
              type="color"
              :value="draft.color"
              title="Mixin colour"
              @input="patchDraft({ color: ($event.target as HTMLInputElement).value })"
            />
          </div>

          <div class="ml-dsub">
            <span class="ml-dtag">{{ draftBadge }}</span>
            <span v-if="activeSavedTime" class="ml-dtime">{{ when(activeSavedTime) }}</span>
            <span v-if="draftDirty" class="ml-dirty">unsaved</span>
          </div>

          <ZenScroll class="ml-dbody">
            <div class="ml-form">
              <FrescoSection label="Description" note="in every box that uses it">
                <ZenInput
                  type="textarea"
                  :rows="3"
                  sm
                  :model-value="draft.desc"
                  placeholder="Who or what this mixin is (appearance, outfit, style…)"
                  @update:model-value="patchDraft({ desc: String($event ?? '') })"
                />
              </FrescoSection>

              <FrescoSection label="Trigger words" note="added to box prompt">
                <ZenInput
                  sm
                  :model-value="draft.trigger"
                  placeholder="e.g. amb3r"
                  @update:model-value="patchDraft({ trigger: String($event ?? '') })"
                />
              </FrescoSection>

              <FrescoSection
                label="References"
                :note="draft.refs.length ? String(draft.refs.length) : 'visual refs'"
              >
                <template #actions>
                  <ZenIconButton
                    size="sm"
                    icon="mdi mdi-upload-outline"
                    title="Upload reference images"
                    :disabled="refsPanel?.busy"
                    @click="refsPanel?.upload()"
                  />
                  <ZenIconButton
                    size="sm"
                    icon="mdi mdi-view-grid-plus-outline"
                    title="Browse images"
                    @click="refsPanel?.openBrowse()"
                  />
                </template>
                <FrescoRefs
                  ref="refsPanel"
                  :model-value="draft.refs"
                  @update:model-value="patchDraft({ refs: $event })"
                />
              </FrescoSection>

              <FrescoSection
                label="LoRAs"
                :note="draft.loras.length ? String(draft.loras.length) : 'regional LoRAs'"
              >
                <template #actions>
                  <ZenIconButton
                    size="sm"
                    icon="mdi mdi-plus"
                    title="Add a LoRA"
                    @click="patchDraft({ loras: [...draft.loras, newLora()] })"
                  />
                </template>
                <FrescoLoras
                  v-if="draft.loras.length"
                  :model-value="draft.loras"
                  @update:model-value="patchDraft({ loras: $event })"
                />
                <p v-else class="ml-hint">No LoRAs yet — click + to attach one.</p>
              </FrescoSection>
            </div>
          </ZenScroll>

          <footer class="ml-dfoot">
            <div class="ml-drow">
              <ZenButton
                v-if="draftSource === 'saved' && savedOriginName"
                sm
                :variant="confirmDelete ? 'danger' : 'default'"
                :icon="confirmDelete ? 'mdi mdi-check' : 'mdi mdi-delete-outline'"
                @click="removeCurrentSaved"
              >
                {{ confirmDelete ? "Delete it" : "Delete" }}
              </ZenButton>
              <ZenIconButton
                v-if="draftSource !== 'new'"
                size="sm"
                icon="mdi mdi-content-copy"
                title="Duplicate as a new library preset"
                @click="duplicateDraft()"
              />
              <span class="ml-grow" />
              <ZenButton
                v-if="draftSource === 'layout' && draftDirty"
                sm
                icon="mdi mdi-check"
                :disabled="!draft.name.trim()"
                title="Apply edits to this mixin in the layout"
                @click="applyLayoutEdits"
              >
                Update in layout
              </ZenButton>
              <ZenButton
                sm
                icon="mdi mdi-content-save-outline"
                :disabled="!draft.name.trim() || saving"
                :title="
                  draftSource === 'saved'
                    ? 'Save changes to library preset'
                    : 'Save to reusable mixin library'
                "
                @click="saveDraft"
              >
                {{ saveLabel }}
              </ZenButton>
            </div>

            <div class="ml-drow ml-primary-row">
              <template v-if="swapTarget">
                <ZenButton
                  sm
                  variant="primary"
                  class="ml-grow"
                  icon="mdi mdi-swap-horizontal"
                  :disabled="isSelfSwap || !draft.name.trim()"
                  @click="commitSwap(draft)"
                >
                  {{ isSelfSwap ? "Same mixin" : `Swap "${swapTarget.name}" with this` }}
                </ZenButton>
              </template>
              <template v-else-if="draftSource !== 'layout'">
                <ZenButton
                  sm
                  :variant="selectedBoxes.length ? 'default' : 'primary'"
                  class="ml-grow"
                  icon="mdi mdi-plus"
                  :disabled="!draft.name.trim()"
                  @click="commitAdd(draft, false)"
                >
                  Add to layout
                </ZenButton>
                <ZenButton
                  v-if="selectedBoxes.length"
                  sm
                  variant="primary"
                  class="ml-grow"
                  icon="mdi mdi-checkbox-marked-circle-plus-outline"
                  :disabled="!draft.name.trim()"
                  @click="commitAdd(draft, true)"
                >
                  Add &amp; assign ({{ selectedBoxes.length }})
                </ZenButton>
              </template>
            </div>
          </footer>
        </template>
        <div v-else class="ml-empty">
          <i class="mdi mdi-account-details-outline" />
          <span
            >Pick a mixin to inspect its references, LoRAs and prompt — or create a new one.</span
          >
          <ZenButton sm icon="mdi mdi-plus" @click="startNew()">New mixin</ZenButton>
        </div>
      </aside>
    </div>

    <ZenContextMenu ref="ctxMenu" :items="ctxItems" />
  </ZenModal>
</template>

<script setup lang="ts">
import {
  ZenButton,
  ZenContextMenu,
  ZenIconButton,
  ZenInput,
  ZenModal,
  ZenScroll,
  type ContextMenuItem,
} from "@nynxz/zenkit-ui";
import { computed, ref, watch } from "vue";

import FrescoLoras from "@/components/fresco/FrescoLoras.vue";
import FrescoRefs from "@/components/fresco/FrescoRefs.vue";
import FrescoSection from "@/components/fresco/FrescoSection.vue";
import MixinAvatar from "@/components/layout/MixinAvatar.vue";
import {
  deleteMixinPreset,
  listSavedMixins,
  renameMixinPreset,
  saveMixinPreset,
  type SavedMixin,
} from "@/layout/library";
import { readMixin, type Mixin } from "@/layout/promptScene";
import type { FrescoLora, FrescoRef } from "@/lib/frescoScene";
import { thumbUrl } from "@/lib/mediaRefs";
import { uid } from "@/lib/sceneUtil";

type ScopeId = "all" | "refs" | "loras" | "layout";
type DraftSource = "saved" | "layout" | "new";

interface MixinCard {
  key: string;
  source: "saved" | "layout";
  savedName?: string;
  savedTime?: number;
  mixin: Mixin;
}

const props = withDefaults(
  defineProps<{
    open: boolean;
    layoutMixins: Mixin[];
    selectedBoxes?: string[];
    /** When set, picking a mixin swaps this layout mixin's content in place. */
    swapTarget?: Mixin | null;
    /** Optional layout mixin to focus immediately on open. */
    initialMixin?: Mixin | null;
  }>(),
  {
    selectedBoxes: () => [],
    swapTarget: null,
    initialMixin: null,
  },
);

const emit = defineEmits<{
  "update:open": [boolean];
  add: [mixin: Mixin, assignSelected: boolean];
  swap: [targetId: string, replacement: Mixin];
  updateLayoutMixin: [mixin: Mixin];
}>();

const saved = ref<SavedMixin[]>([]);
const scope = ref<ScopeId>("all");
const query = ref("");
const note = ref("");
const saving = ref(false);
const confirmDelete = ref(false);
const refsPanel = ref<{ upload: () => void; openBrowse: () => void; busy: boolean } | null>(null);

const activeKey = ref<string | null>(null);
const draft = ref<Mixin | null>(null);
const draftSource = ref<DraftSource>("new");
const savedOriginName = ref<string | null>(null);
const activeSavedTime = ref<number | null>(null);
const baselineJson = ref("");

const modalTitle = computed(() =>
  props.swapTarget ? `Swap "${props.swapTarget.name || "Mixin"}" with…` : "Mixin Library",
);

const savedCards = computed<MixinCard[]>(() =>
  saved.value.map((item, i) => ({
    key: `saved:${item.name}`,
    source: "saved",
    savedName: item.name,
    savedTime: item.saved,
    mixin: readMixin({ ...item.mixin, name: item.name }, i),
  })),
);

const layoutCards = computed<MixinCard[]>(() =>
  props.layoutMixins.map((m) => ({
    key: `layout:${m.id}`,
    source: "layout",
    mixin: cloneMixin(m),
  })),
);

const scopes = computed<
  { id: ScopeId; label: string; icon: string; count: number; sep?: boolean }[]
>(() => [
  { id: "all", label: "All saved", icon: "mdi mdi-bookshelf", count: savedCards.value.length },
  {
    id: "refs",
    label: "With references",
    icon: "mdi mdi-image-multiple-outline",
    count: savedCards.value.filter((c) => activeRefs(c.mixin).length > 0).length,
  },
  {
    id: "loras",
    label: "With LoRAs",
    icon: "mdi mdi-puzzle-outline",
    count: savedCards.value.filter((c) => activeLoras(c.mixin).length > 0).length,
  },
  {
    id: "layout",
    label: "In this layout",
    icon: "mdi mdi-layers-outline",
    count: layoutCards.value.length,
    sep: true,
  },
]);

const visible = computed<MixinCard[]>(() => {
  const pool =
    scope.value === "layout"
      ? layoutCards.value
      : scope.value === "refs"
        ? savedCards.value.filter((c) => activeRefs(c.mixin).length > 0)
        : scope.value === "loras"
          ? savedCards.value.filter((c) => activeLoras(c.mixin).length > 0)
          : savedCards.value;
  const q = query.value.trim().toLowerCase();
  if (!q) return pool;
  return pool.filter(({ mixin }) => {
    const hay = [
      mixin.name,
      mixin.desc,
      mixin.trigger,
      ...mixin.loras.map((l) => l.name),
      ...mixin.refs.map((r) => r.ref),
    ]
      .join("\n")
      .toLowerCase();
    return hay.includes(q);
  });
});

const emptyText = computed(() => {
  if (query.value.trim()) return "No mixins match that search.";
  if (scope.value === "layout") return "No mixins in this layout yet.";
  if (scope.value === "refs") return "No saved mixins with reference images yet.";
  if (scope.value === "loras") return "No saved mixins with LoRAs yet.";
  return "No saved mixins yet — create one here or save one from this layout.";
});

const draftDirty = computed(
  () => !!draft.value && JSON.stringify(draft.value) !== baselineJson.value,
);
const draftBadge = computed(() =>
  draftSource.value === "new"
    ? "New library preset"
    : draftSource.value === "layout"
      ? "In this layout"
      : "Library preset",
);
const saveLabel = computed(() => {
  if (saving.value) return "Saving…";
  if (draftSource.value === "saved") return draftDirty.value ? "Save changes" : "Saved";
  return "Save to library";
});
const isSelfSwap = computed(
  () =>
    !!props.swapTarget && draftSource.value === "layout" && draft.value?.id === props.swapTarget.id,
);

function cloneMixin(m: Mixin): Mixin {
  return JSON.parse(JSON.stringify(m)) as Mixin;
}
function activeRefs(m: Mixin): FrescoRef[] {
  return m.refs.filter((r) => r.on);
}
function activeLoras(m: Mixin): FrescoLora[] {
  return m.loras.filter((l) => l.on && l.name);
}
function firstRef(m: Mixin): FrescoRef | null {
  return activeRefs(m)[0] ?? null;
}
function heroStyle(m: Mixin): Record<string, string> {
  return { background: `color-mix(in srgb, ${m.color} 18%, var(--zen-surface-2, #26262d))` };
}
function cardTitle(card: MixinCard): string {
  const action = props.swapTarget
    ? `double-click to swap "${props.swapTarget.name}" with this`
    : card.source === "layout"
      ? "click to inspect or save to library"
      : "double-click to add to layout";
  return `${card.mixin.name || "Untitled"} — ${action}`;
}
function newLora(): FrescoLora {
  return { id: uid("o"), name: "", strength: 1, on: true };
}

function setOpen(v: boolean) {
  emit("update:open", v);
}

async function refresh() {
  saved.value = await listSavedMixins();
  if (draftSource.value === "saved" && savedOriginName.value) {
    const found = savedCards.value.find((c) => c.savedName === savedOriginName.value);
    if (found && !draftDirty.value) pickCard(found);
    else if (!found) {
      activeKey.value = null;
      draft.value = null;
    }
  }
}

watch(
  () => props.open,
  async (open) => {
    if (!open) return;
    note.value = "";
    query.value = "";
    confirmDelete.value = false;
    await refresh();
    if (props.initialMixin) {
      scope.value = "layout";
      const match = layoutCards.value.find((c) => c.mixin.id === props.initialMixin?.id);
      if (match) pickCard(match);
    } else if (savedCards.value.length) {
      scope.value = "all";
      pickCard(savedCards.value[0]!);
    } else if (layoutCards.value.length) {
      scope.value = "layout";
      pickCard(layoutCards.value[0]!);
    } else {
      scope.value = "all";
      activeKey.value = null;
      draft.value = null;
    }
  },
);

function pickCard(card: MixinCard) {
  activeKey.value = card.key;
  draft.value = cloneMixin(card.mixin);
  draftSource.value = card.source;
  savedOriginName.value = card.savedName ?? null;
  activeSavedTime.value = card.savedTime ?? null;
  baselineJson.value = JSON.stringify(draft.value);
  confirmDelete.value = false;
  note.value = "";
}

function startNew(from?: Mixin) {
  const index = savedCards.value.length + props.layoutMixins.length;
  const fresh = from
    ? {
        ...cloneMixin(from),
        id: uid("c"),
        name: `${from.name || "Mixin"} copy`,
        refs: from.refs.map((r) => ({ ...r, id: uid("r") })),
        loras: from.loras.map((l) => ({ ...l, id: uid("o") })),
      }
    : { ...readMixin({}, index), id: uid("c"), name: `Mixin ${index + 1}` };
  activeKey.value = null;
  draft.value = fresh;
  draftSource.value = "new";
  savedOriginName.value = null;
  activeSavedTime.value = null;
  baselineJson.value = "";
  confirmDelete.value = false;
  note.value = "";
}

function duplicateDraft() {
  if (draft.value) startNew(draft.value);
}

function patchDraft(change: Partial<Mixin>) {
  if (!draft.value) return;
  draft.value = { ...draft.value, ...change };
}

async function saveDraft() {
  const current = draft.value;
  const name = current?.name.trim() ?? "";
  if (!current || !name) return;
  saving.value = true;
  note.value = "";
  try {
    let savedAs: string;
    if (draftSource.value === "saved" && savedOriginName.value && savedOriginName.value !== name) {
      await renameMixinPreset(savedOriginName.value, name);
      savedAs = await saveMixinPreset(name, { ...current, name });
    } else {
      savedAs = await saveMixinPreset(name, { ...current, name });
    }
    await refresh();
    const card = savedCards.value.find((c) => c.savedName === savedAs);
    if (card) {
      if (draftSource.value !== "layout") {
        scope.value = "all";
        pickCard(card);
      } else {
        note.value = `Saved "${savedAs}" to library.`;
      }
    }
    if (draftSource.value !== "layout") note.value = `Saved "${savedAs}".`;
  } catch (e) {
    note.value = `Couldn't save: ${(e as Error).message}`;
  } finally {
    saving.value = false;
  }
}

function applyLayoutEdits() {
  if (!draft.value || draftSource.value !== "layout") return;
  emit("updateLayoutMixin", cloneMixin(draft.value));
  baselineJson.value = JSON.stringify(draft.value);
  note.value = `Updated "${draft.value.name}" in layout.`;
}

async function removeCurrentSaved() {
  if (!savedOriginName.value) return;
  if (!confirmDelete.value) {
    confirmDelete.value = true;
    return;
  }
  const target = savedOriginName.value;
  await deleteMixinPreset(target).catch(() => undefined);
  activeKey.value = null;
  draft.value = null;
  savedOriginName.value = null;
  confirmDelete.value = false;
  await refresh();
  note.value = `Deleted "${target}".`;
}

function freshCopy(source: Mixin): Mixin {
  const copy = cloneMixin(source);
  return {
    ...copy,
    id: uid("c"),
    refs: copy.refs.map((r) => ({ ...r, id: uid("r") })),
    loras: copy.loras.map((l) => ({ ...l, id: uid("o") })),
  };
}

function commitAdd(source: Mixin, assignSelected: boolean) {
  emit("add", freshCopy(source), assignSelected);
  setOpen(false);
}

function commitSwap(source: Mixin) {
  if (!props.swapTarget) return;
  emit("swap", props.swapTarget.id, freshCopy(source));
  setOpen(false);
}

function applyCard(card: MixinCard) {
  if (props.swapTarget) {
    if (card.source === "layout" && card.mixin.id === props.swapTarget.id) return;
    commitSwap(card.mixin);
    return;
  }
  if (card.source === "saved") commitAdd(card.mixin, props.selectedBoxes.length > 0);
}

// --- card right-click context menu --------------------------------------------------------------

const ctxMenu = ref<{ show: (e: MouseEvent) => void } | null>(null);
const ctxCard = ref<MixinCard | null>(null);
function openCardMenu(e: MouseEvent, card: MixinCard) {
  ctxCard.value = card;
  pickCard(card);
  ctxMenu.value?.show(e);
}

const ctxItems = computed<ContextMenuItem[]>(() => {
  const card = ctxCard.value;
  if (!card) return [];
  const items: ContextMenuItem[] = [];
  if (props.swapTarget) {
    const self = card.source === "layout" && card.mixin.id === props.swapTarget.id;
    items.push({
      label: `Swap "${props.swapTarget.name}" with this`,
      icon: "mdi mdi-swap-horizontal",
      disabled: self,
      run: () => commitSwap(card.mixin),
    });
  } else if (card.source === "saved") {
    items.push({
      label: "Add to layout",
      icon: "mdi mdi-plus",
      run: () => commitAdd(card.mixin, false),
    });
    if (props.selectedBoxes.length) {
      items.push({
        label: `Add & assign to ${props.selectedBoxes.length} box(es)`,
        icon: "mdi mdi-checkbox-marked-circle-plus-outline",
        run: () => commitAdd(card.mixin, true),
      });
    }
  }
  items.push({
    label: "Duplicate as new preset",
    icon: "mdi mdi-content-copy",
    run: () => startNew(card.mixin),
  });
  if (card.source === "layout") {
    items.push({
      label: "Save to library",
      icon: "mdi mdi-content-save-outline",
      run: () => {
        pickCard(card);
        void saveDraft();
      },
    });
  } else if (card.savedName) {
    const name = card.savedName;
    items.push("-", {
      label: "Delete from library",
      icon: "mdi mdi-delete-outline",
      danger: true,
      run: async () => {
        await deleteMixinPreset(name).catch(() => undefined);
        if (savedOriginName.value === name) {
          activeKey.value = null;
          draft.value = null;
          savedOriginName.value = null;
        }
        await refresh();
      },
    });
  }
  return items;
});

function when(t: number) {
  return new Date(t * 1000).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}
</script>

<style scoped>
.ml {
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr) 340px;
  gap: 12px;
  height: 100%;
  min-height: 0;
  font-size: 12px;
  color: var(--zen-text, #e5e5ea);
}
.ml-search {
  width: 280px;
}
.ml-grow {
  flex: 1;
  min-width: 0;
}
.ml-nav {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.ml-scope {
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
.ml-scope.sep {
  margin-top: 8px;
  padding-top: 10px;
  border-top: 1px solid var(--zen-border, #34343c);
  border-top-left-radius: 0;
  border-top-right-radius: 0;
}
.ml-scope:hover {
  background: color-mix(in srgb, var(--zen-text, #e5e5ea) 8%, transparent);
}
.ml-scope.on {
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 22%, transparent);
}
.ml-scope i {
  font-size: 15px;
  color: var(--zen-muted, #9aa0aa);
}
.ml-sname {
  flex: 1;
}
.ml-count {
  font-size: 10px;
  color: var(--zen-muted, #9aa0aa);
  font-variant-numeric: tabular-nums;
}
.ml-main {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 0;
}
.ml-gridwrap {
  flex: 1;
  min-height: 0;
}
.ml-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(156px, 1fr));
  gap: 10px;
  padding: 2px;
}
.ml-card {
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 6px;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 7px);
  background: var(--zen-surface, #1e1e24);
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.ml-card:hover {
  border-color: color-mix(in srgb, var(--zen-accent, #6366f1) 50%, var(--zen-border, #34343c));
}
.ml-card.on {
  border-color: var(--zen-accent, #6366f1);
  box-shadow: 0 0 0 1px var(--zen-accent, #6366f1);
}
.ml-card.target {
  opacity: 0.55;
}
.ml-hero {
  position: relative;
  display: grid;
  place-items: center;
  height: 118px;
  border-radius: 5px;
  overflow: hidden;
}
.ml-hero-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.ml-dot {
  position: absolute;
  top: 6px;
  left: 6px;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  box-shadow: 0 0 0 1.5px rgb(0 0 0 / 55%);
}
.ml-badge,
.ml-origin {
  position: absolute;
  padding: 1px 6px;
  border-radius: 999px;
  background: rgb(0 0 0 / 68%);
  color: #fff;
  font-size: 10px;
  font-weight: 600;
}
.ml-badge {
  right: 6px;
  bottom: 6px;
}
.ml-origin {
  top: 6px;
  right: 6px;
}
.ml-cname {
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ml-cmeta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 5px;
  font-size: 10.5px;
  color: var(--zen-muted, #9aa0aa);
}
.ml-chip {
  display: inline-flex;
  align-items: center;
  gap: 2px;
}
.ml-trigger {
  max-width: 100%;
  padding: 0 5px;
  border-radius: 4px;
  background: color-mix(in srgb, var(--zen-text, #fff) 7%, transparent);
  font-family: var(--zen-mono, ui-monospace, monospace);
  font-size: 10px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.ml-muted {
  opacity: 0.75;
}
.ml-detail {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 0;
  padding-left: 12px;
  border-left: 1px solid var(--zen-border, #34343c);
}
.ml-dhead {
  flex: none;
  display: flex;
  align-items: center;
  gap: 8px;
}
.ml-color {
  flex: none;
  width: 24px;
  height: 24px;
  padding: 0;
  border: 0;
  border-radius: 5px;
  background: transparent;
  cursor: pointer;
}
.ml-dsub {
  flex: none;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 10.5px;
  color: var(--zen-muted, #9aa0aa);
}
.ml-dtag {
  padding: 1px 6px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--zen-text, #fff) 8%, transparent);
}
.ml-dtime {
  margin-left: auto;
}
.ml-dirty {
  color: var(--zen-warning, #f5b041);
  font-weight: 600;
}
.ml-dbody {
  flex: 1;
  min-height: 0;
}
.ml-form {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-right: 4px;
}
.ml-hint {
  margin: 0;
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.ml-dfoot {
  flex: none;
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-top: 8px;
  border-top: 1px solid var(--zen-border, #34343c);
}
.ml-drow {
  display: flex;
  align-items: center;
  gap: 6px;
}
.ml-primary-row:empty {
  display: none;
}
.ml-note {
  margin: 0;
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.ml-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  height: 100%;
  min-height: 180px;
  padding: 16px;
  text-align: center;
  color: var(--zen-muted, #9aa0aa);
}
.ml-empty i {
  font-size: 28px;
}
</style>
