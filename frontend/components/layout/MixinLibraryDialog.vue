<!-- Mixin Library modal: ID-keyed, linked 3-column browser (scopes & categories on the left,
     visual card grid in the middle, full inspector + live editor sidebar on the right).
     Layout mixins linked via `libraryId` stay synced with their library presets rather than
     feeling like disconnected copies. -->
<template>
  <ZenModal :open="open" :title="modalTitle" width="1180px" height="82vh" @update:open="setOpen">
    <template #header>
      <span class="ml-search">
        <ZenInput v-model="query" sm placeholder="Search names, categories, triggers, LoRAs…" />
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
                target: swapTarget && card.layoutId === swapTarget.id,
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
                <span
                  v-if="card.state === 'linked'"
                  class="ml-pill linked"
                  title="In this layout and synced with the library"
                >
                  <i class="mdi mdi-check" /> In layout
                </span>
                <span
                  v-else-if="card.state === 'diverged'"
                  class="ml-pill diverged"
                  title="In this layout, with local changes not yet synced"
                >
                  <i class="mdi mdi-sync-alert" /> Modified
                </span>
                <span
                  v-else-if="card.state === 'local'"
                  class="ml-pill local"
                  title="Only in this layout — save to library to reuse across layouts"
                >
                  Local only
                </span>
              </span>
              <span class="ml-cname">{{ card.mixin.name || "Untitled" }}</span>
              <span class="ml-cmeta">
                <span v-if="card.category" class="ml-cat">{{ card.category }}</span>
                <span v-if="activeRefs(card.mixin).length" class="ml-chip">
                  <i class="mdi mdi-image-outline" />{{ activeRefs(card.mixin).length }}
                </span>
                <span v-if="activeLoras(card.mixin).length" class="ml-chip">
                  <i class="mdi mdi-puzzle-outline" />{{ activeLoras(card.mixin).length }}
                </span>
                <span v-if="card.mixin.trigger" class="ml-trigger">{{ card.mixin.trigger }}</span>
                <span
                  v-if="
                    !card.category &&
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
            <span class="ml-dtag" :class="activeLinkState">{{ draftBadge }}</span>
            <span v-if="activeSavedTime" class="ml-dtime">{{ when(activeSavedTime) }}</span>
            <span v-if="draftDirty" class="ml-dirty">unsaved</span>
          </div>

          <div v-if="divergedPair" class="ml-sync-bar">
            <span class="ml-sync-text">
              <i class="mdi mdi-sync-alert" />
              Layout copy differs from the saved library preset.
            </span>
            <div class="ml-drow">
              <ZenButton
                sm
                icon="mdi mdi-cloud-upload-outline"
                title="Update the library preset with the layout's edits"
                @click="pushLayoutToLibrary"
              >
                Push layout → library
              </ZenButton>
              <ZenButton
                sm
                icon="mdi mdi-cloud-download-outline"
                title="Update the layout mixin with the library preset"
                @click="pullLibraryToLayout"
              >
                Pull library → layout
              </ZenButton>
            </div>
          </div>

          <ZenScroll class="ml-dbody">
            <div class="ml-form">
              <FrescoSection label="Category" note="optional folder/group">
                <input
                  v-model="draftCategory"
                  class="ml-cat-input"
                  list="nynxz-mixin-categories"
                  placeholder="Characters, Styles, Props…"
                  spellcheck="false"
                />
                <datalist id="nynxz-mixin-categories">
                  <option v-for="c in allCategories" :key="c" :value="c" />
                </datalist>
              </FrescoSection>

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
                  placeholder="e.g. ohwx, sks style"
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
                <MixinFusionBar
                  :active-count="activeRefs(draft).length"
                  :blend="draft.blend"
                  :softness="draft.softness"
                  :style-mode="draft.styleMode"
                  :style-strength="draft.styleStrength"
                  @update:blend="patchDraft({ blend: $event })"
                  @update:softness="patchDraft({ softness: $event })"
                  @update:style-mode="patchDraft({ styleMode: $event })"
                  @update:style-strength="patchDraft({ styleStrength: $event })"
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
                v-if="activePresetId"
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
                title="Save a separate copy as a new library preset"
                @click="duplicateDraft()"
              />
              <ZenIconButton
                v-if="activeLayoutId && activePresetId"
                size="sm"
                icon="mdi mdi-link-variant-off"
                title="Unlink this layout mixin from the library preset"
                @click="unlinkCurrent"
              />
              <span class="ml-grow" />
              <ZenButton
                sm
                icon="mdi mdi-content-save-outline"
                :disabled="!draft.name.trim() || saving"
                :title="saveTitle"
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
                  @click="commitSwap(draft, activePresetId ?? undefined)"
                >
                  {{ isSelfSwap ? "Same mixin" : `Swap "${swapTarget.name}" with this` }}
                </ZenButton>
              </template>
              <template v-else-if="!alreadyInLayout">
                <ZenButton
                  sm
                  :variant="selectedBoxes.length ? 'default' : 'primary'"
                  class="ml-grow"
                  icon="mdi mdi-plus"
                  :disabled="!draft.name.trim()"
                  @click="commitAdd(draft, activePresetId ?? undefined, false)"
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
                  @click="commitAdd(draft, activePresetId ?? undefined, true)"
                >
                  Add &amp; assign ({{ selectedBoxes.length }})
                </ZenButton>
              </template>
              <template v-else-if="selectedBoxes.length && activeLayoutId">
                <ZenButton
                  sm
                  variant="primary"
                  class="ml-grow"
                  icon="mdi mdi-checkbox-marked-circle-plus-outline"
                  @click="assignExisting"
                >
                  Assign to selected box(es) ({{ selectedBoxes.length }})
                </ZenButton>
              </template>
            </div>
          </footer>
        </template>
        <div v-else class="ml-empty">
          <i class="mdi mdi-account-details-outline" />
          <span>
            Pick a mixin to inspect its references, LoRAs and prompt — or create a new one.
          </span>
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
import MixinFusionBar from "@/components/layout/MixinFusionBar.vue";
import {
  deleteMixinPreset,
  listSavedMixins,
  saveMixinPreset,
  type SavedMixin,
} from "@/layout/library";
import { mixinContentKey, readMixin, type Mixin } from "@/layout/promptScene";
import type { FrescoLora, FrescoRef } from "@/lib/frescoScene";
import { thumbUrl } from "@/lib/mediaRefs";
import { uid } from "@/lib/sceneUtil";

type LinkState = "library" | "linked" | "diverged" | "local";
type DraftSource = "saved" | "local" | "new";

interface MixinCard {
  key: string;
  presetId: string | null;
  layoutId: string | null;
  category: string;
  savedTime?: number;
  state: LinkState;
  mixin: Mixin;
}

const DEFAULT_CATEGORIES = ["Characters", "Styles", "Props", "Settings"];

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
  assignExisting: [layoutMixinId: string];
  /** Sync edits from a saved library preset to any layout mixin linked via `libraryId`. */
  syncFromLibrary: [libraryId: string, mixin: Mixin];
  /** Link or unlink a layout mixin to/from a library preset ID. */
  linkLayoutMixin: [layoutMixinId: string, libraryId: string | undefined, updated?: Mixin];
}>();

const saved = ref<SavedMixin[]>([]);
const scope = ref<string>("all");
const query = ref("");
const note = ref("");
const saving = ref(false);
const confirmDelete = ref(false);
const refsPanel = ref<{ upload: () => void; openBrowse: () => void; busy: boolean } | null>(null);

const activeKey = ref<string | null>(null);
const draft = ref<Mixin | null>(null);
const draftCategory = ref("");
const draftSource = ref<DraftSource>("new");
const activePresetId = ref<string | null>(null);
const activeLayoutId = ref<string | null>(null);
const activeSavedTime = ref<number | null>(null);
const baselineKey = ref("");

const modalTitle = computed(() =>
  props.swapTarget ? `Swap "${props.swapTarget.name || "Mixin"}" with…` : "Mixin Library",
);

const savedById = computed(() => new Map(saved.value.map((s) => [s.id, s])));
const layoutByLibraryId = computed(() => {
  const map = new Map<string, Mixin>();
  for (const m of props.layoutMixins) {
    if (m.libraryId && !map.has(m.libraryId)) map.set(m.libraryId, m);
  }
  return map;
});

function presetAsMixin(item: SavedMixin, index = 0): Mixin {
  return readMixin({ ...item.mixin, id: item.id, libraryId: item.id, name: item.name }, index);
}

const libraryCards = computed<MixinCard[]>(() =>
  saved.value.map((item, i) => {
    const mixin = presetAsMixin(item, i);
    const linked = layoutByLibraryId.value.get(item.id) ?? null;
    const state: LinkState = !linked
      ? "library"
      : mixinContentKey(linked) === mixinContentKey(mixin)
        ? "linked"
        : "diverged";
    return {
      key: `lib:${item.id}`,
      presetId: item.id,
      layoutId: linked?.id ?? null,
      category: item.category || "",
      savedTime: item.saved,
      state,
      mixin,
    };
  }),
);

const layoutCards = computed<MixinCard[]>(() =>
  props.layoutMixins.map((m) => {
    const preset = m.libraryId ? (savedById.value.get(m.libraryId) ?? null) : null;
    if (preset) {
      const savedMixin = presetAsMixin(preset);
      const synced = mixinContentKey(m) === mixinContentKey(savedMixin);
      return {
        key: `lib:${preset.id}`,
        presetId: preset.id,
        layoutId: m.id,
        category: preset.category || "",
        savedTime: preset.saved,
        state: synced ? "linked" : "diverged",
        mixin: cloneMixin(m),
      };
    }
    return {
      key: `local:${m.id}`,
      presetId: null,
      layoutId: m.id,
      category: "",
      state: "local",
      mixin: cloneMixin(m),
    };
  }),
);

const savedCategories = computed(() =>
  [...new Set(saved.value.map((s) => s.category.trim()).filter(Boolean))].sort((a, b) =>
    a.localeCompare(b),
  ),
);
const allCategories = computed(() => [
  ...new Set([...savedCategories.value, ...DEFAULT_CATEGORIES]),
]);

const scopes = computed<
  { id: string; label: string; icon: string; count: number; sep?: boolean }[]
>(() => [
  { id: "all", label: "All library", icon: "mdi mdi-bookshelf", count: libraryCards.value.length },
  {
    id: "layout",
    label: "In this layout",
    icon: "mdi mdi-layers-outline",
    count: layoutCards.value.length,
  },
  {
    id: "refs",
    label: "With references",
    icon: "mdi mdi-image-multiple-outline",
    count: libraryCards.value.filter((c) => activeRefs(c.mixin).length > 0).length,
  },
  {
    id: "loras",
    label: "With LoRAs",
    icon: "mdi mdi-puzzle-outline",
    count: libraryCards.value.filter((c) => activeLoras(c.mixin).length > 0).length,
  },
  ...savedCategories.value.map((cat, idx) => ({
    id: `cat:${cat}`,
    label: cat,
    icon: "mdi mdi-folder-outline",
    count: libraryCards.value.filter((c) => c.category === cat).length,
    sep: idx === 0,
  })),
]);

const visible = computed<MixinCard[]>(() => {
  let pool: MixinCard[];
  if (scope.value === "layout") pool = layoutCards.value;
  else if (scope.value === "refs")
    pool = libraryCards.value.filter((c) => activeRefs(c.mixin).length > 0);
  else if (scope.value === "loras")
    pool = libraryCards.value.filter((c) => activeLoras(c.mixin).length > 0);
  else if (scope.value.startsWith("cat:")) {
    const cat = scope.value.slice(4);
    pool = libraryCards.value.filter((c) => c.category === cat);
  } else {
    pool = libraryCards.value;
  }
  const q = query.value.trim().toLowerCase();
  if (!q) return pool;
  return pool.filter(({ mixin, category }) => {
    const hay = [
      mixin.name,
      category,
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

const currentEditKey = computed(() =>
  draft.value ? `${draftCategory.value.trim()}\n${mixinContentKey(draft.value)}` : "",
);
const draftDirty = computed(() => !!draft.value && currentEditKey.value !== baselineKey.value);

const activeLayoutMixin = computed(() =>
  activeLayoutId.value
    ? (props.layoutMixins.find((m) => m.id === activeLayoutId.value) ?? null)
    : activePresetId.value
      ? (layoutByLibraryId.value.get(activePresetId.value) ?? null)
      : null,
);
const activeSavedPreset = computed(() =>
  activePresetId.value ? (savedById.value.get(activePresetId.value) ?? null) : null,
);
const divergedPair = computed(() => {
  if (!activeLayoutMixin.value || !activeSavedPreset.value) return null;
  const savedM = presetAsMixin(activeSavedPreset.value);
  return mixinContentKey(activeLayoutMixin.value) !== mixinContentKey(savedM)
    ? { layout: activeLayoutMixin.value, preset: activeSavedPreset.value, savedMixin: savedM }
    : null;
});
const activeLinkState = computed<LinkState>(() => {
  if (divergedPair.value) return "diverged";
  if (activePresetId.value && activeLayoutMixin.value) return "linked";
  if (!activePresetId.value && activeLayoutMixin.value) return "local";
  return "library";
});
const draftBadge = computed(() => {
  if (draftSource.value === "new") return "New library preset";
  if (activeLinkState.value === "linked") return "Linked · synced with layout";
  if (activeLinkState.value === "diverged") return "Linked · modified in layout";
  if (activeLinkState.value === "local") return "Local to this layout";
  return "Library preset";
});
const alreadyInLayout = computed(() => !!activeLayoutMixin.value);

const saveLabel = computed(() => {
  if (saving.value) return "Saving…";
  if (activePresetId.value) {
    if (draftDirty.value) return alreadyInLayout.value ? "Save & sync" : "Save changes";
    return "Saved";
  }
  return activeLayoutId.value ? "Save & link" : "Save to library";
});
const saveTitle = computed(() => {
  if (activePresetId.value && alreadyInLayout.value) {
    return "Save changes to the library preset and sync them into this layout";
  }
  if (activePresetId.value) return "Update this library preset (renaming keeps its link intact)";
  if (activeLayoutId.value) return "Save to the Mixin Library and link this layout mixin to it";
  return "Save as a new preset in the Mixin Library";
});
const isSelfSwap = computed(
  () => !!props.swapTarget && activeLayoutMixin.value?.id === props.swapTarget.id,
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
    : card.layoutId
      ? "in this layout — click to inspect or edit"
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
  if (activePresetId.value && !savedById.value.has(activePresetId.value)) {
    activePresetId.value = null;
    if (!activeLayoutId.value) {
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
      const linked = props.initialMixin.libraryId
        ? libraryCards.value.find((c) => c.presetId === props.initialMixin?.libraryId)
        : null;
      const inLayout = layoutCards.value.find((c) => c.layoutId === props.initialMixin?.id);
      scope.value = linked ? "all" : "layout";
      if (inLayout) pickCard(inLayout);
      else if (linked) pickCard(linked);
    } else if (libraryCards.value.length) {
      scope.value = "all";
      pickCard(libraryCards.value[0]!);
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
  draftCategory.value = card.category;
  draftSource.value = card.presetId ? "saved" : "local";
  activePresetId.value = card.presetId;
  activeLayoutId.value = card.layoutId;
  activeSavedTime.value = card.savedTime ?? null;
  baselineKey.value = `${draftCategory.value.trim()}\n${mixinContentKey(draft.value)}`;
  confirmDelete.value = false;
  note.value = "";
}

function startNew(from?: Mixin, category = "") {
  const index = libraryCards.value.length + props.layoutMixins.length;
  const fresh: Mixin = from
    ? {
        ...cloneMixin(from),
        id: uid("c"),
        libraryId: undefined,
        name: `${from.name || "Mixin"} copy`,
        refs: from.refs.map((r) => ({ ...r, id: uid("r") })),
        loras: from.loras.map((l) => ({ ...l, id: uid("o") })),
      }
    : { ...readMixin({}, index), id: uid("c"), name: `Mixin ${index + 1}` };
  activeKey.value = null;
  draft.value = fresh;
  draftCategory.value = category;
  draftSource.value = "new";
  activePresetId.value = null;
  activeLayoutId.value = null;
  activeSavedTime.value = null;
  baselineKey.value = "";
  confirmDelete.value = false;
  note.value = "";
}

function duplicateDraft() {
  if (draft.value) startNew(draft.value, draftCategory.value);
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
    const clean = { ...cloneMixin(current), name };
    delete clean.libraryId;
    const savedRecord = await saveMixinPreset({
      id: activePresetId.value ?? undefined,
      name,
      category: draftCategory.value.trim(),
      mixin: clean,
    });
    const linkedMixin: Mixin = { ...clean, libraryId: savedRecord.id };
    if (activeLayoutId.value && !activePresetId.value) {
      emit("linkLayoutMixin", activeLayoutId.value, savedRecord.id, linkedMixin);
    } else if (layoutByLibraryId.value.has(savedRecord.id)) {
      emit("syncFromLibrary", savedRecord.id, linkedMixin);
    }
    await refresh();
    activePresetId.value = savedRecord.id;
    activeKey.value = `lib:${savedRecord.id}`;
    activeSavedTime.value = savedRecord.saved;
    draftSource.value = "saved";
    draft.value = { ...linkedMixin, id: activeLayoutId.value ?? savedRecord.id };
    draftCategory.value = savedRecord.category;
    baselineKey.value = `${draftCategory.value.trim()}\n${mixinContentKey(draft.value)}`;
    note.value =
      activeLayoutId.value || layoutByLibraryId.value.has(savedRecord.id)
        ? `Saved "${savedRecord.name}" and synced with layout.`
        : `Saved "${savedRecord.name}" to library.`;
  } catch (e) {
    note.value = `Couldn't save: ${(e as Error).message}`;
  } finally {
    saving.value = false;
  }
}

async function pushLayoutToLibrary() {
  const pair = divergedPair.value;
  if (!pair) return;
  const clean = cloneMixin(pair.layout);
  delete clean.libraryId;
  const savedRecord = await saveMixinPreset({
    id: pair.preset.id,
    name: clean.name.trim() || pair.preset.name,
    category: pair.preset.category,
    mixin: clean,
  });
  await refresh();
  const card = libraryCards.value.find((c) => c.presetId === savedRecord.id);
  if (card) pickCard(card);
  note.value = `Updated library preset "${savedRecord.name}" from layout.`;
}

function pullLibraryToLayout() {
  const pair = divergedPair.value;
  if (!pair) return;
  emit("syncFromLibrary", pair.preset.id, {
    ...cloneMixin(pair.savedMixin),
    libraryId: pair.preset.id,
  });
  const card = libraryCards.value.find((c) => c.presetId === pair.preset.id);
  if (card) pickCard(card);
  note.value = `Pulled library preset "${pair.preset.name}" into layout.`;
}

function unlinkCurrent() {
  if (!activeLayoutId.value) return;
  emit("linkLayoutMixin", activeLayoutId.value, undefined);
  activePresetId.value = null;
  draftSource.value = "local";
  activeKey.value = `local:${activeLayoutId.value}`;
  note.value = "Unlinked from library — edits here will stay local to this layout.";
}

async function removeCurrentSaved() {
  if (!activePresetId.value) return;
  if (!confirmDelete.value) {
    confirmDelete.value = true;
    return;
  }
  const targetId = activePresetId.value;
  const targetName = draft.value?.name || "mixin";
  if (activeLayoutId.value) emit("linkLayoutMixin", activeLayoutId.value, undefined);
  await deleteMixinPreset(targetId).catch(() => undefined);
  activePresetId.value = null;
  activeSavedTime.value = null;
  confirmDelete.value = false;
  if (!activeLayoutId.value) {
    activeKey.value = null;
    draft.value = null;
  } else {
    draftSource.value = "local";
    activeKey.value = `local:${activeLayoutId.value}`;
  }
  await refresh();
  note.value = `Deleted "${targetName}" from library.`;
}

function freshCopy(source: Mixin, libraryId?: string): Mixin {
  const copy = cloneMixin(source);
  return {
    ...copy,
    id: uid("c"),
    ...(libraryId ? { libraryId } : {}),
    refs: copy.refs.map((r) => ({ ...r, id: uid("r") })),
    loras: copy.loras.map((l) => ({ ...l, id: uid("o") })),
  };
}

function commitAdd(source: Mixin, libraryId: string | undefined, assignSelected: boolean) {
  emit("add", freshCopy(source, libraryId), assignSelected);
  setOpen(false);
}

function commitSwap(source: Mixin, libraryId: string | undefined) {
  if (!props.swapTarget) return;
  emit("swap", props.swapTarget.id, freshCopy(source, libraryId));
  setOpen(false);
}

function assignExisting() {
  if (!activeLayoutId.value) return;
  emit("assignExisting", activeLayoutId.value);
  setOpen(false);
}

function applyCard(card: MixinCard) {
  if (props.swapTarget) {
    if (card.layoutId === props.swapTarget.id) return;
    commitSwap(card.mixin, card.presetId ?? undefined);
    return;
  }
  if (!card.layoutId)
    commitAdd(card.mixin, card.presetId ?? undefined, props.selectedBoxes.length > 0);
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
    const self = card.layoutId === props.swapTarget.id;
    items.push({
      label: `Swap "${props.swapTarget.name}" with this`,
      icon: "mdi mdi-swap-horizontal",
      disabled: self,
      run: () => commitSwap(card.mixin, card.presetId ?? undefined),
    });
  } else if (!card.layoutId) {
    items.push({
      label: "Add to layout",
      icon: "mdi mdi-plus",
      run: () => commitAdd(card.mixin, card.presetId ?? undefined, false),
    });
    if (props.selectedBoxes.length) {
      items.push({
        label: `Add & assign to ${props.selectedBoxes.length} box(es)`,
        icon: "mdi mdi-checkbox-marked-circle-plus-outline",
        run: () => commitAdd(card.mixin, card.presetId ?? undefined, true),
      });
    }
  } else if (props.selectedBoxes.length) {
    items.push({
      label: `Assign to ${props.selectedBoxes.length} selected box(es)`,
      icon: "mdi mdi-checkbox-marked-circle-plus-outline",
      run: () => assignExisting(),
    });
  }
  items.push({
    label: "Duplicate as new preset",
    icon: "mdi mdi-content-copy",
    run: () => startNew(card.mixin, card.category),
  });
  if (!card.presetId) {
    items.push({
      label: "Save to library & link",
      icon: "mdi mdi-content-save-outline",
      run: () => {
        pickCard(card);
        void saveDraft();
      },
    });
  } else {
    const presetId = card.presetId;
    items.push("-", {
      label: "Delete from library",
      icon: "mdi mdi-delete-outline",
      danger: true,
      run: async () => {
        if (card.layoutId) emit("linkLayoutMixin", card.layoutId, undefined);
        await deleteMixinPreset(presetId).catch(() => undefined);
        if (activePresetId.value === presetId) {
          activePresetId.value = null;
          if (!activeLayoutId.value) {
            activeKey.value = null;
            draft.value = null;
          }
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
  grid-template-columns: 184px minmax(0, 1fr) 352px;
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
  overflow-y: auto;
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
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
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
  border-radius: min(var(--zen-radius, 7px), 5px);
  overflow: hidden;
}
.ml-hero-img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.ml-dot {
  position: absolute;
  top: 6px;
  left: 6px;
  width: 10px;
  height: 10px;
  border-radius: min(var(--zen-radius, 7px), 5px);
  box-shadow: 0 0 0 1.5px rgb(0 0 0 / 55%);
}
.ml-badge,
.ml-pill {
  position: absolute;
  display: inline-flex;
  align-items: center;
  gap: 3px;
  padding: 1px 6px;
  border-radius: min(var(--zen-radius, 7px), 4px);
  font-size: 10px;
  font-weight: 600;
}
.ml-badge {
  right: 6px;
  bottom: 6px;
  background: rgb(0 0 0 / 70%);
  color: #fff;
}
.ml-pill {
  top: 6px;
  right: 6px;
}
.ml-pill.linked {
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 88%, #000);
  color: #fff;
}
.ml-pill.diverged {
  background: color-mix(in srgb, var(--zen-warning, #f5b041) 88%, #000);
  color: #141418;
}
.ml-pill.local {
  background: rgb(0 0 0 / 68%);
  color: #e5e5ea;
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
.ml-cat {
  padding: 0 5px;
  border-radius: min(var(--zen-radius, 7px), 4px);
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 14%, transparent);
  color: var(--zen-text, #e5e5ea);
  font-size: 10px;
}
.ml-chip {
  display: inline-flex;
  align-items: center;
  gap: 2px;
}
.ml-trigger {
  max-width: 100%;
  padding: 0 5px;
  border-radius: min(var(--zen-radius, 7px), 4px);
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
  border-radius: min(var(--zen-radius, 7px), 4px);
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
  padding: 1px 7px;
  border-radius: var(--zen-radius, 7px);
  background: color-mix(in srgb, var(--zen-text, #fff) 8%, transparent);
}
.ml-dtag.linked {
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 20%, transparent);
  color: var(--zen-text, #e5e5ea);
}
.ml-dtag.diverged {
  background: color-mix(in srgb, var(--zen-warning, #f5b041) 20%, transparent);
  color: var(--zen-warning, #f5b041);
}
.ml-dtime {
  margin-left: auto;
}
.ml-dirty {
  color: var(--zen-warning, #f5b041);
  font-weight: 600;
}
.ml-sync-bar {
  flex: none;
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
  border: 1px solid color-mix(in srgb, var(--zen-warning, #f5b041) 45%, transparent);
  border-radius: var(--zen-radius, 7px);
  background: color-mix(in srgb, var(--zen-warning, #f5b041) 10%, transparent);
}
.ml-sync-text {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--zen-warning, #f5b041);
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
.ml-cat-input {
  width: 100%;
  box-sizing: border-box;
  height: 24px;
  padding: 0 8px;
  font: inherit;
  font-size: 11.5px;
  color: var(--zen-text, #e5e5ea);
  background: var(--zen-input, #1b1b20);
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 7px);
  outline: none;
}
.ml-cat-input:focus {
  border-color: var(--zen-accent, #6366f1);
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
