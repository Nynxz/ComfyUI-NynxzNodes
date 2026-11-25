<template>
  <div class="lm">
    <div class="lm-bar">
      <span class="lm-bar-note">Shared across boxes — right-click to swap or manage</span>
      <ZenButton
        sm
        variant="ghost"
        icon="mdi mdi-bookshelf"
        title="Open Mixin Library — browse, inspect, create, and swap saved mixins"
        @click="emit('openLibrary')"
      >
        Library
      </ZenButton>
    </div>

    <div class="lm-grid">
      <button
        v-for="mixin in mixins"
        :key="mixin.id"
        type="button"
        :class="['lm-tile', { on: mixin.id === openId }]"
        :title="tileTooltip(mixin)"
        @click="open(mixin.id)"
        @contextmenu.prevent.stop="openTileMenu($event, mixin)"
      >
        <MixinAvatar :mixin="mixin" :size="40" />
        <span class="lm-tile-name">{{ mixin.name }}</span>
        <span class="lm-tile-meta">
          <span>{{ boxCount(mixin.id) }} box{{ boxCount(mixin.id) === 1 ? "" : "es" }}</span>
          <i
            v-if="refCount(mixin)"
            class="mdi mdi-image-outline"
            :title="`${refCount(mixin)} reference(s)`"
          />
          <i
            v-if="loraCount(mixin)"
            class="mdi mdi-puzzle-outline"
            :title="`${loraCount(mixin)} LoRA(s)`"
          />
        </span>
      </button>
      <button type="button" class="lm-tile lm-new" title="New mixin" @click="emit('create')">
        <i class="mdi mdi-plus" />
        <span class="lm-tile-name">New</span>
      </button>
    </div>

    <p v-if="!mixins.length" class="lm-empty">
      A mixin is something several boxes share — a character, a prop, a style: reference images,
      LoRAs and trigger words given once, then ticked in any box.
    </p>

    <div v-if="current" class="lm-edit">
      <div class="lm-head">
        <MixinAvatar :mixin="current" :size="28" />
        <ZenInput
          sm
          class="lm-name"
          :model-value="current.name"
          placeholder="Name"
          @update:model-value="patch({ name: String($event ?? '') })"
        />
        <input
          class="lm-color"
          type="color"
          :value="current.color"
          title="Its colour"
          @input="patch({ color: ($event.target as HTMLInputElement).value })"
        />
        <ZenIconButton
          size="sm"
          icon="mdi mdi-swap-horizontal"
          :title="`Swap ${current.name} with another mixin from the library`"
          @click="emit('swapFromLibrary', current)"
        />
        <ZenIconButton
          size="sm"
          :icon="savedId === current.id ? 'mdi mdi-check' : 'mdi mdi-content-save-outline'"
          :disabled="!current.name.trim() || saving"
          :title="savedId === current.id ? 'Saved to library' : `Save ${current.name} to library`"
          @click="saveToLibrary(current)"
        />
        <ZenIconButton
          v-if="selectedBoxes.length"
          size="sm"
          icon="mdi mdi-checkbox-marked-circle-plus-outline"
          :title="`Assign ${current.name} to the ${selectedBoxes.length} selected box(es)`"
          @click="emit('assignSelected', current.id)"
        />
        <ZenIconButton
          size="sm"
          icon="mdi mdi-select-group"
          :title="`Select the ${boxCount(current.id)} box(es) it's in`"
          :disabled="!boxCount(current.id)"
          @click="emit('selectBoxes', current.id)"
        />
        <ZenIconButton
          size="sm"
          danger
          icon="mdi mdi-delete-outline"
          :title="`Delete ${current.name} (and take it out of every box)`"
          @click="remove(current.id)"
        />
      </div>
      <FrescoSection label="Description" note="who or what it is — in every box that uses it">
        <ZenInput
          type="textarea"
          :rows="3"
          sm
          :model-value="current.desc"
          placeholder="e.g. Amber, a teenage girl with a short copper-red bob, green eyes and freckles"
          @update:model-value="patch({ desc: String($event ?? '') })"
        />
      </FrescoSection>
      <FrescoSection label="Trigger words" note="added to each box's prompt">
        <ZenInput
          sm
          :model-value="current.trigger"
          placeholder="e.g. amb3r"
          @update:model-value="patch({ trigger: String($event ?? '') })"
        />
      </FrescoSection>
      <FrescoSection
        label="References"
        :note="current.refs.length ? String(current.refs.length) : 'visual refs'"
      >
        <template #actions>
          <ZenIconButton
            size="sm"
            icon="mdi mdi-upload-outline"
            title="Upload images"
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
          :model-value="current.refs"
          @update:model-value="patch({ refs: $event })"
        />
      </FrescoSection>
      <FrescoSection label="LoRAs" :note="current.loras.length ? String(current.loras.length) : ''">
        <template #actions>
          <ZenIconButton
            size="sm"
            icon="mdi mdi-plus"
            title="Add a LoRA"
            @click="patch({ loras: [...current.loras, newLora()] })"
          />
        </template>
        <FrescoLoras
          v-if="current.loras.length"
          :model-value="current.loras"
          @update:model-value="patch({ loras: $event })"
        />
      </FrescoSection>
    </div>

    <ZenContextMenu ref="tileMenu" :items="tileMenuItems" />
  </div>
</template>

<script setup lang="ts">
/**
 * The layout's Mixins tab: every mixin as a tile (right-click to swap, inspect, duplicate, save or
 * delete), a button for the full Mixin Library modal, and the open mixin's inline editor.
 */
import {
  ZenButton,
  ZenContextMenu,
  ZenIconButton,
  ZenInput,
  type ContextMenuItem,
} from "@nynxz/zenkit-ui";
import { computed, ref } from "vue";

import FrescoLoras from "@/components/fresco/FrescoLoras.vue";
import FrescoRefs from "@/components/fresco/FrescoRefs.vue";
import FrescoSection from "@/components/fresco/FrescoSection.vue";
import MixinAvatar from "@/components/layout/MixinAvatar.vue";
import { saveMixinPreset } from "@/layout/library";
import type { Mixin, PromptBox } from "@/layout/promptScene";
import type { FrescoLora } from "@/lib/frescoScene";
import { uid } from "@/lib/sceneUtil";

const props = withDefaults(defineProps<{ boxes: PromptBox[]; selectedBoxes?: string[] }>(), {
  selectedBoxes: () => [],
});
const mixins = defineModel<Mixin[]>({ required: true });
/** The mixin whose editor is open. */
const openId = defineModel<string | null>("open", { default: null });
const emit = defineEmits<{
  create: [];
  selectBoxes: [id: string];
  assignSelected: [id: string];
  openLibrary: [initial?: Mixin];
  swapFromLibrary: [target: Mixin];
  swapBoxes: [idA: string, idB: string];
}>();

const refsPanel = ref<{ upload: () => void; openBrowse: () => void; busy: boolean } | null>(null);
const current = computed(
  () => mixins.value.find((m) => m.id === openId.value) ?? mixins.value[0] ?? null,
);

const saving = ref(false);
const savedId = ref<string | null>(null);

async function saveToLibrary(mixin: Mixin) {
  if (!mixin.name.trim()) return;
  saving.value = true;
  try {
    await saveMixinPreset(mixin.name.trim(), JSON.parse(JSON.stringify(mixin)));
    savedId.value = mixin.id;
    setTimeout(() => {
      if (savedId.value === mixin.id) savedId.value = null;
    }, 1500);
  } finally {
    saving.value = false;
  }
}

function duplicateMixin(source: Mixin) {
  const copy: Mixin = {
    ...JSON.parse(JSON.stringify(source)),
    id: uid("c"),
    name: `${source.name || "Mixin"} copy`,
    refs: source.refs.map((r) => ({ ...r, id: uid("r") })),
    loras: source.loras.map((l) => ({ ...l, id: uid("o") })),
  };
  mixins.value = [...mixins.value, copy];
  openId.value = copy.id;
}

function open(id: string) {
  openId.value = id;
}
function patch(change: Partial<Mixin>) {
  const id = current.value?.id;
  if (!id) return;
  mixins.value = mixins.value.map((m) => (m.id === id ? { ...m, ...change } : m));
}
function remove(id: string) {
  mixins.value = mixins.value.filter((m) => m.id !== id);
  if (openId.value === id) openId.value = null;
}
function boxCount(id: string): number {
  return props.boxes.filter((b) => b.mixins.includes(id)).length;
}
function refCount(mixin: Mixin): number {
  return mixin.refs.filter((r) => r.on).length;
}
function loraCount(mixin: Mixin): number {
  return mixin.loras.filter((l) => l.on && l.name).length;
}
function tileTooltip(mixin: Mixin): string {
  const parts = [`${mixin.name || "Mixin"} (${boxCount(mixin.id)} box(es))`];
  if (mixin.desc.trim()) parts.push(mixin.desc.trim().slice(0, 110));
  if (mixin.trigger.trim()) parts.push(`Trigger: ${mixin.trigger.trim()}`);
  const refs = refCount(mixin);
  const loras = mixin.loras.filter((l) => l.on && l.name);
  if (refs) parts.push(`${refs} reference image(s)`);
  if (loras.length) {
    parts.push(
      `LoRAs: ${loras
        .map(
          (l) =>
            `${l.name
              .split(/[\\/]/)
              .pop()
              ?.replace(/\.[^.]+$/, "")} (${l.strength.toFixed(2)})`,
        )
        .join(", ")}`,
    );
  }
  parts.push("Right-click to swap, inspect or delete");
  return parts.join("\n");
}
function newLora(): FrescoLora {
  return { id: uid("o"), name: "", strength: 1, on: true };
}

// --- right-click menu on mixin tiles ------------------------------------------------------------

const tileMenu = ref<{ show: (e: MouseEvent) => void } | null>(null);
const menuMixin = ref<Mixin | null>(null);
function openTileMenu(e: MouseEvent, mixin: Mixin) {
  menuMixin.value = mixin;
  openId.value = mixin.id;
  tileMenu.value?.show(e);
}

const tileMenuItems = computed<ContextMenuItem[]>(() => {
  const m = menuMixin.value;
  if (!m) return [];
  const others = mixins.value.filter((x) => x.id !== m.id);
  const items: ContextMenuItem[] = [
    {
      label: "Inspect in Library dialog…",
      icon: "mdi mdi-eye-outline",
      run: () => emit("openLibrary", m),
    },
    {
      label: "Swap from Library…",
      icon: "mdi mdi-swap-horizontal",
      run: () => emit("swapFromLibrary", m),
    },
  ];
  if (others.length) {
    items.push("-");
    for (const other of others.slice(0, 6)) {
      items.push({
        label: `Swap places with "${other.name}"`,
        icon: "mdi mdi-swap-horizontal-variant",
        run: () => emit("swapBoxes", m.id, other.id),
      });
    }
  }
  items.push("-");
  if (props.selectedBoxes.length) {
    items.push({
      label: `Assign to ${props.selectedBoxes.length} selected box(es)`,
      icon: "mdi mdi-checkbox-marked-circle-plus-outline",
      run: () => emit("assignSelected", m.id),
    });
  }
  if (boxCount(m.id)) {
    items.push({
      label: `Select ${boxCount(m.id)} box(es) using this`,
      icon: "mdi mdi-select-group",
      run: () => emit("selectBoxes", m.id),
    });
  }
  items.push(
    {
      label: "Duplicate",
      icon: "mdi mdi-content-copy",
      run: () => duplicateMixin(m),
    },
    {
      label: "Save to Library",
      icon: "mdi mdi-content-save-outline",
      disabled: !m.name.trim(),
      run: () => void saveToLibrary(m),
    },
    "-",
    {
      label: "Delete mixin",
      icon: "mdi mdi-delete-outline",
      danger: true,
      run: () => remove(m.id),
    },
  );
  return items;
});
</script>

<style scoped>
.lm {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.lm-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
}
.lm-bar-note {
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.lm-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(76px, 1fr));
  gap: 6px;
}
.lm-tile {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  min-width: 0;
  padding: 8px 4px 6px;
  border: 1px solid color-mix(in srgb, var(--zen-border, #34343c) 70%, transparent);
  border-radius: var(--zen-radius, 7px);
  background: transparent;
  color: var(--zen-text, #e5e5ea);
  cursor: pointer;
}
.lm-tile:hover {
  background: color-mix(in srgb, var(--zen-text, #fff) 4%, transparent);
}
.lm-tile.on {
  border-color: var(--zen-accent, #3b82f6);
  background: color-mix(in srgb, var(--zen-accent, #3b82f6) 12%, transparent);
}
.lm-tile-name {
  max-width: 100%;
  overflow: hidden;
  font-size: 11px;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.lm-tile-meta {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  font-size: 10px;
  color: var(--zen-muted, #9aa0aa);
}
.lm-new {
  justify-content: center;
  border-style: dashed;
  color: var(--zen-muted, #9aa0aa);
}
.lm-new .mdi {
  font-size: 22px;
}
.lm-empty {
  margin: 0;
  font-size: 11px;
  line-height: 1.45;
  color: var(--zen-muted, #9aa0aa);
}
.lm-edit {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-top: 10px;
  border-top: 1px solid color-mix(in srgb, var(--zen-border, #34343c) 60%, transparent);
}
.lm-head {
  display: flex;
  align-items: center;
  gap: 6px;
}
.lm-name {
  flex: 1;
  min-width: 0;
}
.lm-color {
  flex: none;
  width: 22px;
  height: 22px;
  padding: 0;
  border: 0;
  border-radius: 4px;
  background: transparent;
  cursor: pointer;
}
</style>
