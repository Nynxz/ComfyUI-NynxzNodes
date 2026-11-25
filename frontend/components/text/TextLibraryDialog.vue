<!-- The saved-text editor: the library down the left, the picked entry's name, category and text
     on the right. Opened from a Saved Text node, where "Use on this node" picks the entry. -->
<template>
  <ZenModal
    :open="open"
    title="Saved text"
    width="min(960px, 94vw)"
    height="min(680px, 88vh)"
    @update:open="onOpenChange"
  >
    <div class="tl">
      <div class="tl-side">
        <div class="tl-side-top">
          <ZenInput v-model="query" size="sm" placeholder="Search…" class="tl-search" />
          <ZenIconButton size="sm" icon="mdi mdi-plus" title="New text" @click="startNew()" />
        </div>
        <div class="tl-list zen-scroll">
          <template v-for="section in sections" :key="section.category">
            <div class="tl-cat">{{ section.category || "Uncategorised" }}</div>
            <button
              v-for="entry in section.items"
              :key="entry.id"
              class="tl-item"
              :class="{ on: entry.id === draft?.id }"
              @click="select(entry)"
            >
              <span class="tl-item-name">
                {{ entry.name }}
                <i
                  v-if="entry.id === usedId"
                  class="mdi mdi-check-circle tl-used"
                  title="On this node"
                />
              </span>
              <span class="tl-item-sub">{{ firstLine(entry.text) || "Empty" }}</span>
            </button>
          </template>
          <div v-if="!sections.length" class="tl-none">
            {{ query ? "Nothing matches" : "Nothing saved yet" }}
          </div>
        </div>
      </div>

      <div v-if="draft" class="tl-edit">
        <div class="tl-fields">
          <label class="tl-field tl-name">
            <span>Name</span>
            <ZenInput v-model="draft.name" placeholder="What this text is" />
          </label>
          <label class="tl-field">
            <span>Category</span>
            <input
              v-model="draft.category"
              class="tl-cat-input"
              list="nynxz-text-categories"
              placeholder="None"
              spellcheck="false"
            />
            <datalist id="nynxz-text-categories">
              <option v-for="c in lib.categories.value" :key="c" :value="c" />
            </datalist>
          </label>
        </div>
        <textarea
          ref="area"
          v-model="draft.text"
          class="tl-area zen-scroll"
          placeholder="Write the text here…"
          spellcheck="false"
          @keydown.ctrl.s.prevent="save()"
          @keydown.meta.s.prevent="save()"
        />
        <div class="tl-foot">
          <span class="tl-count">
            {{ draft.text.length.toLocaleString() }} characters
            <template v-if="status"> · {{ status }}</template>
          </span>
          <span class="tl-grow" />
          <template v-if="draft.id">
            <ZenButton
              v-if="confirming"
              size="sm"
              variant="danger"
              icon="mdi mdi-delete-outline"
              @click="remove()"
            >
              Delete for good
            </ZenButton>
            <ZenIconButton
              v-else
              size="sm"
              danger
              icon="mdi mdi-delete-outline"
              title="Delete"
              @click="confirming = true"
            />
            <ZenIconButton
              size="sm"
              icon="mdi mdi-content-copy"
              title="Duplicate"
              @click="duplicate()"
            />
          </template>
          <ZenButton size="sm" :disabled="!dirty || !valid" @click="save()">Save</ZenButton>
          <ZenButton
            v-if="pickable"
            size="sm"
            variant="primary"
            icon="mdi mdi-check"
            :disabled="!valid"
            @click="use()"
          >
            Use on this node
          </ZenButton>
        </div>
      </div>
      <ZenEmpty v-else class="tl-edit" icon="mdi mdi-text-box-outline" title="No text picked">
        Pick one on the left, or start a new one.
        <template #actions>
          <ZenButton size="sm" icon="mdi mdi-plus" @click="startNew()">New text</ZenButton>
        </template>
      </ZenEmpty>
    </div>
  </ZenModal>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { ZenButton, ZenEmpty, ZenIconButton, ZenInput, ZenModal } from "@nynxz/zenkit-ui";

import * as lib from "@/lib/textLibrary";
import type { SavedText, SavedTextDraft } from "@/lib/textLibrary";

const props = withDefaults(
  defineProps<{
    open: boolean;
    /** The entry to open on. */
    selected?: string;
    /** The entry the node holds now: marked in the list. */
    usedId?: string;
    /** Text the node holds that the library doesn't: offered as a new entry to save. */
    orphan?: SavedTextDraft | null;
    /** Show "Use on this node". */
    pickable?: boolean;
  }>(),
  { selected: "", usedId: "", orphan: null, pickable: true },
);
const emit = defineEmits<{ "update:open": [boolean]; use: [SavedText] }>();

const query = ref("");
const draft = ref<SavedTextDraft | null>(null);
const confirming = ref(false);
const status = ref("");
const area = ref<HTMLTextAreaElement | null>(null);

const sections = computed(() => {
  const q = query.value.trim().toLowerCase();
  const list = q
    ? lib.texts.value.filter((t) => `${t.name}\n${t.category}\n${t.text}`.toLowerCase().includes(q))
    : lib.texts.value;
  return lib.grouped([...list]);
});
const firstLine = (text: string) => text.trim().split("\n", 1)[0]!.slice(0, 80);

const stored = computed(() => (draft.value?.id ? lib.byId(draft.value.id) : undefined));
const valid = computed(() => !!draft.value?.name.trim());
const dirty = computed(() => {
  const d = draft.value;
  if (!d) return false;
  const s = stored.value;
  if (!s) return !!(d.name.trim() || d.text);
  return d.name !== s.name || d.category !== s.category || d.text !== s.text;
});

function edit(entry: SavedTextDraft) {
  draft.value = { id: entry.id, name: entry.name, category: entry.category, text: entry.text };
  confirming.value = false;
  status.value = "";
}
/** Keep what was typed before leaving it. Nothing typed, or no name yet, is nothing to keep. */
async function keep(): Promise<SavedText | null> {
  const d = draft.value;
  if (!d || !valid.value) return null;
  if (!dirty.value) return stored.value ?? null;
  const saved = await lib.saveText({ ...d, name: d.name.trim(), category: d.category.trim() });
  if (!saved) {
    status.value = "couldn't save";
    return null;
  }
  if (draft.value === d) edit(saved);
  return saved;
}
async function save() {
  if (await keep()) status.value = "saved";
}
async function select(entry: SavedText) {
  if (entry.id === draft.value?.id) return;
  await keep();
  edit(entry);
}
async function startNew(from?: SavedTextDraft) {
  await keep();
  edit({ name: from?.name ?? "", category: from?.category ?? "", text: from?.text ?? "" });
  void nextTick(() => area.value?.focus());
}
async function duplicate() {
  const d = draft.value;
  if (!d) return;
  await startNew({ ...d, name: `${d.name} copy` });
}
async function remove() {
  const id = draft.value?.id;
  if (!id) return;
  if (await lib.deleteText(id)) draft.value = null;
  confirming.value = false;
}
async function use() {
  const saved = await keep();
  if (!saved) return;
  emit("use", saved);
  emit("update:open", false);
}
async function onOpenChange(next: boolean) {
  if (!next) await keep();
  emit("update:open", next);
}

watch(
  () => props.open,
  async (open) => {
    if (!open) return;
    query.value = "";
    await lib.ensure();
    const first = lib.byId(props.selected);
    if (first) edit(first);
    else if (props.orphan?.text) edit({ ...props.orphan, id: undefined });
    else draft.value = null;
  },
  { immediate: true },
);
</script>

<style scoped>
.tl {
  display: grid;
  grid-template-columns: 250px minmax(0, 1fr);
  gap: 14px;
  height: 100%;
  min-height: 0;
}
.tl-side {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 0;
  padding-right: 14px;
  border-right: 1px solid var(--zen-border, #34343c);
}
.tl-side-top {
  display: flex;
  align-items: center;
  gap: 6px;
}
.tl-search {
  flex: 1;
  min-width: 0;
}
.tl-list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.tl-cat {
  padding: 8px 6px 3px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--zen-muted, #9aa0aa);
}
.tl-item {
  display: flex;
  flex-direction: column;
  gap: 1px;
  padding: 6px 8px;
  text-align: left;
  font: inherit;
  color: var(--zen-text, #e5e5ea);
  background: none;
  border: 1px solid transparent;
  border-radius: var(--zen-radius, 7px);
  cursor: pointer;
}
.tl-item:hover {
  background: color-mix(in srgb, var(--zen-text, #fff) 6%, transparent);
}
.tl-item.on {
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 16%, transparent);
  border-color: color-mix(in srgb, var(--zen-accent, #6366f1) 45%, transparent);
}
.tl-item-name {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 12.5px;
  font-weight: 600;
}
.tl-used {
  font-size: 13px;
  color: var(--zen-accent, #6366f1);
}
.tl-item-sub,
.tl-none {
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.tl-item-sub {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.tl-none {
  padding: 16px 8px;
  text-align: center;
}
.tl-edit {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
  min-height: 0;
}
.tl-fields {
  display: flex;
  gap: 10px;
}
.tl-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 0 0 200px;
  min-width: 0;
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.tl-field.tl-name {
  flex: 1;
}
.tl-cat-input,
.tl-area {
  font: inherit;
  color: var(--zen-text, #e5e5ea);
  background: var(--zen-surface, #202026);
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 7px);
  outline: none;
}
.tl-cat-input {
  height: 28px;
  padding: 0 8px;
  font-size: 12px;
}
.tl-cat-input:focus,
.tl-area:focus {
  border-color: var(--zen-accent, #6366f1);
}
.tl-area {
  flex: 1;
  min-height: 0;
  resize: none;
  padding: 10px 12px;
  font-family: var(--zen-font-mono, ui-monospace, "SF Mono", Menlo, monospace);
  font-size: 12.5px;
  line-height: 1.55;
}
.tl-foot {
  display: flex;
  align-items: center;
  gap: 6px;
}
.tl-count {
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.tl-grow {
  flex: 1;
}
@media (max-width: 640px) {
  .tl {
    grid-template-columns: minmax(0, 1fr);
    grid-template-rows: 170px minmax(0, 1fr);
  }
  .tl-side {
    padding: 0 0 10px;
    border-right: 0;
    border-bottom: 1px solid var(--zen-border, #34343c);
  }
}
</style>
