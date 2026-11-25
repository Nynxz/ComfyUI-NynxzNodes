<template>
  <ZenWidget fill :gap="6" pad="2px" class="st">
    <div class="st-pick">
      <ZenCombo
        class="st-combo"
        :model-value="value.id"
        :items="options"
        placeholder="Pick a saved text…"
        empty-text="Nothing saved yet"
        :sorted="false"
        @update:model-value="pick(String($event))"
      >
        <template #selected>
          <span v-if="value.name" class="st-current">{{ value.name }}</span>
          <span v-else class="st-placeholder">Pick a saved text…</span>
        </template>
        <template #option="{ item }">
          <span class="st-opt">
            <span class="st-opt-name">{{ item.name }}</span>
            <span v-if="item.category" class="st-opt-cat">{{ item.category }}</span>
          </span>
        </template>
      </ZenCombo>
      <ZenIconButton
        icon="mdi mdi-pencil-outline"
        :title="value.id ? 'Edit this text' : 'Open the editor'"
        @click="editing = true"
      />
    </div>

    <!-- A press on the preview opens the editor: reading and changing are one step apart. -->
    <button v-if="value.text" class="st-preview" title="Edit" @click="editing = true">
      {{ value.text }}
    </button>
    <div v-else class="st-empty">
      {{ value.id ? "This text is empty" : "Outputs nothing until a text is picked" }}
    </div>

    <div v-if="adrift" class="st-note">
      <i class="mdi mdi-alert-circle-outline" />
      <span>Not in your library</span>
      <button class="st-link" @click="adopt">Save it</button>
    </div>

    <template #footer>
      <WidgetActions
        label="Saved texts"
        icon="mdi mdi-text-box-multiple-outline"
        :count="lib.texts.value.length"
        @primary="editing = true"
      />
    </template>

    <TextLibraryDialog
      v-model:open="editing"
      :selected="value.id"
      :used-id="value.id"
      :orphan="adrift ? { name: value.name, category: '', text: value.text } : null"
      @use="use"
    />
  </ZenWidget>
</template>

<script lang="ts">
import type { WidgetOptions } from "@/framework";

/** Per-widget overrides picked up by discovery; the value default comes from the Python schema
 *  (see nodes/text/saved_text.py).
 *
 *  fill: the node is yours to size, and the text takes whatever room it is given, scrolling
 *  when it is longer than that. */
export const widgetOptions: WidgetOptions = {
  minHeight: 150,
  minWidth: 260,
  fill: true,
};
</script>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { ZenCombo, ZenIconButton, ZenWidget } from "@nynxz/zenkit-ui";

import TextLibraryDialog from "@/components/text/TextLibraryDialog.vue";
import WidgetActions from "@/components/WidgetActions.vue";
import * as lib from "@/lib/textLibrary";
import type { SavedText } from "@/lib/textLibrary";

interface Held {
  id: string;
  name: string;
  text: string;
}

const props = defineProps<{
  widget?: { value: unknown; callback?: (v: unknown) => void };
  node?: { graph?: { setDirtyCanvas?: (a: boolean, b: boolean) => void } };
}>();

// The node keeps the text itself beside the id it was picked by: what runs is what the node
// shows, on any machine, and ComfyUI re-runs it only when that text changes.
function read(raw: unknown): Held {
  const v = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  return { id: String(v.id ?? ""), name: String(v.name ?? ""), text: String(v.text ?? "") };
}
// ComfyUI's host object, not Vue state: writing its value is how a node body persists.
const host = props.widget;
const value = ref<Held>(read(host?.value));
function commit(next: Held) {
  value.value = next;
  if (host) {
    host.value = { ...next };
    try {
      host.callback?.(host.value);
    } catch {
      /* no callback registered */
    }
  }
  props.node?.graph?.setDirtyCanvas?.(true, true);
}

const editing = ref(false);
void lib.ensure();

const options = computed(() =>
  lib.grouped([...lib.texts.value]).flatMap((section) =>
    section.items.map((t) => ({
      value: t.id,
      label: t.name,
      name: t.name,
      category: t.category,
      keywords: `${t.category} ${t.text.slice(0, 400)}`,
    })),
  ),
);

/** Holds text the library doesn't have: a workflow from elsewhere, or an entry since deleted. */
const adrift = computed(() => lib.loaded.value && !!value.value.text && !lib.byId(value.value.id));

const use = (entry: SavedText) => commit({ id: entry.id, name: entry.name, text: entry.text });
function pick(id: string) {
  const entry = lib.byId(id);
  if (entry) use(entry);
}
async function adopt() {
  const saved = await lib.saveText({
    name: value.value.name || "Untitled",
    category: "",
    text: value.value.text,
  });
  if (saved) use(saved);
}

// The library is the source: an entry edited anywhere (this node's editor, another node's)
// reaches every node holding it.
watch(
  () => lib.byId(value.value.id),
  (entry) => {
    if (entry && (entry.text !== value.value.text || entry.name !== value.value.name)) use(entry);
  },
  { deep: true },
);
</script>

<style scoped>
.st-pick {
  display: flex;
  align-items: center;
  gap: 6px;
}
.st-combo {
  flex: 1;
  min-width: 0;
}
/* One line, always: a long name is cut short with an ellipsis, never wrapped under itself. */
.st-current,
.st-placeholder {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.st-current {
  font-weight: 600;
}
.st-placeholder {
  color: var(--zen-muted, #9aa0aa);
}
.st-opt {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  min-width: 0;
  width: 100%;
}
.st-opt-name {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.st-opt-cat {
  flex: none;
  font-size: 10.5px;
  color: var(--zen-muted, #9aa0aa);
}
.st-pick,
.st-note {
  flex: none;
}
/* The text takes the rest of the node, and scrolls inside it when there is more. */
/* ZenWidget's footer spacer grows too; this outweighs it, so the slack goes to the text. */
.st-preview,
.st-empty {
  flex: 1000 1 0;
  min-height: 0;
}
.st-preview {
  display: block;
  overflow-y: auto;
  padding: 7px 9px;
  text-align: left;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font: inherit;
  font-size: 11.5px;
  line-height: 1.45;
  color: var(--zen-muted, #b9bcc4);
  background: color-mix(in srgb, var(--zen-text, #fff) 4%, transparent);
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 7px);
  cursor: pointer;
}
.st-preview:hover {
  border-color: var(--zen-accent, #6366f1);
  color: var(--zen-text, #e5e5ea);
}
.st-empty {
  display: grid;
  place-items: center;
  padding: 10px 8px;
  text-align: center;
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
  border: 1px dashed var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 7px);
}
.st-note {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--zen-warning, #f59e0b);
}
.st-link {
  margin-left: auto;
  padding: 0;
  font: inherit;
  color: var(--zen-accent, #6366f1);
  background: none;
  border: 0;
  cursor: pointer;
  text-decoration: underline;
}
</style>
