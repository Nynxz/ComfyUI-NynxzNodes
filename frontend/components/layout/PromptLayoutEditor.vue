<template>
  <LayoutEditor
    ref="editor"
    v-model:selected="selected"
    :mode="mode"
    :layout="layout"
    :node="node"
    :in-panel="inPanel"
    property-prefix="nynxzLayout"
    :help="HELP"
    @pop-out="emit('popOut')"
  >
    <template #rail>
      <ZenPopover v-model:open="modeOpen" placement="right-start" role="menu">
        <template #trigger="{ toggle }">
          <ZenIconButton
            size="sm"
            :icon="mode.icon"
            :active="modeOpen"
            :title="`Mode: ${mode.label} — the model the prompt is written for`"
            @click="toggle"
          />
        </template>
        <div class="pl-menu">
          <span class="pl-menu-title">Write the prompt for</span>
          <ZenMenuItem
            v-for="m in PROMPT_MODES"
            :key="m.id"
            :icon="m.icon"
            @select="switchMode(m.id as PromptModeId)"
          >
            {{ m.label }}
            <template #hint>{{
              m.id === scene.mode ? "on" : MODE_NOTES[m.id as PromptModeId]
            }}</template>
          </ZenMenuItem>
        </div>
      </ZenPopover>
    </template>

    <template #element>
      <div v-if="box" class="pl-form">
        <FrescoSection label="Kind">
          <ZenToggleGroup
            size="sm"
            :model-value="box.kind"
            :options="kindOptions"
            @update:model-value="patchBox({ kind: $event as BoxKind })"
          />
        </FrescoSection>
        <FrescoSection v-if="box.kind === 'text'" label="Text" note="rendered exactly">
          <ZenInput
            sm
            :model-value="box.text"
            placeholder="The words to render"
            @update:model-value="patchBox({ text: String($event ?? '') })"
          />
        </FrescoSection>
        <FrescoSection label="Description" :note="citeNote">
          <template v-if="citeId" #actions>
            <ZenIconButton
              size="sm"
              icon="mdi mdi-at"
              :title="`Add <${citeId}> to the end of the prompt`"
              @click="cite(citeId)"
            />
          </template>
          <ZenInput
            type="textarea"
            :rows="5"
            sm
            :model-value="box.desc"
            :placeholder="descPlaceholder"
            @update:model-value="patchBox({ desc: String($event ?? '') })"
          />
        </FrescoSection>
        <FrescoSection v-if="scene.mode !== 'klein'" label="Colours" note="up to 5">
          <PaletteEdit
            :model-value="box.palette"
            @update:model-value="patchBox({ palette: $event })"
          />
        </FrescoSection>
        <FrescoSection v-if="scene.mode === 'moss'" label="Tags" note="comma separated">
          <ZenInput
            sm
            :model-value="box.tags.join(', ')"
            placeholder="e.g. 1girl, red scarf"
            @update:model-value="patchBox({ tags: splitList($event) })"
          />
        </FrescoSection>
      </div>
    </template>

    <template #scene>
      <div class="pl-form">
        <FrescoSection label="Mode">
          <ZenToggleGroup
            size="sm"
            :model-value="scene.mode"
            :options="PROMPT_MODES.map((m) => ({ value: m.id, label: m.label, icon: m.icon }))"
            @update:model-value="switchMode($event as PromptModeId)"
          />
        </FrescoSection>
        <FrescoSection :label="PROMPT_LABEL[scene.mode]" :note="PROMPT_NOTE[scene.mode]">
          <ZenInput
            type="textarea"
            :rows="5"
            sm
            :model-value="scene.prompt"
            placeholder="The whole picture"
            @update:model-value="patchScene({ prompt: String($event ?? '') })"
          />
        </FrescoSection>

        <template v-if="scene.mode === 'ideogram'">
          <FrescoSection label="Background">
            <ZenInput
              type="textarea"
              :rows="2"
              sm
              :model-value="scene.background"
              placeholder="What's behind every box"
              @update:model-value="patchScene({ background: String($event ?? '') })"
            />
          </FrescoSection>
          <FrescoSection label="Style">
            <template #actions>
              <ZenSwitch
                :model-value="scene.style.on"
                @update:model-value="patchStyle({ on: $event })"
              />
            </template>
            <template v-if="scene.style.on">
              <ZenToggleGroup
                size="sm"
                :model-value="scene.style.kind"
                :options="[
                  { value: 'photo', label: 'Photo' },
                  { value: 'art', label: 'Art' },
                ]"
                @update:model-value="patchStyle({ kind: $event as 'photo' | 'art' })"
              />
              <ZenInput
                v-for="f in styleFields"
                :key="f.key"
                sm
                :model-value="scene.style[f.key]"
                :placeholder="f.label"
                :title="f.label"
                @update:model-value="patchStyle({ [f.key]: String($event ?? '') })"
              />
              <PaletteEdit
                :model-value="scene.style.palette"
                :max="16"
                @update:model-value="patchStyle({ palette: $event })"
              />
            </template>
          </FrescoSection>
        </template>

        <template v-if="scene.mode === 'moss'">
          <FrescoSection label="Summary">
            <ZenInput
              sm
              :model-value="scene.meta.join(', ')"
              placeholder="Meta, e.g. poster, illustration"
              title="Meta — what kind of image it is"
              @update:model-value="patchScene({ meta: splitList($event) })"
            />
            <ZenSelect
              size="sm"
              :model-value="scene.canvas"
              :options="CANVAS_OPTIONS"
              @update:model-value="patchScene({ canvas: $event as MossCanvas })"
            />
            <PaletteEdit
              :model-value="scene.palette"
              :max="16"
              @update:model-value="patchScene({ palette: $event })"
            />
          </FrescoSection>
        </template>

        <FrescoSection label="Prompt out" :note="compiling ? 'writing…' : `${output.length} chars`">
          <template #actions>
            <ZenIconButton
              size="sm"
              icon="mdi mdi-content-copy"
              title="Copy the prompt"
              :disabled="!output"
              @click="copyOutput"
            />
          </template>
          <pre class="pl-out">{{ output || "—" }}</pre>
          <ul v-if="warnings.length" class="pl-warn">
            <li v-for="w in warnings" :key="w">{{ w }}</li>
          </ul>
        </FrescoSection>
      </div>
    </template>
  </LayoutEditor>
</template>

<script setup lang="ts">
/**
 * The Layout node's editor: the shared layout editor in one of the prompt modes, with a mode
 * switch on the rail, an inspector for what each model reads, and the prompt as the node will
 * output it (written by the backend, so the preview can't drift from the output).
 */
import {
  ZenIconButton,
  ZenInput,
  ZenMenuItem,
  ZenPopover,
  ZenSelect,
  ZenSwitch,
  ZenToggleGroup,
} from "@nynxz/zenkit-ui";
import { computed, onBeforeUnmount, ref, watch } from "vue";

import FrescoSection from "@/components/fresco/FrescoSection.vue";
import LayoutEditor from "@/components/layout/LayoutEditor.vue";
import PaletteEdit from "@/components/layout/PaletteEdit.vue";
import { compileLayout } from "@/layout/library";
import { PROMPT_MODES } from "@/layout/modes";
import { kleinId } from "@/layout/promptScene";
import type {
  BoxKind,
  IdeogramStyle,
  MossCanvas,
  PromptBox,
  PromptModeId,
  PromptScene,
} from "@/layout/promptScene";
import type { SharedLayout } from "@/layout/store";

const props = defineProps<{
  layout: SharedLayout<PromptScene>;
  node?: {
    id?: string | number;
    graph?: { setDirtyCanvas?: (a: boolean, b: boolean) => void };
    properties?: Record<string, unknown>;
  };
  inPanel?: boolean;
}>();
const emit = defineEmits<{ popOut: [] }>();

const HELP: [string, string][] = [
  ["Order", "bottom to top is the order the prompt lists boxes — background first"],
  ["Hidden boxes", "left out of the prompt"],
];
const MODE_NOTES: Record<PromptModeId, string> = {
  ideogram: "caption JSON",
  klein: "FLUX 3 Klein",
  moss: "MOSS-Image",
};
const PROMPT_LABEL: Record<PromptModeId, string> = {
  ideogram: "High-level description",
  klein: "Caption",
  moss: "Caption",
};
const PROMPT_NOTE: Record<PromptModeId, string> = {
  ideogram: "the whole picture",
  klein: "cite each box as <obj_N>",
  moss: "the scene and its style",
};
const CANVAS_OPTIONS = [
  { value: "", label: "Canvas: not said" },
  { value: "single", label: "Single image" },
  { value: "multi_panel", label: "Multi-panel" },
  { value: "collage", label: "Collage" },
];
type StyleText = "aesthetics" | "lighting" | "medium" | "photo" | "art_style";
const styleFields = computed<{ key: StyleText; label: string }[]>(() => [
  { key: "aesthetics", label: "Aesthetics" },
  { key: "lighting", label: "Lighting" },
  { key: "medium", label: "Medium" },
  scene.value.style.kind === "photo"
    ? { key: "photo", label: "Photo — camera, lens, film" }
    : { key: "art_style", label: "Art style" },
]);

const editor = ref<{ update: (next: unknown) => void } | null>(null);
const selected = ref<string[]>([]);
const modeOpen = ref(false);
const scene = props.layout.scene;
const mode = computed(
  () => PROMPT_MODES.find((m) => m.id === scene.value.mode) ?? PROMPT_MODES[0]!,
);

function update(next: PromptScene) {
  editor.value?.update(next);
}
function patchScene(patch: Partial<PromptScene>) {
  update({ ...scene.value, ...patch });
}
function patchStyle(patch: Partial<IdeogramStyle>) {
  patchScene({ style: { ...scene.value.style, ...patch } });
}
function switchMode(id: PromptModeId) {
  modeOpen.value = false;
  if (id !== scene.value.mode) patchScene({ mode: id });
}
const splitList = (v: unknown) =>
  String(v ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

// --- the selected box ---------------------------------------------------------------------------

const box = computed<PromptBox | null>(() =>
  selected.value.length === 1
    ? (scene.value.boxes.find((b) => b.id === selected.value[0]) ?? null)
    : null,
);
function patchBox(patch: Partial<PromptBox>) {
  const id = box.value?.id;
  if (!id) return;
  patchScene({ boxes: scene.value.boxes.map((b) => (b.id === id ? { ...b, ...patch } : b)) });
}
const kindOptions = computed(() => [
  { value: "object", label: "Object", icon: "mdi mdi-cube-outline" },
  ...(scene.value.mode === "moss"
    ? [{ value: "character", label: "Character", icon: "mdi mdi-account-outline" }]
    : []),
  { value: "text", label: "Text", icon: "mdi mdi-format-text" },
]);
const citeId = computed(() =>
  scene.value.mode === "klein" && box.value ? kleinId(scene.value, box.value.id) : null,
);
const citeNote = computed(() => {
  if (!citeId.value) return scene.value.mode === "klein" ? "hidden: not in the prompt" : undefined;
  const cited = scene.value.prompt.includes(`<${citeId.value}>`);
  return `<${citeId.value}>${cited ? "" : " — not cited yet"}`;
});
const descPlaceholder = computed(() =>
  scene.value.mode === "klein"
    ? "Opens with what it is: shape, material, colours, markings — no position"
    : "What it is and what it looks like",
);
function cite(id: string) {
  const prompt = scene.value.prompt.trimEnd();
  patchScene({ prompt: `${prompt}${prompt ? " " : ""}<${id}>` });
}

// --- the prompt out -----------------------------------------------------------------------------

const output = ref("");
const warnings = ref<string[]>([]);
const compiling = ref(false);
let timer: ReturnType<typeof setTimeout> | undefined;
let ticket = 0;
async function recompile() {
  const mine = ++ticket;
  compiling.value = true;
  try {
    const result = await compileLayout(JSON.parse(JSON.stringify(scene.value)));
    if (mine !== ticket) return;
    output.value = result.prompt;
    warnings.value = result.warnings;
  } catch (e) {
    if (mine === ticket) warnings.value = [`Couldn't write the prompt: ${(e as Error).message}`];
  } finally {
    if (mine === ticket) compiling.value = false;
  }
}
watch(
  scene,
  () => {
    clearTimeout(timer);
    timer = setTimeout(recompile, 250);
  },
  { deep: true, immediate: true },
);
onBeforeUnmount(() => clearTimeout(timer));
async function copyOutput() {
  await navigator.clipboard.writeText(output.value).catch(() => undefined);
}
</script>

<style scoped>
.pl-form {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.pl-menu {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 200px;
  padding: 4px;
  font-size: 12px;
  color: var(--zen-text, #e5e5ea);
}
.pl-menu-title {
  padding: 4px 9px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--zen-muted, #9aa0aa);
}
.pl-out {
  margin: 0;
  max-height: 160px;
  overflow: auto;
  padding: 6px 8px;
  border-radius: var(--zen-radius, 7px);
  background: var(--zen-surface-2, #26262d);
  font-family: var(--zen-mono, ui-monospace, monospace);
  font-size: 10.5px;
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
  user-select: text;
}
.pl-warn {
  margin: 0;
  padding-left: 16px;
  font-size: 11px;
  color: var(--zen-warning, #f5b041);
}
</style>
