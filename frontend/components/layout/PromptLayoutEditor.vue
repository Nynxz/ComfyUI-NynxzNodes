<template>
  <div class="pl-host" @contextmenu="keepContextMenu">
    <LayoutEditor
      ref="editor"
      v-model:selected="selected"
      :mode="mode"
      :modes="PROMPT_MODES"
      :layout="layout"
      :node="node"
      :in-panel="inPanel"
      property-prefix="nynxzLayout"
      :help="HELP"
      @pop-out="emit('popOut')"
    >
      <template #rail>
        <ZenIconButton
          size="sm"
          icon="mdi mdi-cog-outline"
          :active="settingsOpen"
          :title="`${mode.label} layout — settings: mode${SETTINGS_NOTE[scene.mode]}`"
          @click="settingsOpen = true"
        />
        <ZenIconButton
          size="sm"
          icon="mdi mdi-creation-outline"
          :active="agentOpen"
          title="Layout agent — a model draws the boxes from the trace image or your words (needs Layout Inputs)"
          @click="agentOpen = true"
        />
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

          <FrescoSection
            v-if="scene.mode === 'ideogram'"
            label="Background"
            note="behind every box"
          >
            <ZenInput
              type="textarea"
              :rows="2"
              sm
              :model-value="scene.background"
              placeholder="What's behind every box"
              @update:model-value="patchScene({ background: String($event ?? '') })"
            />
          </FrescoSection>

          <FrescoSection
            label="Prompt out"
            :note="compiling ? 'writing…' : `${output.length} chars`"
          >
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
    <LayoutAgent
      v-model:open="agentOpen"
      :scene="scene"
      :node-id="String(node?.id ?? '')"
      :image="editor?.backdrop ?? null"
      @apply="applyAgent"
    />
    <LayoutSettings v-model:open="settingsOpen" :scene="scene" @patch="patchScene" />
    <div v-if="choosing" class="pl-choose">
      <div class="pl-choose-card">
        <span class="pl-choose-title">What is this layout for?</span>
        <span class="pl-choose-note">
          Pick the model the prompt is written for. The cog on the left changes it later; the boxes
          stay.
        </span>
        <ModeCards :model-value="null" @update:model-value="pickMode" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * The Layout node's editor: the shared layout editor in the layout's prompt mode. The mode is
 * picked when the node is made and changed in the settings (the cog), along with whatever applies
 * to the whole layout; the sidebar holds only what the mode reads per box and for the scene, and
 * the prompt as the node will output it (written by the backend, so it can't drift).
 */
import { ZenIconButton, ZenInput, ZenToggleGroup } from "@nynxz/zenkit-ui";
import { api } from "@comfy/api";
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";

import FrescoSection from "@/components/fresco/FrescoSection.vue";
import LayoutAgent from "@/components/layout/LayoutAgent.vue";
import LayoutEditor from "@/components/layout/LayoutEditor.vue";
import LayoutSettings from "@/components/layout/LayoutSettings.vue";
import ModeCards from "@/components/layout/ModeCards.vue";
import PaletteEdit from "@/components/layout/PaletteEdit.vue";
import { compileLayout } from "@/layout/library";
import { keepContextMenu } from "@/lib/ownContextMenu";
import { PROMPT_MODES } from "@/layout/modes";
import type { AgentImage, AgentLayout } from "@/layout/agent";
import { boxId, kleinId, nextColor, readBox } from "@/layout/promptScene";
import type { SourceType } from "@/lib/fusionApi";
import type { BoxKind, PromptBox, PromptModeId, PromptScene } from "@/layout/promptScene";
import type { SharedLayout } from "@/layout/store";

const props = defineProps<{
  layout: SharedLayout<PromptScene>;
  node?: {
    id?: string | number;
    graph?: { setDirtyCanvas?: (a: boolean, b: boolean) => void };
    properties?: Record<string, unknown>;
  };
  inPanel?: boolean;
  /** A new layout: ask which model it's for before anything else. */
  chooseMode?: boolean;
}>();
const emit = defineEmits<{ popOut: [] }>();

const HELP: [string, string][] = [
  ["Order", "bottom to top is the order the prompt lists boxes — background first"],
  ["Hidden boxes", "left out of the prompt"],
];
const SETTINGS_NOTE: Record<PromptModeId, string> = {
  ideogram: ", style",
  klein: "",
  moss: ", meta, canvas, palette",
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
const editor = ref<{
  update: (next: unknown) => void;
  offerBackdrop: (image: { ref: string; type: SourceType; size?: [number, number] }) => void;
  backdrop: AgentImage | null;
} | null>(null);
const agentOpen = ref(false);
const selected = ref<string[]>([]);
const settingsOpen = ref(false);
const choosing = ref(!!props.chooseMode);
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
function pickMode(id: PromptModeId) {
  choosing.value = false;
  patchScene({ mode: id });
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

// --- the agent, and the photo a run sends back -------------------------------------------------

function applyAgent(result: AgentLayout, how: "replace" | "add") {
  const kept = how === "add" ? scene.value.boxes : [];
  const boxes = [...kept];
  for (const b of result.boxes) {
    boxes.push(
      readBox(
        { ...b, id: boxId(), name: b.name || `Box ${boxes.length + 1}`, color: nextColor(boxes) },
        boxes.length,
      ),
    );
  }
  patchScene({
    boxes,
    ...(how === "replace" || !scene.value.prompt.trim() ? { prompt: result.prompt } : {}),
    ...(result.background && (how === "replace" || !scene.value.background.trim())
      ? { background: result.background }
      : {}),
  });
  selected.value = [];
}

interface ExecutedOutput {
  layout_node?: string[];
  layout_source?: { filename: string; subfolder?: string; type?: string }[];
  layout_size?: number[];
}
function onExecuted(e: Event) {
  const output = (e as CustomEvent<{ output?: ExecutedOutput }>).detail?.output;
  const mine = String(props.node?.id ?? "");
  if (!mine || !output?.layout_node?.some((id) => id === mine || id.endsWith(`:${mine}`))) return;
  const image = output.layout_source?.[0];
  if (!image) return;
  const size = output.layout_size;
  editor.value?.offerBackdrop({
    ref: image.subfolder ? `${image.subfolder}/${image.filename}` : image.filename,
    type: (image.type ?? "temp") as SourceType,
    size: size?.length === 2 ? [Number(size[0]), Number(size[1])] : undefined,
  });
}
const events = api as unknown as {
  addEventListener(t: string, f: (e: Event) => void): void;
  removeEventListener(t: string, f: (e: Event) => void): void;
};
onMounted(() => events.addEventListener("executed", onExecuted));
onBeforeUnmount(() => events.removeEventListener("executed", onExecuted));

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
.pl-host {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 0;
}
.pl-choose {
  position: absolute;
  inset: 0;
  z-index: 5;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
  border-radius: var(--zen-radius, 7px);
  background: color-mix(in srgb, var(--zen-bg, #141418) 88%, transparent);
  backdrop-filter: blur(2px);
}
.pl-choose-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-width: 720px;
  color: var(--zen-text, #e5e5ea);
}
.pl-choose-title {
  font-size: 16px;
  font-weight: 600;
}
.pl-choose-note {
  font-size: 12px;
  color: var(--zen-muted, #9aa0aa);
}
.pl-form {
  display: flex;
  flex-direction: column;
  gap: 10px;
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
