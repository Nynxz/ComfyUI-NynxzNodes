<!-- The layout agent: ask the model wired into Layout Inputs for a layout — from the trace image, from
     the boxes there are now, or from words alone — see what it drew, then apply it or ask again. -->
<template>
  <ZenModal
    :open="open"
    title="Layout agent"
    width="980px"
    height="min(720px, 86vh)"
    @update:open="emit('update:open', $event)"
  >
    <div class="la">
      <section class="la-ask">
        <div class="la-source">
          <div class="la-pic" :class="{ off: !useImage }">
            <img v-if="imageUrl" :src="imageUrl" alt="" />
            <span v-else class="la-none">
              <i class="mdi mdi-image-off-outline" />
              No trace image. Pick one with the image button on the rail, or wire a photo into
              Layout Inputs and run.
            </span>
          </div>
          <label class="la-toggle" :class="{ dim: !image }">
            <ZenSwitch
              :model-value="useImage && !!image"
              :disabled="!image"
              @update:model-value="useImage = $event"
            />
            Read the trace image
          </label>
          <label class="la-toggle" :class="{ dim: !scene.boxes.length }">
            <ZenSwitch
              :model-value="revise && scene.boxes.length > 0"
              :disabled="!scene.boxes.length"
              @update:model-value="revise = $event"
            />
            Start from the boxes there are now
          </label>
        </div>

        <ZenInput
          v-model="request"
          type="textarea"
          :rows="5"
          sm
          :placeholder="placeholder"
          @keydown.ctrl.enter="ask"
        />
        <div class="la-chips">
          <button v-for="c in CHIPS" :key="c" type="button" class="la-chip" @click="request = c">
            {{ c }}
          </button>
        </div>
        <div class="la-row">
          <label class="la-ctl" title="Higher is freer, lower sticks closer to the request">
            Creativity
            <ZenNumber
              size="sm"
              :model-value="temperature"
              :min="0.1"
              :max="1.5"
              :step="0.05"
              :precision="2"
              @update:model-value="temperature = $event"
            />
          </label>
          <span class="la-grow" />
          <ZenButton v-if="busy" sm icon="mdi mdi-stop" @click="stop">Stop</ZenButton>
          <ZenButton
            v-else
            sm
            variant="primary"
            icon="mdi mdi-creation-outline"
            title="Ctrl+Enter"
            @click="ask"
          >
            {{ result ? "Ask again" : "Ask" }}
          </ZenButton>
        </div>
        <p v-if="busy" class="la-note">
          <i class="mdi mdi-loading mdi-spin" /> Asking… the first time loads the model, which takes
          a while.
        </p>
        <p v-if="error" class="la-error">{{ error }}</p>
      </section>

      <section class="la-result">
        <template v-if="result">
          <div class="la-thumb">
            <LayoutThumb :preview="resultPreview" labels />
          </div>
          <p v-if="result.prompt" class="la-caption">{{ result.prompt }}</p>
          <ul class="la-boxes">
            <li v-for="(b, i) in result.boxes" :key="i">
              <span class="la-dot" :style="{ background: BOX_COLORS[i % BOX_COLORS.length] }" />
              <b>{{ b.name || `Box ${i + 1}` }}</b>
              <span class="la-kind">{{ b.kind }}</span>
              {{ b.kind === "text" && b.text ? `"${b.text}"` : b.desc }}
            </li>
          </ul>
          <div class="la-row">
            <span class="la-grow" />
            <ZenButton
              sm
              icon="mdi mdi-plus"
              title="Keep what's there and add these boxes"
              @click="apply('add')"
            >
              Add boxes
            </ZenButton>
            <ZenButton sm variant="primary" icon="mdi mdi-check" @click="apply('replace')">
              Use this layout
            </ZenButton>
          </div>
        </template>
        <div v-else class="la-empty">
          <i class="mdi mdi-creation-outline" />
          <span>What the agent lays out shows here, to use or ask again.</span>
        </div>
      </section>
    </div>
  </ZenModal>
</template>

<script setup lang="ts">
import { ZenButton, ZenInput, ZenModal, ZenNumber, ZenSwitch } from "@nynxz/zenkit-ui";
import { computed, ref, watch } from "vue";

import LayoutThumb from "@/components/layout/LayoutThumb.vue";
import { askAgent, type AgentImage, type AgentLayout } from "@/layout/agent";
import type { SavedLayoutPreview } from "@/layout/library";
import { BOX_COLORS, type PromptScene } from "@/layout/promptScene";
import { thumbUrl } from "@/lib/fusionApi";

const props = defineProps<{
  open: boolean;
  scene: PromptScene;
  nodeId: string;
  image: AgentImage | null;
}>();
const emit = defineEmits<{
  "update:open": [boolean];
  apply: [layout: AgentLayout, how: "replace" | "add"];
}>();

const CHIPS = [
  "Lay out what's in this picture",
  "Remix it as a bold movie poster with a title",
  "Same scene, as a magazine cover",
  "Keep the people, change the setting",
];

const request = ref("");
const useImage = ref(true);
const revise = ref(false);
const temperature = ref(0.7);
const busy = ref(false);
const error = ref("");
const result = ref<AgentLayout | null>(null);
let controller: AbortController | null = null;

const imageUrl = computed(() => (props.image ? thumbUrl(props.image.ref, props.image.type) : null));
const placeholder = computed(() =>
  props.image
    ? "What to make of it — empty lays out the picture as it is"
    : "What the picture should be",
);
const resultPreview = computed<SavedLayoutPreview | null>(() =>
  result.value
    ? {
        aspect: props.scene.width / props.scene.height,
        boxes: result.value.boxes.map((b, i) => ({
          x: b.x,
          y: b.y,
          w: b.w,
          h: b.h,
          name: b.name,
          color: BOX_COLORS[i % BOX_COLORS.length]!,
        })),
      }
    : null,
);
watch(
  () => props.open,
  (open) => {
    if (open) error.value = "";
  },
);

async function ask() {
  if (busy.value) return;
  busy.value = true;
  error.value = "";
  controller = new AbortController();
  try {
    result.value = await askAgent(
      {
        scene: JSON.parse(JSON.stringify(props.scene)) as PromptScene,
        nodeId: props.nodeId,
        request: request.value,
        image: useImage.value ? props.image : null,
        revise: revise.value && props.scene.boxes.length > 0,
        seed: Math.floor(Math.random() * 2 ** 31),
        temperature: temperature.value,
      },
      controller.signal,
    );
  } catch (e) {
    error.value = (e as Error).message;
  } finally {
    busy.value = false;
    controller = null;
  }
}
function stop() {
  controller?.abort();
}
function apply(how: "replace" | "add") {
  if (!result.value) return;
  emit("apply", result.value, how);
  emit("update:open", false);
}
</script>

<style scoped>
.la {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 16px;
  height: 100%;
  min-height: 0;
  font-size: 12px;
  color: var(--zen-text, #e5e5ea);
}
.la-ask,
.la-result {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 0;
  overflow: auto;
}
.la-result {
  padding-left: 16px;
  border-left: 1px solid var(--zen-border, #34343c);
}
.la-source {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.la-pic {
  height: 180px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--zen-radius, 7px);
  background: var(--zen-surface-2, #26262d);
  overflow: hidden;
}
.la-pic.off img {
  opacity: 0.3;
}
.la-pic img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}
.la-none {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 16px;
  text-align: center;
  color: var(--zen-muted, #9aa0aa);
}
.la-none i {
  font-size: 24px;
}
.la-toggle {
  display: flex;
  align-items: center;
  gap: 8px;
}
.la-toggle.dim {
  opacity: 0.5;
}
.la-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.la-chip {
  padding: 3px 8px;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: 999px;
  background: none;
  color: var(--zen-muted, #9aa0aa);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
}
.la-chip:hover {
  color: var(--zen-text, #e5e5ea);
  border-color: var(--zen-accent, #6366f1);
}
.la-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.la-ctl {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--zen-muted, #9aa0aa);
}
.la-grow {
  flex: 1;
}
.la-note {
  margin: 0;
  color: var(--zen-muted, #9aa0aa);
}
.la-error {
  margin: 0;
  color: var(--zen-danger, #f87171);
  white-space: pre-wrap;
}
.la-thumb {
  height: 260px;
  flex: none;
}
.la-caption {
  margin: 0;
  padding: 6px 8px;
  border-radius: var(--zen-radius, 7px);
  background: var(--zen-surface-2, #26262d);
  line-height: 1.45;
}
.la-boxes {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.la-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  margin-right: 6px;
  border-radius: 2px;
}
.la-kind {
  margin: 0 4px;
  font-size: 10px;
  color: var(--zen-muted, #9aa0aa);
}
.la-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  height: 100%;
  color: var(--zen-muted, #9aa0aa);
  text-align: center;
}
.la-empty i {
  font-size: 28px;
}
</style>
