<!-- Interactive SAM3 cutout modal for any reference image (on a Mixin, Mixin Library preset, Box,
     or Background). Click points directly on the reference photo or type a signed SAM3 prompt
     (`woman`, `jacket, -hands`), preview the clean cutout, then replace or add it alongside. -->
<template>
  <ZenModal
    :open="open"
    title="Cut out reference with SAM3"
    width="920px"
    height="min(680px, 86vh)"
    @update:open="setOpen"
  >
    <div class="rc">
      <section class="rc-left">
        <div class="rc-stage">
          <div
            v-if="sourceUrl"
            class="rc-pic-wrap"
            title="Click to include · Alt+click to exclude · click a dot to remove"
            @click="onImageClick"
          >
            <img :src="sourceUrl" alt="" draggable="false" />
            <span
              v-for="(pt, i) in points"
              :key="i"
              class="rc-pt"
              :class="{ out: !pt.include }"
              :style="{ left: `${pt.x * 100}%`, top: `${pt.y * 100}%` }"
              title="Click to remove · Alt+click to flip include/exclude"
              @click.stop="onPointClick($event, i)"
            />
          </div>
          <span v-else class="rc-empty-txt">No image selected</span>
        </div>

        <div class="rc-pt-bar">
          <span class="rc-hint">
            Click image to pick subject (Alt+click to exclude) — or describe below
          </span>
          <ZenButton
            v-if="points.length"
            sm
            variant="ghost"
            icon="mdi mdi-close"
            @click="points = []"
          >
            Clear ({{ points.length }})
          </ZenButton>
        </div>

        <ZenInput
          v-model="prompt"
          sm
          placeholder="What to keep, e.g. person, red jacket, -hands"
          @keydown.enter="run"
        />

        <div class="rc-chips">
          <button v-for="c in CHIPS" :key="c" type="button" class="rc-chip" @click="prompt = c">
            {{ c }}
          </button>
        </div>

        <div class="rc-opts">
          <label class="rc-field rc-wide">
            <span>SAM3 checkpoint</span>
            <ZenSelect
              v-if="checkpoints.length"
              size="sm"
              :model-value="sam3"
              :options="checkpointOptions"
              @update:model-value="pickCheckpoint(String($event))"
            />
            <span v-else class="rc-warn-inline">
              No SAM3 checkpoint found in ComfyUI checkpoints/
            </span>
          </label>
          <label class="rc-field">
            <span>Background</span>
            <ZenSelect
              size="sm"
              :model-value="background"
              :options="BG_OPTIONS"
              @update:model-value="background = $event as CutoutBackground"
            />
          </label>
          <label class="rc-field">
            <span>Grow</span>
            <ZenNumber
              size="sm"
              :model-value="grow"
              :min="-64"
              :max="64"
              :step="1"
              :precision="0"
              @update:model-value="grow = Math.round($event)"
            />
          </label>
          <label class="rc-field">
            <span>Feather</span>
            <ZenNumber
              size="sm"
              :model-value="feather"
              :min="0"
              :max="32"
              :step="1"
              :precision="0"
              @update:model-value="feather = Math.round($event)"
            />
          </label>
        </div>

        <div class="rc-row">
          <label class="rc-toggle">
            <ZenSwitch v-model="crop" />
            Crop tight to subject
          </label>
          <span class="rc-grow" />
          <ZenButton v-if="busy" sm icon="mdi mdi-stop" @click="stop">Stop</ZenButton>
          <ZenButton
            v-else
            sm
            variant="primary"
            icon="mdi mdi-auto-fix"
            :disabled="!canRun"
            @click="run"
          >
            {{ result ? "Cut out again" : "Cut out" }}
          </ZenButton>
        </div>

        <p v-if="busy" class="rc-status">
          <i class="mdi mdi-loading mdi-spin" /> Running SAM3… the first run loads the model.
        </p>
        <p v-if="error" class="rc-error"><i class="mdi mdi-alert-circle-outline" /> {{ error }}</p>
      </section>

      <section class="rc-right">
        <div class="rc-preview" :class="{ checker: background === 'transparent' }">
          <template v-if="resultUrl">
            <img :src="resultUrl" alt="" />
            <span class="rc-dims">{{ result!.width }}×{{ result!.height }}</span>
          </template>
          <div v-else class="rc-empty">
            <i class="mdi mdi-content-cut" />
            <span>
              Your isolated cutout appears here. Use White for a single reference, or Transparent
              PNG when fusing multiple references together.
            </span>
          </div>
        </div>

        <div class="rc-row">
          <span class="rc-grow" />
          <ZenButton
            sm
            icon="mdi mdi-plus"
            :disabled="!result"
            title="Keep the original reference and add this cutout beside it"
            @click="apply('add')"
          >
            Keep both
          </ZenButton>
          <ZenButton
            sm
            variant="primary"
            icon="mdi mdi-check"
            :disabled="!result"
            title="Swap this reference image to the cutout"
            @click="apply('replace')"
          >
            Replace reference
          </ZenButton>
        </div>
      </section>
    </div>
  </ZenModal>
</template>

<script setup lang="ts">
import { ZenButton, ZenInput, ZenModal, ZenNumber, ZenSelect, ZenSwitch } from "@nynxz/zenkit-ui";
import { computed, onBeforeUnmount, ref, watch } from "vue";

import {
  listSam3Checkpoints,
  rememberSam3Checkpoint,
  runCutout,
  savedSam3Checkpoint,
  type CutoutBackground,
  type CutoutResult,
} from "@/layout/cutout";
import type { FrescoPoint, FrescoRef } from "@/lib/frescoScene";
import { thumbUrl } from "@/lib/mediaRefs";

const props = defineProps<{
  open: boolean;
  source: FrescoRef | null;
}>();

const emit = defineEmits<{
  "update:open": [boolean];
  apply: [result: CutoutResult, how: "replace" | "add"];
}>();

const CHIPS = ["person", "face", "outfit", "hair", "object"];
const BG_OPTIONS = [
  { value: "white", label: "White (best for single ref)" },
  { value: "grey", label: "Neutral grey" },
  { value: "black", label: "Black" },
  { value: "transparent", label: "Transparent PNG (best for multi-ref fusion)" },
];

const checkpoints = ref<string[]>([]);
const sam3 = ref("");
const prompt = ref("person");
const points = ref<FrescoPoint[]>([]);
const background = ref<CutoutBackground>("white");
const crop = ref(true);
const grow = ref(0);
const feather = ref(2);
const busy = ref(false);
const error = ref("");
const result = ref<CutoutResult | null>(null);
let controller: AbortController | null = null;

const checkpointOptions = computed(() => checkpoints.value.map((c) => ({ value: c, label: c })));
const sourceUrl = computed(() =>
  props.source ? thumbUrl(props.source.ref, props.source.type) : null,
);
const resultUrl = computed(() =>
  result.value ? thumbUrl(result.value.ref, result.value.type) : null,
);
const canRun = computed(
  () =>
    !!props.source && !!sam3.value && (points.value.length > 0 || prompt.value.trim().length > 0),
);

function setOpen(v: boolean) {
  if (!v) stop();
  emit("update:open", v);
}

function pickCheckpoint(name: string) {
  sam3.value = name;
  rememberSam3Checkpoint(name);
}

watch(
  () => props.open,
  async (open) => {
    if (!open) return;
    error.value = "";
    result.value = null;
    points.value = [];
    checkpoints.value = await listSam3Checkpoints();
    const remembered = savedSam3Checkpoint();
    if (remembered && checkpoints.value.includes(remembered)) {
      sam3.value = remembered;
    } else if (!checkpoints.value.includes(sam3.value)) {
      sam3.value = checkpoints.value[0] ?? "";
    }
  },
);

function onImageClick(e: MouseEvent) {
  const wrap = e.currentTarget as HTMLElement | null;
  const rect = wrap?.getBoundingClientRect();
  if (!rect?.width || !rect?.height) return;
  const clamp = (v: number) => Math.min(1, Math.max(0, v));
  const pt: FrescoPoint = {
    x: clamp((e.clientX - rect.left) / rect.width),
    y: clamp((e.clientY - rect.top) / rect.height),
    include: !e.altKey,
  };
  points.value = [...points.value, pt];
}

function onPointClick(e: MouseEvent, index: number) {
  if (e.altKey) {
    points.value = points.value.map((p, i) => (i === index ? { ...p, include: !p.include } : p));
  } else {
    points.value = points.value.filter((_, i) => i !== index);
  }
}

async function run() {
  if (busy.value || !canRun.value || !props.source) return;
  busy.value = true;
  error.value = "";
  controller = new AbortController();
  try {
    result.value = await runCutout(
      {
        imageRef: props.source.ref,
        imageType: props.source.type,
        sam3: sam3.value,
        prompt: prompt.value.trim(),
        points: points.value,
        background: background.value,
        crop: crop.value,
        grow: grow.value,
        feather: feather.value,
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
  controller = null;
}

function apply(how: "replace" | "add") {
  if (!result.value) return;
  emit("apply", result.value, how);
  emit("update:open", false);
}

onBeforeUnmount(() => stop());
</script>

<style scoped>
.rc {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
  gap: 16px;
  height: 100%;
  min-height: 0;
  font-size: 12px;
  color: var(--zen-text, #e5e5ea);
}
.rc-left,
.rc-right {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 0;
}
.rc-right {
  padding-left: 16px;
  border-left: 1px solid var(--zen-border, #34343c);
}
.rc-stage {
  flex: 1;
  min-height: 180px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 6px;
  border-radius: var(--zen-radius, 7px);
  background: var(--zen-surface-2, #26262d);
  overflow: hidden;
}
.rc-pic-wrap {
  position: relative;
  display: inline-block;
  max-width: 100%;
  max-height: 100%;
  cursor: crosshair;
  user-select: none;
}
.rc-pic-wrap img {
  display: block;
  max-width: 100%;
  max-height: 290px;
  object-fit: contain;
  border-radius: min(var(--zen-radius, 7px), 4px);
}
.rc-pt {
  position: absolute;
  width: 11px;
  height: 11px;
  margin: -5.5px 0 0 -5.5px;
  border: 2px solid #fff;
  border-radius: 50%;
  background: #22c55e;
  box-shadow: 0 0 0 1px rgb(0 0 0 / 55%);
  cursor: pointer;
}
.rc-pt.out {
  background: #ef4444;
}
.rc-pt-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
}
.rc-hint {
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.rc-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.rc-chip {
  padding: 2px 8px;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 7px);
  background: none;
  color: var(--zen-muted, #9aa0aa);
  font: inherit;
  font-size: 11px;
  cursor: pointer;
}
.rc-chip:hover {
  color: var(--zen-text, #e5e5ea);
  border-color: var(--zen-accent, #6366f1);
}
.rc-opts {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 76px 76px;
  gap: 6px 8px;
}
.rc-field {
  display: flex;
  flex-direction: column;
  gap: 3px;
  font-size: 10.5px;
  color: var(--zen-muted, #9aa0aa);
}
.rc-field.rc-wide {
  grid-column: 1 / -1;
}
.rc-warn-inline {
  padding: 4px 8px;
  border-radius: var(--zen-radius, 6px);
  background: color-mix(in srgb, var(--zen-warning, #f5b041) 14%, transparent);
  color: var(--zen-warning, #f5b041);
  font-size: 11px;
}
.rc-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11.5px;
}
.rc-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.rc-grow {
  flex: 1;
}
.rc-status {
  margin: 0;
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.rc-error {
  display: flex;
  align-items: flex-start;
  gap: 5px;
  margin: 0;
  padding: 5px 8px;
  border-radius: var(--zen-radius, 6px);
  background: color-mix(in srgb, var(--zen-danger, #f87171) 14%, transparent);
  color: var(--zen-danger, #f87171);
  font-size: 11px;
  line-height: 1.35;
}
.rc-preview {
  position: relative;
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 10px;
  border-radius: var(--zen-radius, 7px);
  background: var(--zen-surface-2, #26262d);
  overflow: hidden;
}
.rc-preview.checker {
  background-image:
    linear-gradient(45deg, #2a2a32 25%, transparent 25%),
    linear-gradient(-45deg, #2a2a32 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #2a2a32 75%),
    linear-gradient(-45deg, transparent 75%, #2a2a32 75%);
  background-size: 16px 16px;
  background-position:
    0 0,
    0 8px,
    8px -8px,
    -8px 0;
}
.rc-preview img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  border-radius: min(var(--zen-radius, 7px), 4px);
}
.rc-dims {
  position: absolute;
  right: 8px;
  bottom: 8px;
  padding: 2px 7px;
  border-radius: min(var(--zen-radius, 7px), 4px);
  background: rgb(0 0 0 / 68%);
  color: #fff;
  font-size: 10px;
  font-variant-numeric: tabular-nums;
}
.rc-empty,
.rc-empty-txt {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 20px;
  text-align: center;
  color: var(--zen-muted, #9aa0aa);
}
.rc-empty i {
  font-size: 28px;
}
</style>
