<template>
  <ZenModal
    :open="open"
    title="Layout settings"
    width="760px"
    height="auto"
    @update:open="emit('update:open', $event)"
  >
    <div class="ls">
      <FrescoSection label="Mode" note="the model the prompt is written for — boxes stay">
        <ModeCards
          :model-value="scene.mode"
          @update:model-value="emit('patch', { mode: $event })"
        />
      </FrescoSection>

      <FrescoSection label="Canvas" :note="`${scene.width}×${scene.height}`">
        <ZenResolution
          :width="scene.width"
          :height="scene.height"
          :snap="16"
          :presets="[0.5, 1, 1.5, 2]"
          :max-side="4096"
          :max-mp="4"
          @update:width="emit('patch', { width: $event })"
          @update:height="emit('patch', { height: $event })"
        />
      </FrescoSection>

      <FrescoSection v-if="scene.mode === 'ideogram'" label="Style" note="style_description">
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
          <div class="ls-grid">
            <ZenField v-for="f in styleFields" :key="f.key" :label="f.label" stack>
              <ZenInput
                sm
                :model-value="scene.style[f.key]"
                :placeholder="f.hint"
                @update:model-value="patchStyle({ [f.key]: String($event ?? '') })"
              />
            </ZenField>
          </div>
          <ZenField label="Palette" stack hint="Up to 16. Right-click a colour to drop it.">
            <PaletteEdit
              :model-value="scene.style.palette"
              :max="16"
              @update:model-value="patchStyle({ palette: $event })"
            />
          </ZenField>
        </template>
        <p v-else class="ls-note">Off: the caption has no style_description.</p>
      </FrescoSection>

      <FrescoSection v-if="scene.mode === 'moss'" label="Summary" note="beside the caption">
        <div class="ls-grid">
          <ZenField label="Meta" stack hint="What kind of image, comma separated">
            <ZenInput
              sm
              :model-value="scene.meta.join(', ')"
              placeholder="poster, illustration"
              @update:model-value="emit('patch', { meta: splitList($event) })"
            />
          </ZenField>
          <ZenField label="Canvas" stack hint="One picture, panels, or a collage">
            <ZenSelect
              size="sm"
              :model-value="scene.canvas"
              :options="CANVAS_OPTIONS"
              @update:model-value="emit('patch', { canvas: $event as MossCanvas })"
            />
          </ZenField>
        </div>
        <ZenField label="Palette" stack hint="Up to 16. Right-click a colour to drop it.">
          <PaletteEdit
            :model-value="scene.palette"
            :max="16"
            @update:model-value="emit('patch', { palette: $event })"
          />
        </ZenField>
      </FrescoSection>

      <FrescoSection v-if="scene.mode === 'klein'" label="Klein">
        <p class="ls-note">
          Klein reads only the caption and each box's description — colours go in the words. Cite
          every box in the caption as &lt;obj_N&gt;; the sidebar shows each box's id.
        </p>
      </FrescoSection>
    </div>
    <template #footer>
      <div class="ls-foot">
        <ZenButton sm variant="primary" @click="emit('update:open', false)">Done</ZenButton>
      </div>
    </template>
  </ZenModal>
</template>

<script setup lang="ts">
/** Settings for the whole layout: its mode, the canvas, and what the mode reads once per
 *  layout. Every change applies as it's made (one undo step each). */
import {
  ZenButton,
  ZenField,
  ZenInput,
  ZenModal,
  ZenResolution,
  ZenSelect,
  ZenSwitch,
  ZenToggleGroup,
} from "@nynxz/zenkit-ui";
import { computed } from "vue";

import FrescoSection from "@/components/fresco/FrescoSection.vue";
import ModeCards from "@/components/layout/ModeCards.vue";
import PaletteEdit from "@/components/layout/PaletteEdit.vue";
import type { IdeogramStyle, MossCanvas, PromptScene } from "@/layout/promptScene";

const props = defineProps<{ open: boolean; scene: PromptScene }>();
const emit = defineEmits<{ "update:open": [boolean]; patch: [Partial<PromptScene>] }>();

const CANVAS_OPTIONS = [
  { value: "", label: "Not said" },
  { value: "single", label: "Single image" },
  { value: "multi_panel", label: "Multi-panel" },
  { value: "collage", label: "Collage" },
];
type StyleText = "aesthetics" | "lighting" | "medium" | "photo" | "art_style";
const styleFields = computed<{ key: StyleText; label: string; hint: string }[]>(() => [
  { key: "aesthetics", label: "Aesthetics", hint: "moody, minimal, cinematic" },
  { key: "lighting", label: "Lighting", hint: "soft window light, neon" },
  { key: "medium", label: "Medium", hint: "photograph, oil painting" },
  props.scene.style.kind === "photo"
    ? { key: "photo", label: "Photo", hint: "35mm, shallow depth of field" }
    : { key: "art_style", label: "Art style", hint: "flat vector, ukiyo-e" },
]);

function patchStyle(patch: Partial<IdeogramStyle>) {
  emit("patch", { style: { ...props.scene.style, ...patch } });
}
const splitList = (v: unknown) =>
  String(v ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
</script>

<style scoped>
.ls {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 4px 2px;
  font-size: 12px;
  color: var(--zen-text, #e5e5ea);
}
.ls-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px 12px;
}
.ls-note {
  margin: 0;
  font-size: 11px;
  line-height: 1.45;
  color: var(--zen-muted, #9aa0aa);
}
.ls-foot {
  display: flex;
  justify-content: flex-end;
  width: 100%;
}
</style>
