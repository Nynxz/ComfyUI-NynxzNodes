<!-- Compact Fusion Encode controls for a Mixin's references: multi-ref Blend + Softness
     (when 2+ active refs) and Style Release mode + strength (when 1+ active refs). -->
<template>
  <div v-if="activeCount >= 1" class="mf">
    <div v-if="activeCount > 1" class="mf-row">
      <span class="mf-lbl" :title="BLEND_TIP">Blend</span>
      <ZenSelect
        class="mf-sel"
        size="sm"
        :model-value="blend"
        :options="BLENDS"
        :title="BLEND_TIP"
        @update:model-value="emit('update:blend', $event as Blend)"
      />
      <ZenNumber
        class="mf-num"
        size="sm"
        :model-value="softness"
        :min="0"
        :max="1"
        :step="0.05"
        :precision="2"
        title="Softness between references — 0 hard cells, 1 fully feathered"
        @update:model-value="emit('update:softness', $event)"
      />
    </div>
    <div class="mf-row">
      <span class="mf-lbl" :title="STYLE_TIP">Style</span>
      <ZenSelect
        class="mf-sel"
        size="sm"
        :model-value="styleMode"
        :options="STYLE_MODES"
        :title="STYLE_TIP"
        @update:model-value="onStyleMode($event as RefStyleMode)"
      />
      <ZenNumber
        v-if="styleMode !== 'none'"
        class="mf-num"
        size="sm"
        :model-value="styleStrength"
        :min="0"
        :max="1"
        :step="0.05"
        :precision="2"
        title="Style release strength (0.3–0.5 recommended; 0 = off)"
        @update:model-value="emit('update:styleStrength', $event)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ZenNumber, ZenSelect } from "@nynxz/zenkit-ui";

import { STYLE_MODES, type RefStyleMode } from "@/layout/promptScene";
import { BLENDS, type Blend } from "@/lib/frescoScene";

const props = defineProps<{
  activeCount: number;
  blend: Blend;
  softness: number;
  styleMode: RefStyleMode;
  styleStrength: number;
}>();

const emit = defineEmits<{
  "update:blend": [Blend];
  "update:softness": [number];
  "update:styleMode": [RefStyleMode];
  "update:styleStrength": [number];
}>();

const BLEND_TIP =
  "How Fusion Encode combines multiple references on this mixin (transparent PNG cutouts drop their background out of the blend)";
const STYLE_TIP =
  "Loosen the reference's grip on art style while keeping spatial structure (e.g. anime ref + photoreal prompt/LoRA)";

function onStyleMode(next: RefStyleMode) {
  emit("update:styleMode", next);
  // Picking a style-release mode from "none" with strength 0 would be a silent no-op, so
  // nudge strength to the recommended starting point (0.4).
  if (next !== "none" && props.styleStrength <= 0) {
    emit("update:styleStrength", 0.4);
  }
}
</script>

<style scoped>
.mf {
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding-top: 6px;
}
.mf-row {
  display: flex;
  align-items: center;
  gap: 6px;
}
.mf-lbl {
  flex: none;
  width: 36px;
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.mf-sel {
  flex: 1;
  min-width: 0;
}
.mf-num {
  flex: none;
  width: 62px;
}
</style>
