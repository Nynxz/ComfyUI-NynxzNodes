<!-- Compact Fusion Encode controls for a Mixin's references: multi-ref Blend + Softness
     (when 2+ active refs) and Style Release mode + strength (when 1+ active refs), each with
     a rich info hover explaining what the modes do. -->
<template>
  <div v-if="activeCount >= 1" class="mf">
    <div v-if="activeCount > 1" class="mf-row">
      <span class="mf-lbl-wrap">
        <span class="mf-lbl">Blend</span>
        <InfoHover label="How Blend and Softness work">
          <p class="mf-tip-title">Multi-reference Blend (Fusion Encode)</p>
          <p class="mf-tip-p">
            Combines this mixin's reference images into one visual conditioning grid:
          </p>
          <ul class="mf-tip-list">
            <li :class="{ on: blend === 'mix' }">
              <strong>Mix</strong> — interleaves all references evenly across the grid.
            </li>
            <li :class="{ on: blend === 'weighted' }">
              <strong>Weighted</strong> — allocates grid area proportional to each ref's strength.
            </li>
            <li :class="{ on: blend === 'tiled' }">
              <strong>Tiled</strong> — gives each ref its own grid cell (keeps subjects separate).
            </li>
            <li :class="{ on: blend === 'salient' }">
              <strong>Salient</strong> — takes each area from whichever ref has the strongest detail
              there.
            </li>
          </ul>
          <p class="mf-tip-p">
            <strong>Softness</strong> (0–1): 0 keeps crisp cells per ref; 1 feathers smoothly
            between them. Transparent PNG cutouts automatically drop their background out of the
            blend.
          </p>
        </InfoHover>
      </span>
      <ZenSelect
        class="mf-sel"
        size="sm"
        :model-value="blend"
        :options="BLENDS"
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
      <span class="mf-lbl-wrap">
        <span class="mf-lbl">Style</span>
        <InfoHover label="How Style Release works">
          <p class="mf-tip-title">Style Release (Experimental)</p>
          <p class="mf-tip-p">
            Reference images carry both <em>structure</em> (face, pose, layout) and
            <em>art style</em> (anime, photo, painting). Style release loosens the reference's style
            grip so your prompt or LoRA can restyle it:
          </p>
          <ul class="mf-tip-list">
            <li :class="{ on: styleMode === 'none' }">
              <strong>Keep style</strong> — default; preserves the reference's original look.
            </li>
            <li :class="{ on: styleMode === 'gist' }">
              <strong>Release gist</strong> — gently fades the overall look while keeping identity
              and structure.
            </li>
            <li :class="{ on: styleMode === 'whiten' }">
              <strong>Release stats</strong> — flattens per-channel style statistics (stronger, e.g.
              anime ref → photoreal LoRA).
            </li>
          </ul>
          <p class="mf-tip-p">
            <strong>Strength</strong> (0–1): start around 0.30–0.50; very high values can drift off
            distribution.
          </p>
        </InfoHover>
      </span>
      <ZenSelect
        class="mf-sel"
        size="sm"
        :model-value="styleMode"
        :options="STYLE_MODES"
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

import InfoHover from "@/components/layout/InfoHover.vue";
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
.mf-lbl-wrap {
  flex: none;
  display: inline-flex;
  align-items: center;
  gap: 3px;
  width: 48px;
}
.mf-lbl {
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
.mf-tip-title {
  margin: 0 0 4px;
  font-size: 11.5px;
  font-weight: 600;
  color: var(--zen-text, #e5e5ea);
}
.mf-tip-p {
  margin: 0 0 6px;
  color: var(--zen-muted, #9aa0aa);
}
.mf-tip-p:last-child {
  margin-bottom: 0;
}
.mf-tip-p strong {
  color: var(--zen-text, #e5e5ea);
  font-weight: 600;
}
.mf-tip-list {
  margin: 0 0 6px;
  padding-left: 14px;
  display: flex;
  flex-direction: column;
  gap: 3px;
  color: var(--zen-muted, #9aa0aa);
}
.mf-tip-list strong {
  color: var(--zen-text, #e5e5ea);
  font-weight: 600;
}
.mf-tip-list li.on {
  color: var(--zen-text, #e5e5ea);
}
.mf-tip-list li.on strong {
  color: var(--zen-accent, #818cf8);
}
</style>
