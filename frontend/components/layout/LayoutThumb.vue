<template>
  <svg class="lt" :viewBox="`0 0 ${w} ${h}`" preserveAspectRatio="xMidYMid meet">
    <rect class="lt-canvas" x="0" y="0" :width="w" :height="h" rx="3" />
    <g v-for="(b, i) in preview?.boxes ?? []" :key="i">
      <rect
        :x="b.x * w"
        :y="b.y * h"
        :width="b.w * w"
        :height="b.h * h"
        :fill="b.color"
        fill-opacity="0.28"
        :stroke="b.color"
        stroke-width="1.5"
      />
      <text v-if="labels && b.name" :x="b.x * w + 4" :y="b.y * h + 12" class="lt-label">
        {{ b.name }}
      </text>
    </g>
  </svg>
</template>

<script setup lang="ts">
/** A saved layout's boxes drawn at its aspect, scaled to fit whatever holds it. */
import { computed } from "vue";

import type { SavedLayoutPreview } from "@/layout/library";

const props = defineProps<{ preview: SavedLayoutPreview | null; labels?: boolean }>();
const SIDE = 240;
const aspect = computed(() => props.preview?.aspect || 1);
const w = computed(() => (aspect.value >= 1 ? SIDE : Math.round(SIDE * aspect.value)));
const h = computed(() => (aspect.value >= 1 ? Math.round(SIDE / aspect.value) : SIDE));
</script>

<style scoped>
.lt {
  display: block;
  width: 100%;
  height: 100%;
}
.lt-canvas {
  fill: var(--zen-surface-2, #26262d);
  stroke: var(--zen-border, #34343c);
}
.lt-label {
  font-size: 10px;
  fill: var(--zen-text, #e5e5ea);
  paint-order: stroke;
  stroke: rgb(0 0 0 / 0.6);
  stroke-width: 2px;
}
</style>
