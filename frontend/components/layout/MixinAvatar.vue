<template>
  <span
    class="ma"
    :style="{ width: `${size}px`, height: `${size}px`, '--ma-color': mixin.color }"
    :title="title"
  >
    <img v-if="look" class="ma-img" :src="thumbUrl(look.ref, look.type)" alt="" draggable="false" />
    <span v-else class="ma-letter" :style="{ fontSize: `${Math.round(size * 0.45)}px` }">
      {{ letter }}
    </span>
  </span>
</template>

<script setup lang="ts">
/** A mixin at a glance: its first reference image, or its initial on its colour. */
import { computed } from "vue";

import type { Mixin } from "@/layout/promptScene";
import { thumbUrl } from "@/lib/mediaRefs";

const {
  mixin,
  size = 28,
  title = "",
} = defineProps<{
  mixin: Mixin;
  size?: number;
  title?: string;
}>();

const look = computed(() => mixin.refs.find((r) => r.on));
const letter = computed(() => (mixin.name.trim()[0] ?? "?").toUpperCase());
</script>

<style scoped>
.ma {
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  border-radius: 50%;
  border: 2px solid var(--ma-color);
  background: color-mix(in srgb, var(--ma-color) 35%, var(--zen-bg, #141418));
}
.ma-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.ma-letter {
  font-weight: 700;
  line-height: 1;
  color: var(--zen-text, #e5e5ea);
}
</style>
