<template>
  <div class="fbf" :class="{ bare: !main }">
    <div v-if="main" class="fbf-pic" :style="maskStyle">
      <img class="fbf-img" :src="thumbUrl(main.ref, main.type)" alt="" draggable="false" />
    </div>
    <div v-if="more.length" class="fbf-more">
      <img
        v-for="r in more.slice(0, 3)"
        :key="r.id"
        :src="thumbUrl(r.ref, r.type)"
        alt=""
        draggable="false"
      />
      <span v-if="more.length > 3" class="fbf-plus">+{{ more.length - 3 }}</span>
    </div>
    <span v-if="loras" class="fbf-lora" :title="`${loras} LoRA${loras > 1 ? 's' : ''}`">
      <i class="mdi mdi-puzzle-outline" />{{ loras }}
    </span>
    <p v-if="layer.prompt.trim()" class="fbf-text">{{ layer.prompt.trim() }}</p>
  </div>
</template>

<script setup lang="ts">
/**
 * What a layer carries, drawn inside its box on the canvas: its first reference filling the box,
 * the rest as small thumbs, a LoRA count, and its prompt along the bottom. Detail drops away as
 * the box gets small (container queries on the canvas's box slot).
 */
import { computed } from "vue";

import { thumbUrl } from "@/lib/fusionApi";
import type { FrescoLayer } from "@/lib/frescoScene";

const props = defineProps<{
  layer: FrescoLayer;
  /** The box's painted shape, from the canvas — clips the picture, not the caption. */
  maskStyle?: Record<string, string>;
}>();

const refs = computed(() => props.layer.refs.filter((r) => r.on));
const main = computed(() => refs.value[0]);
const more = computed(() => refs.value.slice(1));
const loras = computed(() => props.layer.loras.filter((r) => r.on && r.name).length);
</script>

<style scoped>
.fbf {
  position: absolute;
  inset: 0;
}
.fbf-pic {
  position: absolute;
  inset: 0;
  -webkit-mask-size: 100% 100%;
  mask-size: 100% 100%;
}
.fbf-img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0.8;
}
.fbf-more {
  position: absolute;
  right: 4px;
  top: 4px;
  display: flex;
  gap: 3px;
}
.fbf-more img,
.fbf-plus {
  width: 22px;
  height: 22px;
  border-radius: 4px;
  object-fit: cover;
  box-shadow: 0 0 0 1px rgb(0 0 0 / 50%);
}
.fbf-plus {
  display: grid;
  place-items: center;
  background: rgb(0 0 0 / 65%);
  color: #fff;
  font-size: 10px;
}
.fbf-lora {
  position: absolute;
  left: 4px;
  top: 20px;
  display: inline-flex;
  align-items: center;
  gap: 2px;
  padding: 0 5px;
  border-radius: 999px;
  background: rgb(0 0 0 / 60%);
  color: #fff;
  font-size: 10px;
  line-height: 16px;
}
.fbf-text {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  margin: 0;
  padding: 14px 6px 4px;
  background: linear-gradient(to top, rgb(0 0 0 / 75%), transparent);
  color: #fff;
  font-size: 11px;
  line-height: 1.3;
  text-shadow: 0 1px 2px rgb(0 0 0 / 60%);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.fbf.bare .fbf-text {
  background: none;
  color: color-mix(in srgb, #fff 85%, transparent);
}

@container (max-height: 70px) {
  .fbf-text,
  .fbf-lora {
    display: none;
  }
}
@container (max-width: 90px) {
  .fbf-more {
    display: none;
  }
}
@container (min-height: 160px) {
  .fbf-text {
    -webkit-line-clamp: 3;
  }
}
</style>
