<template>
  <div class="lbf" :class="{ bare: !main }">
    <div v-if="main" class="lbf-pic" :style="maskStyle">
      <img class="lbf-img" :src="thumbUrl(main.ref, main.type)" alt="" draggable="false" />
    </div>
    <div v-if="mixins.length || more.length" class="lbf-top">
      <MixinAvatar
        v-for="m in mixins.slice(0, 3)"
        :key="m.id"
        :mixin="m"
        :size="18"
        :title="`Mixin: ${m.name}`"
      />
      <div v-if="more.length" class="lbf-more" :class="{ big: !main && !mixins.length }">
        <img
          v-for="r in more.slice(0, 3)"
          :key="r.id"
          :src="thumbUrl(r.ref, r.type)"
          alt=""
          draggable="false"
        />
        <span v-if="more.length > 3" class="lbf-plus">+{{ more.length - 3 }}</span>
      </div>
    </div>
    <span v-if="loras.length" class="lbf-lora" :title="loraTitle">
      <i class="mdi mdi-puzzle-outline" />{{ loras.length }}
    </span>
    <p v-if="caption" class="lbf-text">{{ caption }}</p>
  </div>
</template>

<script setup lang="ts">
/**
 * What a layout box carries, drawn inside its box on the canvas: its first active reference image
 * (from its mixins or directly on the box), extra ref thumbnails, mixin avatars, active LoRA count
 * badge (mixin + direct LoRAs), and its description/text along the bottom. Detail scales down with
 * container queries when boxes are small.
 */
import { computed } from "vue";

import MixinAvatar from "@/components/layout/MixinAvatar.vue";
import {
  boxActiveLoras,
  boxActiveRefs,
  type Mixin,
  type PromptBox,
  type PromptScene,
} from "@/layout/promptScene";
import { thumbUrl } from "@/lib/mediaRefs";

const props = defineProps<{
  scene: PromptScene;
  box: PromptBox;
  /** The box's painted shape, from the canvas — clips the picture, not the caption. */
  maskStyle?: Record<string, string>;
}>();

const mixins = computed<Mixin[]>(() => {
  const byId = new Map(props.scene.mixins.map((m) => [m.id, m]));
  return props.box.mixins.map((id) => byId.get(id)).filter((m): m is Mixin => !!m);
});
const refs = computed(() => boxActiveRefs(props.scene, props.box));
/** A box segmented/painted around part of a photo keeps the photo visible underneath, showing refs
 *  as thumbnails instead of filling the painted area. */
const outlined = computed(() => !!props.box.mask && props.box.shapeFor !== "refs");
const main = computed(() => (outlined.value ? undefined : refs.value[0]));
const more = computed(() => (outlined.value ? refs.value : refs.value.slice(1)));
const loras = computed(() => boxActiveLoras(props.scene, props.box));
const loraTitle = computed(() =>
  loras.value.map((l) => `${l.name} (${l.strength.toFixed(2)})`).join("\n"),
);
const caption = computed(() => {
  if (props.box.kind === "text" && props.box.text.trim()) return `"${props.box.text.trim()}"`;
  return props.box.desc.trim();
});
</script>

<style scoped>
.lbf {
  position: absolute;
  inset: 0;
  pointer-events: none;
}
.lbf-pic {
  position: absolute;
  inset: 0;
  -webkit-mask-size: 100% 100%;
  mask-size: 100% 100%;
}
.lbf-img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0.78;
}
.lbf-top {
  position: absolute;
  right: 4px;
  top: 4px;
  display: flex;
  align-items: center;
  gap: 3px;
}
.lbf-more {
  display: flex;
  gap: 3px;
}
.lbf-more img,
.lbf-plus {
  width: 22px;
  height: 22px;
  border-radius: 4px;
  object-fit: cover;
  box-shadow: 0 0 0 1px rgb(0 0 0 / 50%);
}
.lbf-more.big img,
.lbf-more.big .lbf-plus {
  width: 32px;
  height: 32px;
}
.lbf-plus {
  display: grid;
  place-items: center;
  background: rgb(0 0 0 / 65%);
  color: #fff;
  font-size: 10px;
}
.lbf-lora {
  position: absolute;
  left: 4px;
  top: 20px;
  display: inline-flex;
  align-items: center;
  gap: 2px;
  padding: 0 5px;
  border-radius: 999px;
  background: rgb(0 0 0 / 65%);
  color: #fff;
  font-size: 10px;
  line-height: 16px;
}
.lbf-text {
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
.lbf.bare .lbf-text {
  background: none;
  color: color-mix(in srgb, #fff 85%, transparent);
}

@container (max-height: 70px) {
  .lbf-text,
  .lbf-lora {
    display: none;
  }
}
@container (max-width: 90px) {
  .lbf-more {
    display: none;
  }
}
@container (min-height: 160px) {
  .lbf-text {
    -webkit-line-clamp: 3;
  }
}
</style>
