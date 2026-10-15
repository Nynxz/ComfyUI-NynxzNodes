<template>
  <div class="fsum">
    <div class="fsum-stage">
      <ZenBoxCanvas
        :boxes="boxes"
        :selected="[]"
        :aspect="scene.width / scene.height"
        :draw="false"
      >
        <template #box="{ box, maskStyle }">
          <FrescoBoxFace
            v-if="byId.get(box.id)"
            :layer="byId.get(box.id)!"
            :mask-style="maskStyle"
          />
        </template>
        <template #empty>
          <span>No layers yet</span>
        </template>
      </ZenBoxCanvas>
    </div>
    <div class="fsum-bar">
      <span class="fsum-text">
        {{ scene.layers.length }} layer{{ scene.layers.length === 1 ? "" : "s" }} ·
        {{ scene.width }}×{{ scene.height }} · editing in a panel
      </span>
      <ZenButton sm icon="mdi mdi-open-in-new" @click="emit('focus')">Show editor</ZenButton>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * What the node shows while its editor is in a ZenKit panel: the composition, live and read-only,
 * and a way back to the panel.
 */
import { computed } from "vue";
import { ZenBoxCanvas, ZenButton, type ZenBox } from "@nynxz/zenkit-ui";

import FrescoBoxFace from "@/components/fresco/FrescoBoxFace.vue";
import type { FrescoScene } from "@/lib/frescoScene";

const props = defineProps<{ scene: FrescoScene }>();
const emit = defineEmits<{ focus: [] }>();

const byId = computed(() => new Map(props.scene.layers.map((l) => [l.id, l])));
const boxes = computed<ZenBox[]>(() =>
  props.scene.layers.map((l) => ({
    id: l.id,
    x: l.x,
    y: l.y,
    w: l.w,
    h: l.h,
    color: l.color,
    label: l.name,
    hidden: !l.on,
    locked: true,
    mask: l.mask,
  })),
);
</script>

<style scoped>
.fsum {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
  height: 100%;
  min-height: 0;
}
.fsum-stage {
  flex: 1;
  min-height: 0;
  pointer-events: none;
}
.fsum-bar {
  flex: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
.fsum-text {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
</style>
