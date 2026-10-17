<template>
  <div class="lsum">
    <div class="lsum-stage">
      <ZenBoxCanvas :boxes="boxes" :selected="[]" :aspect="aspect" :draw="false">
        <template #empty>
          <span>No boxes yet</span>
        </template>
      </ZenBoxCanvas>
    </div>
    <div class="lsum-bar">
      <span class="lsum-text">{{ line }} · editing in a panel</span>
      <ZenButton sm icon="mdi mdi-open-in-new" @click="emit('focus')">Show editor</ZenButton>
    </div>
  </div>
</template>

<script setup lang="ts">
/** What a layout node shows while its editor is open in a panel: the boxes, read-only. */
import { ZenBoxCanvas, ZenButton, type ZenBox } from "@nynxz/zenkit-ui";
import { computed } from "vue";

import type { LayoutBox } from "@/layout/mode";

const props = defineProps<{ boxes: LayoutBox[]; aspect: number; line: string }>();
const emit = defineEmits<{ focus: [] }>();

const boxes = computed<ZenBox[]>(() =>
  props.boxes.map((b) => ({ ...b, label: b.name, locked: true })),
);
</script>

<style scoped>
.lsum {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
  height: 100%;
  min-height: 0;
}
.lsum-stage {
  flex: 1;
  min-height: 0;
  pointer-events: none;
}
.lsum-bar {
  flex: none;
  display: flex;
  align-items: center;
  gap: 8px;
}
.lsum-text {
  flex: 1;
  min-width: 0;
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
</style>
