<template>
  <div class="fl">
    <div v-for="(row, i) in loras" :key="row.id" class="fl-row" :class="{ off: !row.on }">
      <ZenDot :model-value="row.on" title="Enable" @update:model-value="patch(i, { on: $event })" />
      <LoraPicker
        class="fl-pick"
        :model-value="row.name"
        @update:model-value="patch(i, { name: $event })"
        @browse="openBrowse(i)"
      />
      <ZenNumber
        class="fl-str"
        :model-value="row.strength"
        :step="0.05"
        :min="-4"
        :max="4"
        @update:model-value="patch(i, { strength: $event })"
      />
      <ZenIconButton size="sm" icon="mdi mdi-close" title="Remove" @click="remove(i)" />
    </div>
    <ZenButton sm icon="mdi mdi-plus" @click="addRow">LoRA</ZenButton>
    <LoraBrowser v-model:open="browse" :selected="loras[target]?.name" @pick="pickFromBrowse" />
  </div>
</template>

<script setup lang="ts">
/** A layer's LoRAs: each runs only on the layer's tokens, at its own strength. */
import { ref } from "vue";
import {
  LoraBrowser,
  LoraPicker,
  ZenButton,
  ZenDot,
  ZenIconButton,
  ZenNumber,
  loraLibrary,
} from "@nynxz/zenkit-ui";

import { uid, type FrescoLora } from "@/lib/frescoScene";

const loras = defineModel<FrescoLora[]>({ required: true });
if (loras.value.some((r) => r.name)) void loraLibrary.ensure();

const browse = ref(false);
const target = ref(-1);

function patch(i: number, fields: Partial<FrescoLora>) {
  loras.value = loras.value.map((r, j) => (j === i ? { ...r, ...fields } : r));
}
function remove(i: number) {
  loras.value = loras.value.filter((_, j) => j !== i);
}
function addRow() {
  loras.value = [...loras.value, { id: uid("o"), name: "", strength: 1, on: true }];
}
function openBrowse(i: number) {
  target.value = i;
  browse.value = true;
}
function pickFromBrowse(name: string) {
  patch(target.value, { name });
  browse.value = false;
}
</script>

<style scoped>
.fl {
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: stretch;
}
.fl > :last-child {
  align-self: flex-start;
}
.fl-row {
  display: flex;
  align-items: center;
  gap: 4px;
}
.fl-row.off .fl-pick {
  opacity: 0.5;
}
.fl-pick {
  flex: 1;
  min-width: 0;
}
.fl-str {
  width: 64px;
  flex: none;
}
</style>
