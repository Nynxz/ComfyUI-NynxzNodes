<template>
  <div class="preset">
    <ZenInput
      class="preset-input"
      sm
      :model-value="modelValue"
      :placeholder="placeholder"
      @update:model-value="emit('update:modelValue', String($event ?? ''))"
    />
    <ZenPopover v-if="presets.length" v-model:open="open" placement="bottom-end" role="menu">
      <template #trigger="{ toggle }">
        <ZenIconButton
          size="sm"
          icon="mdi mdi-chevron-down"
          :active="open"
          title="Suggestions"
          @click="toggle"
        />
      </template>
      <div class="preset-menu">
        <ZenMenuItem v-for="p in presets" :key="p" @select="pick(p)">{{ p }}</ZenMenuItem>
      </div>
    </ZenPopover>
  </div>
</template>

<script setup lang="ts">
/** A text field with a list of suggestions beside it: type anything, or pick one. */
import { ZenIconButton, ZenInput, ZenMenuItem, ZenPopover } from "@nynxz/zenkit-ui";
import { ref } from "vue";

withDefaults(defineProps<{ modelValue: string; presets?: string[]; placeholder?: string }>(), {
  presets: () => [],
  placeholder: "",
});
const emit = defineEmits<{ "update:modelValue": [string] }>();
const open = ref(false);
function pick(p: string) {
  open.value = false;
  emit("update:modelValue", p);
}
</script>

<style scoped>
.preset {
  display: flex;
  align-items: center;
  gap: 4px;
}
.preset-input {
  flex: 1;
  min-width: 0;
}
.preset-menu {
  display: flex;
  flex-direction: column;
  gap: 1px;
  padding: 4px;
  max-height: 280px;
  overflow: auto;
}
</style>
