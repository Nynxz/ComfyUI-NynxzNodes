<template>
  <div class="pal">
    <span v-for="(c, i) in modelValue" :key="i" class="pal-chip" @contextmenu.prevent="remove(i)">
      <ZenColorPicker compact :model-value="c" @update:model-value="set(i, $event)" />
    </span>
    <ZenIconButton
      v-if="modelValue.length < max"
      size="sm"
      icon="mdi mdi-plus"
      :title="`Add a colour (up to ${max}) — right-click one to remove it`"
      @click="emit('update:modelValue', [...modelValue, '#808080'])"
    />
    <span v-if="!modelValue.length" class="pal-none">none</span>
  </div>
</template>

<script setup lang="ts">
/** A short list of colours: click one to change it, right-click to drop it, + to add. */
import { ZenColorPicker, ZenIconButton } from "@nynxz/zenkit-ui";

const props = withDefaults(defineProps<{ modelValue: string[]; max?: number }>(), { max: 5 });
const emit = defineEmits<{ "update:modelValue": [string[]] }>();

function set(i: number, color: string) {
  const next = [...props.modelValue];
  next[i] = color.toUpperCase();
  emit("update:modelValue", next);
}
function remove(i: number) {
  emit(
    "update:modelValue",
    props.modelValue.filter((_, j) => j !== i),
  );
}
</script>

<style scoped>
.pal {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
}
.pal-chip {
  display: inline-flex;
}
.pal-none {
  font-size: 11px;
  color: var(--zen-muted, #9aa0aa);
}
</style>
