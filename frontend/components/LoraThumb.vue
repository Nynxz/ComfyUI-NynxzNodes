<!-- The 40px (or 18px) tile that stands in for a LoRA: its sidecar preview, a placeholder cube
     when it has none, or a warning glyph when the name no longer exists on disk. Its own
     component because the three states and their fallbacks are repeated in the picker, the row
     and the browser, and a preview that 404s has to demote itself in all three. -->
<template>
  <i
    v-if="missing"
    class="mdi mdi-alert lt warn"
    :class="{ sm }"
    :title="`LoRA not found on disk: ${name}`"
  />
  <img
    v-else-if="showImage"
    class="lt"
    :class="{ sm }"
    :src="lib.preview(name)"
    loading="lazy"
    @error="onError"
  />
  <i v-else class="mdi mdi-cube-outline lt ph" :class="{ sm }" />
</template>

<script setup lang="ts">
import { computed } from "vue";

import * as lib from "@/lib/loraLibrary";

const props = withDefaults(
  defineProps<{
    name: string;
    /** 18px inline tile instead of the 40px list tile. */
    sm?: boolean;
    /** Show the "not on disk" state. Off in the browser, where every name came FROM the disk. */
    checkMissing?: boolean;
    /** Trust `previewable` strictly. The default is optimistic — see lib.hasThumb. */
    strict?: boolean;
  }>(),
  { sm: false, checkMissing: false, strict: false },
);

const missing = computed(() => props.checkMissing && lib.isMissing(props.name));
const showImage = computed(
  () => !!props.name && (props.strict ? lib.hasPreview(props.name) : lib.hasThumb(props.name)),
);

function onError(e: Event) {
  lib.onThumbError(props.name);
  lib.onImageError(e);
}
</script>

<style scoped>
.lt {
  flex: none;
  box-sizing: border-box;
  width: 40px;
  height: 40px;
  object-fit: contain;
  border-radius: var(--zen-radius, 6px);
  background: var(--zen-input, #1b1b20);
  border: 1px solid var(--zen-border, #34343c);
}
.lt.sm {
  width: 18px;
  height: 18px;
  border-radius: var(--zen-radius, 5px);
}
.lt.ph {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--zen-muted, #9aa0aa);
  font-size: 20px;
}
.lt.sm.ph {
  font-size: 11px;
}
.lt.warn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--zen-danger, #f5665f);
  background: color-mix(in srgb, var(--zen-danger, #f5665f) 14%, transparent);
  border-color: color-mix(in srgb, var(--zen-danger, #f5665f) 45%, transparent);
  font-size: 20px;
}
.lt.sm.warn {
  font-size: 12px;
}
</style>
