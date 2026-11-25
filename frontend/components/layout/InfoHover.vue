<!-- Rich hover/click info popover backed by ZenPopover so tooltips stay sharp under ComfyUI
     canvas zoom and never clip inside scroll areas or modals. -->
<template>
  <ZenPopover
    v-model:open="open"
    placement="top-start"
    :offset="6"
    role="tooltip"
    @update:open="onOpenChange"
  >
    <template #trigger>
      <button
        type="button"
        class="ih-btn"
        :class="{ on: open }"
        :aria-label="label"
        @mouseenter="enter"
        @mouseleave="leave"
        @focus="enter"
        @blur="leave"
        @click.stop="open = !open"
      >
        <i class="mdi mdi-help-circle-outline" />
      </button>
    </template>
    <div class="ih-card" @mouseenter="enter" @mouseleave="leave">
      <slot />
    </div>
  </ZenPopover>
</template>

<script setup lang="ts">
import { ZenPopover } from "@nynxz/zenkit-ui";
import { onBeforeUnmount, ref } from "vue";

withDefaults(defineProps<{ label?: string }>(), { label: "More info" });

const open = ref(false);
let closeTimer: ReturnType<typeof setTimeout> | null = null;

function clearClose() {
  if (closeTimer) {
    clearTimeout(closeTimer);
    closeTimer = null;
  }
}
function enter() {
  clearClose();
  open.value = true;
}
function leave() {
  clearClose();
  closeTimer = setTimeout(() => {
    open.value = false;
  }, 120);
}
function onOpenChange(v: boolean) {
  if (!v) clearClose();
}
onBeforeUnmount(clearClose);
</script>

<style scoped>
.ih-btn {
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 15px;
  height: 15px;
  padding: 0;
  border: 0;
  border-radius: 50%;
  background: transparent;
  color: var(--zen-muted, #9aa0aa);
  font-size: 13px;
  line-height: 1;
  cursor: pointer;
  transition: color 0.12s ease;
}
.ih-btn:hover,
.ih-btn.on {
  color: var(--zen-text, #e5e5ea);
}
.ih-card {
  width: 268px;
  max-width: min(268px, 85vw);
  padding: 9px 11px;
  font-size: 11px;
  line-height: 1.45;
  color: var(--zen-text, #e5e5ea);
}
</style>
