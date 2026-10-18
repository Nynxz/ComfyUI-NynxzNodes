<template>
  <ZenModal
    :open="open"
    title="Layout mode"
    width="760px"
    height="auto"
    @update:open="emit('update:open', $event)"
  >
    <div class="ls">
      <p class="ls-note">
        The model the prompt is written for. The boxes stay; what each mode reads is in the sidebar.
      </p>
      <ModeCards
        :model-value="mode"
        @update:model-value="
          emit('pick', $event);
          emit('update:open', false);
        "
      />
    </div>
  </ZenModal>
</template>

<script setup lang="ts">
/** Pick the layout's mode. Everything else about the layout lives in the editor's sidebar. */
import { ZenModal } from "@nynxz/zenkit-ui";

import ModeCards from "@/components/layout/ModeCards.vue";
import type { PromptModeId } from "@/layout/promptScene";

defineProps<{ open: boolean; mode: PromptModeId }>();
const emit = defineEmits<{ "update:open": [boolean]; pick: [PromptModeId] }>();
</script>

<style scoped>
.ls {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 4px 2px;
  font-size: 12px;
  color: var(--zen-text, #e5e5ea);
}
.ls-note {
  margin: 0;
  font-size: 11px;
  line-height: 1.45;
  color: var(--zen-muted, #9aa0aa);
}
</style>
