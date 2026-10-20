<template>
  <div class="lc">
    <div v-for="member in cast" :key="member.id" class="lc-card">
      <div class="lc-head">
        <ZenInput
          sm
          class="lc-name"
          :model-value="member.name"
          placeholder="Name"
          @update:model-value="patch(member.id, { name: String($event ?? '') })"
        />
        <span class="lc-count" :title="`${boxCount(member.id)} box(es) name ${member.name}`">
          {{ boxCount(member.id) }}
        </span>
        <ZenIconButton
          size="sm"
          icon="mdi mdi-close"
          :title="`Remove ${member.name} (and from every box)`"
          @click="remove(member.id)"
        />
      </div>
      <ZenInput
        sm
        :model-value="member.trigger"
        placeholder="Trigger words (added to each box's prompt)"
        @update:model-value="patch(member.id, { trigger: String($event ?? '') })"
      />
      <FrescoSection label="Look" :note="member.refs.length ? String(member.refs.length) : ''">
        <template #actions>
          <ZenIconButton
            size="sm"
            icon="mdi mdi-upload-outline"
            title="Upload images"
            :disabled="panels[member.id]?.busy"
            @click="panels[member.id]?.upload()"
          />
          <ZenIconButton
            size="sm"
            icon="mdi mdi-view-grid-plus-outline"
            title="Browse images"
            @click="panels[member.id]?.openBrowse()"
          />
        </template>
        <FrescoRefs
          :ref="(el) => setPanel(member.id, el)"
          :model-value="member.refs"
          @update:model-value="patch(member.id, { refs: $event })"
        />
      </FrescoSection>
      <FrescoSection label="LoRAs" :note="member.loras.length ? String(member.loras.length) : ''">
        <template #actions>
          <ZenIconButton
            size="sm"
            icon="mdi mdi-plus"
            title="Add a LoRA"
            @click="patch(member.id, { loras: [...member.loras, newLora()] })"
          />
        </template>
        <FrescoLoras
          v-if="member.loras.length"
          :model-value="member.loras"
          @update:model-value="patch(member.id, { loras: $event })"
        />
      </FrescoSection>
    </div>
    <p v-if="!cast.length" class="lc-empty">
      Someone who appears in several boxes: give them a look (reference images) and LoRAs once, then
      pick them in each box. Needs Regions Apply after Layout Extras.
    </p>
  </div>
</template>

<script setup lang="ts">
/**
 * A layout's cast: who (or what) several boxes show, with their reference images and LoRAs given
 * once. Boxes name members by id; Layout Extras gives each box its members' LoRAs and reads their
 * images into its prompt.
 */
import { ZenIconButton, ZenInput } from "@nynxz/zenkit-ui";
import { reactive } from "vue";

import FrescoLoras from "@/components/fresco/FrescoLoras.vue";
import FrescoRefs from "@/components/fresco/FrescoRefs.vue";
import FrescoSection from "@/components/fresco/FrescoSection.vue";
import type { FrescoLora } from "@/lib/frescoScene";
import { uid } from "@/lib/frescoScene";
import type { CastMember, PromptBox } from "@/layout/promptScene";

const { boxes } = defineProps<{ boxes: PromptBox[] }>();
const cast = defineModel<CastMember[]>({ required: true });

type RefsPanel = { upload: () => void; openBrowse: () => void; busy: boolean };
const panels = reactive<Record<string, RefsPanel | undefined>>({});
function setPanel(id: string, el: unknown) {
  panels[id] = (el as RefsPanel | null) ?? undefined;
}

function patch(id: string, change: Partial<CastMember>) {
  cast.value = cast.value.map((m) => (m.id === id ? { ...m, ...change } : m));
}
function remove(id: string) {
  cast.value = cast.value.filter((m) => m.id !== id);
}
function boxCount(id: string): number {
  return boxes.filter((b) => b.cast.includes(id)).length;
}
function newLora(): FrescoLora {
  return { id: uid("o"), name: "", strength: 1, on: true };
}
</script>

<style scoped>
.lc {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.lc-card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
  border: 1px solid color-mix(in srgb, var(--zen-border, #34343c) 70%, transparent);
  border-radius: var(--zen-radius, 7px);
}
.lc-head {
  display: flex;
  align-items: center;
  gap: 6px;
}
.lc-name {
  flex: 1;
  min-width: 0;
}
.lc-count {
  min-width: 18px;
  font-size: 11px;
  text-align: center;
  color: var(--zen-muted, #9aa0aa);
}
.lc-empty {
  margin: 0;
  font-size: 11px;
  line-height: 1.45;
  color: var(--zen-muted, #9aa0aa);
}
</style>
