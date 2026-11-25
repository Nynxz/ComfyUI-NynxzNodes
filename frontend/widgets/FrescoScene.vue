<template>
  <div class="fhost">
    <div v-if="!panel" class="fhost-dep">
      <span>Fresco moved into <strong>Layout</strong> (Regions mode).</span>
      <ZenButton sm variant="ghost" icon="mdi mdi-content-copy" @click="copyForLayout">
        {{ copied ? "Copied — paste in Layout JSON" : "Copy for Layout" }}
      </ZenButton>
    </div>
    <div class="fhost-body">
      <FrescoSummary v-if="!panel && poppedOut" :scene="scene" @focus="popOut(true)" />
      <FrescoEditor
        v-else
        :widget="widget"
        :node="node"
        :in-panel="panel"
        @pop-out="popOut(true)"
      />
    </div>
  </div>
</template>

<script lang="ts">
import { nodeId as nodeType, type NodeDef, type WidgetOptions } from "@/framework";

/** The body is the node's, and nothing in it changes size as you work. */
export const widgetOptions: WidgetOptions = { minHeight: 520, fill: true, default: {} };

export const nodeDef: Omit<NodeDef, "widgets"> = {
  is: nodeType("Fresco"),
  minSize: [980, 640],
  slotLinks: [{ output: "fresco", spawn: nodeType("Fresco.Apply") }],
};
</script>

<script setup lang="ts">
/**
 * Legacy Fresco node body: kept working for saved workflows, with a one-click copy into Layout.
 */
import { ref } from "vue";
import { ZenButton } from "@nynxz/zenkit-ui";

import FrescoEditor from "@/components/fresco/FrescoEditor.vue";
import FrescoSummary from "@/components/fresco/FrescoSummary.vue";
import { exportLayout } from "@/layout/jsonIo";
import { layoutRegionsMode } from "@/layout/modes/prompt";
import { readPromptScene } from "@/layout/promptScene";
import { sharedScene } from "@/lib/frescoStore";
import { usePopOut, type PopOutProps } from "@/lib/popOut";

const props = defineProps<PopOutProps>();

const scene = sharedScene(props.widget);
const copied = ref(false);
async function copyForLayout() {
  try {
    const converted = readPromptScene(scene.value);
    await navigator.clipboard.writeText(exportLayout(layoutRegionsMode, converted));
    copied.value = true;
    setTimeout(() => (copied.value = false), 2200);
  } catch {
    copied.value = false;
  }
}

const { poppedOut, popOut } = usePopOut(props, {
  name: "fresco",
  openKey: String(props.node?.id ?? ""),
  property: "nynxzFrescoPoppedOut",
  title: "Fresco",
  icon: "mdi mdi-palette-outline",
  size: { width: 1180, height: 780, minWidth: 760, minHeight: 520 },
});
</script>

<style scoped>
.fhost {
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
  width: 100%;
  height: 100%;
  min-height: 0;
}
.fhost-dep {
  flex: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 4px 8px;
  border-bottom: 1px solid color-mix(in srgb, var(--zen-warning, #f5b041) 35%, transparent);
  background: color-mix(in srgb, var(--zen-warning, #f5b041) 10%, transparent);
  color: var(--zen-text, #e5e5ea);
  font-size: 11px;
}
.fhost-body {
  position: relative;
  flex: 1;
  min-height: 0;
}
.fhost-body > * {
  position: absolute;
  inset: 0;
}
</style>
