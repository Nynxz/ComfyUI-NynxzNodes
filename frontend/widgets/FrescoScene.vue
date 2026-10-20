<template>
  <div class="fhost">
    <FrescoSummary v-if="!panel && poppedOut" :scene="scene" @focus="popOut(true)" />
    <FrescoEditor v-else :widget="widget" :node="node" :in-panel="panel" @pop-out="popOut(true)" />
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
 * The node's body: the Fresco editor — or, while it is popped out into a ZenKit panel, a live
 * summary of the composition. Every view edits one shared scene (lib/frescoStore.ts), written
 * back to this widget.
 */
import FrescoEditor from "@/components/fresco/FrescoEditor.vue";
import FrescoSummary from "@/components/fresco/FrescoSummary.vue";
import { sharedScene } from "@/lib/frescoStore";
import { usePopOut, type PopOutProps } from "@/lib/popOut";

const props = defineProps<PopOutProps>();

const scene = sharedScene(props.widget);
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
  position: relative;
  box-sizing: border-box;
  width: 100%;
  height: 100%;
  min-height: 0;
}
.fhost > * {
  position: absolute;
  inset: 0;
}
</style>
