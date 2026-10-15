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
 * back to this widget. Popped out survives a reload, and the panel keeps its size, place and
 * workspace tile under an id from the graph and the node.
 */
import { computed, getCurrentInstance, onBeforeUnmount, onMounted } from "vue";

import FrescoEditor from "@/components/fresco/FrescoEditor.vue";
import FrescoSummary from "@/components/fresco/FrescoSummary.vue";
import { hasZenPanels, openZenPanel, type ZenPanelHandle } from "@/framework";
import { isPanelOpen, setPanelOpen, sharedScene } from "@/lib/frescoStore";

interface FrescoNode {
  id?: number | string;
  graph?: { id?: string; change?: () => void; setDirtyCanvas?: (a: boolean, b: boolean) => void };
  properties?: Record<string, unknown>;
  onRemoved?: (...args: unknown[]) => unknown;
}
const props = defineProps<{
  widget?: { value: unknown; callback?: (v: unknown) => void };
  node?: FrescoNode;
  panel?: boolean;
}>();

const scene = sharedScene(props.widget);
const id = String(props.node?.id ?? "");
const poppedOut = computed(() => isPanelOpen(id));

// The panel's instance marks itself open while mounted, so closing the panel any way at all
// brings the editor back to the node.
onMounted(() => props.panel && setPanelOpen(id, true));
onBeforeUnmount(() => props.panel && setPanelOpen(id, false));

const POPPED = "nynxzFrescoPoppedOut";
const handles = new WeakMap<object, ZenPanelHandle>();
const quiet = new WeakSet<object>();
const panelId = () => {
  const graph = props.node?.graph?.id;
  return graph ? `nynxz.fresco:${graph}:${id}` : `nynxz.fresco.${id}`;
};
function remember(popped: boolean) {
  const node = props.node;
  if (!node) return;
  node.properties ??= {};
  if (Boolean(node.properties[POPPED]) === popped) return;
  node.properties[POPPED] = popped;
  node.graph?.change?.();
}

const self = getCurrentInstance()!.type;
/** `reveal`: you asked for it, so a panel tiled in a workspace that isn't showing goes there.
 *  The restore on load doesn't: it waits quietly for its workspace. */
function popOut(reveal = false) {
  const node = props.node;
  const handle = openZenPanel(
    {
      id: panelId(),
      title: "Fresco",
      icon: "mdi mdi-palette-outline",
      width: 1180,
      height: 780,
      minWidth: 760,
      minHeight: 520,
      persist: true,
      reveal,
      props: { widget: props.widget, node: props.node, panel: true },
      onClose: () => {
        if (node) handles.delete(node);
        if (node && quiet.delete(node)) return;
        remember(false);
      },
    },
    self,
  );
  if (!handle || !node) return;
  handles.set(node, handle);
  remember(true);
  const previous = node.onRemoved;
  node.onRemoved = function (this: unknown, ...args: unknown[]) {
    const h = handles.get(node);
    if (h) {
      quiet.add(node);
      h.close({ keep: true });
    }
    return previous?.apply(this, args);
  };
}

onMounted(() => {
  if (props.panel || !props.node?.properties?.[POPPED]) return;
  // ZenKit can finish installing after this node mounts; give it a moment rather than give up.
  let tries = 0;
  const attempt = () => {
    if (isPanelOpen(id) || (props.node && handles.has(props.node))) return;
    if (hasZenPanels()) popOut();
    else if (++tries < 40) setTimeout(attempt, 250);
  };
  attempt();
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
