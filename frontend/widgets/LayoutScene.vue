<template>
  <div class="lhost">
    <LayoutSummary
      v-if="!panel && poppedOut"
      :boxes="summaryBoxes"
      :aspect="layout.scene.value.width / layout.scene.value.height"
      :line="summaryLine"
      @focus="popOut(true)"
    />
    <PromptLayoutEditor
      v-else
      :layout="layout"
      :node="node"
      :in-panel="panel"
      :choose-mode="fresh"
      @pop-out="popOut(true)"
    />
  </div>
</template>

<script lang="ts">
import { nodeId as nodeType, type NodeDef, type WidgetOptions } from "@/framework";

export const widgetOptions: WidgetOptions = { minHeight: 480, fill: true, default: {} };

export const nodeDef: Omit<NodeDef, "widgets"> = {
  is: nodeType("Layout"),
  minSize: [900, 560],
  slotLinks: [{ output: "layout", spawn: nodeType("Layout.Extras") }],
};
</script>

<script setup lang="ts">
/**
 * The Layout node's body: the layout editor — or, while it's popped out into a ZenKit panel, the
 * boxes read-only. Every view edits one shared layout, written back to this widget.
 */
import { computed, getCurrentInstance, onBeforeUnmount, onMounted } from "vue";

import LayoutSummary from "@/components/layout/LayoutSummary.vue";
import PromptLayoutEditor from "@/components/layout/PromptLayoutEditor.vue";
import { hasZenPanels, openZenPanel, type ZenPanelHandle } from "@/framework";
import { PROMPT_MODES } from "@/layout/modes";
import { promptLayout } from "@/layout/promptStore";
import { isPanelOpen, setPanelOpen } from "@/layout/store";

function hasMode(value: unknown): boolean {
  return !!value && typeof value === "object" && "mode" in value;
}

interface LayoutNode {
  id?: number | string;
  graph?: { id?: string; change?: () => void; setDirtyCanvas?: (a: boolean, b: boolean) => void };
  properties?: Record<string, unknown>;
  onRemoved?: (...args: unknown[]) => unknown;
}
const props = defineProps<{
  widget?: { value: unknown; callback?: (v: unknown) => void };
  node?: LayoutNode;
  panel?: boolean;
}>();

const fresh = !hasMode(props.widget?.value);
const layout = promptLayout(props.widget);
const id = String(props.node?.id ?? "");
const panelKey = `layout:${id}`;
const poppedOut = computed(() => isPanelOpen(panelKey));

const mode = computed(
  () => PROMPT_MODES.find((m) => m.id === layout.scene.value.mode) ?? PROMPT_MODES[0]!,
);
const summaryBoxes = computed(() => mode.value.boxes(layout.scene.value));
const summaryLine = computed(() => {
  const s = layout.scene.value;
  const n = s.boxes.length;
  return `${mode.value.label} · ${n} box${n === 1 ? "" : "es"} · ${s.width}×${s.height}`;
});

onMounted(() => props.panel && setPanelOpen(panelKey, true));
onBeforeUnmount(() => props.panel && setPanelOpen(panelKey, false));

const POPPED = "nynxzLayoutPoppedOut";
const handles = new WeakMap<object, ZenPanelHandle>();
const quiet = new WeakSet<object>();
const panelId = () => {
  const graph = props.node?.graph?.id;
  return graph ? `nynxz.layout:${graph}:${id}` : `nynxz.layout.${id}`;
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
function popOut(reveal = false) {
  const node = props.node;
  const handle = openZenPanel(
    {
      id: panelId(),
      title: "Layout",
      icon: "mdi mdi-vector-square",
      width: 1120,
      height: 740,
      minWidth: 720,
      minHeight: 480,
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
  let tries = 0;
  const attempt = () => {
    if (isPanelOpen(panelKey) || (props.node && handles.has(props.node))) return;
    if (hasZenPanels()) popOut();
    else if (++tries < 40) setTimeout(attempt, 250);
  };
  attempt();
});
</script>

<style scoped>
.lhost {
  position: relative;
  box-sizing: border-box;
  width: 100%;
  height: 100%;
  min-height: 0;
}
.lhost > * {
  position: absolute;
  inset: 0;
}
</style>
