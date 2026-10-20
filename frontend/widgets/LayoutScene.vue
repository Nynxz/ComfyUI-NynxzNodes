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
      :choose-mode="!hasMode(widget?.value)"
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
import { computed } from "vue";

import LayoutSummary from "@/components/layout/LayoutSummary.vue";
import PromptLayoutEditor from "@/components/layout/PromptLayoutEditor.vue";
import { PROMPT_MODES } from "@/layout/modes";
import { promptLayout } from "@/layout/promptStore";
import { usePopOut, type PopOutProps } from "@/lib/popOut";

function hasMode(value: unknown): boolean {
  return !!value && typeof value === "object" && "mode" in value;
}

const props = defineProps<PopOutProps>();

const layout = promptLayout(props.widget);
const { poppedOut, popOut } = usePopOut(props, {
  name: "layout",
  openKey: `layout:${props.node?.id ?? ""}`,
  property: "nynxzLayoutPoppedOut",
  title: "Layout",
  icon: "mdi mdi-vector-square",
  size: { width: 1120, height: 740, minWidth: 720, minHeight: 480 },
});

const mode = computed(
  () => PROMPT_MODES.find((m) => m.id === layout.scene.value.mode) ?? PROMPT_MODES[0]!,
);
const summaryBoxes = computed(() => mode.value.boxes(layout.scene.value));
const summaryLine = computed(() => {
  const s = layout.scene.value;
  const n = s.boxes.length;
  return `${mode.value.label} · ${n} box${n === 1 ? "" : "es"} · ${s.width}×${s.height}`;
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
