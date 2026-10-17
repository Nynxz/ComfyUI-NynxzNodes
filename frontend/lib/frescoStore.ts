// One scene per Fresco node, shared by every view of it: the editor on the node, the editor in a
// ZenKit panel, and the summary the node shows while the panel is open. It is the node's shared
// layout (layout/store.ts) in Regions mode, so every view also shares one undo history.

import type { Ref } from "vue";

import { regionsMode } from "@/layout/modes/regions";
import { sharedLayout, type SharedLayout } from "@/layout/store";
import type { FrescoScene } from "@/lib/frescoScene";

export { isPanelOpen, setPanelOpen } from "@/layout/store";

interface SceneWidget {
  value: unknown;
  callback?: (v: unknown) => void;
}

/** The node's layout: its scene, how to persist it, and its undo history. */
export function frescoLayout(widget: SceneWidget | undefined): SharedLayout<FrescoScene> {
  return sharedLayout<FrescoScene>(widget, regionsMode);
}

/** The node's scene, parsed from its widget the first time any view asks. */
export function sharedScene(widget: SceneWidget | undefined): Ref<FrescoScene> {
  return frescoLayout(widget).scene;
}
