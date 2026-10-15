// One scene per Fresco node, shared by every view of it: the editor on the node, the editor in a
// ZenKit panel, and the summary the node shows while the panel is open. Each reads and writes the
// same reactive scene, so an edit in the panel shows on the node as it happens.

import { reactive, ref, type Ref } from "vue";

import { parseScene, type FrescoScene } from "@/lib/frescoScene";

interface SceneWidget {
  value: unknown;
  callback?: (v: unknown) => void;
}

const scenes = new WeakMap<object, Ref<FrescoScene>>();

/** The node's scene, parsed from its widget the first time any view asks. */
export function sharedScene(widget: SceneWidget | undefined): Ref<FrescoScene> {
  if (!widget) return ref(parseScene(undefined));
  let scene = scenes.get(widget);
  if (!scene) {
    scene = ref(parseScene(widget.value));
    scenes.set(widget, scene);
  }
  return scene;
}

const openPanels = reactive(new Set<string>());

/** Whether node `id`'s editor is open in a panel right now. */
export function isPanelOpen(id: string): boolean {
  return openPanels.has(id);
}
export function setPanelOpen(id: string, open: boolean): void {
  if (open) openPanels.add(id);
  else openPanels.delete(id);
}
