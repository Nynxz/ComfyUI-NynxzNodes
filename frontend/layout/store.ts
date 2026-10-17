// One layout per node, shared by every view of it — the editor on the node, the editor in a
// ZenKit panel, a summary on the node meanwhile — with one undo history between them.
//
// Every edit goes through `commit`: it writes the widget (how a node body persists into the saved
// graph) and records an undo step. Steps are recorded a moment after edits stop, so typing a
// prompt is one step rather than one per key; undo and redo first record anything pending.

import { createHistory, type History } from "@nynxz/zenkit-ui";
import { reactive, ref, type Ref } from "vue";

import type { AnyLayoutMode } from "./mode";

interface LayoutWidget {
  value: unknown;
  callback?: (v: unknown) => void;
}

export interface SharedLayout<S> {
  scene: Ref<S>;
  /** Persist the scene as it is now, and record an undo step shortly. */
  commit(): void;
  history: Pick<History, "undo" | "redo" | "canUndo" | "canRedo">;
  /** Replace the scene (a load or import): one undo step back to before it. */
  replace(next: S): void;
}

const RECORD_AFTER_MS = 350;
const layouts = new WeakMap<object, SharedLayout<unknown>>();

function persist(
  widget: LayoutWidget | undefined,
  wrap: (scene: unknown) => unknown,
  scene: unknown,
) {
  if (!widget) return;
  const value = wrap(JSON.parse(JSON.stringify(scene)));
  widget.value = value;
  try {
    widget.callback?.(value);
  } catch {
    /* no callback */
  }
}

/** The node's layout, read from its widget by `mode` the first time any view asks. `unwrap` and
 *  `wrap` move between the widget value and the scene when the widget holds more than the scene. */
export function sharedLayout<S>(
  widget: LayoutWidget | undefined,
  mode: AnyLayoutMode,
  { unwrap = (v: unknown) => v, wrap = (s: unknown) => s } = {},
): SharedLayout<S> {
  const known = widget && layouts.get(widget);
  if (known) return known as SharedLayout<S>;

  const scene = ref(mode.read(unwrap(widget?.value))) as Ref<S>;
  const history = createHistory<S>({
    read: () => scene.value,
    write: (s) => {
      scene.value = s;
      persist(widget, wrap, s);
    },
  });
  let pending: ReturnType<typeof setTimeout> | undefined;
  const flush = () => {
    if (pending === undefined) return;
    clearTimeout(pending);
    pending = undefined;
    history.record();
  };

  const shared: SharedLayout<S> = {
    scene,
    commit() {
      persist(widget, wrap, scene.value);
      clearTimeout(pending);
      pending = setTimeout(flush, RECORD_AFTER_MS);
    },
    history: {
      undo: () => (flush(), history.undo()),
      redo: () => (flush(), history.redo()),
      canUndo: history.canUndo,
      canRedo: history.canRedo,
    },
    replace(next) {
      flush();
      scene.value = next;
      persist(widget, wrap, next);
      history.record();
    },
  };
  if (widget) layouts.set(widget, shared as SharedLayout<unknown>);
  return shared;
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
