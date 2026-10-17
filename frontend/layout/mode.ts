// What a layout mode is: everything the shared editor (components/layout/LayoutEditor.vue) needs
// to know about one kind of layout, so canvas mechanics — drawing, undo, the right-click menu,
// copy and paste, saving and JSON — are written once and each mode only says what its boxes mean.
//
// A mode owns its own value `S` (Fresco's scene, an Ideogram caption, …) and shows the editor its
// boxes as LayoutBoxes. Everything the editor does to boxes generically — move, hide, lock, reorder,
// delete, paint — comes back through `setBoxes`; only making boxes needs the mode itself.

import type { BoxRect } from "@nynxz/zenkit-ui";

/** One box as the editor sees it. Geometry is 0..1 of the canvas; `mask` is box-local. */
export interface LayoutBox extends BoxRect {
  id: string;
  name: string;
  color: string;
  hidden: boolean;
  locked: boolean;
  mask: string | null;
}

/** A layout any mode can read: what survives a switch between modes, or an import from another. */
export interface CommonLayout {
  width: number;
  height: number;
  prompt: string;
  boxes: (BoxRect & { name: string; color: string; prompt: string; mask: string | null })[];
}

export interface LayoutMode<S> {
  id: string;
  label: string;
  icon: string;
  /** Boxes can carry a painted shape. */
  paints: boolean;
  /** Called with whatever is stored (anything, from any version) — always returns a valid value. */
  read(raw: unknown): S;
  size(scene: S): { width: number; height: number };
  resize(scene: S, width: number, height: number): S;
  /** Bottom to top. */
  boxes(scene: S): LayoutBox[];
  /** Apply the editor's edits: geometry, order, hidden, locked, name, mask, and removal (a box
   *  missing from `boxes` is gone). Never adds one. */
  setBoxes(scene: S, boxes: LayoutBox[]): S;
  /** A new box over `rect`, on top. */
  add(scene: S, rect: BoxRect): { scene: S; id: string };
  /** Plain-data copies of these boxes, for the clipboard. */
  copy(scene: S, ids: string[]): unknown[];
  /** Copies made by `copy`, landed at `rects` with new ids, on top. */
  paste(scene: S, items: unknown[], rects: BoxRect[]): { scene: S; ids: string[] };
  /** A line for the layer list. */
  describe?(scene: S, id: string): { sub?: string; thumb?: string };
  /** What the pinned Scene row says. */
  sceneLine?(scene: S): string;
  /** Whether the Scene row shows as hidden, and how its eye toggles — when the mode has a
   *  part of the scene that can be switched off (Fresco's background). */
  sceneHidden?(scene: S): boolean;
  toggleScene?(scene: S): S;
  toCommon(scene: S): CommonLayout;
  fromCommon(common: CommonLayout): S;
  /** The model's own format, for "Copy as …" — e.g. Ideogram's caption JSON. */
  exportText?(scene: S): string;
  /** Read the model's own format back, or null when `text` isn't it. */
  importText?(text: string): S | null;
}

/** A mode whose value the caller doesn't know — what the shared editor holds. */
export type AnyLayoutMode = LayoutMode<unknown>;

const modes = new Map<string, AnyLayoutMode>();

export function registerMode(mode: AnyLayoutMode): void {
  modes.set(mode.id, mode);
}
export function layoutMode(id: string): AnyLayoutMode | undefined {
  return modes.get(id);
}
export function layoutModes(): AnyLayoutMode[] {
  return [...modes.values()];
}

/** The boxes as a thumbnail preview: what the library draws without knowing the mode. */
export function previewOf(mode: AnyLayoutMode, scene: unknown) {
  const { width, height } = mode.size(scene);
  return {
    aspect: width / height,
    boxes: mode
      .boxes(scene)
      .filter((b) => !b.hidden)
      .map((b) => ({ x: b.x, y: b.y, w: b.w, h: b.h, color: b.color })),
  };
}
