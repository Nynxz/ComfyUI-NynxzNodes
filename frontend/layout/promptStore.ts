// One layout per Layout node, shared by every view of it (layout/store.ts). Read without forcing a
// mode: the value carries its own, and the editor follows it.

import type { AnyLayoutMode } from "./mode";
import { ideogramMode } from "./modes/prompt";
import { readPromptScene, type PromptScene } from "./promptScene";
import { sharedLayout, type SharedLayout } from "./store";

interface LayoutWidget {
  value: unknown;
  callback?: (v: unknown) => void;
}

const anyMode: AnyLayoutMode = { ...ideogramMode, read: readPromptScene } as AnyLayoutMode;

export function promptLayout(widget: LayoutWidget | undefined): SharedLayout<PromptScene> {
  return sharedLayout<PromptScene>(widget, anyMode);
}
