// Copied boxes, kept for the page: copy in one node's editor, paste in another of the same mode.
// Each copy keeps its rect, so a paste can land the group where the pointer is.

import type { BoxRect } from "@nynxz/zenkit-ui";

interface Copied {
  mode: string;
  items: unknown[];
  rects: BoxRect[];
}

let copied: Copied | null = null;

export function copyBoxes(mode: string, items: unknown[], rects: BoxRect[]): void {
  copied = items.length ? { mode, items, rects } : null;
}

/** What can be pasted into `mode`, if anything. */
export function copiedFor(mode: string): Copied | null {
  return copied?.mode === mode ? copied : null;
}
