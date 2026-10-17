// A layout as JSON, to copy out and paste back in. Our own envelope carries the mode, so a layout
// from one mode can land in another through the common shape; anything else is offered to the
// mode as its model's own format (an Ideogram caption, say). Pasted text is often wrapped — a
// code fence, a sentence before it — so the JSON is dug out of it first.

import type { AnyLayoutMode } from "./mode";
import { layoutMode } from "./mode";

const ENVELOPE = "nynxzLayout";

export function exportLayout(mode: AnyLayoutMode, scene: unknown): string {
  return JSON.stringify({ [ENVELOPE]: 1, mode: mode.id, layout: scene }, null, 2);
}

/** The JSON value inside `text`: as is, inside a ``` fence, or from the first { or [ on. */
export function digJson(text: string): unknown {
  const tries = [text.trim()];
  const fence = /```(?:json)?\s*([\s\S]*?)```/i.exec(text);
  if (fence?.[1]) tries.push(fence[1].trim());
  const start = text.search(/[{[]/);
  if (start > 0) tries.push(text.slice(start).trim());
  for (const attempt of tries) {
    try {
      return JSON.parse(attempt);
    } catch {
      /* try the next */
    }
  }
  return undefined;
}

/** `text` read as a layout for `mode`, or null when nothing in it is one. */
export function importLayout(mode: AnyLayoutMode, text: string, current: unknown): unknown | null {
  const value = digJson(text);
  if (value && typeof value === "object" && ENVELOPE in value) {
    const env = value as { mode?: string; layout?: unknown };
    if (env.mode === mode.id) return mode.read(env.layout);
    const from = env.mode ? layoutMode(env.mode) : undefined;
    return from ? mode.fromCommon(from.toCommon(from.read(env.layout))) : null;
  }
  const native = mode.importText?.(text, current);
  if (native) return native;
  return null;
}
