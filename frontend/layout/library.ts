// Saved layouts (nodes/layout/api.py): one file per layout, listed per mode with a box preview.

import { route } from "@/framework";

export interface SavedLayoutPreview {
  aspect: number;
  boxes: { x: number; y: number; w: number; h: number; color: string; name?: string }[];
  width?: number;
  height?: number;
  prompt?: string;
}
export interface SavedLayout {
  name: string;
  mode: string;
  saved: number;
  preview: SavedLayoutPreview | null;
}

async function call(path: string, init?: RequestInit): Promise<unknown> {
  const res = await fetch(route(path), init);
  const body = (await res.json().catch(() => ({}))) as { error?: string };
  if (!res.ok) throw new Error(body.error ?? `HTTP ${res.status}`);
  return body;
}
const post = (path: string, body: unknown) =>
  call(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

/** Mode "all" lists every mode's. */
export async function listLayouts(mode: string): Promise<SavedLayout[]> {
  try {
    const body = (await call(`layouts?mode=${encodeURIComponent(mode)}`)) as {
      layouts?: SavedLayout[];
    };
    return Array.isArray(body.layouts) ? body.layouts : [];
  } catch {
    return [];
  }
}

export async function loadLayout(mode: string, name: string): Promise<unknown> {
  const q = new URLSearchParams({ mode, name });
  return ((await call(`layouts/one?${q}`)) as { layout?: unknown }).layout;
}

/** Save and return the name it was saved under. */
export async function saveLayout(
  mode: string,
  name: string,
  layout: unknown,
  preview: SavedLayoutPreview,
): Promise<string> {
  return ((await post("layouts", { mode, name, layout, preview })) as { name: string }).name;
}

/** Rename and return the new name. */
export async function renameLayout(mode: string, name: string, to: string): Promise<string> {
  return ((await post("layouts/rename", { mode, name, to })) as { name: string }).name;
}

export async function deleteLayout(mode: string, name: string): Promise<void> {
  await post("layouts/delete", { mode, name });
}

export interface CompiledPrompt {
  prompt: string;
  warnings: string[];
}

/** The prompt the Layout node outputs for `layout` — written by the backend, the one place
 *  that writes it. */
export async function compileLayout(layout: unknown): Promise<CompiledPrompt> {
  const body = (await post("layouts/compile", { layout })) as Partial<CompiledPrompt>;
  return { prompt: body.prompt ?? "", warnings: body.warnings ?? [] };
}

// --- Saved Mixin library ------------------------------------------------------------------------

export interface SavedMixin {
  name: string;
  saved: number;
  mixin: Record<string, unknown>;
}

export async function listSavedMixins(): Promise<SavedMixin[]> {
  try {
    const body = (await call("layouts/mixins")) as { mixins?: SavedMixin[] };
    return Array.isArray(body.mixins) ? body.mixins : [];
  } catch {
    return [];
  }
}

export async function saveMixinPreset(name: string, mixin: unknown): Promise<string> {
  return ((await post("layouts/mixins", { name, mixin })) as { name: string }).name;
}

export async function deleteMixinPreset(name: string): Promise<void> {
  await post("layouts/mixins/delete", { name });
}
