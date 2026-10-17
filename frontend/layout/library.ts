// Saved layouts (nodes/layout/api.py): one file per layout, listed per mode with a box preview.

import { route } from "@/framework";

export interface SavedLayoutPreview {
  aspect: number;
  boxes: { x: number; y: number; w: number; h: number; color: string }[];
}
export interface SavedLayout {
  name: string;
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

export async function deleteLayout(mode: string, name: string): Promise<void> {
  await post("layouts/delete", { mode, name });
}
