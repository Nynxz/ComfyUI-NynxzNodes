// This pack's LoRA backend, shaped as the LoraSource that zenkit-ui's picker consumes.
// Routes are served by nodes/lora/api.py and built from pack identity so the two can't drift.
//
// No caching here: zenkit-ui's shared store fetches once via `ensure()` and keeps the result
// for every widget on the graph, so a second cache layer would only add a way to go stale.

import {
  LoraDetail,
  loraLibrary,
  type LoraInfo,
  type LoraItem,
  type LoraSource,
} from "@nynxz/zenkit-ui";

import { NAMESPACE, openZenPanel, route } from "@/framework";

async function listLoras(): Promise<LoraItem[]> {
  try {
    const data = (await (await fetch(route("loras"))).json()) as { loras?: unknown };
    const arr: unknown[] = Array.isArray(data.loras) ? data.loras : [];
    return arr.map((raw) => {
      const l = raw as {
        name?: unknown;
        has_preview?: unknown;
        preview?: unknown;
        favorite?: unknown;
        size?: unknown;
        mtime?: unknown;
      };
      return {
        name: String(l.name).replace(/\\/g, "/"),
        has_preview: !!l.has_preview,
        preview: l.preview === "video" || l.preview === "image" ? l.preview : null,
        favorite: !!l.favorite,
        size: typeof l.size === "number" ? l.size : null,
        mtime: typeof l.mtime === "number" ? l.mtime : null,
      };
    });
  } catch {
    return [];
  }
}

async function getInfo(name: string): Promise<LoraInfo | null> {
  try {
    const res = await fetch(route("lora/info") + "?name=" + encodeURIComponent(name));
    return res.ok ? ((await res.json()) as LoraInfo) : null;
  } catch {
    return null;
  }
}

async function getFavorites(): Promise<string[]> {
  try {
    const data = (await (await fetch(route("favorites"))).json()) as { loras?: unknown };
    return Array.isArray(data.loras) ? data.loras.map(String) : [];
  } catch {
    return [];
  }
}

async function setFavorite(name: string, pinned: boolean): Promise<string[]> {
  try {
    const res = await fetch(route("favorites"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, pinned }),
    });
    const data = (await res.json()) as { loras?: unknown };
    return Array.isArray(data.loras) ? data.loras.map(String) : [];
  } catch {
    return [];
  }
}

export const loraSource: LoraSource = {
  list: listLoras,
  favorites: getFavorites,
  setFavorite,
  previewUrl: (name) => route("lora/preview") + "?name=" + encodeURIComponent(name),
  mediaUrl: (name) => route("lora/preview") + "?raw=1&name=" + encodeURIComponent(name),
  info: getInfo,
};

/** One ZenKit panel per LoRA, titled with its name. Opened by code only, never listed in the
 *  ZenBar; returns false without ZenKit so zenkit-ui falls back to its modal. */
export function openLoraDetailPanel(name: string): boolean {
  const handle = openZenPanel(
    {
      id: `${NAMESPACE}:lora:${name}`,
      title: loraLibrary.short(name),
      icon: "mdi mdi-information-outline",
      width: 560,
      height: 720,
      minWidth: 360,
      minHeight: 320,
      persist: false,
      props: { name },
    },
    LoraDetail,
  );
  if (!handle) return false;
  void loraLibrary.info(name).then((info) => {
    if (info?.title) handle.setTitle(info.title);
  });
  return true;
}
