// Backend routes for the LoRA widget (same-origin fetch). Registered by
// nodes/lora/api.py.

export interface LoraItem {
  name: string; // relative path, '/'-separated
  has_preview: boolean;
  favorite?: boolean;
}

// Cached across all LoRA-stack instances — fetched on first picker open, NOT eagerly on
// mount, since N stacked LoRA nodes would otherwise each hit these endpoints on graph load.
let lorasCache: Promise<LoraItem[]> | null = null;

export function listLoras(force = false): Promise<LoraItem[]> {
  if (force || !lorasCache) lorasCache = fetchLoras();
  return lorasCache;
}

async function fetchLoras(): Promise<LoraItem[]> {
  try {
    const data = await (await fetch("/nynxz/loras")).json();
    const arr: unknown[] = Array.isArray(data.loras) ? data.loras : [];
    return arr.map((raw) => {
      const l = raw as { name?: unknown; has_preview?: unknown; favorite?: unknown };
      return {
        name: String(l.name).replace(/\\/g, "/"),
        has_preview: !!l.has_preview,
        favorite: !!l.favorite,
      };
    });
  } catch {
    return [];
  }
}

export function previewUrl(name: string): string {
  return "/nynxz/lora/preview?name=" + encodeURIComponent(name);
}

let favCache: Promise<string[]> | null = null;

export function getFavorites(force = false): Promise<string[]> {
  if (force || !favCache) favCache = fetchFavorites();
  return favCache;
}

async function fetchFavorites(): Promise<string[]> {
  try {
    const data = await (await fetch("/nynxz/favorites")).json();
    return Array.isArray(data.loras) ? data.loras.map(String) : [];
  } catch {
    return [];
  }
}

export async function setFavorite(name: string, pinned: boolean): Promise<string[]> {
  try {
    const res = await fetch("/nynxz/favorites", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, pinned }),
    });
    const data = await res.json();
    const next = Array.isArray(data.loras) ? data.loras.map(String) : [];
    favCache = Promise.resolve(next); // keep the shared cache in sync for other instances
    return next;
  } catch {
    return [];
  }
}
