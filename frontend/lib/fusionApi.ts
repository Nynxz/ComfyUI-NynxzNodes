// Backend routes for the Fusion Input grid (same-origin fetch).
//
// Listing comes from this pack's own route so it stands alone; thumbnails and uploads use
// ComfyUI's core /view and /upload/image endpoints. /upload/image only skips a rewrite when
// the SAME name already holds byte-identical content — a same-name/different-content file is
// saved as `foo (1).png`. So the caller (see addFiles in FusionGrid.vue) dedups against the
// input listing first and only uploads files that aren't already there.

export type SourceType = "input" | "temp" | "output";

export interface ImageItem {
  name: string; // path relative to its source dir, '/'-separated
  filename: string;
  subfolder: string;
  type: SourceType;
  size?: number; // bytes (from the listing's os.stat)
  mtime?: number; // epoch seconds (from the listing's os.stat)
}

// Cached per source type: fetched once on first need (dialog open), NOT eagerly on mount,
// so N Fusion Input nodes don't each hit the route on graph load. force=true refreshes
// (e.g. after an upload).
const _cache = new Map<SourceType, Promise<ImageItem[]>>();

export function listImages(type: SourceType = "input", force = false): Promise<ImageItem[]> {
  if (force || !_cache.has(type)) _cache.set(type, fetchImages(type));
  return _cache.get(type)!;
}

async function fetchImages(type: SourceType): Promise<ImageItem[]> {
  try {
    const d = await (await fetch("/nynxz/fusion/images?type=" + type)).json();
    const arr: unknown[] = Array.isArray(d.images) ? d.images : [];
    return arr.map((raw) => {
      const x = raw as Record<string, unknown>;
      return {
        name: String(x.name).replace(/\\/g, "/"),
        filename: String(x.filename),
        subfolder: String(x.subfolder ?? ""),
        type: (x.type as SourceType) ?? type,
        size: typeof x.size === "number" ? x.size : undefined,
        mtime: typeof x.mtime === "number" ? x.mtime : undefined,
      };
    });
  } catch {
    return [];
  }
}

/** Thumbnail URL via ComfyUI's own view endpoint, for whichever folder the image lives in. */
export function thumbUrl(ref: string, type: SourceType = "input"): string {
  const slash = ref.lastIndexOf("/");
  const params = new URLSearchParams({
    filename: slash === -1 ? ref : ref.slice(slash + 1),
    subfolder: slash === -1 ? "" : ref.slice(0, slash),
    type,
  });
  return "/view?" + params.toString();
}

/** Upload a dropped/picked file into the input dir; returns its input-relative ref. */
export async function uploadImage(file: File): Promise<{ ref: string; type: SourceType } | null> {
  try {
    const body = new FormData();
    body.append("image", file, file.name);
    body.append("type", "input");
    const d = await (await fetch("/upload/image", { method: "POST", body })).json();
    if (!d || !d.name) return null;
    const subfolder = String(d.subfolder ?? "");
    const filename = String(d.name);
    _cache.delete("input"); // the listing is now stale
    return { ref: subfolder ? `${subfolder}/${filename}` : filename, type: "input" };
  } catch {
    return null;
  }
}

export function shortName(ref: string): string {
  return ref.split("/").pop() || ref;
}
