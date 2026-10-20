// Image references in ComfyUI's input/temp/output folders: the listing comes from this pack's
// /nynxz/images route (nodes/load_image/api.py); thumbnails and uploads use ComfyUI's own /view
// and /upload/image. /upload/image only skips a rewrite when the SAME name already holds
// byte-identical content — a same-name/different-content file is saved as `foo (1).png` — so
// callers that may re-drop a file (FusionGrid's addFiles) dedup against the listing first.

import { route } from "@/framework";

export type SourceType = "input" | "temp" | "output";

/** An image by its folder-relative, '/'-separated path. */
export interface MediaRef {
  ref: string;
  type: SourceType;
}

export interface ImageItem {
  name: string; // path relative to its source dir, '/'-separated
  filename: string;
  subfolder: string;
  type: SourceType;
  size?: number; // bytes (from the listing's os.stat)
  mtime?: number; // epoch seconds (from the listing's os.stat)
}

// Cached per source type: fetched once on first need (a dialog or combo opening), NOT eagerly on
// mount, so N nodes don't each hit the route on graph load. force=true refreshes.
const _cache = new Map<SourceType, Promise<ImageItem[]>>();

export function listImages(type: SourceType = "input", force = false): Promise<ImageItem[]> {
  if (force || !_cache.has(type)) _cache.set(type, fetchImages(type));
  return _cache.get(type)!;
}

async function fetchImages(type: SourceType): Promise<ImageItem[]> {
  try {
    const d = (await (await fetch(route("images") + "?type=" + type)).json()) as {
      images?: unknown;
    };
    const arr: Record<string, unknown>[] = Array.isArray(d.images) ? d.images : [];
    return arr.map((x) => ({
      name: String(x.name).replace(/\\/g, "/"),
      filename: String(x.filename),
      subfolder: String(x.subfolder ?? ""),
      type: (x.type as SourceType) ?? type,
      size: typeof x.size === "number" ? x.size : undefined,
      mtime: typeof x.mtime === "number" ? x.mtime : undefined,
    }));
  } catch {
    return [];
  }
}

/** Thumbnail URL via ComfyUI's view endpoint, for a file split into filename and subfolder. */
export function fileThumbUrl(file: {
  filename: string;
  subfolder: string;
  type?: SourceType;
}): string {
  const params = new URLSearchParams({
    filename: file.filename,
    subfolder: file.subfolder,
    type: file.type ?? "input",
  });
  return "/view?" + params.toString();
}

/** Thumbnail URL for a folder-relative ref. */
export function thumbUrl(ref: string, type: SourceType = "input"): string {
  const slash = ref.lastIndexOf("/");
  return fileThumbUrl({
    filename: slash === -1 ? ref : ref.slice(slash + 1),
    subfolder: slash === -1 ? "" : ref.slice(0, slash),
    type,
  });
}

/** Upload a dropped/picked file into the input dir; returns its input-relative ref. */
export async function uploadImage(file: File): Promise<MediaRef | null> {
  try {
    const body = new FormData();
    body.append("image", file, file.name);
    body.append("type", "input");
    const d = await (await fetch("/upload/image", { method: "POST", body })).json();
    if (!d || !d.name) return null;
    const subfolder = String(d.subfolder ?? "");
    const filename = String(d.name);
    _cache.delete("input");
    return { ref: subfolder ? `${subfolder}/${filename}` : filename, type: "input" };
  } catch {
    return null;
  }
}

export function shortName(ref: string): string {
  return ref.split("/").pop() || ref;
}

export function fmtSize(bytes?: number): string {
  if (!bytes || bytes <= 0) return "—";
  const u = ["B", "KB", "MB", "GB"];
  let n = bytes;
  let i = 0;
  while (n >= 1024 && i < u.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n < 10 && i > 0 ? n.toFixed(1) : Math.round(n)} ${u[i]}`;
}
