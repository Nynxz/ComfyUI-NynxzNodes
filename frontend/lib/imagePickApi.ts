// Backend for the Load Image picker. The listing route is built from pack identity so it
// always matches nodes/load_image/api.py; /view and /upload/image are ComfyUI's own.

import { route } from "@/framework";

export type SourceType = "input" | "temp" | "output";

export interface ImageItem {
  name: string; // path relative to its source dir, '/'-separated
  filename: string;
  subfolder: string;
  type: SourceType;
  size?: number; // bytes (from the listing's os.stat)
  mtime?: number; // epoch seconds (from the listing's os.stat)
}

// Format helpers for the preview info popover.
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
export function fmtDate(mtime?: number): string {
  if (!mtime) return "—";
  try {
    return new Date(mtime * 1000).toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "—";
  }
}

// Cached per source type: each list is fetched once on first need (combo open), NOT
// eagerly on mount — so N Load-Image nodes don't each hit /nynxz/images on graph load.
// Pass force=true to refresh (e.g. after an upload).
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

// Thumbnails come from ComfyUI's own view endpoint, for whichever source the image lives in.
export function thumbUrl(it: { filename: string; subfolder: string; type?: SourceType }): string {
  const p = new URLSearchParams({
    filename: it.filename,
    type: it.type ?? "input",
    subfolder: it.subfolder,
  }).toString();
  return "/view?" + p;
}

// Use ComfyUI's own upload endpoint — it hash-compares same-named files to avoid
// creating duplicate copies (server.py image_upload), matching the stock Load Image.
// Uploads always land in the input dir.
export async function uploadImage(file: File): Promise<ImageItem | null> {
  try {
    const body = new FormData();
    body.append("image", file, file.name);
    body.append("type", "input");
    const d = await (await fetch("/upload/image", { method: "POST", body })).json();
    if (!d || !d.name) return null;
    const subfolder = String(d.subfolder ?? "");
    const filename = String(d.name);
    return {
      name: subfolder ? `${subfolder}/${filename}` : filename,
      filename,
      subfolder,
      type: "input",
    };
  } catch {
    return null;
  }
}
