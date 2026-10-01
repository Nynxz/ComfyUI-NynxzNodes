// The Save Info panel's backend: list existing project folders (same-origin).
// Route is built from pack identity, so it always matches nodes/save/api.py.

import { route } from "@/framework";

export interface SaveConfig {
  project: string; // '' = output root, else a project folder name
  subfolder: string;
  filename_prefix: string;
  format: "png" | "jpg" | "webp";
  quality: number;
  embed_metadata: boolean;
  save_workflow_json: boolean;
  base_directory: string;
}

// The single source of truth for a fresh Save Info config (mirrors config.py DEFAULTS,
// but with project='' instead of the combo sentinel).
export const DEFAULT_CONFIG: SaveConfig = {
  project: "",
  subfolder: "",
  filename_prefix: "",
  format: "png",
  quality: 90,
  embed_metadata: true,
  save_workflow_json: false,
  base_directory: "",
};

// Coerce an unknown stored value into a complete SaveConfig (unknown keys dropped, missing
// keys defaulted) so the panel never renders against a partial object.
export function normalizeConfig(v: unknown): SaveConfig {
  const o = (v && typeof v === "object" ? v : {}) as Record<string, unknown>;
  const fmt = String(o.format ?? DEFAULT_CONFIG.format);
  return {
    project: String(o.project ?? ""),
    subfolder: String(o.subfolder ?? ""),
    filename_prefix: String(o.filename_prefix ?? ""),
    format: (fmt === "jpg" || fmt === "webp" ? fmt : "png") as SaveConfig["format"],
    quality: Number.isFinite(o.quality as number) ? Number(o.quality) : DEFAULT_CONFIG.quality,
    embed_metadata: o.embed_metadata !== false,
    save_workflow_json: o.save_workflow_json === true,
    base_directory: String(o.base_directory ?? ""),
  };
}

export interface ProjectsResult {
  base: string;
  projects: string[];
}

// Cached per base override: fetched on demand (combo open) and refreshed when the base
// directory changes or after creating a project.
const _cache = new Map<string, Promise<ProjectsResult>>();

export function listProjects(base = "", force = false): Promise<ProjectsResult> {
  if (force || !_cache.has(base)) _cache.set(base, fetchProjects(base));
  return _cache.get(base)!;
}

async function fetchProjects(base: string): Promise<ProjectsResult> {
  try {
    const url = route("projects") + (base ? "?base=" + encodeURIComponent(base) : "");
    const d = await (await fetch(url)).json();
    return {
      base: String(d.base ?? ""),
      projects: Array.isArray(d.projects) ? d.projects.map(String) : [],
    };
  } catch {
    return { base: "", projects: [] };
  }
}

/** ui output records -> durable save-preview records (no baked-in URL).
 *
 *  SavePreview rebuilds a fresh /view URL each render, so a page refresh re-fetches from disk
 *  instead of replaying a stale, cache-busted link. Shared by Save Image and Preview Image,
 *  which differ only in the ui key and the folder the records point at. */
export const previewRecords = (key: string, fallbackType: string) => (o: Record<string, unknown>) =>
  ((o[key] as Record<string, string>[]) ?? []).map((r) => ({
    filename: r.filename,
    subfolder: r.subfolder ?? "",
    type: r.type ?? fallbackType,
  }));
