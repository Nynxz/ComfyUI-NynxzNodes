// The Fresco scene: what the editor edits and what the backend reads (nodes/fresco/_io_types.py).
// Pure — parsing a saved value, defaults, and the shapes — so the widget only deals in state.

import type { SourceType } from "@/lib/fusionApi";

export type RefFit = "contain" | "cover" | "stretch";
export const REF_FITS: RefFit[] = ["contain", "cover", "stretch"];

export interface FrescoRef {
  id: string;
  ref: string;
  type: SourceType;
  strength: number;
  fit: RefFit;
  on: boolean;
}

export interface FrescoLora {
  id: string;
  name: string;
  strength: number;
  on: boolean;
}

/** What a layer carries besides its box — the background is one of these with no box. */
export interface FrescoContent {
  on: boolean;
  prompt: string;
  refs: FrescoRef[];
  loras: FrescoLora[];
  denoise: number;
}

export interface FrescoLayer extends FrescoContent {
  id: string;
  name: string;
  color: string;
  x: number;
  y: number;
  w: number;
  h: number;
  mask: string | null;
  feather: number;
  locked: boolean;
}

export interface FrescoScene {
  width: number;
  height: number;
  prompt: string;
  background: FrescoContent;
  /** Bottom to top. */
  layers: FrescoLayer[];
}

export const BACKGROUND_ID = "background";

/** Layer colours, in the order new layers take them. */
export const PALETTE = [
  "#f05a5a",
  "#4fb8ff",
  "#7be36a",
  "#ffc94f",
  "#c47dff",
  "#ff8fc0",
  "#4fe0c8",
  "#ff9a4f",
];

let seq = 0;
export function uid(prefix = "l"): string {
  return `${prefix}${Date.now().toString(36)}${(seq++).toString(36)}`;
}

function num(v: unknown, fallback: number): number {
  const n = +(v as number);
  return Number.isFinite(n) ? n : fallback;
}
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const obj = (v: unknown): Record<string, unknown> =>
  v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
const list = (v: unknown): Record<string, unknown>[] =>
  Array.isArray(v)
    ? v.filter((x): x is Record<string, unknown> => !!x && typeof x === "object")
    : [];

function parseContent(raw: Record<string, unknown>): FrescoContent {
  return {
    on: raw.on !== false,
    prompt: String(raw.prompt ?? ""),
    refs: list(raw.refs)
      .filter((r) => r.ref)
      .map((r) => ({
        id: String(r.id ?? uid("r")),
        ref: String(r.ref),
        type: (["input", "temp", "output"].includes(r.type as string)
          ? r.type
          : "input") as SourceType,
        strength: Math.max(0, num(r.strength, 1)),
        fit: (REF_FITS.includes(r.fit as RefFit) ? r.fit : "contain") as RefFit,
        on: r.on !== false,
      })),
    loras: list(raw.loras).map((r) => ({
      id: String(r.id ?? uid("o")),
      name: String(r.name ?? ""),
      strength: num(r.strength, 1),
      on: r.on !== false,
    })),
    denoise: clamp01(num(raw.denoise, 1)),
  };
}

export function emptyContent(denoise = 1): FrescoContent {
  return { on: true, prompt: "", refs: [], loras: [], denoise };
}

export function parseScene(value: unknown): FrescoScene {
  let v = value;
  if (typeof v === "string") {
    try {
      v = JSON.parse(v || "{}");
    } catch {
      v = {};
    }
  }
  const raw = obj(v);
  return {
    width: Math.max(64, Math.round(num(raw.width, 1024))),
    height: Math.max(64, Math.round(num(raw.height, 1024))),
    prompt: String(raw.prompt ?? ""),
    background: parseContent(obj(raw.background)),
    layers: list(raw.layers).map((l, i) => {
      const x = clamp01(num(l.x, 0.25));
      const y = clamp01(num(l.y, 0.25));
      return {
        ...parseContent(l),
        id: String(l.id ?? uid()),
        name: String(l.name ?? `Layer ${i + 1}`),
        color: String(l.color ?? PALETTE[i % PALETTE.length]),
        x,
        y,
        w: Math.min(1 - x, clamp01(num(l.w, 0.5))),
        h: Math.min(1 - y, clamp01(num(l.h, 0.5))),
        mask: typeof l.mask === "string" && l.mask.startsWith("data:image/") ? l.mask : null,
        feather: Math.max(0, Math.round(num(l.feather, 16))),
        locked: l.locked === true,
      };
    }),
  };
}

/** A new layer over `rect`, coloured by how many there already are. */
export function newLayer(
  rect: { x: number; y: number; w: number; h: number },
  existing: FrescoLayer[],
): FrescoLayer {
  const used = new Set(existing.map((l) => l.color));
  const color = PALETTE.find((c) => !used.has(c)) ?? PALETTE[existing.length % PALETTE.length]!;
  let n = existing.length + 1;
  const names = new Set(existing.map((l) => l.name));
  while (names.has(`Layer ${n}`)) n++;
  return {
    ...emptyContent(),
    id: uid(),
    name: `Layer ${n}`,
    color,
    ...rect,
    mask: null,
    feather: 16,
    locked: false,
  };
}

/** A short line describing what a layer carries, for the layer list. */
export function summarize(content: FrescoContent): string {
  const parts: string[] = [];
  const refs = content.refs.filter((r) => r.on).length;
  const loras = content.loras.filter((r) => r.on && r.name).length;
  if (content.prompt.trim()) parts.push(content.prompt.trim().slice(0, 40));
  if (refs) parts.push(`${refs} ref${refs > 1 ? "s" : ""}`);
  if (loras) parts.push(`${loras} LoRA${loras > 1 ? "s" : ""}`);
  return parts.join(" · ") || "empty";
}
