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
  /** How strongly the area reads its own prompt and references. 1 = as weighted by its claim. */
  influence: number;
  /** How several references blend (nodes/fresco/_scene.py BLENDS). */
  blend: Blend;
  /** 0 = hard cells, 1 = fully feathered between references. */
  softness: number;
}

export type Blend = "mix" | "weighted" | "tiled" | "salient";
export const BLENDS: { value: Blend; label: string; title: string }[] = [
  { value: "mix", label: "Mix", title: "Every reference interleaved in equal shares" },
  {
    value: "weighted",
    label: "Weighted",
    title: "A stronger reference covers more — the strength numbers set each one's share",
  },
  { value: "tiled", label: "Tiled", title: "Each reference keeps one part to itself" },
  {
    value: "salient",
    label: "Salient",
    title: "Each part takes from whichever reference shows something there",
  },
];

export interface FrescoPoint {
  x: number;
  y: number;
  /** Include this spot, or exclude it. */
  include: boolean;
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
  /** What the painted shape limits: the whole layer, or only where its references are read. */
  shapeFor: "layer" | "refs";
  /** SAM3: what to capture in the box, by text (signed: "jacket, -hands")… */
  capture: string;
  /** …and/or click points, as fractions of the canvas. */
  points: FrescoPoint[];
  /** The segmentation the current mask came from (set by the backend). */
  maskSource: string;
  /** Canvas pixels the shape grows by (negative shrinks), within the box. */
  grow: number;
  feather: number;
  locked: boolean;
}

export type Overlap = "top" | "share";

export interface FrescoSettings {
  /** top: a layer covers what is under it. share: overlaps are split. */
  overlap: Overlap;
  /** Resolution references are read at. */
  visual_size: number;
}

export interface FrescoScene {
  width: number;
  height: number;
  prompt: string;
  background: FrescoContent;
  /** Bottom to top. */
  layers: FrescoLayer[];
  settings: FrescoSettings;
}

export const BACKGROUND_ID = "background";
/** Mirrors nodes/fresco/_scene.py: an edit keeps the scene unless asked. */
export const BACKGROUND_DENOISE = 0.05;

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

function parseContent(raw: Record<string, unknown>, denoise = 1): FrescoContent {
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
    denoise: clamp01(num(raw.denoise, denoise)),
    influence: Math.min(4, Math.max(0, num(raw.influence, 1))),
    blend: (BLENDS.some((b) => b.value === raw.blend) ? raw.blend : "mix") as Blend,
    softness: clamp01(num(raw.softness, 0.5)),
  };
}

export function emptyContent(denoise = 1): FrescoContent {
  return {
    on: true,
    prompt: "",
    refs: [],
    loras: [],
    denoise,
    influence: 1,
    blend: "mix",
    softness: 0.5,
  };
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
    background: parseContent(obj(raw.background), BACKGROUND_DENOISE),
    settings: {
      overlap: obj(raw.settings).overlap === "share" ? "share" : "top",
      visual_size: Math.min(
        1024,
        Math.max(128, Math.round(num(obj(raw.settings).visual_size, 384))),
      ),
    },
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
        grow: Math.round(num(l.grow, 0)),
        shapeFor: l.shapeFor === "refs" ? "refs" : "layer",
        capture: String(l.capture ?? ""),
        points: list(l.points).map((pt) => ({
          x: clamp01(num(pt.x, 0)),
          y: clamp01(num(pt.y, 0)),
          include: pt.include !== false,
        })),
        maskSource: String(l.maskSource ?? ""),
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
    shapeFor: "layer",
    capture: "",
    points: [],
    maskSource: "",
    grow: 0,
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

/** A media ref as nodekit's drop helpers give it (`input/sub/file.png`) as a reference's file. */
export function refFromMedia(media: string): { ref: string; type: SourceType } {
  const [head, ...rest] = media.split("/");
  return ["input", "temp", "output"].includes(head!)
    ? { ref: rest.join("/"), type: head as SourceType }
    : { ref: media, type: "input" };
}

/** A new reference row for a file. */
export function newRef(file: { ref: string; type: SourceType }): FrescoRef {
  return { id: uid("r"), ...file, strength: 1, fit: "contain", on: true };
}

/** Whether a media ref is an image a reference can use. */
export function isImage(media: string): boolean {
  return /\.(png|jpe?g|webp|gif|bmp|avif)$/i.test(media);
}
