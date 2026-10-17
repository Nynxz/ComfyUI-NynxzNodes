// The Layout node's value: boxes for a model that reads a layout from its prompt — Ideogram 4,
// FLUX 3 Image Klein, MOSS-Image. The three share every field, so switching mode keeps it all;
// nodes/layout/_prompt.py reads the same shape and writes each model's prompt (it's the one
// place that does, so what you copy is what the node outputs).
//
// Reading back goes the other way: a model's own prompt — an Ideogram caption, a MOSS layout,
// Klein's caption and table, or core's prompt elements — becomes a layout.

import type { BoxRect } from "@nynxz/zenkit-ui";

import { digJson } from "./jsonIo";

export type PromptModeId = "ideogram" | "klein" | "moss";
export type BoxKind = "object" | "character" | "text";
export type MossCanvas = "" | "single" | "multi_panel" | "collage";

export interface PromptBox extends BoxRect {
  id: string;
  name: string;
  color: string;
  hidden: boolean;
  locked: boolean;
  kind: BoxKind;
  desc: string;
  text: string;
  palette: string[];
  tags: string[];
}

export interface IdeogramStyle {
  on: boolean;
  kind: "photo" | "art";
  aesthetics: string;
  lighting: string;
  medium: string;
  photo: string;
  art_style: string;
  palette: string[];
}

export interface PromptScene {
  mode: PromptModeId;
  width: number;
  height: number;
  prompt: string;
  background: string;
  style: IdeogramStyle;
  meta: string[];
  canvas: MossCanvas;
  palette: string[];
  /** Bottom to top: background first, foreground last — the order the prompt lists them. */
  boxes: PromptBox[];
}

export const BOX_COLORS = [
  "#3B82F6",
  "#F59E0B",
  "#10B981",
  "#EF4444",
  "#8B5CF6",
  "#EC4899",
  "#14B8A6",
  "#F97316",
  "#84CC16",
  "#06B6D4",
];

const MODES: PromptModeId[] = ["ideogram", "klein", "moss"];
const KINDS: BoxKind[] = ["object", "character", "text"];
const CANVASES: MossCanvas[] = ["", "single", "multi_panel", "collage"];
const SNAP = 16;
const GRID = 1000;

let seq = 0;
export function boxId(): string {
  seq += 1;
  return `b${Date.now().toString(36)}${seq.toString(36)}`;
}

export function nextColor(boxes: { color: string }[]): string {
  const used = new Set(boxes.map((b) => b.color));
  return BOX_COLORS.find((c) => !used.has(c)) ?? BOX_COLORS[boxes.length % BOX_COLORS.length]!;
}

const str = (v: unknown) => (typeof v === "string" ? v : "");
const num = (v: unknown, d: number) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};
const strings = (v: unknown) =>
  Array.isArray(v) ? v.filter((s): s is string => typeof s === "string" && !!s.trim()) : [];
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const side = (v: unknown) =>
  Math.max(SNAP, Math.min(16384, Math.round(num(v, 1024) / SNAP) * SNAP));

export function emptyStyle(): IdeogramStyle {
  return {
    on: false,
    kind: "photo",
    aesthetics: "",
    lighting: "",
    medium: "photograph",
    photo: "",
    art_style: "",
    palette: [],
  };
}

export function readBox(raw: unknown, index = 0): PromptBox {
  const b = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const x = clamp01(num(b.x, 0.3));
  const y = clamp01(num(b.y, 0.3));
  return {
    id: str(b.id) || boxId(),
    name: str(b.name) || `Box ${index + 1}`,
    color: str(b.color) || BOX_COLORS[index % BOX_COLORS.length]!,
    x,
    y,
    w: Math.min(1 - x, Math.max(0, num(b.w, 0.4))),
    h: Math.min(1 - y, Math.max(0, num(b.h, 0.4))),
    hidden: !!b.hidden,
    locked: !!b.locked,
    kind: KINDS.includes(b.kind as BoxKind) ? (b.kind as BoxKind) : "object",
    desc: str(b.desc),
    text: str(b.text),
    palette: strings(b.palette),
    tags: strings(b.tags),
  };
}

export function readPromptScene(raw: unknown): PromptScene {
  const s = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const style = (s.style && typeof s.style === "object" ? s.style : {}) as Record<string, unknown>;
  const base = emptyStyle();
  return {
    mode: MODES.includes(s.mode as PromptModeId) ? (s.mode as PromptModeId) : "ideogram",
    width: side(s.width),
    height: side(s.height),
    prompt: str(s.prompt),
    background: str(s.background),
    style: {
      on: !!style.on,
      kind: style.kind === "art" ? "art" : "photo",
      aesthetics: str(style.aesthetics),
      lighting: str(style.lighting),
      medium: typeof style.medium === "string" ? style.medium : base.medium,
      photo: str(style.photo),
      art_style: str(style.art_style),
      palette: strings(style.palette),
    },
    meta: strings(s.meta),
    canvas: CANVASES.includes(s.canvas as MossCanvas) ? (s.canvas as MossCanvas) : "",
    palette: strings(s.palette),
    boxes: Array.isArray(s.boxes) ? s.boxes.map((b, i) => readBox(b, i)) : [],
  };
}

/** The boxes the prompt lists: not hidden, with some area. */
export function shownBoxes(scene: PromptScene): PromptBox[] {
  return scene.boxes.filter((b) => !b.hidden && b.w > 0 && b.h > 0);
}

/** The id Klein's caption cites box `id` by, or null when it isn't in the prompt. */
export function kleinId(scene: PromptScene, id: string): string | null {
  const at = shownBoxes(scene).findIndex((b) => b.id === id);
  return at < 0 ? null : `obj_${at + 1}`;
}

// --- reading a model's own prompt -----------------------------------------------------------------

type Raw = Record<string, unknown>;
const isObj = (v: unknown): v is Raw => !!v && typeof v === "object" && !Array.isArray(v);

/** A short name from what a box says it is. */
function nameFrom(desc: string, text: string, index: number): string {
  const words = (text || desc).replace(/["<>]/g, "").trim().split(/\s+/).slice(0, 4).join(" ");
  return words ? words.slice(0, 32) : `Box ${index + 1}`;
}

/** `[y_min, x_min, y_max, x_max]` on the 0-1000 grid — or 0-1, when nothing is over 1. */
function fromGrid(bbox: unknown): BoxRect | null {
  if (!Array.isArray(bbox) || bbox.length !== 4) return null;
  const v = bbox.map((n) => Number(n));
  if (v.some((n) => !Number.isFinite(n))) return null;
  const scale = Math.max(...v) <= 1 ? 1 : GRID;
  const [y0, x0, y1, x1] = v.map((n) => clamp01(n / scale)) as [number, number, number, number];
  return { x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0), h: Math.abs(y1 - y0) };
}

function makeBox(
  rect: BoxRect,
  index: number,
  fields: Partial<Pick<PromptBox, "kind" | "desc" | "text" | "palette" | "tags">>,
): PromptBox {
  const desc = fields.desc ?? "";
  const text = fields.text ?? "";
  return readBox(
    {
      ...rect,
      ...fields,
      id: boxId(),
      name: nameFrom(desc, text, index),
      color: BOX_COLORS[index % BOX_COLORS.length],
    },
    index,
  );
}

function blank(mode: PromptModeId, size?: { width: number; height: number }): PromptScene {
  return { ...readPromptScene({}), mode, ...(size ?? {}) };
}

function fromIdeogram(v: Raw, size?: { width: number; height: number }): PromptScene | null {
  const comp = v.compositional_deconstruction;
  if (!isObj(comp) && typeof v.high_level_description !== "string") return null;
  const scene = blank("ideogram", size);
  scene.prompt = str(v.high_level_description);
  const sd = v.style_description;
  if (isObj(sd)) {
    scene.style = {
      ...emptyStyle(),
      on: true,
      kind: str(sd.art_style).trim() && !str(sd.photo).trim() ? "art" : "photo",
      aesthetics: str(sd.aesthetics),
      lighting: str(sd.lighting),
      medium: str(sd.medium),
      photo: str(sd.photo),
      art_style: str(sd.art_style),
      palette: strings(sd.color_palette),
    };
  }
  if (isObj(comp)) {
    scene.background = str(comp.background);
    scene.boxes = fromElements(comp.elements) ?? [];
  }
  return scene;
}

/** Core's prompt elements, or an Ideogram caption's: `{type, bbox, desc, text?, color_palette?}`. */
function fromElements(list: unknown): PromptBox[] | null {
  if (!Array.isArray(list)) return null;
  const boxes: PromptBox[] = [];
  for (const el of list) {
    if (!isObj(el)) continue;
    const rect = fromGrid(el.bbox);
    if (!rect) continue;
    boxes.push(
      makeBox(rect, boxes.length, {
        kind: el.type === "text" ? "text" : "object",
        desc: str(el.desc),
        text: str(el.text),
        palette: strings(el.color_palette),
      }),
    );
  }
  return boxes.length ? boxes : null;
}

function fromMoss(v: Raw, size?: { width: number; height: number }): PromptScene | null {
  if (!isObj(v.summary) || !Array.isArray(v.items)) return null;
  const width = side(v.width ?? size?.width);
  const height = side(v.height ?? size?.height);
  const centred = v.bbox_coord_space !== "abs";
  const yxyx = v.bbox_format !== "xyxy";
  const scene = blank("moss", { width, height });
  scene.prompt = str(v.summary.caption);
  scene.meta = strings(v.summary.meta);
  scene.canvas = CANVASES.includes(v.summary.canvas as MossCanvas)
    ? (v.summary.canvas as MossCanvas)
    : "";
  scene.palette = strings(v.summary.palette);
  for (const item of v.items) {
    if (!isObj(item) || !Array.isArray(item.bbox) || item.bbox.length !== 4) continue;
    const n = item.bbox.map((c) => Number(c));
    if (n.some((c) => !Number.isFinite(c))) continue;
    const [y0, x0, y1, x1] = (yxyx ? n : [n[1], n[0], n[3], n[2]]) as number[] as [
      number,
      number,
      number,
      number,
    ];
    const ox = centred ? width / 2 : 0;
    const oy = centred ? height / 2 : 0;
    const left = clamp01((x0 + ox) / width);
    const top = clamp01((y0 + oy) / height);
    const kind = KINDS.includes(item.label as BoxKind) ? (item.label as BoxKind) : "object";
    scene.boxes.push(
      makeBox(
        {
          x: left,
          y: top,
          w: clamp01((x1 + ox) / width) - left,
          h: clamp01((y1 + oy) / height) - top,
        },
        scene.boxes.length,
        {
          kind,
          desc: str(item.caption),
          text: str(item.text),
          tags: strings(item.tags),
          palette: strings(item.palette),
        },
      ),
    );
  }
  return scene;
}

/** Klein's "caption [table]": the table is the last JSON array of rows with a bbox. */
function fromKlein(text: string, size?: { width: number; height: number }): PromptScene | null {
  for (let at = text.lastIndexOf("["); at >= 0; at = text.lastIndexOf("[", at - 1)) {
    let rows: unknown;
    try {
      rows = JSON.parse(text.slice(at));
    } catch {
      continue;
    }
    if (!Array.isArray(rows) || !rows.length || !rows.every((r) => isObj(r) && "bbox" in r)) {
      continue;
    }
    const scene = blank("klein", size);
    scene.prompt = text.slice(0, at).trim();
    const ids: [string, string][] = [];
    for (const row of rows as Raw[]) {
      const rect = fromGrid(row.bbox ?? row.tgt_bbox);
      if (!rect) continue;
      const quoted = /^(.*?)\s*"([^"]*)"\s*$/.exec(str(row.desc));
      const isText = /_text_\d+$/i.test(str(row.id)) && !!quoted;
      const box = makeBox(rect, scene.boxes.length, {
        kind: isText ? "text" : "object",
        desc: isText ? quoted![1]! : str(row.desc),
        text: isText ? quoted![2]! : "",
      });
      scene.boxes.push(box);
      ids.push([str(row.id), `obj_${scene.boxes.length}`]);
    }
    for (const [from, to] of ids) {
      if (from && from !== to) scene.prompt = scene.prompt.split(`<${from}>`).join(`<${to}>`);
    }
    return scene;
  }
  return null;
}

/** A model's own layout prompt in `text`, as a layout — or null when it's none of them. */
export function readModelPrompt(
  text: string,
  size?: { width: number; height: number },
): PromptScene | null {
  const value = digJson(text);
  if (isObj(value)) {
    const scene = fromMoss(value, size) ?? fromIdeogram(value, size);
    if (scene) return scene;
  }
  if (Array.isArray(value)) {
    const boxes = fromElements(value);
    if (boxes) return { ...blank("ideogram", size), boxes };
  }
  return fromKlein(text.trim(), size);
}
