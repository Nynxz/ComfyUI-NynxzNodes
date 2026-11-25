// The unified Layout node's value: boxes on a canvas, for plain regional prompting/LoRAs/refs
// ("regions") or a model's native layout prompt (Ideogram 4, FLUX 3 Image Klein, MOSS-Image).
// Every mode shares every field — including painted shapes, SAM3 capture, mixins, per-box
// references/LoRAs, denoise and background — so switching mode loses nothing.
//
// `nodes/layout/_prompt.py` reads the same shape (including `cast`-era layouts and Fresco scenes).

import type { BoxRect } from "@nynxz/zenkit-ui";

import type { Blend, FrescoLora, FrescoPoint, FrescoRef, Overlap } from "@/lib/frescoScene";
import { BACKGROUND_DENOISE, BLENDS, readLoras, readRefs } from "@/lib/frescoScene";
import { clamp01, nextColor as nextColorOf, num, uid } from "@/lib/sceneUtil";

import { digJson } from "./jsonIo";

export type PromptModeId = "regions" | "ideogram" | "klein" | "moss";
export type BoxKind = "object" | "character" | "text";
export type MossCanvas = "" | "single" | "multi_panel" | "collage";
export type ShapeFor = "layer" | "refs";

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
  /** Ids of the mixins in this box. */
  mixins: string[];
  /** Direct per-box reference images (combined with any mixin references). */
  refs: FrescoRef[];
  /** Direct per-box LoRAs (combined with any mixin LoRAs). */
  loras: FrescoLora[];
  /** How much of a source image this box may rewrite (0..1, for edits). */
  denoise: number;
  /** How strongly this box follows its own prompt/refs against the rest of the picture. */
  influence: number;
  /** How multiple references on this box combine. */
  blend: Blend;
  /** Softness between references (0..1). */
  softness: number;
  /** Optional box-local painted/segmented mask (PNG data URL). */
  mask: string | null;
  /** What the painted shape limits: the whole box, or only its references. */
  shapeFor: ShapeFor;
  /** Canvas pixels to grow/shrink the shape within the box. */
  grow: number;
  /** Edge feathering in canvas pixels. */
  feather: number;
  /** SAM3 signed capture text ("jacket, -hands"). */
  capture: string;
  /** SAM3 click points in canvas fractions. */
  points: FrescoPoint[];
  /** Signature of the segmentation that produced `mask`. */
  maskSource: string;
}

/** Something several boxes share — a character, an object, a style: reference images, LoRAs and
 *  trigger words given once. In each box that names it, the LoRAs run on that box only and the
 *  images are read into its prompt. */
export interface Mixin {
  id: string;
  name: string;
  /** Its avatar's colour where it has no image, and its badge on the canvas. */
  color: string;
  /** Who or what it is, in words: added to the description of every box that uses it. */
  desc: string;
  /** Words added to the prompt of every box that names them — a LoRA's trigger. */
  trigger: string;
  refs: FrescoRef[];
  loras: FrescoLora[];
}

/** Background region (wherever no box sits), for Regions mode and source-image edits. */
export interface LayoutBackdrop {
  on: boolean;
  prompt: string;
  refs: FrescoRef[];
  loras: FrescoLora[];
  denoise: number;
  influence: number;
  blend: Blend;
  softness: number;
}

export interface LayoutRegionSettings {
  overlap: Overlap;
  visual_size: number;
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
  mixins: Mixin[];
  backdrop: LayoutBackdrop;
  settings: LayoutRegionSettings;
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

const MODES: PromptModeId[] = ["regions", "ideogram", "klein", "moss"];
const KINDS: BoxKind[] = ["object", "character", "text"];
const CANVASES: MossCanvas[] = ["", "single", "multi_panel", "collage"];
const SNAP = 16;
const GRID = 1000;

export const boxId = () => uid("b");

export const nextColor = (boxes: { color: string }[]) => nextColorOf(boxes, BOX_COLORS);

const str = (v: unknown) => (typeof v === "string" ? v : "");
const strings = (v: unknown) =>
  Array.isArray(v) ? v.filter((s): s is string => typeof s === "string" && !!s.trim()) : [];
const side = (v: unknown) =>
  Math.max(SNAP, Math.min(16384, Math.round(num(v, 1024) / SNAP) * SNAP));
const readBlend = (v: unknown): Blend => (BLENDS.some((b) => b.value === v) ? (v as Blend) : "mix");
const readPoints = (v: unknown): FrescoPoint[] =>
  Array.isArray(v)
    ? v
        .filter((p): p is Record<string, unknown> => !!p && typeof p === "object")
        .map((p) => ({
          x: clamp01(num(p.x, 0)),
          y: clamp01(num(p.y, 0)),
          include: p.include !== false,
        }))
    : [];

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

export function emptyBackdrop(on = false): LayoutBackdrop {
  return {
    on,
    prompt: "",
    refs: [],
    loras: [],
    denoise: BACKGROUND_DENOISE,
    influence: 1,
    blend: "mix",
    softness: 0.5,
  };
}

export function readBackdrop(raw: unknown, defaultOn = false): LayoutBackdrop {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return emptyBackdrop(defaultOn);
  const r = raw as Record<string, unknown>;
  return {
    on: r.on !== false,
    prompt: str(r.prompt),
    refs: readRefs(r.refs),
    loras: readLoras(r.loras),
    denoise: clamp01(num(r.denoise, BACKGROUND_DENOISE)),
    influence: Math.min(4, Math.max(0, num(r.influence, 1))),
    blend: readBlend(r.blend),
    softness: clamp01(num(r.softness, 0.5)),
  };
}

export function readBox(raw: unknown, index = 0, fromFresco = false): PromptBox {
  const b = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const x = clamp01(num(b.x, 0.3));
  const y = clamp01(num(b.y, 0.3));
  const rawMixins = Array.isArray(b.mixins) ? b.mixins : b.cast;
  return {
    id: str(b.id) || boxId(),
    name: str(b.name) || `Box ${index + 1}`,
    color: str(b.color) || BOX_COLORS[index % BOX_COLORS.length]!,
    x,
    y,
    w: Math.min(1 - x, Math.max(0, num(b.w, 0.4))),
    h: Math.min(1 - y, Math.max(0, num(b.h, 0.4))),
    hidden: fromFresco ? b.on === false : !!b.hidden,
    locked: !!b.locked,
    kind: KINDS.includes(b.kind as BoxKind) ? (b.kind as BoxKind) : "object",
    desc: fromFresco ? str(b.prompt) || str(b.desc) : str(b.desc),
    text: str(b.text),
    palette: strings(b.palette),
    tags: strings(b.tags),
    mixins: strings(rawMixins),
    refs: readRefs(b.refs),
    loras: readLoras(b.loras),
    denoise: clamp01(num(b.denoise, 1)),
    influence: Math.min(4, Math.max(0, num(b.influence, 1))),
    blend: readBlend(b.blend),
    softness: clamp01(num(b.softness, 0.5)),
    mask: typeof b.mask === "string" && b.mask.startsWith("data:image/") ? b.mask : null,
    shapeFor: b.shapeFor === "refs" || b.shape_for === "refs" ? "refs" : "layer",
    grow: Math.round(num(b.grow, 0)),
    feather: Math.max(0, Math.round(num(b.feather, 0))),
    capture: str(b.capture),
    points: readPoints(b.points),
    maskSource: str(b.maskSource ?? b.mask_source),
  };
}

export function readMixin(raw: unknown, index = 0): Mixin {
  const m = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  return {
    id: str(m.id) || uid("c"),
    name: str(m.name) || `Mixin ${index + 1}`,
    color: str(m.color) || BOX_COLORS[(index + 3) % BOX_COLORS.length]!,
    desc: str(m.desc),
    trigger: str(m.trigger),
    refs: readRefs(m.refs),
    loras: readLoras(m.loras),
  };
}

export function isFrescoRaw(v: unknown): boolean {
  return !!v && typeof v === "object" && !Array.isArray(v) && "layers" in v && !("boxes" in v);
}

export function readPromptScene(raw: unknown): PromptScene {
  let v = raw;
  if (typeof v === "string") {
    try {
      v = JSON.parse(v || "{}");
    } catch {
      v = {};
    }
  }
  const s = (v && typeof v === "object" && !Array.isArray(v) ? v : {}) as Record<string, unknown>;
  const fresco = isFrescoRaw(s);
  const style = (s.style && typeof s.style === "object" ? s.style : {}) as Record<string, unknown>;
  const settings = (s.settings && typeof s.settings === "object" ? s.settings : {}) as Record<
    string,
    unknown
  >;
  const base = emptyStyle();
  const rawMixins = Array.isArray(s.mixins) ? s.mixins : s.cast;
  const rawBoxes = fresco ? s.layers : s.boxes;
  const rawBg = s.background;
  const rawBackdrop =
    s.backdrop && typeof s.backdrop === "object"
      ? s.backdrop
      : rawBg && typeof rawBg === "object"
        ? rawBg
        : undefined;
  return {
    mode: fresco
      ? "regions"
      : MODES.includes(s.mode as PromptModeId)
        ? (s.mode as PromptModeId)
        : "ideogram",
    width: side(s.width),
    height: side(s.height),
    prompt: str(s.prompt),
    background: typeof rawBg === "string" ? rawBg : "",
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
    mixins: Array.isArray(rawMixins) ? rawMixins.map((m, i) => readMixin(m, i)) : [],
    backdrop: readBackdrop(rawBackdrop, fresco),
    settings: {
      overlap: settings.overlap === "share" ? "share" : "top",
      visual_size: Math.min(1024, Math.max(128, Math.round(num(settings.visual_size, 384)))),
    },
    boxes: Array.isArray(rawBoxes) ? rawBoxes.map((b, i) => readBox(b, i, fresco)) : [],
  };
}

/** Swap the content of boxes `a` and `b`, keeping each box's position, size, shape and colour. */
const SWAP_KEYS = [
  "name",
  "kind",
  "desc",
  "text",
  "palette",
  "tags",
  "mixins",
  "refs",
  "loras",
  "denoise",
  "influence",
  "blend",
  "softness",
] as const;

export function swapBoxContent(boxes: PromptBox[], a: string, b: string): PromptBox[] {
  const first = boxes.find((x) => x.id === a);
  const second = boxes.find((x) => x.id === b);
  if (!first || !second || a === b) return boxes;
  const take = (from: PromptBox) =>
    Object.fromEntries(SWAP_KEYS.map((k) => [k, from[k]])) as Pick<
      PromptBox,
      (typeof SWAP_KEYS)[number]
    >;
  return boxes.map((x) =>
    x.id === a ? { ...x, ...take(second) } : x.id === b ? { ...x, ...take(first) } : x,
  );
}

/** Active references on a box: its mixins' active refs followed by its own active refs. */
export function boxActiveRefs(scene: PromptScene, box: PromptBox): FrescoRef[] {
  const byId = new Map(scene.mixins.map((m) => [m.id, m]));
  const fromMixins = box.mixins
    .map((id) => byId.get(id))
    .filter((m): m is Mixin => !!m)
    .flatMap((m) => m.refs.filter((r) => r.on));
  return [...fromMixins, ...box.refs.filter((r) => r.on)];
}

/** Active LoRAs on a box: its mixins' active LoRAs followed by its own active LoRAs. */
export function boxActiveLoras(scene: PromptScene, box: PromptBox): FrescoLora[] {
  const byId = new Map(scene.mixins.map((m) => [m.id, m]));
  const fromMixins = box.mixins
    .map((id) => byId.get(id))
    .filter((m): m is Mixin => !!m)
    .flatMap((m) => m.loras.filter((l) => l.on && l.name));
  return [...fromMixins, ...box.loras.filter((l) => l.on && l.name)];
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
