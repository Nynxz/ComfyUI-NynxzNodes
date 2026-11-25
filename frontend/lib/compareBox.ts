// Geometry for Image Compare's box mode. The box is kept in fractions of the image (0..1), not
// of the widget, so it stays on the same part of the picture when the node is resized; the
// image itself sits letterboxed (object-fit: contain) inside the widget.

/** A rectangle in image fractions: x/y the top-left, w/h the size, all 0..1. */
export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}
/** Where the contained image lands inside the viewport, in viewport pixels. */
export interface Fit {
  left: number;
  top: number;
  width: number;
  height: number;
}
export interface Edges {
  l: boolean;
  r: boolean;
  t: boolean;
  b: boolean;
}
/** What a press at a point would do: resize from some edges, move the box, or draw a new one. */
export type BoxHit = { kind: "resize"; edges: Edges } | { kind: "move" } | { kind: "draw" };

export const DEFAULT_BOX: Box = { x: 0.3, y: 0.3, w: 0.4, h: 0.4 };
/** Grab distance around an edge, and the smallest a box may get, in viewport pixels. */
export const EDGE_PX = 8;
export const MIN_PX = 16;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function isBox(v: unknown): v is Box {
  if (!v || typeof v !== "object") return false;
  const b = v as Record<string, unknown>;
  return (["x", "y", "w", "h"] as const).every(
    (k) => typeof b[k] === "number" && Number.isFinite(b[k]),
  );
}

/** The rect an image of `iw`×`ih` fills inside a `vw`×`vh` viewport under object-fit: contain.
 *  Without a known image size it fills the viewport. */
export function containFit(vw: number, vh: number, iw: number, ih: number): Fit {
  if (!(iw > 0 && ih > 0)) return { left: 0, top: 0, width: vw, height: vh };
  const s = Math.min(vw / iw, vh / ih);
  const width = iw * s;
  const height = ih * s;
  return { left: (vw - width) / 2, top: (vh - height) / 2, width, height };
}

/** A box pulled back inside the image, no smaller than `minW`×`minH` (fractions). */
export function clampBox(b: Box, minW = 0, minH = 0): Box {
  const w = clamp(b.w, minW, 1);
  const h = clamp(b.h, minH, 1);
  return { x: clamp(b.x, 0, 1 - w), y: clamp(b.y, 0, 1 - h), w, h };
}

/** The box in viewport pixels. */
export function boxRect(b: Box, fit: Fit) {
  const left = fit.left + b.x * fit.width;
  const top = fit.top + b.y * fit.height;
  return { left, top, right: left + b.w * fit.width, bottom: top + b.h * fit.height };
}

/** A viewport point as image fractions, clamped to the image. */
export function toImage(px: number, py: number, fit: Fit) {
  return {
    x: fit.width > 0 ? clamp((px - fit.left) / fit.width, 0, 1) : 0,
    y: fit.height > 0 ? clamp((py - fit.top) / fit.height, 0, 1) : 0,
  };
}

export function hitBox(px: number, py: number, b: Box, fit: Fit, edge = EDGE_PX): BoxHit {
  const r = boxRect(b, fit);
  const within = (v: number, lo: number, hi: number) => v >= lo - edge && v <= hi + edge;
  if (!within(px, r.left, r.right) || !within(py, r.top, r.bottom)) return { kind: "draw" };
  const edges = {
    l: Math.abs(px - r.left) <= edge,
    r: Math.abs(px - r.right) <= edge,
    t: Math.abs(py - r.top) <= edge,
    b: Math.abs(py - r.bottom) <= edge,
  };
  // A box narrower than two grab zones: the nearer edge wins, so it can still be resized.
  if (edges.l && edges.r) edges[px - r.left < r.right - px ? "r" : "l"] = false;
  if (edges.t && edges.b) edges[py - r.top < r.bottom - py ? "b" : "t"] = false;
  if (edges.l || edges.r || edges.t || edges.b) return { kind: "resize", edges };
  return { kind: "move" };
}

export function cursorFor(hit: BoxHit): string {
  if (hit.kind === "move") return "move";
  if (hit.kind === "draw") return "crosshair";
  const { l, r, t, b } = hit.edges;
  if ((l && t) || (r && b)) return "nwse-resize";
  if ((r && t) || (l && b)) return "nesw-resize";
  return l || r ? "ew-resize" : "ns-resize";
}

/** The box spanned by two image points, at least `minW`×`minH`. */
export function spanBox(
  a: { x: number; y: number },
  p: { x: number; y: number },
  minW: number,
  minH: number,
): Box {
  const w = Math.max(Math.abs(p.x - a.x), minW);
  const h = Math.max(Math.abs(p.y - a.y), minH);
  const x = p.x < a.x ? a.x - w : a.x;
  const y = p.y < a.y ? a.y - h : a.y;
  return clampBox({ x, y, w, h }, minW, minH);
}

/** `start` moved by `dx`/`dy` (fractions), kept whole and inside the image. */
export function moveBox(start: Box, dx: number, dy: number): Box {
  return clampBox({ ...start, x: start.x + dx, y: start.y + dy });
}

/** `start` with the grabbed edges moved by `dx`/`dy`; the opposite edges stay put. */
export function resizeBox(
  start: Box,
  edges: Edges,
  dx: number,
  dy: number,
  minW: number,
  minH: number,
): Box {
  let left = start.x;
  let top = start.y;
  let right = start.x + start.w;
  let bottom = start.y + start.h;
  if (edges.l) left = clamp(left + dx, 0, right - minW);
  if (edges.r) right = clamp(right + dx, left + minW, 1);
  if (edges.t) top = clamp(top + dy, 0, bottom - minH);
  if (edges.b) bottom = clamp(bottom + dy, top + minH, 1);
  return { x: left, y: top, w: right - left, h: bottom - top };
}

/** `b` re-centred on an image point, its size kept. */
export function centreBox(b: Box, p: { x: number; y: number }): Box {
  return clampBox({ ...b, x: p.x - b.w / 2, y: p.y - b.h / 2 });
}
