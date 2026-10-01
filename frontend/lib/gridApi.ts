// Grid Style contract (frontend mirror of nodes/grid/config.py) — layout/border/shadow.
// Backgrounds live entirely in the Background Maker now (see src/background/api.ts).

export interface GridStyle {
  // layout
  columns: number;
  size_mode: "cells" | "canvas";
  cell_width: number;
  cell_height: number;
  canvas_width: number;
  canvas_height: number;
  gap_x: number;
  gap_y: number;
  padding: number;
  fit: "cover" | "contain";
  // border
  border_width: number;
  border_mode: "solid" | "gradient" | "auto";
  border_color: string;
  border_color2: string;
  border_angle: number;
  border_radius: number;
  // shadow / glow
  shadow: boolean;
  shadow_mode: "drop" | "glow";
  shadow_blur: number;
  shadow_offset_x: number;
  shadow_offset_y: number;
  shadow_color: string;
  shadow_opacity: number;
  shadow_match: boolean;
}

export const DEFAULT_GRID_STYLE: GridStyle = {
  columns: 3,
  size_mode: "cells",
  cell_width: 320,
  cell_height: 180,
  canvas_width: 1280,
  canvas_height: 720,
  gap_x: 12,
  gap_y: 12,
  padding: 24,
  fit: "cover",
  border_width: 0,
  border_mode: "solid",
  border_color: "#ffffff",
  border_color2: "#6366f1",
  border_angle: 45,
  border_radius: 10,
  shadow: true,
  shadow_mode: "drop",
  shadow_blur: 18,
  shadow_offset_x: 0,
  shadow_offset_y: 10,
  shadow_color: "#000000",
  shadow_opacity: 0.45,
  shadow_match: false,
};

// Decoration = everything the Vue panel still owns after geometry moved to native widgets on
// the node. The panel commits ONLY these keys, so its value never collides with the native
// geometry widgets (whose values the node reads directly). Keep in sync with GridStyle above.
export const GRID_DECO_KEYS = [
  "border_width",
  "border_mode",
  "border_color",
  "border_color2",
  "border_angle",
  "border_radius",
  "shadow",
  "shadow_mode",
  "shadow_blur",
  "shadow_offset_x",
  "shadow_offset_y",
  "shadow_color",
  "shadow_opacity",
  "shadow_match",
] as const satisfies readonly (keyof GridStyle)[];

export type GridDeco = Pick<GridStyle, (typeof GRID_DECO_KEYS)[number]>;

/** The decoration subset of a full style — what the panel persists into its widget value. */
export function gridDecoOnly(s: GridStyle): GridDeco {
  const out = {} as GridDeco;
  for (const k of GRID_DECO_KEYS) (out as Record<string, unknown>)[k] = s[k];
  return out;
}

/** Default panel value: decoration only (geometry defaults live on the native widgets). */
export const DEFAULT_GRID_DECO: GridDeco = gridDecoOnly(DEFAULT_GRID_STYLE);

const num = (v: unknown, d: number) => (Number.isFinite(Number(v)) ? Number(v) : d);
const str = (v: unknown, d: string) => (typeof v === "string" && v ? v : d);
const pick = <T extends string>(v: unknown, opts: T[], d: T): T =>
  (opts as string[]).includes(v as string) ? (v as T) : d;

export function normalizeGridStyle(v: unknown): GridStyle {
  const o = (v && typeof v === "object" ? v : {}) as Record<string, unknown>;
  const D = DEFAULT_GRID_STYLE;
  return {
    columns: num(o.columns, D.columns),
    size_mode: pick(o.size_mode, ["cells", "canvas"], D.size_mode),
    cell_width: num(o.cell_width, D.cell_width),
    cell_height: num(o.cell_height, D.cell_height),
    canvas_width: num(o.canvas_width, D.canvas_width),
    canvas_height: num(o.canvas_height, D.canvas_height),
    gap_x: num(o.gap_x, D.gap_x),
    gap_y: num(o.gap_y, D.gap_y),
    padding: num(o.padding, D.padding),
    fit: pick(o.fit, ["cover", "contain"], D.fit),
    border_width: num(o.border_width, D.border_width),
    border_mode: pick(o.border_mode, ["solid", "gradient", "auto"], D.border_mode),
    border_color: str(o.border_color, D.border_color),
    border_color2: str(o.border_color2, D.border_color2),
    border_angle: num(o.border_angle, D.border_angle),
    border_radius: num(o.border_radius, D.border_radius),
    shadow: o.shadow !== false,
    shadow_mode: pick(o.shadow_mode, ["drop", "glow"], D.shadow_mode),
    shadow_blur: num(o.shadow_blur, D.shadow_blur),
    shadow_offset_x: num(o.shadow_offset_x, D.shadow_offset_x),
    shadow_offset_y: num(o.shadow_offset_y, D.shadow_offset_y),
    shadow_color: str(o.shadow_color, D.shadow_color),
    shadow_opacity: num(o.shadow_opacity, D.shadow_opacity),
    shadow_match: o.shadow_match === true,
  };
}
