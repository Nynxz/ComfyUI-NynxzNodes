// Background Maker style contract (mirror of nodes/background/config.py). No canvas size —
// the spec is resolution-independent; the grid renders it at its own size.

export interface BgStyle {
  bg_mode: "solid" | "gradient" | "pattern" | "icons" | "transparent";
  bg_color: string;
  bg_color2: string;
  bg_angle: number;
  // procedural pattern
  pattern: string;
  pattern_color: string;
  pattern_scale: number;
  pattern_opacity: number;
  // icon grid (square, sized to the output)
  columns: number;
  scale: number;
  tint: boolean;
  tint_color: string;
  invert_alpha: boolean;
  rotate_jitter: number;
  scale_jitter: number;
  opacity: number;
}

export const DEFAULT_BG: BgStyle = {
  bg_mode: "solid",
  bg_color: "#16161c",
  bg_color2: "#262633",
  bg_angle: 45,
  pattern: "dots",
  pattern_color: "#ffffff",
  pattern_scale: 26,
  pattern_opacity: 0.1,
  columns: 12,
  scale: 0.7,
  tint: true,
  tint_color: "#ffffff",
  invert_alpha: false,
  rotate_jitter: 0,
  scale_jitter: 0,
  opacity: 0.16,
};

const num = (v: unknown, d: number) => (Number.isFinite(Number(v)) ? Number(v) : d);
const pick = <T extends string>(v: unknown, opts: T[], d: T): T =>
  (opts as string[]).includes(v as string) ? (v as T) : d;

// A colour from a workflow goes into CSS (`background`, a gradient): only a hex, rgb()/hsl() of
// numbers, or a colour name gets through, so a shared workflow can't slip in a url() that the
// browser would fetch.
const COLOR = /^(#[0-9a-f]{3,8}|(rgb|hsl)a?\([\d\s.,%/+-]*\)|[a-z]{3,24})$/i;
const color = (v: unknown, d: string) =>
  typeof v === "string" && COLOR.test(v.trim()) ? v.trim() : d;
const word = (v: unknown, d: string) =>
  typeof v === "string" && /^[a-z0-9_-]{1,40}$/i.test(v) ? v : d;

export function normalizeBg(v: unknown): BgStyle {
  const o = (v && typeof v === "object" ? v : {}) as Record<string, unknown>;
  const D = DEFAULT_BG;
  return {
    bg_mode: pick(o.bg_mode, ["solid", "gradient", "pattern", "icons", "transparent"], D.bg_mode),
    bg_color: color(o.bg_color, D.bg_color),
    bg_color2: color(o.bg_color2, D.bg_color2),
    bg_angle: num(o.bg_angle, D.bg_angle),
    pattern: word(o.pattern, D.pattern),
    pattern_color: color(o.pattern_color, D.pattern_color),
    pattern_scale: num(o.pattern_scale, D.pattern_scale),
    pattern_opacity: num(o.pattern_opacity, D.pattern_opacity),
    columns: num(o.columns, D.columns),
    scale: num(o.scale, D.scale),
    tint: o.tint !== false,
    tint_color: color(o.tint_color, D.tint_color),
    invert_alpha: o.invert_alpha === true,
    rotate_jitter: num(o.rotate_jitter, D.rotate_jitter),
    scale_jitter: num(o.scale_jitter, D.scale_jitter),
    opacity: num(o.opacity, D.opacity),
  };
}
