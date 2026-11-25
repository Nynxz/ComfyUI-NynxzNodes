// Small pure helpers for reading stored widget values and for making scene items.

/** `v` as a finite number, else `fallback`. */
export function num(v: unknown, fallback: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/** `v` when it is one of `options`, else `fallback`. */
export const pick = <T extends string>(v: unknown, options: T[], fallback: T): T =>
  (options as string[]).includes(v as string) ? (v as T) : fallback;

let seq = 0;
/** A fresh id, unique in this session and practically across sessions. */
export function uid(prefix = "l"): string {
  return `${prefix}${Date.now().toString(36)}${(seq++).toString(36)}`;
}

/** The first of `colors` no item uses yet; cycles once they all are. */
export function nextColor(items: { color: string }[], colors: readonly string[]): string {
  const used = new Set(items.map((i) => i.color));
  return colors.find((c) => !used.has(c)) ?? colors[items.length % colors.length]!;
}
