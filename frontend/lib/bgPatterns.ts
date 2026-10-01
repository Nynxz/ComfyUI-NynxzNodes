// Inline-SVG previews for the pattern picker — these mirror the backend's procedural
// patterns so the swatch you click matches what gets rendered. The list mirrors PATTERNS in
// nodes/background/config.py; the motifs below are frontend-only, which is why there is no
// route for this.

export const PATTERNS = ["dots", "grid", "checker", "stripes", "diagonal", "crosshatch"];

const TILE = 16;
const MOTIF: Record<string, string> = {
  dots: `<circle cx='8' cy='8' r='2.4' fill='{c}'/>`,
  grid: `<path d='M0 0H16M0 0V16' stroke='{c}' stroke-width='1.4' fill='none'/>`,
  checker: `<rect width='8' height='8' fill='{c}'/><rect x='8' y='8' width='8' height='8' fill='{c}'/>`,
  stripes: `<rect width='2' height='16' fill='{c}'/>`,
  diagonal: `<path d='M0 16L16 0' stroke='{c}' stroke-width='2' fill='none'/>`,
  crosshatch: `<path d='M0 16L16 0M0 0L16 16' stroke='{c}' stroke-width='1.3' fill='none'/>`,
};

/** A `url(...)` CSS value tiling the pattern in `color`, for use as a background-image. */
export function patternCss(id: string, color = "#9aa0aa"): string {
  const motif = MOTIF[id];
  if (!motif) return "none";
  const svg =
    `<svg xmlns='http://www.w3.org/2000/svg' width='100%' height='100%'>` +
    `<defs><pattern id='p' width='${TILE}' height='${TILE}' patternUnits='userSpaceOnUse'>` +
    motif.replace(/\{c\}/g, color) +
    `</pattern></defs><rect width='100%' height='100%' fill='url(#p)'/></svg>`;
  return `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
}
