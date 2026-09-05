<template>
  <div class="fi">
    <div v-if="!payload || !payload.num_sources" class="fi-empty">
      <i class="mdi mdi-chart-scatter-plot" />
      <span>Run the fusion to inspect it</span>
      <small>wire the encode node's <b>fusion_inspect</b> output here, then queue</small>
    </div>

    <template v-else>
      <div class="fi-bar" @pointerdown.stop>
        <ZenToggleGroup
          :model-value="view"
          :options="VIEW_OPTS"
          @update:model-value="setView($event)"
        />
      </div>

      <div class="fi-main">
        <div ref="wrapEl" class="fi-gridwrap">
          <canvas
            ref="canvasEl"
            class="fi-canvas"
            :style="{ width: cssW + 'px', height: cssH + 'px' }"
            @pointermove="onHover"
            @pointerleave="hovered = -1"
            @pointerdown.stop
          />
          <div v-if="view === 'contest'" class="fi-caption">
            bright = contested (strength · feather · blend act here) · dark = one source owns it
          </div>
        </div>

        <div class="fi-side" @pointerdown.stop>
          <div class="fi-side-tabs">
            <button
              class="fi-tab"
              :class="{ on: sideTab === 'legend' }"
              @click="sideTab = 'legend'"
            >
              Sources
            </button>
            <button
              class="fi-tab"
              :class="{ on: sideTab === 'settings' }"
              @click="sideTab = 'settings'"
            >
              Settings
            </button>
          </div>

          <!-- SETTINGS tab: the full fusion recipe, grouped (dim = off/none/0) -->
          <div v-if="sideTab === 'settings'" class="fi-settings">
            <div v-for="g in settingGroups" :key="g.title" class="fi-sgroup">
              <div class="fi-sgroup-h">{{ g.title }}</div>
              <div v-for="r in g.rows" :key="r.k" class="fi-srow" :class="{ dim: r.dim }">
                <span class="fi-sk">{{ r.k }}</span
                ><span class="fi-svv">{{ r.v }}</span>
              </div>
            </div>
          </div>

          <!-- SOURCES tab: hover breakdown, else the source legend -->
          <template v-else>
            <template v-if="hovered >= 0">
              <div class="fi-side-head">Cell {{ cellRow }},{{ cellCol }}</div>
              <div class="fi-bars">
                <div v-for="r in cellBreakdown" :key="r.i" class="fi-brow">
                  <span class="fi-thumb sm" :style="{ borderColor: r.color }">
                    <img v-if="r.thumb" :src="r.thumb" alt="" />
                    <span v-else class="fi-thumb-fill" :style="{ background: r.color }" />
                  </span>
                  <span class="fi-blabel" :title="r.label">{{ r.label }}</span>
                  <span class="fi-btrack"
                    ><span class="fi-bfill" :style="{ width: r.pct + '%', background: r.color }"
                  /></span>
                  <span class="fi-bval">{{ r.w.toFixed(2) }}</span>
                </div>
              </div>
            </template>
            <template v-else>
              <div class="fi-legend">
                <button
                  v-for="(s, i) in payload.sources"
                  :key="i"
                  class="fi-lrow"
                  :class="{ on: view === 'isolate' && selected === i }"
                  :title="`Isolate ${s.label}`"
                  @click="isolate(i)"
                >
                  <span class="fi-thumb" :style="{ borderColor: s.color }">
                    <img v-if="s.thumb" :src="s.thumb" alt="" />
                    <span v-else class="fi-thumb-fill" :style="{ background: s.color }" />
                  </span>
                  <span class="fi-blabel" :title="s.label">{{ s.label }}</span>
                  <span class="fi-lstr" title="strength you set on the slider"
                    >×{{ s.strength.toFixed(1) }}</span
                  >
                  <span class="fi-lmeta" title="this source's share of the grid"
                    >{{ Math.round(s.share * 100) }}%</span
                  >
                </button>
              </div>
              <p class="fi-lhint"><b>×</b> strength you set · <b>%</b> its share of the grid</p>
            </template>
          </template>
        </div>
      </div>
    </template>
  </div>
</template>

<script lang="ts">
import { nodeId, type NodeDef, type WidgetOptions } from "@/framework";

/**
 * Display-only, and `serialize: false` is the load-bearing part: the blend payload describes one
 * RUN, is stale the moment the graph reopens, and carries a full weight field per token — saving
 * it would put megabytes of dead numbers into every workflow containing this node.
 */
export const widgetOptions: WidgetOptions = {
  minHeight: 260,
  fill: true,
  serialize: false,
  default: {},
};

export const nodeDef: Omit<NodeDef, "widgets"> = {
  is: nodeId("Fusion.Inspector"),
  minSize: [460, 380],
  hideOutputImages: true,
  // On execute, map the node's ui `fusion_inspect` payload onto the inspector widget. ComfyUI
  // wraps every ui value in a list, so unwrap [0] back to the payload dict. The component also
  // listens to `executed` itself and apply() is idempotent, so both paths firing is harmless.
  output: {
    widget: "inspector",
    from: (o) => (Array.isArray(o.fusion_inspect) ? o.fusion_inspect[0] : o.fusion_inspect),
  },
};
</script>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { ZenToggleGroup } from "@nynxz/zenkit-ui";
import { api } from "@comfy/api";

interface Source {
  label: string;
  color: string;
  strength: number;
  fit: string;
  share: number;
  thumb: string;
}
interface Payload {
  grid: [number, number];
  num_sources: number;
  sources: Source[];
  weights: number[][];
  settings: Record<string, unknown>;
}
interface DOMWidget {
  value: unknown;
  callback?: (v: unknown) => void;
}
const props = defineProps<{ widget?: DOMWidget; node?: unknown }>();

type View = "dominant" | "per-source" | "contest" | "blend" | "isolate";
const VIEWS: View[] = ["dominant", "per-source", "contest", "blend", "isolate"];
const VIEW_OPTS = [
  {
    value: "dominant",
    label: "Dominant",
    title: "Winner colour per cell, dimmed by how contested it is",
  },
  {
    value: "per-source",
    label: "Per-source",
    title: "One small panel per source — that source’s weight as a ramp, nothing mixed",
  },
  {
    value: "contest",
    label: "Contest",
    title:
      "Where the blend is contested — the ONLY cells where strength / feather / blend have any effect (bright = contested, dark = one source owns it)",
  },
  { value: "blend", label: "Blend", title: "Weighted average of all source colours" },
  {
    value: "isolate",
    label: "Isolate",
    title: "One source’s weight as a ramp (pick it in the legend)",
  },
];
type SideTab = "legend" | "settings";

function parse(v: unknown): Payload | null {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const o = v as any;
  if (!o || typeof o !== "object" || !Array.isArray(o.weights) || !o.num_sources) return null;
  const g = Array.isArray(o.grid) ? o.grid : [0, 0];
  return {
    grid: [Number(g[0]) || 0, Number(g[1]) || 0],
    num_sources: Number(o.num_sources) || 0,
    sources: Array.isArray(o.sources) ? o.sources : [],
    weights: o.weights,
    settings: o.settings && typeof o.settings === "object" ? o.settings : {},
  };
}

const payload = ref<Payload | null>(parse(props.widget?.value));
const view = ref<View>("dominant");
const sideTab = ref<SideTab>("legend");
const selected = ref(0);
const hovered = ref(-1);

const wrapEl = ref<HTMLElement | null>(null);
const canvasEl = ref<HTMLCanvasElement | null>(null);
const cssW = ref(0);
const cssH = ref(0);

const gridH = computed(() => payload.value?.grid[0] ?? 0);
const gridW = computed(() => payload.value?.grid[1] ?? 0);
const cellRow = computed(() => (gridW.value ? Math.floor(hovered.value / gridW.value) : 0));
const cellCol = computed(() => (gridW.value ? hovered.value % gridW.value : 0));

function setView(v: unknown) {
  view.value = (VIEWS.includes(v as View) ? v : "dominant") as View;
  if (view.value === "per-source") hovered.value = -1;
}
function isolate(i: number) {
  selected.value = i;
  view.value = "isolate";
  sideTab.value = "legend";
}

// the full fusion recipe, grouped — dim rows that are off/none/0 so what's ACTIVE stands out
const settingGroups = computed(() => {
  const p = payload.value;
  if (!p) return [];
  const s = p.settings;
  const num = (k: string, d = 0) => (Number.isFinite(Number(s[k])) ? Number(s[k]) : d);
  const str = (k: string, d = "") => (s[k] == null ? d : String(s[k]));
  const method = str("fusion_method", "spatial-checkerboard");
  const cMode = str("content_mode", "none");
  const yMode = str("style_mode", "none");
  const f2 = (k: string) => num(k).toFixed(2);
  return [
    {
      title: "Pattern",
      rows: [
        { k: "method", v: method.replace("spatial-", "") },
        { k: "block", v: String(num("block_size")), dim: method !== "spatial-block-interleave" },
        { k: "dither", v: f2("dither_ratio"), dim: method !== "spatial-dither-random" },
        { k: "jitter", v: f2("pattern_jitter"), dim: num("pattern_jitter") <= 0 },
      ],
    },
    {
      title: "Blend",
      rows: [
        { k: "blend", v: f2("blend_strength"), dim: num("blend_strength") <= 0 },
        { k: "feather", v: f2("feather"), dim: num("feather") <= 0 },
        { k: "norm", v: s.preserve_norm ? "on" : "off", dim: !s.preserve_norm },
      ],
    },
    {
      title: "Content",
      rows: [
        { k: "mode", v: cMode, dim: cMode === "none" },
        {
          k: "strength",
          v: f2("content_strength"),
          dim: cMode === "none" || num("content_strength") <= 0,
        },
      ],
    },
    {
      title: "Style",
      rows: [
        { k: "mode", v: yMode, dim: yMode === "none" },
        {
          k: "strength",
          v: f2("style_strength"),
          dim: yMode === "none" || num("style_strength") <= 0,
        },
      ],
    },
    {
      title: "Regions",
      rows: [{ k: "strength", v: f2("region_strength"), dim: num("region_strength") <= 0 }],
    },
    {
      title: "Variation",
      rows: [
        { k: "seed", v: String(num("seed")) },
        { k: "roll", v: f2("strength_roll"), dim: num("strength_roll") <= 0 },
      ],
    },
    {
      title: "Grid",
      rows: [
        { k: "size", v: String(num("visual_size")) },
        { k: "aspect", v: str("visual_aspect", "auto") },
        { k: "tokens", v: `${p.grid[1]}×${p.grid[0]}` },
        { k: "sources", v: String(p.num_sources) },
      ],
    },
  ];
});

// per-cell breakdown for the hovered token, sorted by weight desc
const cellBreakdown = computed(() => {
  const p = payload.value;
  if (!p || hovered.value < 0) return [];
  const rows = p.sources.map((s, i) => ({
    i,
    label: s.label,
    color: s.color,
    thumb: s.thumb,
    w: p.weights[i]?.[hovered.value] ?? 0,
  }));
  const max = Math.max(1e-6, ...rows.map((r) => r.w));
  return rows.map((r) => ({ ...r, pct: (r.w / max) * 100 })).sort((a, b) => b.w - a.w);
});

// --- colour mapping (mirrors _weightmap.py so the inspector and weight_map agree) ---
function hexRgb(hex: string): [number, number, number] {
  const s = hex.replace("#", "");
  return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
}
let srcRgb: [number, number, number][] = [];
function colorAt(token: number): [number, number, number] {
  const p = payload.value!;
  const n = p.num_sources;
  if (view.value === "isolate") {
    const amt = Math.min(1, Math.max(0, p.weights[selected.value]?.[token] ?? 0));
    const c = srcRgb[selected.value] ?? [0, 0, 0];
    return [c[0] * amt, c[1] * amt, c[2] * amt];
  }
  if (view.value === "blend") {
    let r = 0,
      g = 0,
      b = 0;
    for (let i = 0; i < n; i++) {
      const w = p.weights[i]?.[token] ?? 0;
      const c = srcRgb[i];
      r += w * c[0];
      g += w * c[1];
      b += w * c[2];
    }
    return [r, g, b];
  }
  if (view.value === "contest") {
    // 1 - winner's share = how contested the cell is. This is the ONLY place strength/feather
    // act: a cell one source owns outright (contest 0) ignores strength entirely. dark → amber.
    let max = 0;
    for (let i = 0; i < n; i++) {
      const w = p.weights[i]?.[token] ?? 0;
      if (w > max) max = w;
    }
    const t = Math.pow(Math.min(1, Math.max(0, 1 - max)), 0.65);
    return [20 + (251 - 20) * t, 20 + (191 - 20) * t, 24 + (36 - 24) * t];
  }
  // dominant
  let owner = 0,
    share = 0;
  for (let i = 0; i < n; i++) {
    const w = p.weights[i]?.[token] ?? 0;
    if (w > share) {
      share = w;
      owner = i;
    }
  }
  const conf = 0.22 + 0.78 * Math.min(1, Math.max(0, share));
  const c = srcRgb[owner] ?? [0, 0, 0];
  return [c[0] * conf, c[1] * conf, c[2] * conf];
}

function fit() {
  const wrap = wrapEl.value;
  const p = payload.value;
  if (!wrap || !p || !gridW.value || !gridH.value) return;
  const availW = wrap.clientWidth;
  const availH = wrap.clientHeight;
  if (availW <= 0 || availH <= 0) return;
  if (view.value === "per-source") {
    // small-multiples fill the whole area; panels are tiled inside
    cssW.value = availW;
    cssH.value = availH;
    return;
  }
  const cell = Math.max(3, Math.floor(Math.min(availW / gridW.value, availH / gridH.value)));
  cssW.value = cell * gridW.value;
  cssH.value = cell * gridH.value;
}

function draw() {
  const p = payload.value;
  const canvas = canvasEl.value;
  if (!p || !canvas || !cssW.value || !cssH.value) return;
  srcRgb = p.sources.map((s) => hexRgb(s.color));
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW.value * dpr);
  canvas.height = Math.round(cssH.value * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW.value, cssH.value);
  if (view.value === "per-source") {
    drawPanels(ctx);
    return;
  }
  const w = gridW.value;
  const h = gridH.value;
  const cw = cssW.value / w;
  const ch = cssH.value / h;
  for (let row = 0; row < h; row++) {
    for (let col = 0; col < w; col++) {
      const [r, g, b] = colorAt(row * w + col);
      ctx.fillStyle = `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
      ctx.fillRect(col * cw, row * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }
  if (cw >= 10) {
    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.lineWidth = 1;
    for (let gx = 1; gx < w; gx++) {
      ctx.beginPath();
      ctx.moveTo(gx * cw, 0);
      ctx.lineTo(gx * cw, cssH.value);
      ctx.stroke();
    }
    for (let gy = 1; gy < h; gy++) {
      ctx.beginPath();
      ctx.moveTo(0, gy * ch);
      ctx.lineTo(cssW.value, gy * ch);
      ctx.stroke();
    }
  }
  if (hovered.value >= 0) {
    const col = hovered.value % w;
    const row = Math.floor(hovered.value / w);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.strokeRect(col * cw + 1, row * ch + 1, cw - 2, ch - 2);
  }
}

function trunc(label: string, widthPx: number): string {
  const max = Math.max(2, Math.floor(widthPx / 6));
  return label.length > max ? label.slice(0, max - 1) + "…" : label;
}
/** Pick the columns×rows split whose per-panel image (at the grid's true aspect) comes out
 *  largest — so panels never have to stretch to fill an awkwardly-shaped cell. */
function panelLayout(
  n: number,
  availW: number,
  availH: number,
  aspect: number,
  pad: number,
  titleH: number,
) {
  let best = {
    cols: Math.max(1, Math.ceil(Math.sqrt(n))),
    rows: Math.ceil(n / Math.max(1, Math.ceil(Math.sqrt(n)))),
    area: -1,
  };
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols);
    const cellW = (availW - pad * (cols + 1)) / cols;
    const areaH = (availH - pad * (rows + 1)) / rows - titleH;
    if (cellW <= 0 || areaH <= 0) continue;
    let dw = cellW;
    let dh = dw / aspect;
    if (dh > areaH) {
      dh = areaH;
      dw = dh * aspect;
    }
    const area = dw * dh;
    if (area > best.area) best = { cols, rows, area };
  }
  return best;
}

/** Small multiples: one panel per source, its own weight as a black→colour ramp (nothing mixed).
 *  Each panel's grid is drawn at the token grid's true aspect ratio, centred in its cell — no
 *  stretching (every other view already preserves aspect via the min() in fit()). */
function drawPanels(ctx: CanvasRenderingContext2D) {
  const p = payload.value;
  if (!p) return;
  const n = p.num_sources;
  const w = gridW.value;
  const h = gridH.value;
  const aspect = w / h;
  const pad = 6;
  const titleH = 13;
  const { cols, rows } = panelLayout(n, cssW.value, cssH.value, aspect, pad, titleH);
  const cellW = (cssW.value - pad * (cols + 1)) / cols;
  const cellH = (cssH.value - pad * (rows + 1)) / rows;
  const areaH = Math.max(1, cellH - titleH);
  ctx.font = "600 10px sans-serif";
  ctx.textBaseline = "alphabetic";
  for (let s = 0; s < n; s++) {
    const cx = pad + (s % cols) * (cellW + pad);
    const cy = pad + Math.floor(s / cols) * (cellH + pad);
    // fit the token grid (aspect w:h) inside the cell's image area, then centre it horizontally
    let dw = cellW;
    let dh = dw / aspect;
    if (dh > areaH) {
      dh = areaH;
      dw = dh * aspect;
    }
    const dx = cx + (cellW - dw) / 2;
    const dy = cy + titleH;
    ctx.fillStyle = p.sources[s]?.color ?? "#888";
    ctx.fillText(trunc(p.sources[s]?.label ?? `src ${s + 1}`, dw), dx + 1, cy + 10);
    const cw = dw / w;
    const ch = dh / h;
    const c = srcRgb[s] ?? [0, 0, 0];
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        const amt = Math.min(1, Math.max(0, p.weights[s]?.[row * w + col] ?? 0));
        ctx.fillStyle = `rgb(${Math.round(c[0] * amt)},${Math.round(c[1] * amt)},${Math.round(c[2] * amt)})`;
        ctx.fillRect(dx + col * cw, dy + row * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }
    ctx.strokeStyle = "rgba(255,255,255,0.14)";
    ctx.lineWidth = 1;
    ctx.strokeRect(dx + 0.5, dy + 0.5, dw - 1, dh - 1);
  }
}

function onHover(e: PointerEvent) {
  const canvas = canvasEl.value;
  const p = payload.value;
  if (!canvas || !p || view.value === "per-source") return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  const col = Math.min(
    gridW.value - 1,
    Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * gridW.value)),
  );
  const row = Math.min(
    gridH.value - 1,
    Math.max(0, Math.floor(((e.clientY - rect.top) / rect.height) * gridH.value)),
  );
  hovered.value = row * gridW.value + col;
}

/** Set the payload from whatever the host hands us (unwrapping ComfyUI's ui list). */
function apply(v: unknown) {
  const p = Array.isArray(v) ? v[0] : v;
  payload.value = parse(p);
  if (selected.value >= (payload.value?.num_sources ?? 0)) selected.value = 0;
}

let ro: ResizeObserver | undefined;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let onExec: ((e: any) => void) | undefined;
onMounted(() => {
  // Two ways the blend payload can arrive — use whichever the host provides:
  //  1) the NodeDef `output` seam sets widget.value + fires this callback (value already unwrapped);
  //  2) a direct listen to ComfyUI's `executed` event (the proven refSync/previewSync pattern).
  // Both call apply(); it's idempotent, so a host that fires both is harmless.
  if (props.widget) {
    // eslint-disable-next-line vue/no-mutating-props
    props.widget.callback = (v) => apply(v);
  }
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    onExec = (e: any) => {
      const d = e?.detail;
      if (!d || String(d.node) !== String((props.node as { id?: unknown })?.id)) return;
      if (d.output && "fusion_inspect" in d.output) apply(d.output.fusion_inspect);
    };
    api.addEventListener("executed", onExec);
  } catch {
    /* api unavailable */
  }
  fit();
  draw();
  if (wrapEl.value && typeof ResizeObserver !== "undefined") {
    ro = new ResizeObserver(() => {
      fit();
      draw();
    });
    ro.observe(wrapEl.value);
  }
  [0, 80, 300].forEach((t) =>
    window.setTimeout(() => {
      fit();
      draw();
    }, t),
  );
});
watch([payload, view, selected, hovered], () => {
  fit();
  draw();
});
onBeforeUnmount(() => {
  ro?.disconnect();
  if (onExec) {
    try {
      api.removeEventListener("executed", onExec);
    } catch {
      /* ignore */
    }
  }
});
</script>

<style scoped>
.fi {
  display: flex;
  flex-direction: column;
  gap: 6px;
  height: 100%;
  min-height: 0;
  padding: 2px;
  box-sizing: border-box;
  font-size: 12px;
  color: var(--zen-text, #e5e5ea);
  /* same radius hierarchy as Fusion Studio (panels vs controls). */
  --fs-r: var(--zen-radius, 8px);
  --fs-rs: var(--radius-sm, 5px);
}
.fi-empty {
  height: 100%;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 5px;
  color: var(--zen-muted, #9aa0aa);
  text-align: center;
}
.fi-empty > .mdi {
  font-size: 30px;
  opacity: 0.8;
}
.fi-empty small {
  font-size: 10px;
  opacity: 0.75;
}
.fi-bar {
  flex: none;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}
.fi-main {
  flex: 1 1 auto;
  min-height: 0;
  display: flex;
  gap: 6px;
}
.fi-gridwrap {
  position: relative;
  flex: 1 1 auto;
  min-width: 0;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 8px);
  background: #0e0e12;
}
.fi-caption {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  padding: 3px 6px;
  font-size: 9px;
  line-height: 1.25;
  text-align: center;
  color: var(--zen-text, #e5e5ea);
  background: linear-gradient(transparent, rgba(0, 0, 0, 0.72));
  pointer-events: none;
}
.fi-canvas {
  display: block;
  image-rendering: pixelated;
}
.fi-side {
  flex: none;
  width: 190px;
  display: flex;
  flex-direction: column;
  min-height: 0;
  border: 1px solid var(--zen-border, #34343c);
  border-radius: var(--zen-radius, 8px);
  background: color-mix(in srgb, var(--zen-text, #fff) 2%, transparent);
  overflow: hidden;
}
.fi-side-head {
  flex: none;
  padding: 5px 8px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--zen-muted, #9aa0aa);
  border-bottom: 1px solid var(--zen-border, #34343c);
}
.fi-bars,
.fi-legend {
  flex: 1 1 0;
  min-height: 0;
  overflow-y: auto;
  padding: 5px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.fi-brow,
.fi-lrow {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 10.5px;
}
.fi-lrow {
  padding: 3px;
  border: 1px solid transparent;
  border-radius: var(--fs-rs);
  background: none;
  color: inherit;
  font: inherit;
  cursor: pointer;
  text-align: left;
}
.fi-lrow:hover {
  border-color: color-mix(in srgb, var(--zen-accent, #6366f1) 50%, transparent);
}
.fi-lrow.on {
  border-color: var(--zen-accent, #6366f1);
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 16%, transparent);
}
.fi-thumb {
  flex: none;
  width: 24px;
  height: 24px;
  border-radius: 4px;
  border: 1.5px solid;
  overflow: hidden;
  background: var(--zen-surface, #202026);
  display: flex;
  align-items: center;
  justify-content: center;
}
.fi-thumb.sm {
  width: 17px;
  height: 17px;
  border-radius: 3px;
}
.fi-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.fi-thumb-fill {
  width: 100%;
  height: 100%;
}
.fi-blabel {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fi-btrack {
  flex: 1.4;
  height: 6px;
  border-radius: 3px;
  background: color-mix(in srgb, var(--zen-text, #fff) 10%, transparent);
  overflow: hidden;
}
.fi-bfill {
  display: block;
  height: 100%;
  border-radius: 3px;
}
.fi-bval {
  flex: none;
  width: 26px;
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--zen-muted, #9aa0aa);
}
.fi-lstr {
  flex: none;
  font-variant-numeric: tabular-nums;
  font-size: 10px;
  color: var(--zen-text, #e5e5ea);
}
.fi-lmeta {
  flex: none;
  min-width: 30px;
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--zen-muted, #9aa0aa);
}
.fi-lhint {
  flex: none;
  margin: 0;
  padding: 5px 7px;
  font-size: 9.5px;
  line-height: 1.3;
  color: var(--zen-muted, #9aa0aa);
  border-top: 1px solid var(--zen-border, #34343c);
}
.fi-lhint b {
  color: var(--zen-text, #e5e5ea);
}

/* side-panel tabs */
.fi-side-tabs {
  flex: none;
  display: flex;
  gap: 3px;
  padding: 4px;
  border-bottom: 1px solid var(--zen-border, #34343c);
}
.fi-tab {
  flex: 1;
  padding: 3px 6px;
  border: 1px solid transparent;
  border-radius: var(--zen-radius, 5px);
  background: none;
  color: var(--zen-muted, #9aa0aa);
  font: inherit;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  cursor: pointer;
}
.fi-tab:hover {
  color: var(--zen-text, #e5e5ea);
}
.fi-tab.on {
  color: var(--zen-text, #e5e5ea);
  background: color-mix(in srgb, var(--zen-accent, #6366f1) 16%, transparent);
}

/* settings tab */
.fi-settings {
  flex: 1 1 0;
  min-height: 0;
  overflow-y: auto;
  padding: 4px 6px 6px;
}
.fi-sgroup {
  margin-top: 5px;
}
.fi-sgroup-h {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--zen-accent, #6366f1);
  margin-bottom: 2px;
}
.fi-srow {
  display: flex;
  justify-content: space-between;
  gap: 6px;
  font-size: 10.5px;
  padding: 1px 0;
}
.fi-srow.dim {
  opacity: 0.4;
}
.fi-sk {
  color: var(--zen-muted, #9aa0aa);
}
.fi-svv {
  color: var(--zen-text, #e5e5ea);
  font-variant-numeric: tabular-nums;
  text-align: right;
}
</style>
