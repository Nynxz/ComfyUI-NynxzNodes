// A reference sheet for a mixin: a new layout in the same mode, with full-body turnaround views and
// expression close-ups, every figure using the mixin. Generated like any layout, its output becomes
// the mixin's Look, so every box that uses it reads one consistent design.

import { nodeId } from "@/framework";
import { uid } from "@/lib/frescoScene";

import type { Mixin, PromptBox, PromptScene } from "./promptScene";
import { BOX_COLORS, readBox, readPromptScene } from "./promptScene";

const WIDTH = 1536;
const HEIGHT = 1024;

/** `[name, x, y, w, h, what the box shows]`, on a 1536x1024 sheet. */
const VIEWS: [string, number, number, number, number, string][] = [
  [
    "Front",
    0.03,
    0.05,
    0.19,
    0.9,
    "Full body, front view, standing straight, arms relaxed at the sides, neutral expression.",
  ],
  [
    "Side",
    0.245,
    0.05,
    0.19,
    0.9,
    "Full body, side view in profile facing left, standing straight, neutral expression.",
  ],
  [
    "Back",
    0.46,
    0.05,
    0.19,
    0.9,
    "Full body, back view, standing straight, arms relaxed at the sides.",
  ],
  [
    "Smile",
    0.68,
    0.05,
    0.29,
    0.28,
    "Head and shoulders close-up, facing the viewer, smiling warmly.",
  ],
  [
    "Surprise",
    0.68,
    0.36,
    0.29,
    0.28,
    "Head and shoulders close-up, facing the viewer, surprised, eyes wide, mouth open.",
  ],
  [
    "Anger",
    0.68,
    0.67,
    0.29,
    0.28,
    "Head and shoulders close-up, facing the viewer, angry, frowning.",
  ],
];

/** The sheet for `mixin`, from the layout it is in: same mode and caption (its style), the mixin
 *  carried over, a plain background at the bottom and every view using the mixin. */
export function referenceSheet(from: PromptScene, mixin: Mixin): PromptScene {
  const background: PromptBox = readBox(
    {
      id: uid("b"),
      name: "Background",
      color: BOX_COLORS[0],
      x: 0,
      y: 0,
      w: 1,
      h: 1,
      desc: "A plain, even light grey background, nothing else.",
    },
    0,
  );
  const views = VIEWS.map(([name, x, y, w, h, desc], i) =>
    readBox(
      {
        id: uid("b"),
        name,
        color: BOX_COLORS[(i + 1) % BOX_COLORS.length],
        x,
        y,
        w,
        h,
        kind: from.mode === "moss" ? "character" : "object",
        desc,
        mixins: [mixin.id],
      },
      i + 1,
    ),
  );
  return {
    ...readPromptScene({}),
    mode: from.mode,
    width: WIDTH,
    height: HEIGHT,
    prompt: from.prompt,
    style: from.style,
    meta: ["character reference sheet", "turnaround"],
    canvas: "collage",
    palette: [],
    mixins: [{ ...mixin }],
    boxes: [background, ...views],
  };
}

interface GraphNodeLike {
  pos: [number, number];
  size: [number, number];
  widgets?: { name: string; value: unknown; callback?: (value: unknown) => void }[];
}
interface GraphLike {
  add: (node: GraphNodeLike) => void;
  setDirtyCanvas?: (a: boolean, b: boolean) => void;
}
interface LiteGraphLike {
  createNode: (type: string) => GraphNodeLike | null;
}

/** A new Layout node holding `scene`, placed to the right of `beside`; null if it can't be made. */
export function addLayoutNode(
  scene: PromptScene,
  beside: { pos?: [number, number]; size?: [number, number]; graph?: unknown },
): GraphNodeLike | null {
  const liteGraph = (window as unknown as { LiteGraph?: LiteGraphLike }).LiteGraph;
  const graph = beside.graph as GraphLike | undefined;
  const node = liteGraph?.createNode(nodeId("Layout"));
  if (!node || !graph) return null;
  const [x, y] = beside.pos ?? [0, 0];
  const [width] = beside.size ?? [400, 0];
  node.pos = [x + width + 60, y];
  graph.add(node);
  const widget = node.widgets?.find((w) => w.name === "layout") ?? node.widgets?.[0];
  if (widget) {
    widget.value = scene;
    widget.callback?.(scene);
  }
  graph.setDirtyCanvas?.(true, true);
  return node;
}
