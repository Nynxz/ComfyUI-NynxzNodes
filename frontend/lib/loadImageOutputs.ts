// Load Image's optional outputs, and the sockets that follow them.
//
// Which outputs show after `image` is the node's `outputs` widget ("mask,width"), hidden and set
// from the header cog. Outputs are matched to links by POSITION, so the layout here and the tuple
// nodes/load_image/load_image.py returns must agree: `image`, then the switched-on outputs in
// OPTIONAL_OUTPUTS order. The backend declares them all as MASK,INT,STRING; each slot gets its real
// type here, so the graph only lets a MASK reach a MASK input.

import { app } from "@comfy/app";

export type OptionalOutput = "mask" | "width" | "height" | "filename";

export const OPTIONAL_OUTPUTS: { name: OptionalOutput; type: string; icon: string }[] = [
  { name: "mask", type: "MASK", icon: "mdi mdi-checkerboard" },
  { name: "width", type: "INT", icon: "mdi mdi-arrow-left-right" },
  { name: "height", type: "INT", icon: "mdi mdi-arrow-up-down" },
  { name: "filename", type: "STRING", icon: "mdi mdi-format-text" },
];

const LOAD_IMAGE = "nynxz.IO.LoadImage";
const OUTPUTS_WIDGET = "outputs";
const DEFAULT_OUTPUTS = "mask";
const IMAGE = { name: "image", type: "IMAGE" };

interface LGLink {
  target_id: number | string;
  target_slot: number;
}
interface LGSlot {
  name: string;
  type: string;
  label?: string;
  localized_name?: string;
  links?: number[] | null;
}
interface LGWidget {
  name: string;
  value: unknown;
  hidden?: boolean;
}
interface LGGraphLike {
  links?: Map<number, LGLink> | Record<number, LGLink>;
  getLink?(id: number): LGLink | undefined;
  getNodeById(id: number | string): unknown;
  beforeChange?(): void;
  afterChange?(): void;
}
export interface LoadImageNodeLike {
  outputs?: LGSlot[];
  widgets?: LGWidget[];
  graph?: LGGraphLike | null;
  addOutput(name: string, type: string): void;
  removeOutput(slot: number): void;
  connect(slot: number, target: unknown, targetSlot: number): unknown;
  setDirtyCanvas?(fg: boolean, bg: boolean): void;
}

const outputsWidget = (node: LoadImageNodeLike) =>
  node.widgets?.find((w) => w.name === OUTPUTS_WIDGET);

/** The optional outputs `node` shows, in slot order. */
export function shownOutputs(raw: unknown): OptionalOutput[] {
  const node = raw as LoadImageNodeLike;
  const value = outputsWidget(node)?.value;
  const wanted = new Set(String(typeof value === "string" ? value : DEFAULT_OUTPUTS).split(","));
  return OPTIONAL_OUTPUTS.filter((o) => wanted.has(o.name)).map((o) => o.name);
}

function layoutOf(shown: OptionalOutput[]) {
  return [IMAGE, ...OPTIONAL_OUTPUTS.filter((o) => shown.includes(o.name))];
}

function link(graph: LGGraphLike, id: number): LGLink | undefined {
  if (graph.getLink) return graph.getLink(id);
  const links = graph.links;
  return links instanceof Map ? links.get(id) : links?.[id];
}

/** Lay the outputs out for `shown`, keeping every link a kept output had. */
function syncOutputs(node: LoadImageNodeLike, shown: OptionalOutput[]) {
  const want = layoutOf(shown);
  const current = (node.outputs ?? []).map((s) => `${s.name}:${s.type}`).join(",");
  if (current === want.map((s) => `${s.name}:${s.type}`).join(",")) return;

  const graph = node.graph ?? null;
  const outgoing = new Map<string, LGLink[]>();
  if (graph) {
    for (const slot of node.outputs ?? []) {
      const links = (slot.links ?? []).map((id) => link(graph, id)).filter((l) => !!l);
      outgoing.set(
        slot.name,
        links.map((l) => ({ ...l })),
      );
    }
  }

  graph?.beforeChange?.();
  for (let i = (node.outputs?.length ?? 0) - 1; i >= 0; i--) node.removeOutput(i);
  for (const slot of want) node.addOutput(slot.name, slot.type);
  if (graph) {
    want.forEach((slot, index) => {
      for (const l of outgoing.get(slot.name) ?? []) {
        const target = graph.getNodeById(l.target_id);
        if (target) node.connect(index, target, l.target_slot);
      }
    });
  }
  graph?.afterChange?.();
  node.setDirtyCanvas?.(true, true);
}

/** Name the outputs for `shown` without moving any links. On load ComfyUI lays the outputs out
 *  in the definition's order (image, mask, width, height, filename) while each link keeps the slot
 *  index it was saved at, so only the names and types are wrong: rename in place and drop the
 *  unlinked extras from the end. */
function relabelOutputs(node: LoadImageNodeLike, shown: OptionalOutput[]) {
  const want = layoutOf(shown);
  const outputs = () => node.outputs ?? [];
  while (outputs().length > want.length && !outputs().at(-1)?.links?.length) {
    node.removeOutput(outputs().length - 1);
  }
  outputs().forEach((slot, index) => {
    const wanted = want[index];
    if (!wanted) return;
    slot.name = wanted.name;
    slot.type = wanted.type;
    slot.label = undefined;
    slot.localized_name = wanted.name;
  });
  for (const slot of want.slice(outputs().length)) node.addOutput(slot.name, slot.type);
}

/** Switch `output` on or off on `node`, and lay its sockets out to match. */
export function toggleOutput(raw: unknown, output: OptionalOutput) {
  const node = raw as LoadImageNodeLike;
  const widget = outputsWidget(node);
  if (!widget) return;
  const shown = shownOutputs(node);
  const next = shown.includes(output) ? shown.filter((o) => o !== output) : [...shown, output];
  const ordered = OPTIONAL_OUTPUTS.map((o) => o.name).filter((name) => next.includes(name));
  widget.value = ordered.join(",");
  syncOutputs(node, ordered);
}

app.registerExtension({
  name: `${LOAD_IMAGE}.Outputs`,
  nodeCreated(raw: unknown) {
    const node = raw as LoadImageNodeLike & {
      constructor: { comfyClass?: string };
      onConfigure?: (...args: unknown[]) => void;
    };
    if (node.constructor.comfyClass !== LOAD_IMAGE) return;
    const widget = outputsWidget(node);
    if (!widget) return;
    widget.hidden = true;
    syncOutputs(node, shownOutputs(node));
    const prev = node.onConfigure;
    node.onConfigure = function (this: unknown, ...args: unknown[]) {
      prev?.apply(this, args);
      const shown = shownOutputs(node);
      relabelOutputs(node, shown);
      syncOutputs(node, shown);
    };
  },
} as never);
