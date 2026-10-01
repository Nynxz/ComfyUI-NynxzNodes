// LoRA Loader's "apply to" setting, and the sockets that follow it.
//
// The mode is the node's `apply_to` widget, hidden and set from the stack widget's settings menu.
// Each mode shows only the inputs and outputs it uses, in the order nodes/lora/loader.py returns
// them — outputs are matched to links by position, so the layout here and the tuple there must
// agree. The backend declares both outputs as MODEL,CLIP; here each slot gets its real type.

import { app } from "@comfy/app";

import { nodeId } from "@/framework";

export type LoraMode = "model" | "clip" | "both";
type Slot = "model" | "clip";

export const LORA_MODES: { value: LoraMode; label: string; icon: string }[] = [
  { value: "model", label: "Model only", icon: "mdi mdi-cube-outline" },
  { value: "clip", label: "CLIP only", icon: "mdi mdi-format-text" },
  { value: "both", label: "Model + CLIP", icon: "mdi mdi-layers-outline" },
];

const SLOTS: Record<LoraMode, Slot[]> = {
  model: ["model"],
  clip: ["clip"],
  both: ["model", "clip"],
};
const TYPES: Record<Slot, string> = { model: "MODEL", clip: "CLIP" };
const MODE_WIDGET = "apply_to";

const LOADER_TYPE = nodeId("Lora.Loader");

interface LGLink {
  origin_id: number | string;
  origin_slot: number;
  target_id: number | string;
  target_slot: number;
}
interface LGSlot {
  name: string;
  type: string;
  link?: number | null;
  links?: number[] | null;
}
interface LGWidget {
  name: string;
  value: unknown;
  hidden?: boolean;
  callback?: (v: unknown) => void;
}
export interface LoraNodeLike {
  inputs?: LGSlot[];
  outputs?: LGSlot[];
  widgets?: LGWidget[];
  graph?: LGGraphLike | null;
  addInput(name: string, type: string): void;
  addOutput(name: string, type: string): void;
  removeInput(slot: number): void;
  removeOutput(slot: number): void;
  connect(slot: number, target: unknown, targetSlot: number): unknown;
  setDirtyCanvas?(fg: boolean, bg: boolean): void;
}
interface LGGraphLike {
  links?: Map<number, LGLink> | Record<number, LGLink>;
  getNodeById(id: number | string): LoraNodeLike | null;
  getLink?(id: number): LGLink | undefined;
  beforeChange?(): void;
  afterChange?(): void;
}

const modeWidget = (node: LoraNodeLike) => node.widgets?.find((w) => w.name === MODE_WIDGET);

export function loraMode(node: LoraNodeLike): LoraMode | null {
  const value = modeWidget(node)?.value;
  return value === "clip" || value === "both" || value === "model" ? value : null;
}

function link(graph: LGGraphLike, id: number): LGLink | undefined {
  if (graph.getLink) return graph.getLink(id);
  const links = graph.links;
  return links instanceof Map ? links.get(id) : links?.[id];
}

const layout = (slots: LGSlot[] | undefined) =>
  (slots ?? []).map((s) => `${s.name}:${s.type}`).join(",");

const isSocket = (s: LGSlot): s is LGSlot & { name: Slot } =>
  s.name === "model" || s.name === "clip";

function rebuild(node: LoraNodeLike, want: Slot[]) {
  for (let i = (node.inputs?.length ?? 0) - 1; i >= 0; i--)
    if (isSocket(node.inputs![i]!)) node.removeInput(i);
  for (let i = (node.outputs?.length ?? 0) - 1; i >= 0; i--) node.removeOutput(i);
  for (const name of want) node.addInput(name, TYPES[name]);
  for (const name of want) node.addOutput(name, TYPES[name]);
}

/** Lay the node's model/clip sockets out for `mode`, keeping every link a kept slot had. */
function syncSlots(node: LoraNodeLike, mode: LoraMode) {
  const want = SLOTS[mode];
  const wanted = want.map((name) => `${name}:${TYPES[name]}`).join(",");
  if (layout(node.inputs?.filter(isSocket)) === wanted && layout(node.outputs) === wanted) return;

  const graph = node.graph ?? null;
  const incoming = new Map<Slot, LGLink>();
  const outgoing = new Map<Slot, LGLink[]>();
  if (graph) {
    for (const s of node.inputs?.filter(isSocket) ?? []) {
      const l = s.link != null ? link(graph, s.link) : undefined;
      if (l) incoming.set(s.name, { ...l });
    }
    for (const s of node.outputs?.filter(isSocket) ?? []) {
      const ls = (s.links ?? []).map((id) => link(graph, id)).filter((l): l is LGLink => !!l);
      outgoing.set(
        s.name,
        ls.map((l) => ({ ...l })),
      );
    }
  }

  graph?.beforeChange?.();
  rebuild(node, want);
  if (graph) {
    for (const name of want) {
      const inLink = incoming.get(name);
      const origin = inLink && graph.getNodeById(inLink.origin_id);
      const inIndex = node.inputs!.findIndex((s) => s.name === name);
      if (origin && inIndex >= 0) origin.connect(inLink.origin_slot, node, inIndex);
      const outIndex = want.indexOf(name);
      for (const l of outgoing.get(name) ?? []) {
        const target = graph.getNodeById(l.target_id);
        if (target) node.connect(outIndex, target, l.target_slot);
      }
    }
  }
  graph?.afterChange?.();
  node.setDirtyCanvas?.(true, true);
}

/** Name the sockets for `want` without moving any links. On load ComfyUI lays a node's sockets
 *  out in its definition's order (model, clip) while each link keeps the slot index it was saved
 *  at, so a CLIP-only node's clip link sits under a socket called "model". The links are already
 *  where the saved layout put them; only the names are wrong, so rename in place and drop the
 *  unlinked extras from the end. */
function relabelSlots(node: LoraNodeLike, want: Slot[]) {
  const rename = (
    slots: () => LGSlot[] | undefined,
    linked: (s: LGSlot) => boolean,
    remove: (i: number) => void,
    add: (name: Slot) => void,
  ) => {
    const sockets = () =>
      (slots() ?? []).map((s, i) => [s, i] as const).filter(([s]) => isSocket(s));
    for (let current = sockets(); current.length > want.length; current = sockets()) {
      const spare = [...current].reverse().find(([s]) => !linked(s));
      if (!spare) break;
      remove(spare[1]);
    }
    sockets().forEach(([slot], k) => {
      const name = want[k];
      if (!name) return;
      slot.name = name;
      slot.type = TYPES[name];
      (slot as { localized_name?: string }).localized_name = name;
    });
    for (const name of want.slice(sockets().length)) add(name);
  };
  rename(
    () => node.inputs,
    (s) => s.link != null,
    (i) => node.removeInput(i),
    (name) => node.addInput(name, TYPES[name]),
  );
  rename(
    () => node.outputs,
    (s) => !!s.links?.length,
    (i) => node.removeOutput(i),
    (name) => node.addOutput(name, TYPES[name]),
  );
  node.setDirtyCanvas?.(true, true);
}

export function setLoraMode(node: LoraNodeLike, mode: LoraMode) {
  const widget = modeWidget(node);
  if (!widget) return;
  widget.value = mode;
  try {
    widget.callback?.(mode);
  } catch {
    /* no callback registered */
  }
  syncSlots(node, mode);
}

app.registerExtension({
  name: `${nodeId("Lora")}.Mode`,
  nodeCreated(raw: unknown) {
    const node = raw as LoraNodeLike & {
      constructor: { comfyClass?: string };
      onConfigure?: (...args: unknown[]) => void;
    };
    if (node.constructor.comfyClass !== LOADER_TYPE) return;
    const widget = modeWidget(node);
    if (!widget) return;
    widget.hidden = true;
    // A new node lays out its default mode now; a loaded one gets its saved socket names back
    // once configure has restored the mode and the links.
    syncSlots(node, loraMode(node) ?? "model");
    const prev = node.onConfigure;
    node.onConfigure = function (this: unknown, ...args: unknown[]) {
      prev?.apply(this, args);
      const mode = loraMode(node) ?? "model";
      relabelSlots(node, SLOTS[mode]);
      syncSlots(node, mode);
    };
  },
} as never);
