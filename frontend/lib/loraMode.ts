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
const SELECT = "select";

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
  properties?: Record<string, unknown>;
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
  keepLast(node, "select");
}

/** Put the input called `name` back below the model/clip sockets, which were just re-added after
 *  it. Only while nothing is wired to it: a linked input stays where its link points. */
function keepLast(node: LoraNodeLike, name: string) {
  const at = node.inputs?.findIndex((s) => s.name === name) ?? -1;
  if (at < 0 || at === node.inputs!.length - 1) return;
  const old = node.inputs![at]!;
  if (old.link != null) return;
  node.removeInput(at);
  node.addInput(name, old.type);
  // The re-added slot is a bare one: give it back what the definition had put on it (its
  // tooltip, its label, that it is an input only).
  Object.assign(node.inputs![node.inputs!.length - 1]!, { ...old, link: null });
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
 *  out in its definition's order (model, clip, select) while each link keeps the slot index it
 *  was saved at, so a CLIP-only node's clip link sits under a socket called "model" and its
 *  select link under one called "clip". The links are already where the saved layout put them;
 *  only the names are wrong, so rename in place and drop the unlinked extras.
 *
 *  Inputs are named from `saved`, the node as it was stored, which says what each link was wired
 *  to. Outputs have no select among them, so their order alone tells. */
function relabelSlots(node: LoraNodeLike, want: Slot[], saved: unknown) {
  const savedName = new Map<number, string>();
  for (const s of (saved as { inputs?: LGSlot[] } | undefined)?.inputs ?? []) {
    if (s?.link != null && (isSocket(s) || s.name === SELECT)) savedName.set(s.link, s.name);
  }
  const ours = (s: LGSlot) => isSocket(s) || s.name === SELECT;
  const linkedAs = new Set<string>();
  for (const slot of node.inputs?.filter(ours) ?? []) {
    if (slot.link == null) continue;
    const name = savedName.get(slot.link);
    if (name && name !== slot.name) {
      slot.name = name;
      slot.type = name === SELECT ? "STRING" : TYPES[name as Slot];
      Object.assign(slot, { localized_name: name, label: undefined, tooltip: undefined });
    }
    linkedAs.add(slot.name);
  }
  // An unlinked socket that a linked one now stands for, or that this mode doesn't have, goes.
  for (let i = (node.inputs?.length ?? 0) - 1; i >= 0; i--) {
    const slot = node.inputs![i]!;
    if (!ours(slot) || slot.link != null) continue;
    const spare = linkedAs.has(slot.name) || (isSocket(slot) && !want.includes(slot.name));
    if (spare) node.removeInput(i);
  }
  for (const name of want) {
    if (!node.inputs?.some((s) => s.name === name)) node.addInput(name, TYPES[name]);
  }

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
    () => node.outputs,
    (s) => !!s.links?.length,
    (i) => node.removeOutput(i),
    (name) => node.addOutput(name, TYPES[name]),
  );
  node.setDirtyCanvas?.(true, true);
}

// The `select` input (choose a group's LoRAs from the graph) is off until asked for: most
// loaders never use it, and a socket nobody wired is a row of the node spent on nothing. Whether
// it is on is a node property; a graph saved with it wired keeps it either way.
const SELECT_KEY = "nynxzLoraSelect";

export const selectShown = (node: LoraNodeLike) => !!node.inputs?.some((s) => s.name === SELECT);

function syncSelect(node: LoraNodeLike) {
  const at = node.inputs?.findIndex((s) => s.name === SELECT) ?? -1;
  const linked = at >= 0 && node.inputs![at]!.link != null;
  const want = node.properties?.[SELECT_KEY] === true || linked;
  if (want && at < 0) node.addInput(SELECT, "STRING");
  else if (!want && at >= 0) node.removeInput(at);
  if (want) keepLast(node, SELECT);
  node.setDirtyCanvas?.(true, true);
}

/** Show or hide the `select` input. Hiding it drops whatever was wired to it. */
export function setSelectShown(node: LoraNodeLike, shown: boolean) {
  const at = node.inputs?.findIndex((s) => s.name === SELECT) ?? -1;
  node.graph?.beforeChange?.();
  if (!shown && at >= 0) node.removeInput(at);
  if (node.properties) node.properties[SELECT_KEY] = shown;
  syncSelect(node);
  node.graph?.afterChange?.();
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
    syncSlots(node, loraMode(node) ?? "both");
    syncSelect(node);
    const prev = node.onConfigure;
    node.onConfigure = function (this: unknown, ...args: unknown[]) {
      prev?.apply(this, args);
      const mode = loraMode(node) ?? "both";
      relabelSlots(node, SLOTS[mode], args[0]);
      syncSlots(node, mode);
      syncSelect(node);
    };
  },
} as never);
