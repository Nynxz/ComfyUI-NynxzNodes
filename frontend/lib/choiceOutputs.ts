// Choice's optional outputs, and the sockets that follow them.
//
// Which outputs show is the node's `outputs` widget ("text,index"), hidden and set from the
// header cog. Any can be off, `text` included, as long as one is left. Outputs are matched to
// links by POSITION, so the layout here and the list nodes/text/choice.py returns must agree:
// the switched-on outputs, in OPTIONAL_OUTPUTS order. The backend declares them all as INT,STRING; each slot gets its real
// type here. The same arrangement as Load Image's (lib/loadImageOutputs.ts).

import { app } from "@comfy/app";

import { nodeId } from "@/framework";

export type ChoiceOutput = "text" | "index" | "count" | "options";

export const OPTIONAL_OUTPUTS: { name: ChoiceOutput; type: string; icon: string; hint: string }[] =
  [
    { name: "text", type: "STRING", icon: "mdi mdi-format-text", hint: "The chosen option" },
    {
      name: "index",
      type: "INT",
      icon: "mdi mdi-numeric",
      hint: "Place of the chosen option, from 0",
    },
    { name: "count", type: "INT", icon: "mdi mdi-counter", hint: "How many options there are" },
    {
      name: "options",
      type: "STRING",
      icon: "mdi mdi-format-list-bulleted",
      hint: "Every option, one per line",
    },
  ];

/** How `index` is given. Kept in the same setting as the outputs, and mirrored by choice.py. */
export type IndexFlag = "index_from_1" | "index_as_text";
export const INDEX_FLAGS: { name: IndexFlag; label: string; icon: string }[] = [
  { name: "index_from_1", label: "Count from 1", icon: "mdi mdi-numeric-1-box-outline" },
  { name: "index_as_text", label: "As text, not a number", icon: "mdi mdi-format-text" },
];

const CHOICE = nodeId("Text.Choice");
const OUTPUTS_WIDGET = "outputs";
/** What a node shows before anything is switched: also what an empty setting means. */
const DEFAULT_OUTPUTS: ChoiceOutput[] = ["text"];

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
  options?: { values?: unknown };
}
interface LGGraphLike {
  rootGraph?: LGGraphLike;
  subgraphs?: Map<unknown, LGGraphLike>;
  nodes?: ChoiceNodeLike[];
  /** A subgraph's inputs: each is a socket on the subgraph's node, linked to nodes inside. */
  inputNode?: { slots?: { name: string; linkIds?: number[] }[] };
  links?: Map<number, LGLink> | Record<number, LGLink>;
  getLink?(id: number): LGLink | undefined;
  getNodeById(id: number | string): unknown;
  beforeChange?(): void;
  afterChange?(): void;
}
interface ChoiceNodeLike {
  id?: number | string;
  inputs?: { name: string }[];
  subgraph?: LGGraphLike;
  properties?: Record<string, unknown>;
  outputs?: LGSlot[];
  widgets?: LGWidget[];
  graph?: LGGraphLike | null;
  addOutput(name: string, type: string): void;
  removeOutput(slot: number): void;
  connect(slot: number, target: unknown, targetSlot: number): unknown;
  setDirtyCanvas?(fg: boolean, bg: boolean): void;
}

/** Where a node keeps its options and whether its list is open, in node.properties. */
export const STATE_KEY = "nynxzChoice";

/** Each node's own options list, by its combo widget. */
const ownValues = new WeakMap<object, string[]>();

/** The options saved on `node`: the rows with something typed in them. */
function savedOptions(node: ChoiceNodeLike): string[] {
  const state = node.properties?.[STATE_KEY] as { options?: unknown } | undefined;
  const rows = Array.isArray(state?.options) ? state.options : [];
  return rows
    .map((o) => String((o as { text?: unknown } | null)?.text ?? ""))
    .filter((text) => text.trim());
}

/** Give `node`'s combo its options (`texts`, or the ones saved on the node), in a list that is
 *  this node's alone. ComfyUI builds every Choice node's combo from the one node definition, so
 *  they start out sharing a single options list: filling that one in place put the last node's
 *  options on all of them. The node's own list is made once and refilled after that, so whatever
 *  already holds it sees the new options.
 *
 *  This runs when the node is made and when it is loaded, not only from the list widget's body:
 *  a node inside a subgraph has no body until the subgraph is opened, and the copy of its combo
 *  promoted onto the subgraph's node takes its options when the workflow loads. */
function linkOf(graph: LGGraphLike, id: number): LGLink | undefined {
  if (graph.getLink) return graph.getLink(id);
  const links = graph.links;
  return links instanceof Map ? links.get(id) : links?.[id];
}

/** The copies of `node`'s `inputName` widget promoted onto subgraph nodes: on every node that is
 *  an instance of the subgraph `node` sits in, and on from there if that one is promoted again. */
function promotedCopies(node: ChoiceNodeLike, inputName: string, depth = 0): LGWidget[] {
  const graph = node.graph;
  const root = graph?.rootGraph;
  if (!graph || !root || graph === root || depth > 8) return [];
  const slot = node.inputs?.findIndex((i) => i.name === inputName) ?? -1;
  const names = (graph.inputNode?.slots ?? [])
    .filter((s) =>
      (s.linkIds ?? []).some((id) => {
        const l = linkOf(graph, id);
        return !!l && String(l.target_id) === String(node.id) && l.target_slot === slot;
      }),
    )
    .map((s) => s.name);
  if (!names.length) return [];
  const hosts = [root, ...(root.subgraphs?.values() ?? [])]
    .flatMap((g) => g.nodes ?? [])
    .filter((n) => n.subgraph === graph);
  return hosts.flatMap((host) =>
    names.flatMap((name) => {
      const copy = host.widgets?.find((w) => w.name === name);
      return [...(copy ? [copy] : []), ...promotedCopies(host, name, depth + 1)];
    }),
  );
}

export function offerOptions(raw: unknown, texts?: string[]) {
  const node = raw as ChoiceNodeLike;
  const combo = node.widgets?.find((w) => w.name === "choice");
  if (!combo) return;
  let own = ownValues.get(combo);
  if (!own) {
    own = [];
    ownValues.set(combo, own);
  }
  own.splice(0, own.length, ...(texts ?? savedOptions(node)));
  if (combo.options?.values !== own) combo.options = { ...combo.options, values: own };
  // A promoted copy holds its own copy of the options, taken when it was promoted or loaded,
  // so options changed inside the subgraph are handed to it here.
  for (const copy of promotedCopies(node, "choice")) {
    copy.options = { ...copy.options, values: [...own] };
  }
}

const outputsWidget = (node: ChoiceNodeLike) =>
  node.widgets?.find((w) => w.name === OUTPUTS_WIDGET);

const settingOf = (raw: unknown) => {
  const value = outputsWidget(raw as ChoiceNodeLike)?.value;
  return new Set(String(typeof value === "string" ? value : "").split(","));
};

/** The outputs `node` shows, in slot order. A node always has one. */
export function shownOutputs(raw: unknown): ChoiceOutput[] {
  const wanted = settingOf(raw);
  const shown = OPTIONAL_OUTPUTS.filter((o) => wanted.has(o.name)).map((o) => o.name);
  return shown.length ? shown : DEFAULT_OUTPUTS;
}

/** The index flags switched on for `node`. */
export function shownFlags(raw: unknown): IndexFlag[] {
  const wanted = settingOf(raw);
  return INDEX_FLAGS.filter((f) => wanted.has(f.name)).map((f) => f.name);
}

const layoutOf = (shown: ChoiceOutput[], flags: IndexFlag[]) => [
  ...OPTIONAL_OUTPUTS.filter((o) => shown.includes(o.name)).map((o) =>
    o.name === "index" && flags.includes("index_as_text") ? { ...o, type: "STRING" } : o,
  ),
];

/** Lay the outputs out for `shown`, keeping every link a kept output had. */
function syncOutputs(node: ChoiceNodeLike, shown: ChoiceOutput[]) {
  const want = layoutOf(shown, shownFlags(node));
  const current = (node.outputs ?? []).map((s) => `${s.name}:${s.type}`).join(",");
  if (current === want.map((s) => `${s.name}:${s.type}`).join(",")) return;

  const graph = node.graph ?? null;
  const outgoing = new Map<string, LGLink[]>();
  if (graph) {
    for (const slot of node.outputs ?? []) {
      const links = (slot.links ?? []).map((id) => linkOf(graph, id)).filter((l) => !!l);
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
 *  in the definition's order while each link keeps the slot index it was saved at, so only the
 *  names and types are wrong: rename in place and drop the unlinked extras from the end. */
function relabelOutputs(node: ChoiceNodeLike, shown: ChoiceOutput[]) {
  const want = layoutOf(shown, shownFlags(node));
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

function write(node: ChoiceNodeLike, shown: ChoiceOutput[], flags: IndexFlag[]) {
  const widget = outputsWidget(node);
  if (!widget) return;
  const ordered = OPTIONAL_OUTPUTS.map((o) => o.name).filter((name) => shown.includes(name));
  const kept = INDEX_FLAGS.map((f) => f.name).filter((name) => flags.includes(name));
  widget.value = [...ordered, ...kept].join(",");
  syncOutputs(node, ordered);
}

/** Switch `output` on or off on `node`, and lay its sockets out to match. */
export function toggleOutput(raw: unknown, output: ChoiceOutput) {
  const node = raw as ChoiceNodeLike;
  const shown = shownOutputs(node);
  const next = shown.includes(output) ? shown.filter((o) => o !== output) : [...shown, output];
  if (!next.length) return; // the last one stays: a node with no outputs is no use to anything
  write(node, next, shownFlags(node));
}

/** Switch one of the index flags on or off. "As text" changes the index socket's type, which
 *  drops a wire the new type can't keep. */
export function toggleFlag(raw: unknown, flag: IndexFlag) {
  const node = raw as ChoiceNodeLike;
  const flags = shownFlags(node);
  const next = flags.includes(flag) ? flags.filter((f) => f !== flag) : [...flags, flag];
  write(node, shownOutputs(node), next);
}

app.registerExtension({
  name: `${CHOICE}.Outputs`,
  nodeCreated(raw: unknown) {
    const node = raw as ChoiceNodeLike & {
      constructor: { comfyClass?: string };
      onConfigure?: (...args: unknown[]) => void;
    };
    if (node.constructor.comfyClass !== CHOICE) return;
    offerOptions(node);
    const widget = outputsWidget(node);
    if (!widget) return;
    widget.hidden = true;
    syncOutputs(node, shownOutputs(node));
    const prev = node.onConfigure;
    node.onConfigure = function (this: unknown, ...args: unknown[]) {
      prev?.apply(this, args);
      offerOptions(node);
      const shown = shownOutputs(node);
      relabelOutputs(node, shown);
      syncOutputs(node, shown);
    };
  },
} as never);
