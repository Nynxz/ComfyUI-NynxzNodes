// The graph side of the Group Switcher: read the groups a switcher can see, and set the mode of
// the nodes inside them. Kept free of Vue so the widget only deals in plain snapshots.
//
// State is never stored here or on the switcher: whether a group is "on" is read straight from
// its nodes' modes every time. That keeps the switcher honest when a node is bypassed by hand,
// after undo, or when another switcher touches the same group.

/** LiteGraph node modes (LGraphEventMode). */
export const MODE_ALWAYS = 0;
export const MODE_NEVER = 2;
export const MODE_BYPASS = 4;

/** What turning a group off does to its nodes. */
export type OffMode = "bypass" | "mute";

/** The switcher's own settings — the widget's serialized value. */
export interface SwitcherSettings {
  off: OffMode;
  /** Group colours this switcher controls; empty means every colour. `NO_COLOR` = uncoloured. */
  colors: string[];
  /** Keys of groups whose node list is expanded. */
  expanded: string[];
}

export const NO_COLOR = "none";

export const DEFAULT_SETTINGS: SwitcherSettings = { off: "bypass", colors: [], expanded: [] };

/** Persist the settings on ComfyUI's widget host object — how a node body saves with the graph. */
export function writeSettings(
  widget: { value: unknown; callback?: (v: unknown) => void } | undefined,
  value: SwitcherSettings,
) {
  if (!widget) return;
  widget.value = value;
  try {
    widget.callback?.(value);
  } catch {
    /* no callback registered */
  }
}

export function readSettings(value: unknown): SwitcherSettings {
  const v = (value && typeof value === "object" ? value : {}) as Partial<SwitcherSettings>;
  return {
    off: v.off === "mute" ? "mute" : "bypass",
    colors: Array.isArray(v.colors) ? v.colors.map(String) : [],
    expanded: Array.isArray(v.expanded) ? v.expanded.map(String) : [],
  };
}

// The bits of LiteGraph this touches.
export interface LGNode {
  id: number | string;
  title?: string;
  type?: string;
  mode: number;
}
interface LGGroup {
  id?: number;
  title: string;
  color?: string;
  pos: ArrayLike<number>;
  nodes: LGNode[];
  recomputeInsideNodes: () => void;
}
export interface LGGraph {
  groups?: LGGroup[];
  _groups?: LGGroup[];
  setDirtyCanvas?: (fg: boolean, bg: boolean) => void;
  beforeChange?: () => void;
  afterChange?: () => void;
}

export interface GroupNodeEntry {
  id: number | string;
  title: string;
  on: boolean;
  node: LGNode;
}
export interface GroupEntry {
  key: string;
  title: string;
  /** Normalised colour (lower-case hex), or `NO_COLOR`. */
  color: string;
  nodes: GroupNodeEntry[];
  /** "on" when every node runs, "off" when none do, "mixed" in between. */
  state: "on" | "off" | "mixed";
}

const normColor = (c: string | undefined) => (c ? c.trim().toLowerCase() : NO_COLOR);

/** Every group in `graph`, in canvas reading order (top to bottom, then left to right), with the
 *  nodes inside it. `self` — the switcher — is never listed as one of its own targets. */
export function readGroups(graph: LGGraph | null | undefined, self: unknown): GroupEntry[] {
  const groups = graph?.groups ?? graph?._groups ?? [];
  return [...groups]
    .sort((a, b) => a.pos[1] - b.pos[1] || a.pos[0] - b.pos[0])
    .map((group, index) => {
      // Membership isn't tracked live — nodes dragged in or out only show up after this.
      group.recomputeInsideNodes();
      const nodes = group.nodes
        .filter((n) => n !== self)
        .map((n) => ({
          id: n.id,
          title: n.title || n.type || `#${n.id}`,
          on: n.mode === MODE_ALWAYS,
          node: n,
        }));
      const onCount = nodes.filter((n) => n.on).length;
      return {
        key: group.id != null ? `g${group.id}` : `${index}:${group.title}`,
        title: group.title || "Untitled group",
        color: normColor(group.color),
        nodes,
        state: onCount === nodes.length ? "on" : onCount === 0 ? "off" : "mixed",
      } satisfies GroupEntry;
    });
}

/** A string that changes whenever anything the widget shows changes — so a refresh can skip
 *  re-rendering when the graph is unchanged. */
export function signature(groups: GroupEntry[]): string {
  return groups
    .map(
      (g) => `${g.key}|${g.title}|${g.color}|${g.nodes.map((n) => `${n.id}${n.on ? "+" : "-"}`)}`,
    )
    .join("\n");
}

// Assigned directly, as ComfyUI's own bypass command does: LGraphNode.changeMode() has no case
// for BYPASS and silently refuses it.
function setMode(node: LGNode, mode: number) {
  if (node.mode !== mode) node.mode = mode;
}

/** Turn nodes on or off, as one undoable change. */
export function setNodesOn(
  graph: LGGraph | null | undefined,
  nodes: LGNode[],
  on: boolean,
  off: OffMode,
) {
  const mode = on ? MODE_ALWAYS : off === "mute" ? MODE_NEVER : MODE_BYPASS;
  graph?.beforeChange?.();
  for (const node of nodes) setMode(node, mode);
  graph?.afterChange?.();
  graph?.setDirtyCanvas?.(true, true);
}
