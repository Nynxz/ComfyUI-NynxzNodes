// Group Switcher: a frontend-only node that turns the groups in its graph on and off.
//
// It is a VIRTUAL node (like rgthree's Fast Groups Bypasser): there is no Python class behind it,
// and `isVirtualNode` keeps it out of the prompt entirely, so it can never affect execution or
// validation. All it does is change the mode of other nodes. That is also why it lives here
// rather than in frontend/widgets/ — that folder is for widgets backed by a Python io type.

import { app } from "@comfy/app";

import { CATEGORY, NAMESPACE, mountWidget, nodeId } from "@/framework";
import GroupSwitcher from "./GroupSwitcher.vue";
import { DEFAULT_SETTINGS } from "./groups";

const NODE_TYPE = nodeId("Utils.GroupSwitcher");
const TITLE = "Group Switcher";
const NODE_CATEGORY = `${CATEGORY}/Utils`;

// LiteGraph's classes, as ComfyUI exposes them to extensions on window.
interface LGraphNodeLike {
  size: [number, number];
  isVirtualNode?: boolean;
  serialize_widgets?: boolean;
  onAdded?(graph: unknown): void;
}
type LGraphNodeClass = new (title?: string) => LGraphNodeLike;
// The fields of a node-library definition this touches.
interface NodeDefLike {
  name: string;
  display_name?: string;
  category?: string;
  python_module?: string;
}
interface LiteGraphGlobals {
  LGraphNode: LGraphNodeClass & { category?: string };
  LiteGraph: {
    registerNodeType: (type: string, ctor: unknown) => void;
  };
}

app.registerExtension({
  name: `${NAMESPACE}.GroupSwitcher`,
  registerCustomNodes() {
    const { LGraphNode, LiteGraph } = window as unknown as LiteGraphGlobals;

    class GroupSwitcherNode extends LGraphNode {
      static title = TITLE;
      static comfyClass = NODE_TYPE;

      #mounted = false;

      constructor(title?: string) {
        super(title ?? GroupSwitcherNode.title);
        this.isVirtualNode = true;
        this.serialize_widgets = true;
        this.size = [300, 140];
      }

      // Mounted on joining a graph rather than in the constructor: a loaded workflow writes the
      // saved settings into the widget right after this, before the widget's first render.
      onAdded() {
        if (this.#mounted) return;
        this.#mounted = true;
        mountWidget(this as never, {
          widgetName: "switcher",
          widgetType: "NYNXZ_GROUP_SWITCHER",
          component: GroupSwitcher,
          defaultValue: { ...DEFAULT_SETTINGS },
        });
      }
    }

    LiteGraph.registerNodeType(NODE_TYPE, GroupSwitcherNode);
    // registerNodeType derives category from a "/" in the type string, and ours has none — so
    // it wipes any category set beforehand. Set it after.
    (GroupSwitcherNode as { category?: string }).category = NODE_CATEGORY;
  },
  // A node with no Python class gets a stand-in definition for the node library, named after its
  // raw type string and filed under a "frontend_only" source. Give ours its real name, and the
  // same source as the pack's Python nodes so the library shows it with the rest of the pack.
  beforeRegisterVueAppNodeDefs(defs: NodeDefLike[]) {
    const def = defs.find((d) => d.name === NODE_TYPE);
    if (!def) return;
    def.display_name = TITLE;
    def.category = NODE_CATEGORY;
    const sibling = defs.find((d) => d.name.startsWith(`${NAMESPACE}.`) && d !== def);
    if (sibling?.python_module) def.python_module = sibling.python_module;
  },
});
