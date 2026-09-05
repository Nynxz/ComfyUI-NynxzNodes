// The pack's one crossing into @nynxz/zenkit-nodekit. Identity is bound here from pack.json;
// everything else imports from this file.

import {
  createNodekit,
  defineNode,
  useDragSurface,
  type DOMWidget,
  type NodeDef,
  type NodeWidgetDef,
  type WidgetFileExports,
  type WidgetOptions,
} from "@nynxz/zenkit-nodekit";

import manifest from "../pack.json";

export const {
  NAMESPACE,
  DISPLAY_NAME,
  CATEGORY,
  nodeId,
  route,
  settingId,
  typeId,
  registerNodes,
  discoverNodes,
  discoverWidgets,
  widgetTypes,
  mountWidget,
} = createNodekit(manifest);

export { defineNode, useDragSurface };
export type { DOMWidget, NodeDef, NodeWidgetDef, WidgetFileExports, WidgetOptions };
