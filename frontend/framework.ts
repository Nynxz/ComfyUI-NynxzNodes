// The pack's one crossing into @nynxz/zenkit-nodekit. Identity is bound here from pack.json;
// everything else imports from this file.

import {
  createNodekit,
  defineNode,
  openZenPanel,
  useDragSurface,
  viewUrl,
  type DOMWidget,
  type NodeDef,
  type NodeHeaderButtonHandle,
  type NodeHeaderButtonOptions,
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
  addNodeHeaderButton,
} = createNodekit(manifest);

// Identity-free helpers pass straight through.
export { defineNode, openZenPanel, useDragSurface, viewUrl };
export type {
  DOMWidget,
  NodeDef,
  NodeHeaderButtonHandle,
  NodeHeaderButtonOptions,
  NodeWidgetDef,
  WidgetFileExports,
  WidgetOptions,
};
