// A pack widget promoted out of a subgraph is drawn a second time, on the subgraph's node. That
// copy is handed a stand-in for the widget (see nodekit's widgetMirror), where the body on the
// node itself is handed the node's own widget.

/** True for the widget body on its own node; false for a copy promoted onto a subgraph's node.
 *  A copy shouldn't put a settings button in that other node's header: the settings are the
 *  inner node's, and are changed there. */
export function isOwnMount(props: { widget?: unknown; node?: unknown }): boolean {
  const widgets = (props.node as { widgets?: unknown[] } | undefined)?.widgets;
  return !widgets || !props.widget || widgets.includes(props.widget);
}
