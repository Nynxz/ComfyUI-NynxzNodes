// A node body that can pop out into a ZenKit panel: the panel's own instance marks itself open
// while mounted (so closing it any way brings the editor back to the node), popped-out survives a
// reload in a node property, and the panel keeps its size, place and workspace tile under an id
// from the graph and the node. Call from the body component's setup; the panel mounts the same
// component with `panel: true`.

import { computed, getCurrentInstance, onBeforeUnmount, onMounted } from "vue";

import { hasZenPanels, openZenPanel, type ZenPanelHandle } from "@/framework";
import { isPanelOpen, setPanelOpen } from "@/layout/store";

export interface PopOutNode {
  id?: number | string;
  graph?: { id?: string; change?: () => void; setDirtyCanvas?: (a: boolean, b: boolean) => void };
  properties?: Record<string, unknown>;
  onRemoved?: (...args: unknown[]) => unknown;
}

export interface PopOutProps {
  widget?: { value: unknown; callback?: (v: unknown) => void };
  node?: PopOutNode;
  panel?: boolean;
}

export interface PopOutOptions {
  /** Names the panel id (`nynxz.<name>:<graph>:<node>`). */
  name: string;
  /** The panel-open registry's key for this node. */
  openKey: string;
  /** The node property that remembers it was popped out. */
  property: string;
  title: string;
  icon: string;
  size: { width: number; height: number; minWidth: number; minHeight: number };
}

export function usePopOut(props: PopOutProps, options: PopOutOptions) {
  const id = String(props.node?.id ?? "");
  const poppedOut = computed(() => isPanelOpen(options.openKey));

  onMounted(() => props.panel && setPanelOpen(options.openKey, true));
  onBeforeUnmount(() => props.panel && setPanelOpen(options.openKey, false));

  const handles = new WeakMap<object, ZenPanelHandle>();
  const quiet = new WeakSet<object>();
  const panelId = () => {
    const graph = props.node?.graph?.id;
    return graph ? `nynxz.${options.name}:${graph}:${id}` : `nynxz.${options.name}.${id}`;
  };
  function remember(popped: boolean) {
    const node = props.node;
    if (!node) return;
    node.properties ??= {};
    if (Boolean(node.properties[options.property]) === popped) return;
    node.properties[options.property] = popped;
    node.graph?.change?.();
  }

  const self = getCurrentInstance()!.type;
  /** `reveal`: you asked for it, so a panel tiled in a workspace that isn't showing goes there.
   *  The restore on load doesn't: it waits quietly for its workspace. */
  function popOut(reveal = false) {
    const node = props.node;
    const handle = openZenPanel(
      {
        id: panelId(),
        title: options.title,
        icon: options.icon,
        ...options.size,
        persist: true,
        reveal,
        props: { widget: props.widget, node: props.node, panel: true },
        onClose: () => {
          if (node) handles.delete(node);
          if (node && quiet.delete(node)) return;
          remember(false);
        },
      },
      self,
    );
    if (!handle || !node) return;
    handles.set(node, handle);
    remember(true);
    const previous = node.onRemoved;
    node.onRemoved = function (this: unknown, ...args: unknown[]) {
      const h = handles.get(node);
      if (h) {
        quiet.add(node);
        h.close({ keep: true });
      }
      return previous?.apply(this, args);
    };
  }

  onMounted(() => {
    if (props.panel || !props.node?.properties?.[options.property]) return;
    // ZenKit can finish installing after this node mounts; give it a moment rather than give up.
    let tries = 0;
    const attempt = () => {
      if (isPanelOpen(options.openKey) || (props.node && handles.has(props.node))) return;
      if (hasZenPanels()) popOut();
      else if (++tries < 40) setTimeout(attempt, 250);
    };
    attempt();
  });

  return { poppedOut, popOut };
}
