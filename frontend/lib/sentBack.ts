// What an editor node's run sends back (nodes/_sent_back.py): `<key>_node` names the node it is
// for — its id, or a subgraph path ending in it — and `<key>_source`/`<key>_size` the image to
// offer as the trace image.

import { onBeforeUnmount, onMounted } from "vue";

import { api } from "@comfy/api";

import type { MediaRef, SourceType } from "@/lib/mediaRefs";

export interface SentImage extends MediaRef {
  size?: [number, number];
}

type Output = Record<string, unknown[] | undefined>;

function sentImage(output: Output, key: string): SentImage | null {
  const image = output[`${key}_source`]?.[0] as
    { filename: string; subfolder?: string; type?: string } | undefined;
  if (!image) return null;
  const size = output[`${key}_size`];
  return {
    ref: image.subfolder ? `${image.subfolder}/${image.filename}` : image.filename,
    type: (image.type ?? "temp") as SourceType,
    size: size?.length === 2 ? [Number(size[0]), Number(size[1])] : undefined,
  };
}

/** Calls `take` with each run output sent back to this node, while the component is mounted. */
export function useSentBack(
  key: string,
  nodeId: () => string,
  take: (output: Output, image: SentImage | null) => void,
): void {
  function onExecuted(e: Event) {
    const output = (e as CustomEvent<{ output?: Output }>).detail?.output;
    const mine = nodeId();
    const nodes = output?.[`${key}_node`];
    if (
      !mine ||
      !output ||
      !nodes?.some((id) => {
        const theirs = String(id ?? "");
        return theirs === mine || theirs.endsWith(`:${mine}`);
      })
    )
      return;
    take(output, sentImage(output, key));
  }
  const events = api as unknown as {
    addEventListener(t: string, f: (e: Event) => void): void;
    removeEventListener(t: string, f: (e: Event) => void): void;
  };
  onMounted(() => events.addEventListener("executed", onExecuted));
  onBeforeUnmount(() => events.removeEventListener("executed", onExecuted));
}
