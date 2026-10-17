// Regions — Fresco's mode: each box is a layer with a prompt, references, LoRAs and a painted
// shape, and the scene goes to Fresco, which turns it into the Regions wire.

import { thumbUrl } from "@/lib/fusionApi";
import {
  PALETTE,
  newLayer,
  parseScene,
  summarize,
  uid,
  type FrescoLayer,
  type FrescoScene,
} from "@/lib/frescoScene";

import type { LayoutBox, LayoutMode } from "../mode";

function nextColor(layers: FrescoLayer[]): string {
  const used = new Set(layers.map((l) => l.color));
  return PALETTE.find((c) => !used.has(c)) ?? PALETTE[layers.length % PALETTE.length]!;
}

export const regionsMode: LayoutMode<FrescoScene> = {
  id: "regions",
  label: "Regions",
  icon: "mdi mdi-palette-outline",
  paints: true,
  read: parseScene,
  size: (s) => ({ width: s.width, height: s.height }),
  resize: (s, width, height) => ({ ...s, width: Math.round(width), height: Math.round(height) }),
  boxes: (s) =>
    s.layers.map((l) => ({
      id: l.id,
      name: l.name,
      color: l.color,
      x: l.x,
      y: l.y,
      w: l.w,
      h: l.h,
      hidden: !l.on,
      locked: l.locked,
      mask: l.mask,
    })),
  setBoxes(s, boxes: LayoutBox[]) {
    const byId = new Map(s.layers.map((l) => [l.id, l]));
    return {
      ...s,
      layers: boxes.flatMap((b) => {
        const l = byId.get(b.id);
        if (!l) return [];
        const { x, y, w, h, name, locked, mask } = b;
        return [{ ...l, x, y, w, h, name, locked, mask, on: !b.hidden }];
      }),
    };
  },
  add(s, rect) {
    const layer = newLayer(rect, s.layers);
    return { scene: { ...s, layers: [...s.layers, layer] }, id: layer.id };
  },
  copy: (s, ids) =>
    JSON.parse(JSON.stringify(s.layers.filter((l) => ids.includes(l.id)))) as unknown[],
  paste(s, items, rects) {
    const layers = [...s.layers];
    const ids: string[] = [];
    (items as FrescoLayer[]).forEach((item, i) => {
      const rect = rects[i];
      if (!rect) return;
      const layer: FrescoLayer = {
        ...item,
        ...rect,
        id: uid(),
        name: `${item.name} copy`,
        color: nextColor(layers),
        points: item.points.map((p) => ({
          ...p,
          x: rect.x + (p.x - item.x),
          y: rect.y + (p.y - item.y),
        })),
        refs: item.refs.map((r) => ({ ...r, id: uid("r") })),
        loras: item.loras.map((o) => ({ ...o, id: uid("o") })),
      };
      layers.push(layer);
      ids.push(layer.id);
    });
    return { scene: { ...s, layers }, ids };
  },
  describe(s, id) {
    const l = s.layers.find((layer) => layer.id === id);
    if (!l) return {};
    return {
      sub: summarize(l),
      thumb: l.refs[0] ? thumbUrl(l.refs[0].ref, l.refs[0].type) : undefined,
    };
  },
  sceneHidden: (s) => !s.background.on,
  toggleScene: (s) => ({ ...s, background: { ...s.background, on: !s.background.on } }),
  sceneLine: (s) => s.prompt.trim().slice(0, 48) || "prompt, background, settings",
  toCommon: (s) => ({
    width: s.width,
    height: s.height,
    prompt: s.prompt,
    boxes: s.layers.map((l) => ({
      x: l.x,
      y: l.y,
      w: l.w,
      h: l.h,
      name: l.name,
      color: l.color,
      prompt: l.prompt,
      mask: l.mask,
    })),
  }),
  fromCommon: (c) =>
    parseScene({
      width: c.width,
      height: c.height,
      prompt: c.prompt,
      layers: c.boxes.map((b) => ({ ...b, id: uid() })),
    }),
};
