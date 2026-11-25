// Regions, MOSS, Ideogram and Klein — the unified Layout modes. They share one value
// (layout/promptScene.ts), so switching mode keeps everything — boxes, painted shapes, mixins,
// references, LoRAs and settings; they differ in the prompt written out and mode-specific fields.

import type { BoxRect } from "@nynxz/zenkit-ui";

import { thumbUrl } from "@/lib/mediaRefs";
import { uid } from "@/lib/sceneUtil";

import { compileLayout } from "../library";
import type { CommonLayout, LayoutBox, LayoutMode } from "../mode";
import {
  boxActiveLoras,
  boxActiveRefs,
  boxId,
  isFrescoRaw,
  nextColor,
  readBox,
  readModelPrompt,
  readPromptScene,
  type PromptBox,
  type PromptModeId,
  type PromptScene,
} from "../promptScene";
import { digJson } from "../jsonIo";

function describe(scene: PromptScene, box: PromptBox): { sub?: string; thumb?: string } {
  const parts: string[] = [];
  if (box.kind !== "object" && scene.mode !== "regions") parts.push(box.kind);
  const what = box.kind === "text" && box.text ? `"${box.text}"` : box.desc.trim();
  if (what) parts.push(what.slice(0, 40));
  const refs = boxActiveRefs(scene, box);
  const loras = boxActiveLoras(scene, box);
  if (refs.length) parts.push(`${refs.length} ref${refs.length > 1 ? "s" : ""}`);
  if (loras.length) parts.push(`${loras.length} LoRA${loras.length > 1 ? "s" : ""}`);
  const firstRef = refs[0];
  return {
    sub: parts.join(" · ") || "empty",
    thumb: firstRef ? thumbUrl(firstRef.ref, firstRef.type) : undefined,
  };
}

function promptMode(id: PromptModeId, label: string, icon: string): LayoutMode<PromptScene> {
  return {
    id,
    label,
    icon,
    paints: true,
    read: (raw) => ({ ...readPromptScene(raw), mode: id }),
    size: (s) => ({ width: s.width, height: s.height }),
    resize: (s, width, height) => ({ ...s, width: Math.round(width), height: Math.round(height) }),
    boxes: (s) =>
      s.boxes.map((b) => ({
        id: b.id,
        name: b.name,
        color: b.color,
        x: b.x,
        y: b.y,
        w: b.w,
        h: b.h,
        hidden: b.hidden,
        locked: b.locked,
        mask: b.mask,
      })),
    setBoxes(s, boxes: LayoutBox[]) {
      const byId = new Map(s.boxes.map((b) => [b.id, b]));
      return {
        ...s,
        boxes: boxes.flatMap((b) => {
          const known = byId.get(b.id);
          if (!known) return [];
          const { x, y, w, h, name, hidden, locked, mask } = b;
          return [{ ...known, x, y, w, h, name, hidden, locked, mask }];
        }),
      };
    },
    add(s, rect: BoxRect) {
      const box = readBox(
        { ...rect, id: boxId(), name: `Box ${s.boxes.length + 1}`, color: nextColor(s.boxes) },
        s.boxes.length,
      );
      return { scene: { ...s, boxes: [...s.boxes, box] }, id: box.id };
    },
    copy: (s, ids) =>
      JSON.parse(JSON.stringify(s.boxes.filter((b) => ids.includes(b.id)))) as unknown[],
    paste(s, items, rects) {
      const boxes = [...s.boxes];
      const ids: string[] = [];
      items.forEach((item, i) => {
        const rect = rects[i];
        if (!rect) return;
        const copied = readBox(item, boxes.length);
        const box: PromptBox = {
          ...copied,
          ...rect,
          id: boxId(),
          name: `${copied.name} copy`,
          color: nextColor(boxes),
          points: copied.points.map((p) => ({
            ...p,
            x: rect.x + (p.x - copied.x),
            y: rect.y + (p.y - copied.y),
          })),
          refs: copied.refs.map((r) => ({ ...r, id: uid("r") })),
          loras: copied.loras.map((o) => ({ ...o, id: uid("o") })),
        };
        boxes.push(box);
        ids.push(box.id);
      });
      return { scene: { ...s, boxes }, ids };
    },
    describe(s, boxIdToFind) {
      const box = s.boxes.find((b) => b.id === boxIdToFind);
      return box ? describe(s, box) : {};
    },
    sceneHidden: (s) => (s.mode === "regions" ? !s.backdrop.on : false),
    toggleScene: (s) =>
      s.mode === "regions" ? { ...s, backdrop: { ...s.backdrop, on: !s.backdrop.on } } : s,
    sceneLine: (s) => s.prompt.trim().slice(0, 48) || `${label} — the whole picture`,
    toCommon: (s): CommonLayout => ({
      width: s.width,
      height: s.height,
      prompt: s.prompt,
      boxes: s.boxes.map((b) => ({
        x: b.x,
        y: b.y,
        w: b.w,
        h: b.h,
        name: b.name,
        color: b.color,
        prompt: b.desc,
        mask: b.mask,
        hidden: b.hidden,
        locked: b.locked,
      })),
    }),
    fromCommon: (c) =>
      readPromptScene({
        mode: id,
        width: c.width,
        height: c.height,
        prompt: c.prompt,
        boxes: c.boxes.map((b) => ({ ...b, id: boxId(), desc: b.prompt })),
      }),
    exportText: async (s) => (await compileLayout(s)).prompt,
    importText: (text, current) => {
      const parsed = digJson(text);
      if (isFrescoRaw(parsed)) return { ...readPromptScene(parsed), mode: id };
      return readModelPrompt(text, current);
    },
  };
}

export const layoutRegionsMode = promptMode("regions", "Regions", "mdi mdi-palette-outline");
export const mossMode = promptMode("moss", "MOSS", "mdi mdi-grid");
export const ideogramMode = promptMode("ideogram", "Ideogram", "mdi mdi-format-text-variant");
export const kleinMode = promptMode("klein", "Klein", "mdi mdi-alpha-k-box-outline");

export const PROMPT_MODES = [layoutRegionsMode, mossMode, ideogramMode, kleinMode];
