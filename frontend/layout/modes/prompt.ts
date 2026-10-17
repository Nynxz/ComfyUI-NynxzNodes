// Ideogram, Klein and MOSS — the modes for models that read a layout from their prompt. They
// share one value (layout/promptScene.ts), so a switch between them keeps everything; they
// differ in the prompt written out and in what the inspector asks for.

import type { BoxRect } from "@nynxz/zenkit-ui";

import { compileLayout } from "../library";
import type { CommonLayout, LayoutBox, LayoutMode } from "../mode";
import {
  boxId,
  nextColor,
  readBox,
  readModelPrompt,
  readPromptScene,
  type PromptBox,
  type PromptModeId,
  type PromptScene,
} from "../promptScene";

function describe(box: PromptBox): string {
  const kind = box.kind === "object" ? "" : `${box.kind} · `;
  const what = box.kind === "text" && box.text ? `"${box.text}"` : box.desc;
  return `${kind}${what.trim().slice(0, 48) || "no description"}`;
}

function promptMode(id: PromptModeId, label: string, icon: string): LayoutMode<PromptScene> {
  return {
    id,
    label,
    icon,
    paints: false,
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
        mask: null,
      })),
    setBoxes(s, boxes: LayoutBox[]) {
      const byId = new Map(s.boxes.map((b) => [b.id, b]));
      return {
        ...s,
        boxes: boxes.flatMap((b) => {
          const known = byId.get(b.id);
          if (!known) return [];
          const { x, y, w, h, name, hidden, locked } = b;
          return [{ ...known, x, y, w, h, name, hidden, locked }];
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
        const box = {
          ...copied,
          ...rect,
          id: boxId(),
          name: `${copied.name} copy`,
          color: nextColor(boxes),
        };
        boxes.push(box);
        ids.push(box.id);
      });
      return { scene: { ...s, boxes }, ids };
    },
    describe(s, boxIdToFind) {
      const box = s.boxes.find((b) => b.id === boxIdToFind);
      return box ? { sub: describe(box) } : {};
    },
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
        mask: null,
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
    importText: (text, current) => readModelPrompt(text, current),
  };
}

export const ideogramMode = promptMode("ideogram", "Ideogram", "mdi mdi-format-text-variant");
export const kleinMode = promptMode("klein", "Klein", "mdi mdi-alpha-k-box-outline");
export const mossMode = promptMode("moss", "MOSS", "mdi mdi-grid");

export const PROMPT_MODES = [ideogramMode, kleinMode, mossMode];
