// The saved-text library, shared by every Saved Text node on the graph: fetched once, and
// reactive, so an edit made in one node's editor shows on the others at once. Routes are served
// by nodes/text/api.py.

import { computed, ref } from "vue";

import { route } from "@/framework";

export interface SavedText {
  id: string;
  name: string;
  /** Free-form heading the entry is filed under ("System prompts"), or "" for none. */
  category: string;
  text: string;
  /** When it was last saved, epoch seconds. */
  updated: number;
}
export type SavedTextDraft = Pick<SavedText, "name" | "category" | "text"> & { id?: string };

export const texts = ref<SavedText[]>([]);
/** False until the first fetch lands: an entry missing before then isn't known to be missing. */
export const loaded = ref(false);

function take(data: unknown): SavedText[] | null {
  const list = (data as { texts?: unknown } | null)?.texts;
  if (!Array.isArray(list)) return null;
  return list
    .filter((e): e is Record<string, unknown> => !!e && typeof e === "object")
    .map((e) => ({
      id: String(e.id ?? ""),
      name: String(e.name ?? ""),
      category: String(e.category ?? ""),
      text: String(e.text ?? ""),
      updated: Number(e.updated) || 0,
    }))
    .filter((e) => e.id);
}

async function post(path: string, body: unknown): Promise<unknown> {
  const res = await fetch(route(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

let pending: Promise<void> | null = null;
/** Fetch the library unless it is already here (or on its way). */
export function ensure(): Promise<void> {
  if (loaded.value) return Promise.resolve();
  pending ??= (async () => {
    try {
      const list = take(await (await fetch(route("texts"))).json());
      if (list) {
        texts.value = list;
        loaded.value = true;
      }
    } catch {
      /* no server, or not this pack's: the node still shows the text it holds */
    } finally {
      pending = null;
    }
  })();
  return pending;
}

/** Add `draft`, or replace the entry with its id; the stored entry, or null if it didn't save. */
export async function saveText(draft: SavedTextDraft): Promise<SavedText | null> {
  try {
    const data = (await post("texts", draft)) as { text?: { id?: unknown } };
    const list = take(data);
    if (list) texts.value = list;
    return byId(String(data.text?.id ?? "")) ?? null;
  } catch {
    return null;
  }
}

export async function deleteText(id: string): Promise<boolean> {
  try {
    const list = take(await post("texts/delete", { id }));
    if (list) texts.value = list;
    return !!list;
  } catch {
    return false;
  }
}

export const byId = (id: string) => (id ? texts.value.find((t) => t.id === id) : undefined);

/** Every category in use, A to Z. */
export const categories = computed(() =>
  [...new Set(texts.value.map((t) => t.category).filter(Boolean))].sort((a, b) =>
    a.localeCompare(b),
  ),
);

/** The library as headed sections: uncategorised first, then each category A to Z. */
export function grouped(list: SavedText[]): { category: string; items: SavedText[] }[] {
  const byName = (a: SavedText, b: SavedText) => a.name.localeCompare(b.name);
  const sections = new Map<string, SavedText[]>();
  for (const entry of list) {
    if (!sections.has(entry.category)) sections.set(entry.category, []);
    sections.get(entry.category)!.push(entry);
  }
  return [...sections]
    .sort(([a], [b]) => (a === "" ? -1 : b === "" ? 1 : a.localeCompare(b)))
    .map(([category, items]) => ({ category, items: items.sort(byName) }));
}
