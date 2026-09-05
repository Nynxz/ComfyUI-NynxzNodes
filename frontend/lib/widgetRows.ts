// Row bookkeeping for list-shaped node bodies: parse the saved value into rows, keep a stable id
// per row, write back to both the widget value and node.properties.
//
// ComfyUI hashes the widget value to decide re-runs, so display order must not reach it —
// reordering would invalidate a cached execution for nothing. The value is a normalized
// projection; order and row ids live in node.properties, which serialises but isn't hashed.

import { ref, type Ref } from "vue";

/** ComfyUI's DOM widget, as much of it as a node body touches. */
export interface WidgetLike {
  value: unknown;
  callback?: (value: unknown) => void;
}

/** The `{ widget, node }` props `mountWidget` hands every widget component. */
export interface WidgetHost {
  widget?: WidgetLike;
  node?: {
    graph?: { setDirtyCanvas?: (a: boolean, b: boolean) => void };
    properties?: Record<string, unknown>;
  };
}

export interface RowStoreOptions<T> {
  /** `node.properties` key holding the display order. Unique per widget type. */
  orderKey: string;
  /** Build a row from a saved entry. `id` is already allocated and must be kept. */
  make: (raw: Record<string, unknown>, id: number) => T;
  /** Row -> the entry the backend receives. Drop anything presentational (ids included). */
  project: (row: T) => unknown;
  /** Optional total order for the emitted value, so display order can't change the hash. */
  sort?: (a: T, b: T) => number;
}

export interface RowStore<T> {
  rows: Ref<T[]>;
  /** A fresh row id, above every id restored from the graph. */
  nextId: () => number;
  /** Write the rows to `node.properties` and the projected value to the widget. */
  commit: () => void;
  /** Merge `fields` into row `index` and commit. No-op for an index that isn't there. */
  patch: (index: number, fields: Partial<T>) => void;
}

/** Coerce a saved widget value (list, or the JSON string an older graph may hold) into entries. */
function entriesOf(value: unknown): Record<string, unknown>[] {
  let arr = value;
  if (typeof arr === "string") {
    try {
      arr = JSON.parse(arr || "[]");
    } catch {
      arr = [];
    }
  }
  if (!Array.isArray(arr)) return [];
  return arr.filter((r): r is Record<string, unknown> => !!r && typeof r === "object");
}

export function rowsFrom<T extends { id: number }>(
  host: WidgetHost,
  opts: RowStoreOptions<T>,
): RowStore<T> {
  let seq = 0;
  const nextId = () => ++seq;

  function normalize(value: unknown): T[] {
    return entriesOf(value).map((raw) => {
      const saved = +(raw.id as number);
      const id = Number.isFinite(saved) ? saved : nextId();
      if (id > seq) seq = id; // keep nextId() above any id restored from the graph
      return opts.make(raw, id);
    });
  }

  // Prefer the ordered rows in node.properties — they carry the display order and the ids. Fall
  // back to the widget value for an older graph, or the first load of a fresh node, where the only
  // thing saved is the backend's own (sorted) projection.
  const stored = host.node?.properties?.[opts.orderKey];
  const rows = ref(
    Array.isArray(stored) && stored.length ? normalize(stored) : normalize(host.widget?.value),
  ) as Ref<T[]>;

  function commit() {
    if (host.node?.properties) {
      host.node.properties[opts.orderKey] = rows.value.map((r) => ({ ...r }));
    }
    if (host.widget) {
      const ordered = opts.sort ? [...rows.value].sort(opts.sort) : rows.value;
      const value = ordered.map(opts.project);
      // `widget` is ComfyUI's host object, not Vue state; writing its value IS how a node body
      // persists, and `serializeValue` reads it back at prompt-queue time.
      host.widget.value = value;
      try {
        host.widget.callback?.(value);
      } catch {
        /* no callback registered */
      }
    }
    host.node?.graph?.setDirtyCanvas?.(true, true);
  }

  function patch(index: number, fields: Partial<T>) {
    const row = rows.value[index];
    if (!row) return;
    Object.assign(row, fields);
    commit();
  }

  return { rows, nextId, commit, patch };
}
