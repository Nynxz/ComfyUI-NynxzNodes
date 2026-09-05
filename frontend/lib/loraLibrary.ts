// One shared view of the LoRA folder for every widget that picks from it. Module-level rather
// than a composable: a graph can hold a dozen LoRA nodes, and per-component state would mean a
// dozen fetches and a dozen places for a fresh bookmark to be stale.

import { computed, ref } from "vue";

import { getFavorites, listLoras, previewUrl, setFavorite, type LoraItem } from "@/lib/loraApi";

/** Every LoRA on disk. Empty until `ensure()` has resolved at least once. */
export const loras = ref<LoraItem[]>([]);
/** Bookmarked names. */
export const favorites = ref<string[]>([]);
/** Whether the listing has arrived; before that, "not in the list" means "not loaded yet". */
export const listLoaded = ref(false);

const favSet = computed(() => new Set(favorites.value));
const known = computed(() => new Set(loras.value.map((l) => l.name)));

// Sets are replaced rather than mutated: Vue's reactivity does not track Set membership, so a
// `.add()` on a ref'd Set updates nothing on screen.
const previewable = ref<Set<string>>(new Set());
const thumbFailed = ref<Set<string>>(new Set());

let pending: Promise<void> | null = null;

/**
 * Load the listing and favourites once, shared across every caller.
 *
 * Call it on first picker open rather than on mount — a graph full of LoRA nodes would otherwise
 * hit both endpoints on every graph load — and on mount only for a node that already has a name
 * to validate.
 */
export function ensure(force = false): Promise<void> {
  if (force) pending = null;
  if (!pending) {
    pending = Promise.all([listLoras(force), getFavorites(force)]).then(([ls, fs]) => {
      loras.value = ls;
      favorites.value = fs;
      previewable.value = new Set(ls.filter((l) => l.has_preview).map((l) => l.name));
      listLoaded.value = true;
    });
  }
  return pending;
}

/** `characters/ada_v2.safetensors` -> `ada_v2`. */
export function short(name: unknown): string {
  const base = String(name).split("/").pop() || String(name);
  return base.replace(/\.(safetensors|pt|ckpt|bin|lora)$/i, "");
}

/** `characters/ada_v2.safetensors` -> `characters`, or "" for a LoRA at the root. */
export function folder(name: unknown): string {
  const parts = String(name).split("/");
  return parts.length > 1 ? parts.slice(0, -1).join("/") : "";
}

export function hasPreview(name: unknown): boolean {
  return previewable.value.has(String(name));
}

export function preview(name: unknown): string {
  return previewUrl(String(name));
}

export function isFav(name: unknown): boolean {
  return favSet.value.has(String(name));
}

export async function toggleFav(name: unknown): Promise<void> {
  favorites.value = await setFavorite(String(name), !isFav(name));
}

/** A chosen LoRA that is no longer on disk. False until the listing has loaded. */
export function isMissing(name: unknown): boolean {
  const n = String(name);
  return listLoaded.value && !!n && !known.value.has(n);
}

/**
 * Whether to try a thumbnail for a SELECTED name.
 *
 * Optimistic before the listing loads, so a saved LoRA's thumb appears on mount instead of
 * flashing a placeholder; `onThumbError` demotes it if the request 404s. Once the listing is in,
 * `previewable` is authoritative.
 */
export function hasThumb(name: unknown): boolean {
  const n = String(name);
  if (!n || thumbFailed.value.has(n)) return false;
  return listLoaded.value ? previewable.value.has(n) : true;
}

/** A selected-row thumbnail that failed to load — stop offering it. */
export function onThumbError(name: unknown): void {
  thumbFailed.value = new Set(thumbFailed.value).add(String(name));
}

/** An <img> in a list that failed to load — drop it from `previewable` and hide the element. */
export function onImageError(e: Event): void {
  const img = e.target as HTMLImageElement;
  const name = decodeURIComponent(new URL(img.src).searchParams.get("name") || "");
  if (name && previewable.value.has(name)) {
    const next = new Set(previewable.value);
    next.delete(name);
    previewable.value = next;
  }
  img.style.display = "none";
}
