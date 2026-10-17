// A right-click inside one of our editors is ours: it must not reach the graph and open ComfyUI's
// node menu over the editor. Text fields keep the browser's own menu (copy, paste, spelling).

const TEXT_FIELD = "input, textarea, [contenteditable=''], [contenteditable='true']";

export function keepContextMenu(e: MouseEvent): void {
  e.stopPropagation();
  if (!(e.target as Element | null)?.closest?.(TEXT_FIELD)) e.preventDefault();
}
