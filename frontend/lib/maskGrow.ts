/**
 * Growing a box-local mask, as Fresco's backend does (a square max-pool, or min-pool to shrink),
 * for showing the grown shape on the canvas.
 */

function pass(
  src: Uint8ClampedArray,
  w: number,
  h: number,
  r: number,
  horizontal: boolean,
  pick: (a: number, b: number) => number,
) {
  const out = new Uint8ClampedArray(src.length);
  const [outer, inner] = horizontal ? [h, w] : [w, h];
  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < inner; i++) {
      let v = src[horizontal ? o * w + i : i * w + o]!;
      for (let k = Math.max(0, i - r); k <= Math.min(inner - 1, i + r); k++) {
        v = pick(v, src[horizontal ? o * w + k : k * w + o]!);
      }
      out[horizontal ? o * w + i : i * w + o] = v;
    }
  }
  return out;
}

/** `alpha` (`w`×`h`, row-major) grown by `rx`, `ry` pixels — shrunk where they are negative. */
export function growAlpha(
  alpha: Uint8ClampedArray,
  w: number,
  h: number,
  rx: number,
  ry: number,
): Uint8ClampedArray {
  let out = alpha;
  if (rx) out = pass(out, w, h, Math.abs(rx), true, rx > 0 ? Math.max : Math.min);
  if (ry) out = pass(out, w, h, Math.abs(ry), false, ry > 0 ? Math.max : Math.min);
  return out;
}

const cache = new Map<string, Promise<string>>();

/** A mask data URL (white on alpha) grown by `fx`, `fy` of its own width and height. */
export function growMask(url: string, fx: number, fy: number): Promise<string> {
  const key = `${fx.toFixed(4)},${fy.toFixed(4)},${url}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const made = new Promise<string>((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const alpha = new Uint8ClampedArray(canvas.width * canvas.height);
      for (let i = 0; i < alpha.length; i++) alpha[i] = data.data[i * 4 + 3]!;
      const rx = Math.round(fx * canvas.width);
      const ry = Math.round(fy * canvas.height);
      const grown = growAlpha(alpha, canvas.width, canvas.height, rx, ry);
      for (let i = 0; i < grown.length; i++) {
        data.data.set([255, 255, 255, grown[i]!], i * 4);
      }
      ctx.putImageData(data, 0, 0);
      resolve(canvas.toDataURL("image/png"));
    };
    img.onerror = reject;
    img.src = url;
  });
  if (cache.size > 64) cache.delete(cache.keys().next().value!);
  cache.set(key, made);
  return made;
}
