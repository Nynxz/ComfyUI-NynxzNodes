// Sidecar mask storage: the mask is saved as its OWN grayscale file (white = masked) via the
// stock `POST /upload/image` endpoint — the original image is NEVER modified. `nynxz.IO.LoadImage`
// loads the original RGB and this separate mask. Saved into temp/clipspace (wiped on restart) with
// ONE stable name per source image, OVERWRITTEN each save (`overwrite=true`) so nothing piles up.
// `baseKey` strips our own prefixes so re-editing reuses the same file instead of growing the name.
import { thumbUrl, type SourceType } from "@/lib/imagePickApi";

export interface OriginalRef {
  filename: string; // bare filename (no subfolder)
  subfolder: string;
  type: SourceType;
}
export interface MaskUploadResult {
  name: string;
  subfolder: string;
  type: string;
}

// Save masked copies into the TEMP dir (ComfyUI wipes temp/ on restart via cleanup_temp() in
// main.py) so they don't permanently clutter — unlike ComfyUI's own mask editor, which writes to
// input/clipspace and never cleans up. Change to 'input' if you'd rather masks survive restarts.
const SAVE_TYPE = "temp";

const stripExt = (f: string) => f.replace(/\.[^./\\]+$/, "");
const sanitize = (s: string) =>
  s.replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "image";
// A stable, prefix-free key for a source image, so masked copies don't accumulate.
function baseKey(ref: OriginalRef): string {
  const name = stripExt(ref.filename)
    .replace(/^clipspace-paint-/, "")
    .replace(/^clipspace-/, "");
  const sub = ref.subfolder && ref.subfolder !== "clipspace" ? ref.subfolder : "";
  return sanitize(sub ? `${sub}/${name}` : name);
}

async function postUpload(
  route: string,
  uploadName: string,
  originalRef: OriginalRef | null,
  blob: Blob,
  opts: { type?: string; subfolder?: string; overwrite?: boolean } = {},
): Promise<MaskUploadResult | null> {
  try {
    const body = new FormData();
    body.append("image", blob, uploadName);
    if (originalRef)
      body.append(
        "original_ref",
        JSON.stringify({
          filename: originalRef.filename,
          subfolder: originalRef.subfolder,
          type: originalRef.type,
        }),
      );
    body.append("type", opts.type ?? SAVE_TYPE);
    body.append("subfolder", opts.subfolder ?? "clipspace");
    if (opts.overwrite ?? true) body.append("overwrite", "true"); // reuse stable names when requested
    const res = await fetch(route, { method: "POST", body });
    const d = await res.json();
    if (!d || !d.name) return null;
    return {
      name: String(d.name),
      subfolder: String(d.subfolder ?? opts.subfolder ?? "clipspace"),
      type: String(d.type ?? opts.type ?? "input"),
    };
  } catch {
    return null;
  }
}

// maskCanvas: RGB white, alpha = coverage. Flatten over black → a grayscale mask (white = masked,
// fully opaque), which the node reads directly. The base image is NEVER touched (sidecar model).
function maskCanvasToGray(maskCanvas: HTMLCanvasElement): HTMLCanvasElement {
  const out = document.createElement("canvas");
  out.width = maskCanvas.width;
  out.height = maskCanvas.height;
  const c = out.getContext("2d")!;
  c.fillStyle = "#000";
  c.fillRect(0, 0, out.width, out.height);
  c.drawImage(maskCanvas, 0, 0);
  return out;
}
function canvasToBlob(c: HTMLCanvasElement): Promise<Blob> {
  return new Promise((res, rej) =>
    c.toBlob((b) => (b ? res(b) : rej(new Error("toBlob failed"))), "image/png"),
  );
}
function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((res, rej) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => res(img);
    img.onerror = () => rej(new Error(`Failed to load image: ${url}`));
    img.src = url;
  });
}

// Save the mask as its OWN grayscale file in temp/clipspace, one stable name per source image
// (overwritten each save → no spam, wiped on restart). Returns the saved ref.
export async function uploadMaskFile(
  maskCanvas: HTMLCanvasElement,
  ref: OriginalRef,
): Promise<MaskUploadResult | null> {
  const blob = await canvasToBlob(maskCanvasToGray(maskCanvas));
  return postUpload("/upload/image", `mask-${baseKey(ref)}.png`, null, blob);
}

// Migrate the old Nynxz sidecar mask model to ComfyUI's native clipspace model. The
// sidecar is grayscale where white = masked; /upload/mask expects an alpha channel where
// transparent = masked, so alpha is inverted before upload. The result is a normal
// input/clipspace image with alpha, which ComfyUI's stock mask editor can load/re-edit.
export async function uploadSidecarAsClipspaceImage(
  maskRef: OriginalRef,
  originalRef: OriginalRef,
): Promise<MaskUploadResult | null> {
  try {
    const maskImg = await loadImage(thumbUrl(maskRef));
    const canvas = document.createElement("canvas");
    canvas.width = maskImg.naturalWidth || maskImg.width;
    canvas.height = maskImg.naturalHeight || maskImg.height;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(maskImg, 0, 0);
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < data.data.length; i += 4) {
      const masked = data.data[i];
      data.data[i] = 0;
      data.data[i + 1] = 0;
      data.data[i + 2] = 0;
      data.data[i + 3] = 255 - masked;
    }
    ctx.putImageData(data, 0, 0);
    const blob = await canvasToBlob(canvas);
    return postUpload(
      "/upload/mask",
      `clipspace-nynxz-masked-${Date.now()}.png`,
      originalRef,
      blob,
      { type: "input", subfolder: "clipspace", overwrite: false },
    );
  } catch {
    return null;
  }
}

// When the RGB paint layer is used, the painted result becomes the new base IMAGE. Save the
// (opaque) composite as its own file; stable name, overwritten. `blob` is a PNG of base+paint.
export async function uploadPaintedImage(
  blob: Blob,
  ref: OriginalRef,
): Promise<MaskUploadResult | null> {
  return postUpload("/upload/image", `paint-${baseKey(ref)}.png`, null, blob);
}

// ComfyUI's annotated value format: `subfolder/filename [type]`.
export function annotatedValue(r: MaskUploadResult): string {
  const path = r.subfolder ? `${r.subfolder}/${r.name}` : r.name;
  return `${path} [${r.type}]`;
}
