"""LoRA sidecars: preview files, video poster frames, and ComfyUI-Lora-Manager metadata.

Lora Manager writes `<lora>.metadata.json` beside each LoRA; its `civitai` block carries the
version info and the example images with their generation params. Examples it has downloaded
are preferred over the Civitai URLs. Without the sidecar, the safetensors header (kohya `ss_*`
keys) is the only source.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
from html.parser import HTMLParser

from . import lora_manager

IMAGE_EXTS = (
    ".preview.png",
    ".preview.jpg",
    ".preview.jpeg",
    ".preview.webp",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
)
VIDEO_EXTS = (".preview.mp4", ".preview.webm", ".mp4", ".webm")

POSTER_MAX = 512
_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".cache"
)
_POSTER_DIR = os.path.join(_CACHE, "lora_posters")
_THUMB_DIR = os.path.join(_CACHE, "lora_thumbs")
#: The thumbnail sizes served (longest side, px); a request is rounded up to one of them, so the
#: cache holds a few files per preview however the browser is sized.
THUMB_SIZES = (64, 128, 256, 512)

# Values Civitai's `meta` carries that are too large to ship to the browser.
_META_DROP = {"comfy", "workflow", "hashes", "resources", "civitaiResources", "additionalResources"}
_META_MAX_STR = 4000


# --- previews ------------------------------------------------------------------------------


def find_preview(lora_path: str | None, siblings: set[str] | None = None) -> tuple[str, str] | None:
    """(path, "image" | "video") for the LoRA's sidecar preview, images first.

    `siblings` is the directory's file names, so a listing can scan each folder once.
    """
    if not lora_path:
        return None
    folder, file = os.path.split(lora_path)
    stem = os.path.splitext(file)[0]
    for exts, kind in ((IMAGE_EXTS, "image"), (VIDEO_EXTS, "video")):
        for ext in exts:
            name = stem + ext
            if siblings is not None:
                if name in siblings:
                    return os.path.join(folder, name), kind
            elif os.path.isfile(os.path.join(folder, name)):
                return os.path.join(folder, name), kind
    return None


def poster(video_path: str) -> str | None:
    """Cached JPEG of the video's first frame, or None if it can't be decoded."""
    try:
        st = os.stat(video_path)
    except OSError:
        return None
    key = hashlib.sha1(f"{video_path}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()
    out = os.path.join(_POSTER_DIR, key + ".jpg")
    if os.path.isfile(out):
        return out
    os.makedirs(_POSTER_DIR, exist_ok=True)
    tmp = out + ".tmp"
    try:
        if _poster_av(video_path, tmp) or _poster_ffmpeg(video_path, tmp):
            os.replace(tmp, out)
            return out
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return None


def thumb(image_path: str, size: int) -> str | None:
    """Cached JPEG of a preview image, its longest side `size` rounded up to one of THUMB_SIZES:
    a list row or a card decodes a few KB instead of the full-size preview. None if it can't be
    read."""
    size = next((s for s in THUMB_SIZES if s >= size), THUMB_SIZES[-1])
    try:
        st = os.stat(image_path)
    except OSError:
        return None
    key = hashlib.sha1(f"{image_path}|{st.st_mtime_ns}|{st.st_size}|{size}".encode()).hexdigest()
    out = os.path.join(_THUMB_DIR, key + ".jpg")
    if os.path.isfile(out):
        return out
    os.makedirs(_THUMB_DIR, exist_ok=True)
    tmp = out + ".tmp"
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            img.draft("RGB", (size, size))
            img = img.convert("RGB")
            img.thumbnail((size, size))
            img.save(tmp, "JPEG", quality=85)
        os.replace(tmp, out)
        return out
    except Exception:  # noqa: BLE001 - an unreadable preview just has no thumbnail
        return None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _poster_av(src: str, dst: str) -> bool:
    try:
        import av
    except ImportError:
        return False
    try:
        with av.open(src) as container:
            for frame in container.decode(video=0):
                img = frame.to_image().convert("RGB")
                img.thumbnail((POSTER_MAX, POSTER_MAX))
                img.save(dst, "JPEG", quality=85)
                return True
    except Exception:  # noqa: BLE001 - any decode failure just means no poster
        return False
    return False


def _poster_ffmpeg(src: str, dst: str) -> bool:
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    scale = f"scale={POSTER_MAX}:{POSTER_MAX}:force_original_aspect_ratio=decrease"
    cmd = [exe, "-v", "error", "-y", "-i", src, "-frames:v", "1", "-vf", scale]
    cmd += ["-f", "image2", "-c:v", "mjpeg", dst]
    try:
        subprocess.run(cmd, check=True, timeout=30, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return os.path.isfile(dst) and os.path.getsize(dst) > 0


# --- info ----------------------------------------------------------------------------------


def read_info(lora_path: str) -> dict:
    """Everything the details panel shows for one LoRA. Missing sources are skipped."""
    info: dict = {"size": _size(lora_path), "images": []}
    md = _load_json(os.path.splitext(lora_path)[0] + ".metadata.json")
    if isinstance(md, dict):
        info.update(_from_lora_manager(md))
    header = _safetensors_metadata(lora_path)
    if header:
        info["training"] = _from_header(header)
    return info


def sha256_of(lora_path: str) -> str | None:
    md = _load_json(os.path.splitext(lora_path)[0] + ".metadata.json")
    return md.get("sha256") if isinstance(md, dict) else None


def _size(path: str) -> int | None:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _load_json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _from_lora_manager(md: dict) -> dict:
    civ = md.get("civitai") if isinstance(md.get("civitai"), dict) else {}
    model = civ.get("model") if isinstance(civ.get("model"), dict) else {}
    creator = civ.get("creator") if isinstance(civ.get("creator"), dict) else {}
    url = None
    if civ.get("modelId"):
        url = f"https://civitai.com/models/{civ['modelId']}"
        if civ.get("id"):
            url += f"?modelVersionId={civ['id']}"
    return {
        "title": md.get("model_name") or model.get("name"),
        "version": civ.get("name"),
        "base_model": md.get("base_model") or civ.get("baseModel"),
        "sha256": md.get("sha256"),
        "trigger_words": [str(w) for w in civ.get("trainedWords") or [] if w],
        "tags": [str(t) for t in md.get("tags") or model.get("tags") or []],
        "description": html_to_text(md.get("modelDescription") or model.get("description")),
        "version_description": html_to_text(civ.get("description")),
        "notes": str(md.get("notes") or ""),
        "creator": creator.get("username"),
        "url": url,
        "usage": _usage(md.get("usage_tips")),
        "images": _examples(civ, md.get("sha256")),
    }


def _examples(civ: dict, sha256: str | None) -> list[dict]:
    """Civitai examples, each with `file` set when Lora Manager has it on disk, then the
    user's imported examples (which only exist locally)."""
    files = lora_manager.local_files(lora_manager.example_folder(sha256))
    out = []
    for i, img in enumerate(civ.get("images") or []):
        if isinstance(img, dict) and (img.get("url") or f"image_{i}" in files):
            out.append(_image(img, files.get(f"image_{i}")))
    for img in civ.get("customImages") or []:
        if isinstance(img, dict) and f"custom_{img.get('id')}" in files:
            out.append(_image(img, files[f"custom_{img['id']}"]))
    return out


def _usage(tips) -> dict:
    if isinstance(tips, str):
        try:
            tips = json.loads(tips)
        except ValueError:
            return {}
    if not isinstance(tips, dict):
        return {}
    return {k: v for k, v in tips.items() if isinstance(v, (str, int, float, bool))}


def _image(img: dict, file: str | None) -> dict:
    meta = img.get("meta") if isinstance(img.get("meta"), dict) else {}
    kind = "video" if img.get("type") == "video" else "image"
    if file:
        kind = lora_manager.MEDIA_EXTS.get(os.path.splitext(file)[1].lower(), kind)
    return {
        "url": str(img.get("url") or ""),
        "file": file,
        "kind": kind,
        "width": img.get("width"),
        "height": img.get("height"),
        "nsfw_level": img.get("nsfwLevel") or 0,
        "meta": {k: v for k, v in meta.items() if k not in _META_DROP and _small(v)},
    }


def _small(v) -> bool:
    if isinstance(v, str):
        return len(v) <= _META_MAX_STR
    if isinstance(v, (int, float, bool)):
        return True
    if isinstance(v, list):
        return all(isinstance(x, (str, int, float)) for x in v) and len(v) <= 32
    return False


def _safetensors_metadata(path: str) -> dict | None:
    if not path.lower().endswith(".safetensors"):
        return None
    try:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            if n > 64 * 1024 * 1024:
                return None
            header = json.loads(f.read(n))
    except (OSError, ValueError, struct.error):
        return None
    meta = header.get("__metadata__") if isinstance(header, dict) else None
    return meta if isinstance(meta, dict) else None


def _from_header(meta: dict) -> dict:
    out = {}
    for key, label in (
        ("ss_base_model_version", "base_model"),
        ("modelspec.architecture", "architecture"),
        ("ss_network_module", "network"),
        ("ss_network_dim", "dim"),
        ("ss_network_alpha", "alpha"),
        ("ss_num_train_images", "train_images"),
        ("ss_num_epochs", "epochs"),
        ("ss_steps", "steps"),
        ("ss_learning_rate", "learning_rate"),
        ("ss_resolution", "resolution"),
        ("modelspec.trigger_phrase", "trigger_phrase"),
    ):
        v = meta.get(key)
        if v not in (None, "", "None"):
            out[label] = str(v)
    tags = _top_tags(meta.get("ss_tag_frequency"))
    if tags:
        out["top_tags"] = tags
    return out


def _top_tags(raw, limit: int = 24) -> list[str]:
    try:
        freq = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return []
    if not isinstance(freq, dict):
        return []
    totals: dict[str, int] = {}
    for bucket in freq.values():
        if isinstance(bucket, dict):
            for tag, n in bucket.items():
                if isinstance(n, int):
                    totals[tag.strip()] = totals.get(tag.strip(), 0) + n
    return [t for t, _ in sorted(totals.items(), key=lambda kv: -kv[1])[:limit] if t]


# --- html ----------------------------------------------------------------------------------

_BLOCK = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "tr"}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _BLOCK:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("• ")

    def handle_endtag(self, tag):
        if tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(src) -> str:
    """Civitai descriptions are HTML; the panel shows them as plain text."""
    if not src:
        return ""
    p = _Text()
    p.feed(str(src))
    lines = [ln.strip() for ln in "".join(p.parts).splitlines()]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()
