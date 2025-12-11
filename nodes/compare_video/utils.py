from PIL import Image, ImageOps

from ..enums import ResizeModeEnum


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert a hex color string to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    return tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def get_larger_and_smaller_images(img1, img2):
    """Return (larger_image, smaller_image) tuple based on pixel area."""
    area1 = img1.width * img1.height
    area2 = img2.width * img2.height
    if area1 >= area2:
        return (img1, img2)
    else:
        return (img2, img1)


def enforce_target(img, target, color=(0, 0, 0)):
    if img.size == target:
        return img

    w, h = img.size
    tw, th = target

    # If bigger → crop center
    if w > tw or h > th:
        left = (w - tw) // 2
        top = (h - th) // 2
        return img.crop((left, top, left + tw, top + th))

    # If smaller → pad center
    padded = Image.new("RGB", target, color)
    padded.paste(img, ((tw - w) // 2, (th - h) // 2))
    return padded


def resize_fit(img, target_size, resample_method, mode: ResizeModeEnum):
    tw, th = target_size

    if mode == ResizeModeEnum.STRETCH.value:
        return img.resize(target_size, resample_method)

    elif mode == ResizeModeEnum.COVER.value:
        # fill area, cropping excess
        return ImageOps.fit(img, target_size, resample_method, centering=(0.5, 0.5))
    elif mode == ResizeModeEnum.CONTAIN.value:
        # fit inside, preserve aspect ratio, add black borders
        return ImageOps.contain(img, target_size, resample_method)

    elif mode == ResizeModeEnum.PAD.value:
        # same as contain but pads to exact size
        img2 = ImageOps.contain(img, target_size, resample_method)
        padded = Image.new("RGB", target_size, (0, 0, 0))
        padded.paste(img2, ((tw - img2.width)//2, (th - img2.height)//2))
        return padded

    elif mode == ResizeModeEnum.CROP_CENTER.value:
        # crop center region to match size
        w, h = img.size
        left = (w - tw) // 2
        top = (h - th) // 2
        right = left + tw
        bottom = top + th
        return img.crop((left, top, right, bottom))

    else:
        raise ValueError(f"Unknown fit_mode: {mode}")
