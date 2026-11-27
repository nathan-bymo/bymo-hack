"""Image preparation, identical for training and inference."""

import base64
import io

from PIL import Image

PATCH = 16  # Apertus v1.5 makes one image token per 16 x 16 pixels.
MAX_ASPECT = 8
JPEG_QUALITY = 92


def opaque_rgb_on_white(image):
    """RGB with transparency composited onto white (from the Kaggle data page)."""
    if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in image.info:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def target_size(width, height, side):
    """About side x side pixels, aspect ratio kept (capped for thin strips), both sides multiples of PATCH."""
    aspect = min(MAX_ASPECT, max(1 / MAX_ASPECT, width / height))
    return tuple(max(PATCH, round(side * factor / PATCH) * PATCH) for factor in (aspect**0.5, aspect**-0.5))


def prepared(path, side=512):
    """The image as the model gets it: transparency on white and resized."""
    with Image.open(path) as image:
        rgb = opaque_rgb_on_white(image)
    return rgb.resize(target_size(*rgb.size, side), Image.LANCZOS)


def data_url(path, side=512):
    """The prepared image as a base64 JPEG data URL."""
    buffer = io.BytesIO()
    prepared(path, side).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def prepared_jpeg(path, side=512):
    """Same pixels as data_url after JPEG decoding, so local models and the API read the same image."""
    return Image.open(io.BytesIO(base64.b64decode(data_url(path, side).split(",", 1)[1]))).convert("RGB")
