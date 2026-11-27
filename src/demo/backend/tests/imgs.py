"""Immagini sintetiche generate in memoria: nessun file di dati nel repo."""

import io

from PIL import Image


def make(fmt="JPEG", size=(48, 32), color=(200, 30, 30), exif_orientation=None, mode="RGB"):
    image = Image.new(mode, size, color if mode == "RGB" else color + (255,))
    buffer = io.BytesIO()
    kwargs = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def animated_gif():
    frames = [Image.new("P", (16, 16), i) for i in (0, 1)]
    buffer = io.BytesIO()
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:])
    return buffer.getvalue()


def animated_webp():
    frames = [Image.new("RGB", (16, 16), c) for c in ((255, 0, 0), (0, 255, 0))]
    buffer = io.BytesIO()
    frames[0].save(buffer, format="WEBP", save_all=True, append_images=frames[1:])
    return buffer.getvalue()
