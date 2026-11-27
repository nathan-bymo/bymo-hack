"""Validazione e normalizzazione delle immagini caricate (allegato C, C.4): nessun byte inatteso entra nella demo."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass

from PIL import Image, ImageOps

from .errors import ApiError
from .repo_code import images_module, prompt_module

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
EXIF_ORIENTATION = 0x0112
THUMB_SIDE = 256


@dataclass
class Ingested:
    data: bytes             # byte da salvare (originali, o PNG raddrizzato se c'era EXIF)
    sha256: str
    format: str             # JPEG | PNG | WEBP
    ext: str
    mime: str
    width: int
    height: int
    has_alpha: bool
    exif_transposed: bool


def sniff(data: bytes) -> str | None:
    """Formato dai magic byte: JPEG FF D8 FF, PNG 89 50 4E 47 0D 0A 1A 0A, WebP RIFF????WEBP."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def ingest(data: bytes, declared_mime: str | None, max_bytes: int, max_pixels: int) -> Ingested:
    if len(data) > max_bytes:
        raise ApiError("FILE_TOO_LARGE", f"The file is larger than {max_bytes // (1024 * 1024)} MB.")
    if declared_mime not in ALLOWED_MIME or sniff(data) != declared_mime:
        raise ApiError("UNSUPPORTED_MEDIA_TYPE",
                       "Only JPEG, PNG and WebP images are accepted (export phone photos to JPEG).",
                       {"declared": declared_mime, "detected": sniff(data)})
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            if width * height > max_pixels:
                raise ApiError("IMAGE_TOO_LARGE", "The image has too many pixels.",
                               {"pixels": width * height, "max_pixels": max_pixels})
            probe.verify()
        with Image.open(io.BytesIO(data)) as image:
            if getattr(image, "n_frames", 1) > 1 or getattr(image, "is_animated", False):
                raise ApiError("UNSUPPORTED_MEDIA_TYPE", "Animated images are not supported.")
            image.load()
            fmt = image.format or ""
            orientation = image.getexif().get(EXIF_ORIENTATION, 1)
            has_alpha = image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info
            if orientation not in (None, 1):
                fixed = ImageOps.exif_transpose(image)
                buffer = io.BytesIO()
                fixed.save(buffer, format="PNG")  # senza perdita; il preprocessing del repo non legge l'EXIF
                return Ingested(buffer.getvalue(), hashlib.sha256(buffer.getvalue()).hexdigest(), "PNG", "png",
                                "image/png", fixed.width, fixed.height, has_alpha, True)
            return Ingested(data, hashlib.sha256(data).hexdigest(), fmt, {"JPEG": "jpg", "PNG": "png",
                            "WEBP": "webp"}[fmt], declared_mime, image.width, image.height, has_alpha, False)
    except ApiError:
        raise
    except Image.DecompressionBombError:
        raise ApiError("IMAGE_TOO_LARGE", "The image has too many pixels.") from None
    except Exception:  # noqa: BLE001 - qualsiasi errore di decodifica = file corrotto
        raise ApiError("CORRUPT_IMAGE", "The image cannot be decoded (corrupt or truncated file).") from None


def make_thumbnail(data: bytes) -> bytes:
    with Image.open(io.BytesIO(data)) as image:
        rgb = images_module().opaque_rgb_on_white(image)
    rgb.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.LANCZOS)
    out = io.BytesIO()
    rgb.save(out, format="JPEG", quality=85)
    return out.getvalue()


def clean_context(raw: dict | None) -> dict[str, str]:
    """Solo le chiavi di `prompt.CONTEXT_FIELDS`; stringhe vuote = campo assente."""
    keys = [k for k, _ in prompt_module().CONTEXT_FIELDS]
    out = {}
    for key in keys:
        value = (raw or {}).get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()[:500]
    return out


def context_hash(context: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(context, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
