"""Small encoders shared by the exporters: atomic writes, gzip buffers, RGBM light, PNG."""

from __future__ import annotations

import gzip
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image

# Exported buffers are gzip files; the dev server sends them with Content-Encoding: gzip, so the
# browser hands the viewer the raw bytes. Level 1: 3-6x smaller for a fraction of a second per buffer.
GZIP_LEVEL = 1
# RGBM: rgb in 0..1 times a shared multiplier in alpha, times RGBM_RANGE. 8 bits per channel cover
# the bake's HDR light (1.0 = white, sunlit texels around 2-4) with headroom for lamps.
RGBM_RANGE = 16.0


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_buffer(folder: Path, name: str, array: np.ndarray, dtype: str, components: int, **extra: object) -> dict:
    """Write ``array`` as gzip-compressed ``<name>.bin.gz``; returns its manifest entry."""
    data = np.ascontiguousarray(array, dtype=dtype)
    write_atomic(folder / f"{name}.bin.gz", gzip.compress(data.tobytes(), compresslevel=GZIP_LEVEL, mtime=0))
    entry = {"uri": f"{name}.bin.gz", "dtype": np.dtype(dtype).name, "components": components}
    if np.dtype(dtype).itemsize > 1:
        entry["byte_order"] = "little"
    entry.update(extra)
    return entry


def rgbm(rgb: np.ndarray) -> np.ndarray:
    """Linear HDR rgb (..., 3) -> uint8 RGBM (..., 4); decode with ``rgb * m * RGBM_RANGE``."""
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, RGBM_RANGE)
    m = np.clip(rgb.max(axis=-1, keepdims=True) / RGBM_RANGE, 1.0 / 255.0, 1.0)
    m = np.ceil(m * 255.0) / 255.0
    encoded = np.concatenate([rgb / (m * RGBM_RANGE), m], axis=-1)
    return np.clip(np.round(encoded * 255.0), 0, 255).astype(np.uint8)


def unrgbm(encoded: np.ndarray) -> np.ndarray:
    """The inverse of :func:`rgbm` (for tests and checks)."""
    values = encoded.astype(np.float64) / 255.0
    return values[..., :3] * values[..., 3:4] * RGBM_RANGE


def png_bytes(pixels: np.ndarray) -> bytes:
    """uint8 (h, w, 3 or 4) -> PNG."""
    mode = "RGBA" if pixels.shape[-1] == 4 else "RGB"
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(pixels), mode).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()
