"""Decoder for ``.pct`` textures.

Layout of a ``.pct`` file (the separate ``.pct_header`` entry is a copy of the
header)::

    0x00   header: 64 fixed bytes, then 8 bytes per mip (a mip table)
             0x06  "TCIP" magic
             0x10  u32 width, u32 height, u32 depth, u32 faces/array size
             0x26  u32 format code
             0x30  u32 mip count
    0x40 + 8 * mips   mip chain, largest mip first, each mip tightly packed
    end    6 trailing bytes

Verified on 9-mip (136-byte header), 10-mip (144) and 8-mip (128) textures:
``file size == 64 + 8 * mips + mip chain + 6`` exactly.

Format codes, established by decoding and checking image smoothness (the
codes are the engine's own enum, not DXGI values):

=========  ===========  =============================================
code       format       used for
=========  ===========  =============================================
``0x24``   BC5          normal maps (``*_nm``), 5,238 textures
``0x33``   BC7          colour textures, 2,216
``0x34``   BC7          colour / emissive textures, 1,849
``0x11``   BC3          decals with alpha (e.g. ``blood_spots_01``), 153
``0x31``   BC6H         HDR lightmaps (``[scene]/lm/*_rnm0``, ``*_dircolor``)
``0x16``   RGBA8        small uncompressed textures
``0x00``   RGBA8
=========  ===========  =============================================

Previews decode BC6H with Pillow, which clamps the HDR values to 0..1;
:func:`decode_hdr` keeps the full range (see ``bc6h``).

Other block-compressed codes (``0x25``, ``0x0c``, ``0x0d``: 0.5 bytes per
pixel) are resolved at runtime by decoding each candidate format and keeping
the smoothest result.

Two wrong readings along the way, both producing structured noise: data at
"file size minus mip chain" (off by the 6-byte trailer), and a fixed 136-byte
header (right only for 9-mip textures).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
from PIL import Image

from extractor.formats import bc6h
from extractor.ser.errors import SerFormatError

FIXED_HEADER_SIZE = 0x40
MIP_TABLE_ENTRY_SIZE = 8
TRAILER_SIZE = 6
MAGIC = b"TCIP"
MAGIC_OFFSET = 0x06

# (bytes per 4x4 block or per pixel, is_block_compressed)
_FORMAT_LAYOUT = {
    "BC1": (8, True),
    "BC4": (8, True),
    "BC2": (16, True),
    "BC3": (16, True),
    "BC5": (16, True),
    "BC6H": (16, True),
    "BC7": (16, True),
    "RGBA8": (4, False),
}
KNOWN_CODES = {0x24: "BC5", 0x33: "BC7", 0x34: "BC7", 0x11: "BC3", 0x31: "BC6H", 0x16: "RGBA8", 0x00: "RGBA8"}
CANDIDATES_BY_BLOCK_BYTES = {8: ("BC1", "BC4"), 16: ("BC7", "BC3", "BC5")}
_PILLOW_BCN = {
    "BC1": (1, "RGBA"),
    "BC2": (2, "RGBA"),
    "BC3": (3, "RGBA"),
    "BC4": (4, "L"),
    "BC5": (5, "RGB"),
    "BC6H": (6, "RGB"),
    "BC7": (7, "RGBA"),
}

# Formats chosen at runtime for codes outside KNOWN_CODES, shared by all textures of that code.
_detected: dict[int, str] = {}


@dataclass(frozen=True)
class PctHeader:
    width: int
    height: int
    depth: int
    faces: int
    format_code: int
    mip_count: int


def header_size(header: PctHeader) -> int:
    """Bytes before the first mip: the fixed part plus the mip table."""
    return FIXED_HEADER_SIZE + MIP_TABLE_ENTRY_SIZE * header.mip_count


def parse_header(data: bytes | bytearray | memoryview) -> PctHeader:
    """Parse the fixed part of the header; needs at least the first 64 bytes."""
    raw = bytes(data[:FIXED_HEADER_SIZE])
    if len(raw) < FIXED_HEADER_SIZE or raw[MAGIC_OFFSET:MAGIC_OFFSET + 4] != MAGIC:
        raise SerFormatError("not a .pct texture (missing TCIP magic)", offset=MAGIC_OFFSET)
    width, height, depth, faces = struct.unpack_from("<4I", raw, 0x10)
    (format_code,) = struct.unpack_from("<I", raw, 0x26)
    (mip_count,) = struct.unpack_from("<I", raw, 0x30)
    return PctHeader(width, height, max(depth, 1), max(faces, 1), format_code, max(mip_count, 1))


def _mip_bytes(fmt: str, width: int, height: int) -> int:
    unit, compressed = _FORMAT_LAYOUT[fmt]
    if compressed:
        return max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * unit
    return width * height * unit


def mip_offsets(header: PctHeader, fmt: str) -> list[tuple[int, int, int, int]]:
    """(offset, size, width, height) of each mip of the first face/slice, largest first."""
    out, offset = [], header_size(header)
    for level in range(header.mip_count):
        w, h = max(1, header.width >> level), max(1, header.height >> level)
        size = _mip_bytes(fmt, w, h)
        out.append((offset, size, w, h))
        offset += size
    return out


def decode_payload(fmt: str, payload: bytes, width: int, height: int) -> Image.Image:
    """Decode one mip's bytes in format ``fmt`` to an RGBA image."""
    if fmt == "RGBA8":
        return Image.frombytes("RGBA", (width, height), payload)
    number, mode = _PILLOW_BCN[fmt]
    # Pillow decodes whole blocks; pad tiny mips up to one block.
    padded_w, padded_h = max(4, (width + 3) // 4 * 4), max(4, (height + 3) // 4 * 4)
    image = Image.frombytes(mode, (padded_w, padded_h), payload, "bcn", number)
    if fmt == "BC5":
        # Two-channel normal map: rebuild blue from the unit-length constraint for display.
        rg = np.asarray(image, dtype=np.float32)[..., :2] / 255.0 * 2.0 - 1.0
        b = np.sqrt(np.clip(1.0 - (rg ** 2).sum(-1), 0.0, 1.0))
        rgb = np.dstack([(rg + 1.0) * 127.5, (b + 1.0) * 127.5]).astype(np.uint8)
        image = Image.fromarray(rgb, "RGB")
    return image.convert("RGBA").crop((0, 0, width, height))


def _smoothness(image: Image.Image) -> float:
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    return float(np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean())


def resolve_format(header: PctHeader, total_size: int, sample: tuple[bytes, int, int] | None = None) -> str:
    """The pixel format for a header, detecting unknown codes from the file size and, if needed, the data.

    ``sample`` is a small mip as (bytes from its start, width, height).
    """
    if header.format_code in KNOWN_CODES:
        return KNOWN_CODES[header.format_code]
    if header.format_code in _detected:
        return _detected[header.format_code]
    pixels = sum(max(4, header.width >> i) * max(4, header.height >> i) for i in range(header.mip_count))
    per_pixel = (total_size - header_size(header) - TRAILER_SIZE) / max(pixels * header.faces * header.depth, 1)
    block_bytes = 8 if per_pixel < 0.75 else 16
    candidates = CANDIDATES_BY_BLOCK_BYTES[block_bytes]
    if sample is None:
        return candidates[0]
    # Decode the same small mip with every candidate and keep the smoothest image.
    payload, width, height = sample
    best = min(
        candidates,
        key=lambda fmt: _smoothness(decode_payload(fmt, payload[: _mip_bytes(fmt, width, height)], width, height)),
    )
    _detected[header.format_code] = best
    return best


def decode_mip(data: bytes, max_size: int = 1024) -> tuple[Image.Image, PctHeader, str]:
    """Decode the largest mip that fits within ``max_size`` pixels on its longest side."""
    header = parse_header(data)
    fmt = KNOWN_CODES.get(header.format_code)
    if fmt is None:
        probe_fmt = "BC1" if (len(data) - header_size(header)) / max(header.width * header.height, 1) < 0.75 else "BC7"
        # The largest mip within 64 x 64 (64 x 16 for a 4:1 texture), decoded at its own size.
        sample_mips = [m for m in mip_offsets(header, probe_fmt) if 16 <= max(m[2], m[3]) and m[2] <= 64 and m[3] <= 64]
        sample = (data[sample_mips[0][0]:], sample_mips[0][2], sample_mips[0][3]) if sample_mips else None
        fmt = resolve_format(header, len(data), sample)
    offset, size, width, height = choose_mip(mip_offsets(header, fmt), max_size)
    if offset + size > len(data):
        raise SerFormatError(f"mip {width}x{height} runs past the end of the texture", offset=offset)
    image = decode_payload(fmt, bytes(data[offset:offset + size]), width, height)
    return fit(image, max_size), header, fmt


def decode_hdr(data: bytes) -> np.ndarray:
    """The largest mip of a BC6H texture as float32 RGB (height, width, 3), without clamping."""
    header = parse_header(data)
    if KNOWN_CODES.get(header.format_code) != "BC6H":
        raise SerFormatError(f"format {header.format_code:#x} is not BC6H", offset=0x26)
    offset, size, width, height = mip_offsets(header, "BC6H")[0]
    if offset + size > len(data):
        raise SerFormatError(f"mip {width}x{height} runs past the end of the texture", offset=offset)
    return bc6h.decode(bytes(data[offset:offset + size]), width, height)


def choose_mip(mips: list[tuple[int, int, int, int]], max_size: int) -> tuple[int, int, int, int]:
    """The largest mip within ``max_size``, or the smallest mip when none fits (e.g. single-mip sky domes)."""
    if not mips:
        raise SerFormatError("texture has no mips")
    for mip in mips:
        if max(mip[2], mip[3]) <= max_size:
            return mip
    return mips[-1]


def fit(image: Image.Image, max_size: int) -> Image.Image:
    """Downscale so the longest side is at most ``max_size``."""
    if max(image.size) <= max_size:
        return image
    copy = image.copy()
    copy.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return copy


def average_color(image: Image.Image) -> tuple[int, int, int]:
    """Mean sRGB colour, ignoring fully transparent texels."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32).reshape(-1, 4)
    visible = rgba[rgba[:, 3] > 8]
    pixels = visible if len(visible) else rgba
    r, g, b = pixels[:, :3].mean(axis=0)
    return int(round(r)), int(round(g)), int(round(b))
