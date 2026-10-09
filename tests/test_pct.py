from __future__ import annotations

import struct

import pytest

from extractor.formats import pct
from extractor.ser import SerFormatError

RGBA8 = 0x16


def build_pct(width: int, height: int, mips: int, code: int = RGBA8, fill=lambda level: (255, 0, 0, 255)) -> bytes:
    """A .pct with the observed layout: 64 fixed bytes + 8 per mip, mips largest first, 6-byte trailer."""
    header = bytearray(pct.FIXED_HEADER_SIZE + pct.MIP_TABLE_ENTRY_SIZE * mips)
    header[0:6] = b"\xf0\x00\x0a\x00\x00\x00"
    header[6:10] = b"TCIP"
    struct.pack_into("<4I", header, 0x10, width, height, 1, 1)
    struct.pack_into("<I", header, 0x26, code)
    struct.pack_into("<I", header, 0x30, mips)
    body = bytearray()
    for level in range(mips):
        w, h = max(1, width >> level), max(1, height >> level)
        body += bytes(fill(level)) * (w * h)
    return bytes(header) + bytes(body) + bytes(pct.TRAILER_SIZE)


def test_header_size_grows_with_mip_count() -> None:
    for mips, expected in ((8, 128), (9, 136), (10, 144)):
        assert pct.header_size(pct.parse_header(build_pct(512, 512, mips))) == expected


def test_file_size_formula_matches_layout() -> None:
    data = build_pct(64, 32, 4)
    header = pct.parse_header(data)
    chain = sum(size for _, size, _, _ in pct.mip_offsets(header, "RGBA8"))
    assert len(data) == pct.header_size(header) + chain + pct.TRAILER_SIZE


def test_decode_picks_largest_mip_that_fits() -> None:
    colors = {0: (255, 0, 0, 255), 1: (0, 255, 0, 255), 2: (0, 0, 255, 255)}
    data = build_pct(64, 64, 3, fill=lambda level: colors[level])
    image, header, fmt = pct.decode_mip(data, max_size=32)
    assert fmt == "RGBA8" and header.width == 64
    assert image.size == (32, 32)
    assert image.getpixel((5, 5)) == (0, 255, 0, 255)


def test_single_mip_textures_are_scaled_down() -> None:
    image, _, _ = pct.decode_mip(build_pct(128, 64, 1), max_size=32)
    assert image.size == (32, 16)


def test_average_color_ignores_transparent_texels() -> None:
    data = build_pct(8, 8, 1, fill=lambda level: (10, 20, 30, 255))
    image, _, _ = pct.decode_mip(data)
    assert pct.average_color(image) == (10, 20, 30)


def test_ambiguous_block_format_resolves_on_non_square_textures() -> None:
    # Codes 0x0c/0x0d are BC1 or BC4, told apart by decoding a small mip. For a 4:1
    # texture that mip is 64 x 16, not 64 x 64 (girder_03, 2048 x 512, failed this way).
    width, height, mips = 256, 64, 7
    block = struct.pack("<HHI", 0xF800, 0xF800, 0)  # one solid BC1 block
    header = bytearray(pct.FIXED_HEADER_SIZE + pct.MIP_TABLE_ENTRY_SIZE * mips)
    header[6:10] = b"TCIP"
    struct.pack_into("<4I", header, 0x10, width, height, 1, 1)
    struct.pack_into("<I", header, 0x26, 0x0C)
    struct.pack_into("<I", header, 0x30, mips)
    body = b"".join(block * (max(1, (width >> k) // 4) * max(1, (height >> k) // 4)) for k in range(mips))
    pct._detected.clear()  # the resolved format is cached per code
    try:
        image, _, fmt = pct.decode_mip(bytes(header) + body + bytes(pct.TRAILER_SIZE), max_size=64)
    finally:
        pct._detected.clear()
    assert fmt in ("BC1", "BC4") and image.size == (64, 16)


def test_missing_magic_is_rejected() -> None:
    data = bytearray(build_pct(8, 8, 1))
    data[6:10] = b"XXXX"
    with pytest.raises(SerFormatError, match="TCIP"):
        pct.parse_header(bytes(data))


def test_choose_mip_requires_mips() -> None:
    with pytest.raises(SerFormatError, match="no mips"):
        pct.choose_mip([], 64)
