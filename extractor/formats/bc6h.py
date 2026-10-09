"""BC6H (unsigned) block decoder producing full-range HDR floats.

Pillow decodes BC6H too, but clamps the result to 8-bit 0..1, which flattens
the game's lightmaps (about a fifth of their texels are brighter than 1).

BC6H in short (Direct3D 11 specification): 128-bit blocks of 4 x 4 texels.
The mode (2 bits, or 5 when the first two are 1x) selects one of 14 layouts:
one or two subsets of texels, how many bits each endpoint has, and whether
endpoints after the first are stored as signed deltas from it ("transformed").
The endpoint bits are scattered through the block in a fixed order per mode
(``_LAYOUTS``). Two-subset modes then store a 5-bit partition number (shared
with BC7's first 32 two-subset partitions), followed by 3-bit indices (4-bit
with one subset). The first texel of each subset (the "anchor") drops its
index's top bit. Endpoints are unquantized to 16 bits, interpolated with fixed
weights, scaled by 31/64 and read as half floats.

The whole image is decoded at once with numpy, one mode at a time.
"""

from __future__ import annotations

import re

import numpy as np

# Per mode, keyed by its value (the bits read first are the low bits):
# (endpoint bits, delta bits per channel or None when not transformed, subsets, endpoint bit layout).
# The layout lists, in stream order, which endpoint (0..3: subset 0 A, B, subset 1 A, B) and
# channel each bit belongs to. "r0[9:0]" is read from bit 0 up; "r0[10:15]" (as the
# specification writes the reversed fields) from bit 15 down.
_MODES: dict[int, tuple[int, tuple[int, int, int] | None, int, str]] = {
    0b00: (10, (5, 5, 5), 2,
           "g2[4] b2[4] b3[4] r0[9:0] g0[9:0] b0[9:0] r1[4:0] g3[4] g2[3:0] g1[4:0] b3[0] g3[3:0] "
           "b1[4:0] b3[1] b2[3:0] r2[4:0] b3[2] r3[4:0] b3[3]"),
    0b01: (7, (6, 6, 6), 2,
           "g2[5] g3[4] g3[5] r0[6:0] b3[0] b3[1] b2[4] g0[6:0] b2[5] b3[2] g2[4] b0[6:0] b3[3] b3[5] "
           "b3[4] r1[5:0] g2[3:0] g1[5:0] g3[3:0] b1[5:0] b2[3:0] r2[5:0] r3[5:0]"),
    0b00010: (11, (5, 4, 4), 2,
              "r0[9:0] g0[9:0] b0[9:0] r1[4:0] r0[10] g2[3:0] g1[3:0] g0[10] b3[0] g3[3:0] b1[3:0] "
              "b0[10] b3[1] b2[3:0] r2[4:0] b3[2] r3[4:0] b3[3]"),
    0b00110: (11, (4, 5, 4), 2,
              "r0[9:0] g0[9:0] b0[9:0] r1[3:0] r0[10] g3[4] g2[3:0] g1[4:0] g0[10] g3[3:0] b1[3:0] "
              "b0[10] b3[1] b2[3:0] r2[3:0] b3[0] b3[2] r3[3:0] g2[4] b3[3]"),
    0b01010: (11, (4, 4, 5), 2,
              "r0[9:0] g0[9:0] b0[9:0] r1[3:0] r0[10] b2[4] g2[3:0] g1[3:0] g0[10] b3[0] g3[3:0] "
              "b1[4:0] b0[10] b2[3:0] r2[3:0] b3[1] b3[2] r3[3:0] b3[4] b3[3]"),
    0b01110: (9, (5, 5, 5), 2,
              "r0[8:0] b2[4] g0[8:0] g2[4] b0[8:0] b3[4] r1[4:0] g3[4] g2[3:0] g1[4:0] b3[0] g3[3:0] "
              "b1[4:0] b3[1] b2[3:0] r2[4:0] b3[2] r3[4:0] b3[3]"),
    0b10010: (8, (6, 5, 5), 2,
              "r0[7:0] g3[4] b2[4] g0[7:0] b3[2] g2[4] b0[7:0] b3[3] b3[4] r1[5:0] g2[3:0] g1[4:0] "
              "b3[0] g3[3:0] b1[4:0] b3[1] b2[3:0] r2[5:0] r3[5:0]"),
    0b10110: (8, (5, 6, 5), 2,
              "r0[7:0] b3[0] b2[4] g0[7:0] g2[5] g2[4] b0[7:0] g3[5] b3[4] r1[4:0] g3[4] g2[3:0] "
              "g1[5:0] g3[3:0] b1[4:0] b3[1] b2[3:0] r2[4:0] b3[2] r3[4:0] b3[3]"),
    0b11010: (8, (5, 5, 6), 2,
              "r0[7:0] b3[1] b2[4] g0[7:0] b2[5] g2[4] b0[7:0] b3[5] b3[4] r1[4:0] g3[4] g2[3:0] "
              "g1[4:0] b3[0] g3[3:0] b1[5:0] b2[3:0] r2[4:0] b3[2] r3[4:0] b3[3]"),
    0b11110: (6, None, 2,
              "r0[5:0] g3[4] b3[0] b3[1] b2[4] g0[5:0] g2[5] b2[5] b3[2] g2[4] b0[5:0] g3[5] b3[3] "
              "b3[5] b3[4] r1[5:0] g2[3:0] g1[5:0] g3[3:0] b1[5:0] b2[3:0] r2[5:0] r3[5:0]"),
    0b00011: (10, None, 1, "r0[9:0] g0[9:0] b0[9:0] r1[9:0] g1[9:0] b1[9:0]"),
    0b00111: (11, (9, 9, 9), 1, "r0[9:0] g0[9:0] b0[9:0] r1[8:0] r0[10] g1[8:0] g0[10] b1[8:0] b0[10]"),
    0b01011: (12, (8, 8, 8), 1,
              "r0[9:0] g0[9:0] b0[9:0] r1[7:0] r0[10:11] g1[7:0] g0[10:11] b1[7:0] b0[10:11]"),
    0b01111: (16, (4, 4, 4), 1,
              "r0[9:0] g0[9:0] b0[9:0] r1[3:0] r0[10:15] g1[3:0] g0[10:15] b1[3:0] b0[10:15]"),
}

# BC7's first 32 two-subset partitions: bit t is the subset of texel t (row by row).
_PARTITIONS = np.array(
    [
        0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80,
        0xC800, 0xFFEC, 0xFE80, 0xE800, 0xFFE8, 0xFF00, 0xFFF0, 0xF000,
        0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310, 0x3100, 0x8CCE,
        0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C,
    ],
    dtype=np.int64,
)
# Anchor texel of subset 1 for each partition (subset 0's anchor is texel 0).
_ANCHORS = np.array(
    [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
     15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2],
    dtype=np.int64,
)
_WEIGHTS = {
    3: np.array([0, 9, 18, 27, 37, 46, 55, 64], dtype=np.int64),
    4: np.array([0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64], dtype=np.int64),
}
_FIELD = re.compile(r"([rgb])([0-3])\[(\d+)(?::(\d+))?\]")


def _parse_layout(layout: str) -> list[tuple[int, int, int]]:
    """(endpoint, channel, bit) for each stream bit, in stream order."""
    bits = []
    for channel, endpoint, first, last in _FIELD.findall(layout):
        high, low = int(first), int(last) if last else int(first)
        # The right-hand index is read first: [9:0] from bit 0 up, [10:15] from bit 15 down.
        order = range(low, high + 1) if high >= low else range(low, high - 1, -1)
        bits.extend((int(endpoint), "rgb".index(channel), b) for b in order)
    return bits


_PARSED = {mode: _parse_layout(spec[3]) for mode, spec in _MODES.items()}


def _bits_value(bits: np.ndarray, start: np.ndarray | int, count: np.ndarray | int) -> np.ndarray:
    """Little-endian unsigned values of ``count`` bits starting at ``start``, per block (row of ``bits``)."""
    count = np.broadcast_to(np.asarray(count), (len(bits),))
    start = np.broadcast_to(np.asarray(start), (len(bits),))
    value = np.zeros(len(bits), dtype=np.int64)
    rows = np.arange(len(bits))
    for k in range(int(count.max(initial=0))):
        take = k < count
        value[take] |= bits[rows[take], start[take] + k].astype(np.int64) << k
    return value


def _sign_extend(value: np.ndarray, bits: int) -> np.ndarray:
    sign = 1 << (bits - 1)
    return (value & (sign - 1)) - (value & sign)


def _unquantize(value: np.ndarray, bits: int) -> np.ndarray:
    """Endpoint to 16 bits, unsigned format."""
    if bits >= 15:
        return value
    top = (1 << bits) - 1
    scaled = ((value << 16) + 0x8000) >> bits
    return np.where(value == 0, 0, np.where(value == top, 0xFFFF, scaled))


def _decode_mode(bits: np.ndarray, mode: int) -> np.ndarray:
    """(n, 16, 3) float32 texels for blocks that all use ``mode``."""
    precision, deltas, subsets, _ = _MODES[mode]
    n = len(bits)
    mode_bits = 2 if mode < 2 else 5
    endpoints = np.zeros((n, 4, 3), dtype=np.int64)
    for position, (endpoint, channel, bit) in enumerate(_PARSED[mode]):
        endpoints[:, endpoint, channel] |= bits[:, mode_bits + position].astype(np.int64) << bit
    cursor = mode_bits + len(_PARSED[mode])

    if deltas is not None:
        mask = (1 << precision) - 1
        for endpoint in range(1, 2 * subsets):
            for channel in range(3):
                delta = _sign_extend(endpoints[:, endpoint, channel], deltas[channel])
                endpoints[:, endpoint, channel] = (endpoints[:, 0, channel] + delta) & mask
    endpoints = _unquantize(endpoints, precision)

    if subsets == 2:
        partition = _bits_value(bits, cursor, 5)
        cursor += 5
        subset_of = (_PARTITIONS[partition][:, None] >> np.arange(16)) & 1  # (n, 16)
        anchor = _ANCHORS[partition]
        index_bits = 3
    else:
        subset_of = np.zeros((n, 16), dtype=np.int64)
        anchor = np.full(n, -1)
        index_bits = 4

    indices = np.zeros((n, 16), dtype=np.int64)
    offset = np.full(n, cursor, dtype=np.int64)
    for texel in range(16):
        width = np.full(n, index_bits) - ((texel == 0) | (anchor == texel))
        indices[:, texel] = _bits_value(bits, offset, width)
        offset += width

    weight = _WEIGHTS[index_bits][indices]  # (n, 16)
    a = np.take_along_axis(endpoints, (2 * subset_of)[..., None].repeat(3, axis=2), axis=1)
    b = np.take_along_axis(endpoints, (2 * subset_of + 1)[..., None].repeat(3, axis=2), axis=1)
    interpolated = ((64 - weight)[..., None] * a + weight[..., None] * b + 32) >> 6
    halves = ((interpolated * 31) >> 6).astype(np.uint16)
    return halves.view(np.float16).astype(np.float32)


def decode(payload: bytes, width: int, height: int) -> np.ndarray:
    """Decode one BC6H (unsigned) mip to float32 RGB of shape (height, width, 3)."""
    blocks_x, blocks_y = max(1, (width + 3) // 4), max(1, (height + 3) // 4)
    count = blocks_x * blocks_y
    raw = np.frombuffer(payload, dtype=np.uint8, count=count * 16).reshape(count, 16)
    bits = np.unpackbits(raw, axis=1, bitorder="little")
    first = bits[:, 0].astype(np.int64) | (bits[:, 1].astype(np.int64) << 1)
    mode = np.where(first < 2, first, _bits_value(bits, 0, 5))
    texels = np.zeros((count, 16, 3), dtype=np.float32)  # reserved modes decode to black
    for value in np.unique(mode):
        if int(value) in _MODES:
            chosen = mode == value
            texels[chosen] = _decode_mode(bits[chosen], int(value))
    image = texels.reshape(blocks_y, blocks_x, 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(blocks_y * 4, blocks_x * 4, 3)
    return image[:height, :width]
