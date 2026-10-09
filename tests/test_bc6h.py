from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from extractor.formats import bc6h

VALID_MODES = sorted(bc6h._MODES)


def random_blocks(mode: int, count: int, seed: int) -> bytes:
    """``count`` random blocks forced to ``mode`` (2 mode bits for modes 0 and 1, else 5)."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(count, 16), dtype=np.uint8)
    mask = 0b11 if mode < 2 else 0b11111
    blocks[:, 0] = (blocks[:, 0] & ((~mask) & 0xFF)) | mode
    return blocks.tobytes()


@pytest.mark.parametrize("mode", VALID_MODES)
def test_every_mode_matches_pillow_within_its_clamped_range(mode: int) -> None:
    # Pillow decodes BC6H correctly but clamps to 8-bit 0..1; compare inside that range.
    width, height = 32, 32
    payload = random_blocks(mode, (width // 4) * (height // 4), seed=mode)
    ours = np.clip(bc6h.decode(payload, width, height), 0.0, 1.0) * 255.0
    pillow = np.asarray(Image.frombuffer("RGB", (width, height), payload, "bcn", (6, "BC6H")), dtype=np.float64)
    assert np.abs(ours - pillow).max() <= 1.5


def test_full_range_is_kept_beyond_one() -> None:
    # Mode 0b00011 (one subset, 10-bit endpoints stored as is) with every endpoint bit set:
    # endpoints unquantize to 0xFFFF and the texels to the largest half float.
    block = bytearray(16)
    bits = [1, 1, 0, 0, 0] + [1] * 60
    for position, bit in enumerate(bits):
        block[position // 8] |= bit << (position % 8)
    image = bc6h.decode(bytes(block), 4, 4)
    assert image.shape == (4, 4, 3)
    assert np.all(image == 65504.0)


def test_blocks_tile_left_to_right_then_down() -> None:
    white = bytearray(16)
    for position, bit in enumerate([1, 1, 0, 0, 0] + [1] * 60):
        white[position // 8] |= bit << (position % 8)
    black = bytes(16)  # mode 0, all endpoints zero
    image = bc6h.decode(bytes(white) + black + black + bytes(white), 8, 8)
    assert image[0:4, 0:4].min() == 65504.0 and image[0:4, 4:8].max() == 0.0
    assert image[4:8, 0:4].max() == 0.0 and image[4:8, 4:8].min() == 65504.0


def test_reserved_modes_decode_to_black() -> None:
    payload = random_blocks(0b10011, 4, seed=1)
    assert bc6h.decode(payload, 8, 8).max() == 0.0
