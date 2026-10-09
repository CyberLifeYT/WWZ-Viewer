from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from extractor.export.encoding import RGBM_RANGE, png_bytes, rgbm, unrgbm, write_buffer


def test_rgbm_keeps_hdr_light_within_a_few_percent() -> None:
    light = np.array([[0.0, 0.0, 0.0], [0.25, 0.5, 1.0], [3.0, 2.0, 0.1], [12.0, 15.0, 1.0]])
    decoded = unrgbm(rgbm(light))
    assert np.allclose(decoded[0], 0.0)
    assert np.allclose(decoded[1:], light[1:], rtol=0.03, atol=0.02)


def test_rgbm_clamps_above_its_range() -> None:
    decoded = unrgbm(rgbm(np.array([[100.0, 1.0, 0.0]])))
    assert np.isclose(decoded[0, 0], RGBM_RANGE, rtol=0.01)


def test_buffers_are_gzip_and_identical_on_every_export(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    entry = write_buffer(tmp_path, "geometry_positions", values, "<f4", 3, extra_flag=True)
    assert entry == {
        "uri": "geometry_positions.bin.gz",
        "dtype": "float32",
        "components": 3,
        "byte_order": "little",
        "extra_flag": True,
    }
    first = (tmp_path / "geometry_positions.bin.gz").read_bytes()
    assert np.array_equal(np.frombuffer(gzip.decompress(first), dtype="<f4").reshape(4, 3), values)
    write_buffer(tmp_path, "geometry_positions", values, "<f4", 3, extra_flag=True)
    assert (tmp_path / "geometry_positions.bin.gz").read_bytes() == first  # no timestamp inside
    assert "byte_order" not in write_buffer(tmp_path, "bytes", np.zeros((2, 4)), "u1", 4)


def test_png_bytes_is_a_png() -> None:
    assert png_bytes(np.zeros((2, 2, 4), dtype=np.uint8)).startswith(b"\x89PNG")
