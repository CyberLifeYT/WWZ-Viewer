from __future__ import annotations

import numpy as np

from extractor.export import box_oracle
from extractor.export.box_oracle import BoxOracle
from extractor.formats.obb import ObbSet


def boxes(*specs: tuple[tuple[float, float, float], tuple[float, float, float]]) -> ObbSet:
    origins = np.array([origin for origin, _ in specs], dtype=np.float32)
    extents = np.array([size for _, size in specs], dtype=np.float32)
    axes = np.repeat(np.eye(3, dtype=np.float32)[None], len(specs), axis=0)
    return ObbSet(np.array([0, len(specs)], dtype=np.int64), origins, axes, extents)


def test_points_inside_and_outside() -> None:
    oracle = BoxOracle(boxes(((0, 0, 0), (10, 5, 10)), ((100, 0, 100), (4, 4, 4))))
    points = np.array([[5, 2, 5], [10.1, 2, 5], [102, 2, 102], [50, 50, 50], [-1, 0, 0]])
    assert oracle.inside(points).tolist() == [True, True, True, False, False]


def test_huge_boxes_are_ignored() -> None:
    oracle = BoxOracle(boxes(((-5000, -100, -5000), (10000, 1, 10000))))
    assert not oracle.inside(np.array([[0, -99.5, 0]])).any()


def test_chunked_and_single_pass_agree(monkeypatch) -> None:
    oracle = BoxOracle(boxes(((0, 0, 0), (20, 20, 20))))
    points = np.random.default_rng(0).uniform(-10, 30, size=(500, 3))
    single = oracle.inside(points)
    monkeypatch.setattr(box_oracle, "CHUNK_POINTS", 37)
    assert np.array_equal(oracle.inside(points), single)
