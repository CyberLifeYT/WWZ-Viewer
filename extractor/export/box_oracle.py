"""Point-in-bounding-box tests against a map's ``.obb`` boxes, vectorized.

The game's own oriented bounding boxes enclose its geometry: 97.4% of the
vertices of world-space (float-position) draws in ny_01_pvp fall inside one.
That makes them a reliable judge of where a piece of geometry belongs, which
the geometry exporter uses to place quantized draws whose quantization frame
it cannot otherwise determine (see ``geometry_export``).
"""

from __future__ import annotations

import numpy as np

from extractor.formats.obb import ObbSet

CELL_SIZE = 16.0  # metres per grid cell
MAX_BOX_SIZE = 300.0  # boxes larger than this (world boundaries, sky) say nothing useful
MIN_THICKNESS = 0.3  # flat boxes get some thickness so surface points count as inside
TOLERANCE = 0.25  # metres of slack around every box
CHUNK_POINTS = 50_000  # points tested per vectorized pass
_KEY_SCALE = np.int64(1 << 20)  # packs three cell coordinates (each within +-2^19) into one int64


def _cell_keys(cells: np.ndarray) -> np.ndarray:
    offset = cells.astype(np.int64) + (_KEY_SCALE // 2)
    return (offset[..., 0] * _KEY_SCALE + offset[..., 1]) * _KEY_SCALE + offset[..., 2]


class BoxOracle:
    """Answers "how much of this geometry lies inside the map's boxes?" for many points at once."""

    def __init__(self, boxes: ObbSet) -> None:
        origins = boxes.origins.astype(np.float64)
        axes = boxes.axes.astype(np.float64)
        extents = np.maximum(boxes.extents.astype(np.float64), MIN_THICKNESS)
        corners = np.stack(
            [origins + (np.array(c, dtype=np.float64)[:, None] * extents[:, :, None] * axes).sum(1) for c in np.ndindex(2, 2, 2)],
            axis=1,
        )
        low, high = corners.min(1), corners.max(1)
        usable = np.flatnonzero((high - low).max(1) < MAX_BOX_SIZE)
        self._origins, self._axes, self._extents = origins, axes, extents

        # CSR table: sorted cell keys -> list of box indices overlapping that cell.
        keys, owners = [], []
        first, last = np.floor(low / CELL_SIZE).astype(np.int64), np.floor(high / CELL_SIZE).astype(np.int64)
        for i in usable:
            ranges = [np.arange(first[i, k], last[i, k] + 1) for k in range(3)]
            grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
            keys.append(_cell_keys(grid))
            owners.append(np.full(len(grid), i, dtype=np.int64))
        all_keys = np.concatenate(keys) if keys else np.zeros(0, dtype=np.int64)
        all_owners = np.concatenate(owners) if owners else np.zeros(0, dtype=np.int64)
        order = np.argsort(all_keys, kind="stable")
        self._sorted_keys = all_keys[order]
        self._owners = all_owners[order]
        self._unique_keys, self._starts, self._counts = np.unique(self._sorted_keys, return_index=True, return_counts=True)

    def inside(self, points: np.ndarray) -> np.ndarray:
        """Boolean per point: inside (with tolerance) at least one usable box."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) <= CHUNK_POINTS:
            return self._inside_chunk(points)
        # Dense cells can hold dozens of boxes; bound the (point, box) pairs held in memory.
        return np.concatenate([self._inside_chunk(points[i:i + CHUNK_POINTS]) for i in range(0, len(points), CHUNK_POINTS)])

    def _inside_chunk(self, points: np.ndarray) -> np.ndarray:
        result = np.zeros(len(points), dtype=bool)
        if len(points) == 0 or len(self._unique_keys) == 0:
            return result
        keys = _cell_keys(np.floor(points / CELL_SIZE))
        slot = np.searchsorted(self._unique_keys, keys)
        slot = np.clip(slot, 0, len(self._unique_keys) - 1)
        found = self._unique_keys[slot] == keys
        point_ids = np.flatnonzero(found)
        if len(point_ids) == 0:
            return result
        counts = self._counts[slot[point_ids]]
        starts = self._starts[slot[point_ids]]
        # Expand every (point, candidate box) pair and test them all at once.
        pair_point = np.repeat(point_ids, counts)
        within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        pair_box = self._owners[np.repeat(starts, counts) + within]
        local = np.einsum("nij,nj->ni", self._axes[pair_box], points[pair_point] - self._origins[pair_box])
        hit = ((local >= -TOLERANCE) & (local <= self._extents[pair_box] + TOLERANCE)).all(1)
        result[np.unique(pair_point[hit])] = True
        return result

    def inside_fraction(self, points: np.ndarray) -> float:
        points = np.asarray(points).reshape(-1, 3)
        return float(self.inside(points).mean()) if len(points) else 0.0
