"""Decoder for ``.obb`` resources: oriented bounding boxes, grouped per object.

Payload layout, after the 1SER header (which has no dependencies)::

    u32 group_count
    group[group_count]:
        u32 box_count                      1..16 in shipped data
        box[box_count], 60 bytes each:
            f32[3]  origin                 one corner of the box, world space, metres
            f32[9]  axes                   3x3 rotation; row i is local axis i in world space
            f32[3]  extents                full edge length along each local axis, never negative

A box covers ``origin + t0*e0*axes[0] + t1*e1*axes[1] + t2*e2*axes[2]`` for
``t`` in [0, 1]. One extent is 0 on flat boxes (25% of them).

How the convention was established: ny_01_pvp's group 1 is a five-box world
boundary (four edge walls and a floor) whose origins sit exactly on the corners
of a 6,901 m square. Only "origin is a corner, rows are axes, extents are full
lengths" makes every box end exactly on the opposite corner; the column-major
reading misses by kilometres, and a centred box would overshoot by 3.4 km.

Verified across all 44 scene paks, 2,963,534 boxes (client and server copies
are byte-identical): each payload is consumed exactly, and every ``axes``
matrix is orthonormal to within 4e-7 with determinant +1. Y is up: in 43 maps
the 1st-99th percentile spread of box positions along Y is 5-20x smaller than
along X or Z. The exception, ``ny_02_pve``, spans 233 m vertically against
215-268 m horizontally, which looks like a genuinely tall level.

Still open: the world's handedness. A mirrored map is just as self-consistent,
so it has to be settled against the game itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from extractor.ser.container import SerResource, load
from extractor.ser.errors import SerFormatError

RESOURCE_TYPE = "obb"
FLOATS_PER_BOX = 15
BOX_SIZE = FLOATS_PER_BOX * 4


@dataclass(frozen=True)
class ObbSet:
    """Every box in an ``.obb`` resource, as flat arrays with group boundaries."""

    group_starts: np.ndarray
    """int64, shape (groups + 1,): boxes of group ``g`` are ``group_starts[g]:group_starts[g + 1]``."""

    origins: np.ndarray
    """float32, shape (boxes, 3): the corner the axes start from."""

    axes: np.ndarray
    """float32, shape (boxes, 3, 3): row ``i`` is the box's local axis ``i`` in world space."""

    extents: np.ndarray
    """float32, shape (boxes, 3): full edge length along each local axis."""

    @property
    def group_count(self) -> int:
        return len(self.group_starts) - 1

    @property
    def box_count(self) -> int:
        return len(self.origins)

    def group_sizes(self) -> np.ndarray:
        return np.diff(self.group_starts)

    def group(self, index: int) -> slice:
        """Index range of one group's boxes, for slicing ``origins``, ``axes`` and ``extents``."""
        if not 0 <= index < self.group_count:
            raise IndexError(f"group {index} out of range 0..{self.group_count - 1}")
        return slice(int(self.group_starts[index]), int(self.group_starts[index + 1]))

    def group_ids(self) -> np.ndarray:
        """uint32, shape (boxes,): the group each box belongs to."""
        return np.repeat(np.arange(self.group_count, dtype=np.uint32), self.group_sizes())

    def centers(self) -> np.ndarray:
        """float64, shape (boxes, 3): the geometric centre of each box."""
        half_spans = np.einsum("ni,nij->nj", self.extents.astype(np.float64) * 0.5, self.axes.astype(np.float64))
        return self.origins.astype(np.float64) + half_spans


def decode(resource: SerResource) -> ObbSet:
    """Decode a parsed ``obb`` resource.

    Raises:
        SerFormatError: if the resource is not an ``obb``, is truncated, or has
            bytes left over after the declared groups.
    """
    if resource.type_name != RESOURCE_TYPE:
        raise SerFormatError(f"expected a {RESOURCE_TYPE!r} resource, got {resource.type_name!r}", offset=4)

    reader = resource.reader()
    group_count = reader.u32()
    sizes = np.empty(group_count, dtype=np.int64)
    chunks = []
    for index in range(group_count):
        box_count = reader.u32()
        sizes[index] = box_count
        chunks.append(reader.raw(box_count * BOX_SIZE))

    if not reader.at_end():
        raise SerFormatError(
            f"{reader.remaining()} unexpected bytes after {group_count} box groups",
            offset=reader.absolute(),
        )

    flat = np.frombuffer(b"".join(chunks), dtype="<f4").reshape(-1, FLOATS_PER_BOX)
    group_starts = np.zeros(group_count + 1, dtype=np.int64)
    np.cumsum(sizes, out=group_starts[1:])
    return ObbSet(
        group_starts=group_starts,
        origins=np.ascontiguousarray(flat[:, 0:3]),
        axes=np.ascontiguousarray(flat[:, 3:12]).reshape(-1, 3, 3),
        extents=np.ascontiguousarray(flat[:, 12:15]),
    )


def decode_bytes(data: bytes | bytearray | memoryview) -> ObbSet:
    """Parse the 1SER container and decode it in one step."""
    return decode(load(data))
