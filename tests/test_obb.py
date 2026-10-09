from __future__ import annotations

import struct

import numpy as np
import pytest

from extractor.formats import obb
from extractor.ser import SerFormatError

IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
QUARTER_TURN_Y = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)


def box(origin: tuple[float, float, float], axes: tuple[float, ...], extents: tuple[float, float, float]) -> bytes:
    return struct.pack("<15f", *origin, *axes, *extents)


def payload(*groups: list[bytes]) -> bytes:
    out = struct.pack("<I", len(groups))
    for boxes in groups:
        out += struct.pack("<I", len(boxes)) + b"".join(boxes)
    return out


def test_decodes_groups_into_flat_arrays(ser_builder) -> None:
    data = ser_builder(
        "obb",
        payload=payload(
            [box((-6.16, 17.96, -34.08), IDENTITY, (0.0, 2.412, 4.242))],
            [box((1, 2, 3), IDENTITY, (1, 1, 1)), box((4, 5, 6), QUARTER_TURN_Y, (0.5, 2, 3))],
        ),
    )
    boxes = obb.decode_bytes(data)
    assert boxes.group_count == 2
    assert boxes.box_count == 3
    assert boxes.group_sizes().tolist() == [1, 2]
    assert boxes.group(1) == slice(1, 3)
    np.testing.assert_allclose(boxes.origins[0], (-6.16, 17.96, -34.08), rtol=1e-6)
    np.testing.assert_array_equal(boxes.axes[2], np.array(QUARTER_TURN_Y, dtype=np.float32).reshape(3, 3))
    np.testing.assert_array_equal(boxes.extents[2], (0.5, 2, 3))


def test_centers_follow_rows_from_the_origin_corner(ser_builder) -> None:
    boxes = obb.decode_bytes(
        ser_builder(
            "obb",
            payload=payload([box((0, 0, 0), IDENTITY, (2, 4, 6)), box((4, 5, 6), QUARTER_TURN_Y, (0.5, 2, 3))]),
        )
    )
    # Rows are the local axes: (0,0,1), (0,1,0), (-1,0,0) for the quarter turn.
    np.testing.assert_allclose(boxes.centers(), [(1, 2, 3), (4 - 1.5, 5 + 1, 6 + 0.25)])


def test_world_boundary_fence_closes_with_row_axes(ser_builder) -> None:
    # Shape of ny_01_pvp group 1: edges of a square start at its corners and
    # run its full side length, so each far end must land on another corner.
    side = 6901.0
    west, south = -2885.7, -3371.2
    east, north = west + side, south + side
    edge_along_minus_z = (-1, 0, 0, 0, 0, -1, 0, -1, 0)
    edge_along_minus_x = (0, 0, 1, -1, 0, 0, 0, -1, 0)
    boxes = obb.decode_bytes(
        ser_builder(
            "obb",
            payload=payload(
                [
                    box((west, -93.1, north), edge_along_minus_z, (0, side, 1)),
                    box((east, -93.1, north), edge_along_minus_x, (0, side, 1)),
                ]
            ),
        )
    )
    far_ends = boxes.origins.astype(np.float64) + np.einsum("ni,nij->nj", boxes.extents, boxes.axes)
    np.testing.assert_allclose(far_ends[0], (west, -94.1, south), atol=1e-2)
    np.testing.assert_allclose(far_ends[1], (west, -94.1, north), atol=1e-2)


def test_group_ids_label_each_box(ser_builder) -> None:
    data = ser_builder("obb", payload=payload([box((0, 0, 0), IDENTITY, (1, 1, 1))], [box((1, 1, 1), IDENTITY, (1, 1, 1))] * 2))
    assert obb.decode_bytes(data).group_ids().tolist() == [0, 1, 1]


def test_empty_resource(ser_builder) -> None:
    boxes = obb.decode_bytes(ser_builder("obb", payload=struct.pack("<I", 0)))
    assert boxes.group_count == 0
    assert boxes.box_count == 0
    assert boxes.origins.shape == (0, 3)
    assert boxes.centers().shape == (0, 3)
    assert boxes.axes.shape == (0, 3, 3)


def test_group_index_out_of_range(ser_builder) -> None:
    boxes = obb.decode_bytes(ser_builder("obb", payload=payload([box((0, 0, 0), IDENTITY, (1, 1, 1))])))
    with pytest.raises(IndexError):
        boxes.group(1)


def test_trailing_bytes_are_rejected(ser_builder) -> None:
    data = ser_builder("obb", payload=payload([box((0, 0, 0), IDENTITY, (1, 1, 1))]) + b"\x00" * 4)
    with pytest.raises(SerFormatError, match="4 unexpected bytes"):
        obb.decode_bytes(data)


def test_truncated_group_is_rejected(ser_builder) -> None:
    data = ser_builder("obb", payload=payload([box((0, 0, 0), IDENTITY, (1, 1, 1))])[:-8])
    with pytest.raises(SerFormatError, match="runs past end"):
        obb.decode_bytes(data)


def test_wrong_resource_type_is_rejected(ser_builder) -> None:
    with pytest.raises(SerFormatError, match="expected a 'obb' resource"):
        obb.decode_bytes(ser_builder("cdt", payload=struct.pack("<I", 0)))
