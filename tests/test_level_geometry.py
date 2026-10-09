from __future__ import annotations

import struct

import numpy as np

import pytest

from extractor.formats import level_geometry as lgm
from extractor.ser import SerFormatError


def key(name: str) -> bytes:
    return struct.pack("<I", len(name)) + name.encode("ascii")


def value_int(v: int) -> bytes:
    return struct.pack("<Ii", 1, v)


def value_float(v: float) -> bytes:
    return struct.pack("<If", 2, v)


def value_str(s: str) -> bytes:
    return struct.pack("<II", 4, len(s)) + s.encode("utf-8")


def test_typed_material_values_round_trip() -> None:
    # The material encoding seen in the game: dict bodies with typed values.
    layer = struct.pack("<II", 7, 3) + key("texName") + value_str("bricks_01") + key("tilingU") + value_float(2.0) + key(
        "useLayerAlpha"
    ) + struct.pack("<IB", 3, 1)
    tint = struct.pack("<II", 6, 2) + value_float(255.0) + value_float(128.0)
    body = struct.pack("<I", 3) + key("version") + value_int(2) + key("layer0") + layer + key("tint") + tint
    material, end = lgm.read_dict_body(body, 0)
    assert end == len(body)
    assert material == {
        "version": 2,
        "layer0": {"texName": "bricks_01", "tilingU": 2.0, "useLayerAlpha": True},
        "tint": [255.0, 128.0],
    }


def test_unknown_value_type_is_rejected() -> None:
    with pytest.raises(SerFormatError, match="unknown value type 9"):
        lgm.read_value(struct.pack("<II", 9, 0), 0)


def field(fid: int, start: int, payload: bytes) -> bytes:
    return struct.pack("<HI", fid, start + 6 + len(payload)) + payload


def test_find_section_locates_counts_and_buffer_table() -> None:
    prefix = b"\x11" * 13  # arbitrary, unaligned bytes before the section
    p = len(prefix)
    counts = field(0, p, struct.pack("<HIIIII", 0, 5, 2, 1, 1, 0) + bytes(4))
    assert len(counts) == 32
    q = p + 32
    children, at = b"", q + 6
    for cid, payload in ((0, bytes(22)), (4, bytes(6)), (1, struct.pack("<HH", 16, 6)), (2, struct.pack("<II", 32, 12))):
        child = field(cid, at, payload)
        children += child
        at += len(child)
    table = field(2, q, children)
    blob = prefix + counts + table + b"\x00" * 8
    assert lgm.find_section(blob) == p


def test_find_section_reports_absence() -> None:
    with pytest.raises(SerFormatError, match="no geometry section"):
        lgm.find_section(bytes(256))


def test_batch_index_ranges_end_where_the_next_batch_starts() -> None:
    import numpy as np

    buffers = [lgm.Buffer(0, 16, 1600, 0, bytes(11)), lgm.Buffer(1, 6, 600, 0, bytes(11))]
    batches = [((0, 0), (1, 300)), ((0, 160), (1, 0)), ((0, 800), (1, 120))]
    tables = lgm.GeometryTables(0, 0, buffers, batches, [], [], np.zeros((0, 6)))
    assert lgm.batch_index_ranges(tables) == {1: (1, 0, 120), 2: (1, 120, 300), 0: (1, 300, 600)}


# -- vertex encodings ----------------------------------------------------------------------


def descriptor(*middle: int) -> bytes:
    """An 11-byte buffer descriptor with bytes 2..6 set (the rest as in the game's files)."""
    return bytes([0x45, 0x00, *middle]) + bytes(11 - 2 - len(middle))


@pytest.mark.parametrize(
    ("middle", "stride", "tangents", "colours", "layout"),
    [
        ((0x00, 0x10, 0x02, 0x42, 0x00), 8, 1, 0, ((0, 1, 1),)),  # tangent, compressed UV0
        ((0x00, 0x10, 0x02, 0xC6, 0x00), 12, 1, 0, ((0, 1, 1), (1, 2, 1))),  # and a compressed lightmap set
        ((0x00, 0x10, 0xC2, 0xC6, 0x00), 20, 1, 2, ((0, 3, 1), (1, 4, 1))),  # two colour words before the UVs
        ((0x00, 0x10, 0x02, 0x06, 0x00), 20, 1, 0, ((0, 1, 2), (1, 3, 2))),  # float UV sets: two words each
        ((0x00, 0x10, 0x02, 0x4A, 0x00), 16, 1, 0, ((0, 1, 1), (2, 2, 2))),  # sparse sets: 0 and 2
        ((0x00, 0xF0, 0x43, 0xFE, 0x07), 44, 4, 2, tuple((k, 6 + k, 1) for k in range(5))),  # five compressed sets
    ],
)
def test_vertex_format_follows_the_descriptor(middle, stride, tangents, colours, layout) -> None:
    fmt = lgm.VertexFormat.of(lgm.Buffer(0, stride, stride * 4, 1, descriptor(*middle)))
    assert (fmt.tangents, fmt.colour_words, fmt.uv_layout) == (tangents, colours, layout)
    assert fmt.expected_words == fmt.words == stride // 4  # the descriptor predicts the stride
    assert fmt.uv_sets == len(layout)
    for number, word, width in layout:
        assert fmt.uv_set(number) == (word, width)
    for missing in set(range(5)) - {number for number, _, _ in layout}:
        assert fmt.uv_set(missing) is None and not fmt.has_uv_set(missing)


def test_a_uv_set_past_the_record_is_absent() -> None:
    # Flags announce a lightmap set, but the stride has no room for it.
    fmt = lgm.VertexFormat.of(lgm.Buffer(0, 8, 32, 1, descriptor(0x00, 0x10, 0x02, 0xC6, 0x00)))
    assert fmt.has_uv_set(0) and not fmt.has_uv_set(1)


def test_grid_normals_decode_the_axis_values_seen_in_the_game() -> None:
    normals = lgm.decode_grid_normals(np.array([16108, 15929, 16380, -16380, 32671, 90, -16108]))
    expected = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 0, 0)]
    assert np.allclose(normals, expected, atol=0.03)


def test_byte_normals_and_tangents_map_bytes_to_unit_vectors() -> None:
    packed = np.array([255 | 128 << 8 | 128 << 16, 128 | 255 << 8 | 128 << 16])
    assert np.allclose(lgm.decode_byte_normals(packed), [(1, 0, 0), (0, 1, 0)], atol=0.01)
    tangents = lgm.decode_tangents(np.array([[255, 128, 128, 255], [128, 128, 0, 0]], dtype=np.uint8))
    assert np.allclose(tangents, [(1, 0, 0, 1), (0, 0, -1, -1)], atol=0.01)


def grid_mesh(size: float, cells: int) -> tuple[np.ndarray, np.ndarray]:
    """A flat square of ``size`` metres in the xz plane, split into ``cells`` x ``cells`` quads."""
    steps = np.linspace(0.0, size, cells + 1)
    x, z = np.meshgrid(steps, steps)
    vertices = np.stack([x.ravel(), np.zeros(x.size), z.ravel()], axis=1)
    triangles = []
    for row in range(cells):
        for col in range(cells):
            a = row * (cells + 1) + col
            b, c, d = a + 1, a + cells + 2, a + cells + 1
            triangles += [(a, c, b), (a, d, c)]
    return vertices, np.array(triangles)


@pytest.mark.parametrize(("uv_per_metre", "expected"), [(0.09, 1.0), (0.5, 5.0), (1.3, 13.0)])
def test_uv_scale_recovers_the_games_range_from_the_stored_density(uv_per_metre: float, expected: float) -> None:
    vertices, triangles = grid_mesh(10.0, 4)
    uv = vertices[:, [0, 2]] * uv_per_metre
    k = float(np.ceil(np.abs(uv).max()))  # the game's choice
    raw = np.round(uv / k * lgm.UV_RANGE)
    density = (uv_per_metre * lgm.DENSITY_BIAS,) * 2
    assert lgm.uv_scale(vertices, triangles, raw, density) == expected == k


def test_uv_scale_defaults_to_one_without_density() -> None:
    vertices, triangles = grid_mesh(1.0, 1)
    assert lgm.uv_scale(vertices, triangles, vertices[:, [0, 2]] * 1000, None) == 1.0


# -- physical layout -------------------------------------------------------------------------


def synthetic_level() -> tuple[lgm.GeometryTables, bytes, dict[int, int]]:
    """One quad in three buffers, stored out of table order and with zero padding, like the game does.

    Table order: 0 positions, 1 indices, 2 attributes. Blob order: 0, then 16 zero
    bytes, 2, then 1 at its stride alignment.
    """
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    up = 128 | 255 << 8 | 128 << 16  # normal +y as packed bytes
    positions = b"".join(struct.pack("<4f", x, 0.0, z, float(up)) for x, z in corners)
    attributes = b"".join(bytes([255, 128, 128, 255]) + struct.pack("<2h", int(x * 32767), int(z * 32767)) for x, z in corners)
    indices = struct.pack("<6H", 0, 2, 1, 0, 3, 2)  # counter-clockwise seen from +y
    buffers = [
        lgm.Buffer(0, 16, len(positions), 0, descriptor(0x03, 0x0C)),
        lgm.Buffer(1, 6, len(indices), 0, descriptor()),
        lgm.Buffer(2, 8, len(attributes), 1, descriptor(0x00, 0x10, 0x02, 0x42, 0x00)),
    ]
    batches = [((0, 0), (2, 0), (1, 0))]
    draws = [lgm.Draw(0, 4, 0, 2, 1, 0, (1.0 * lgm.DENSITY_BIAS,) * 2)]
    tables = lgm.GeometryTables(0, 1, buffers, batches, draws, [{}], np.zeros((0, 6)))
    guid = bytes(range(1, 17))
    blob = guid + positions + bytes(16) + attributes
    blob += bytes(-(len(blob) - lgm.GUID_SIZE) % 6) + indices
    expected = {0: 16, 2: 16 + 64 + 16, 1: len(blob) - len(indices)}
    return tables, blob, expected


def test_solve_layout_finds_out_of_order_buffers_behind_zero_padding() -> None:
    tables, blob, expected = synthetic_level()
    assert lgm.solve_layout(tables, blob) == expected


def test_decoded_draw_has_positions_normals_tangents_and_uvs() -> None:
    tables, blob, starts = synthetic_level()
    vertices, triangles = lgm.draw_mesh(tables, starts, blob, 0)
    assert np.allclose(vertices[:, [0, 2]], [(0, 0), (1, 0), (1, 1), (0, 1)])
    assert triangles.tolist() == [[0, 2, 1], [0, 3, 2]]
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    faces = np.cross(b - a, c - a)
    normals = lgm.draw_normals(tables, starts, blob, 0)
    assert np.all(np.sum(normals[triangles[:, 0]] * faces, axis=1) > 0)  # winding agrees with the normals
    attributes = lgm.draw_attributes(tables, starts, blob, 0)
    assert np.allclose(attributes.tangents[:, :3], (1, 0, 0), atol=0.01)
    scale = lgm.uv_scale(vertices, triangles, attributes.uv_raw, tables.draws[0].uv_density)
    assert scale == 1.0
    assert np.allclose(attributes.uv_raw / lgm.UV_RANGE * scale, vertices[:, [0, 2]], atol=1e-4)
    assert attributes.lightmap_uv is None  # no lightmap set asked for, and this format has none
    assert all(score > 0.99 for score in lgm.layout_quality(tables, starts, blob).values())
