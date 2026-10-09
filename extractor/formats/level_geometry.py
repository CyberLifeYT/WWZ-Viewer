"""Level geometry: the geometry section of a ``.lg`` plus its ``.lg_data`` blob.

How the pieces fit (verified on ny_01_pvp, located by pattern in every map):

**Field encoding.** The ``.lg`` payload is a tree of fields, each
``u16 id, u32 absolute end offset, payload``. The geometry section is a run of
sibling fields:

* id 0 (26 bytes): counts. ``u16 0, u32 objects, u32 buffers, u32 batches, u32 draws, ...``
* id 2: the buffer table, with children
  id 0 (11-byte descriptor per buffer, see below), id 4 (3 bytes per buffer;
  byte 2 is the memory pool), id 1 (u16 stride per buffer; 6 means a triangle
  of u16 indices), id 2 (u32 byte size per buffer).
* id 3: child id 2 holds one record per batch, ``u8 n`` then ``n`` pairs of
  ``u32 buffer, u32 byte offset``. The first stream is positions, the last is
  the index buffer, the one between holds the other vertex attributes.
* id 4: per-draw tables. Child id 0: draw records, ``u16 first vertex,
  u16 vertex count, u16 first triangle, u16 triangle count, u16 0, u16 0xffff,
  u32 LOD mask`` followed by two floats per bit set in the mask: the draw's UV
  density (UV units per metre along u and v) at that LOD. Ranges are relative
  to the batch: vertex numbers to its position stream, triangle numbers to its
  index range. The indices of every non-empty draw stay inside its vertex
  range. Child id 1: u32 batch index per draw. Child id 8: one material per
  draw (a typed key/value tree, see :func:`read_value`). Child id 5: one
  quantization frame per quantized draw, six i16 (centre xyz in metres,
  half-size xyz in metres); the k-th quantized draw uses the k-th frame.

**The blob.** ``.lg_data`` is the level's 16-byte GUID followed by the buffers.
Counted from the end of the GUID, each buffer starts at the next multiple of
its own stride (6 for index buffers); a few have extra zero padding (32 bytes
before one buffer in ny_01_pvp). The physical order is the table order with
a few buffers moved later. :func:`solve_layout` places the buffers one by one
and checks each candidate start against the content: index ranges must fit
their draws, stored normals must match the triangles, tangents must be unit
vectors. In ny_01_pvp the solved layout ends exactly at the end of the file.

**Positions** (first stream). Descriptor byte 2 is ``0x03`` for float and
``0x0b`` for quantized positions.

* Float: ``f32 x, y, z, n`` in world space. ``n`` is an integer stored as a
  float, packing the normal as three bytes (x, y, z) mapped from 0..255 to -1..1.
* Quantized: ``i16 x, y, z, w``, dequantized as
  ``centre + q / 32767 * half_size`` with the draw's frame. ``w`` packs the
  normal on a 181 x 181 grid: ``|w| = row * 181 + col``, ``z = row / 90 - 1``,
  ``x = (col - 90.5) / 89.5`` (fitted: columns 1 and 180 are -x and +x), and the
  sign of ``w`` is the sign of ``y``. Median error against the triangles: 2.6
  degrees; walls facing +-x sit one or two rows below the equator, which the
  grid does not explain.

Stored normals agree with the triangles' winding (counter-clockwise front
faces), 0.99 on average.

**Attributes** (middle stream). Records of 4-byte words whose layout follows
the descriptor (:class:`VertexFormat`): the tangent frames (bytes x, y, z
mapped to -1..1, then 0 or 255 for the bitangent sign), the colour words
(four 8-bit channels), then the UV sets that are present, each either
compressed (``i16 u, v``, one word) or plain (``f32 u, v``, two words). Material
layers and lightmaps name UV sets by number (``uvSetIdx``); vertex-lit
materials use a compressed set whose values point at lightmap texel centres.

**UVs.** ``uv = raw / 32767 * k`` with one scale ``k`` per draw. The game
picks ``k = ceil(max |uv|)`` (1 for atlases, 2 for most walls, 8..25 for
long tiling runs), so ``k`` is not stored as such, but the draw's stored UV
density pins it down: :func:`uv_scale` estimates it from the density and the
geometry, then snaps it to the integer the ceil rule allows. The UV gradients
of the decoded triangles point along the stored tangents (0.94..0.99).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from extractor.ser.errors import SerFormatError

GUID_SIZE = 16
INDEX_STRIDE = 6
POSITION_FLOAT = 0x03
POSITION_QUANTIZED = 0x0B
QUANTIZED_RANGE = 32767.0
UV_RANGE = 32767.0
BLOCKER_TEXTURE = "red"  # the game ships no texture of this name; it marks invisible blockers
LIGHTMAP_UV_RANGE = 32768.0  # lightmap UVs index texels: raw / 32768 * size lands on a texel centre
NORMAL_GRID = 181  # quantized normals: 181 x 181 grid over x and z
NORMAL_GRID_HALF = 90.0  # rows: z = row / 90 - 1
NORMAL_COLUMN_CENTRE = 90.5  # columns: x = (col - 90.5) / 89.5
NORMAL_COLUMN_HALF = 89.5
# Where a draw's UVs span exactly 0..1, the stored density is 0.82% above the
# density of raw / 32767. Dividing it out makes k exactly 1 there.
DENSITY_BIAS = 1.0082
# Padding between buffers is zero and can be long (2,960 bytes once in rome_03_pve), so a
# buffer starts at most this many bytes before the first non-zero byte after the previous one
# (a buffer's own first values can be zero, e.g. the first index of a batch).
MAX_LEADING_ZEROS = 64
GOOD_FIT = 0.9  # a buffer's own records must look right this often...
FLOAT_POSITION_FIT = 0.75  # ...float positions less: some vertices carry degenerate packed normals
# ...and, once the buffers it needs are placed, its draws must fit at least this well. Right
# placements score 0.75-1.0 (some meshes carry unusual tangents); wrong ones 0-0.35.
GEOMETRIC_FIT = 0.5
SAMPLE_RECORDS = 256
SAMPLE_DRAWS = 24


# -- the typed key/value encoding used by materials -----------------------------------


def read_value(buf: bytes, p: int) -> tuple[object, int]:
    """One typed value: u32 type, then 1 int32 | 2 float32 | 3 bool8 | 4 string | 6 array | 7 dict."""
    (kind,) = struct.unpack_from("<I", buf, p)
    p += 4
    if kind == 1:
        return struct.unpack_from("<i", buf, p)[0], p + 4
    if kind == 2:
        return struct.unpack_from("<f", buf, p)[0], p + 4
    if kind == 3:
        return buf[p] != 0, p + 1
    if kind == 4:
        (length,) = struct.unpack_from("<I", buf, p)
        return buf[p + 4:p + 4 + length].decode("utf-8", "replace"), p + 4 + length
    if kind == 6:
        (count,) = struct.unpack_from("<I", buf, p)
        p += 4
        items = []
        for _ in range(count):
            item, p = read_value(buf, p)
            items.append(item)
        return items, p
    if kind == 7:
        return read_dict_body(buf, p)
    raise SerFormatError(f"unknown value type {kind}", offset=p - 4)


def read_dict_body(buf: bytes, p: int) -> tuple[dict[str, object], int]:
    (count,) = struct.unpack_from("<I", buf, p)
    p += 4
    out: dict[str, object] = {}
    for _ in range(count):
        (length,) = struct.unpack_from("<I", buf, p)
        key = buf[p + 4:p + 4 + length].decode("ascii", "replace")
        value, p = read_value(buf, p + 4 + length)
        out[key] = value
    return out, p


# -- tables -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Buffer:
    index: int
    stride: int
    size: int
    pool: int
    descriptor: bytes

    @property
    def position_format(self) -> int:
        return self.descriptor[2]

    @property
    def alignment(self) -> int:
        return max(self.stride, 1)


@dataclass(frozen=True)
class VertexFormat:
    """Word layout of an attribute record (4-byte words), read from the buffer descriptor.

    Byte 3's high nibble has one bit per tangent frame; byte 4 one bit per colour
    word (0x02 is always set and adds nothing). Bytes 5-6 list the UV sets: bit
    ``0x02 << k`` of byte 5 marks set ``k`` (0-4) present, bit ``0x40 << k`` of
    bytes 5-6 (read as one little-endian u16) marks it compressed to one word
    (``i16 u, v``) instead of two (``f32 u, v``). Sets can be sparse (0 and 2).
    Records are the tangents, the colours, then the present sets in order; this
    predicts the stride of all 30 formats in the game's 44 maps.
    """

    tangents: int
    colour_words: int
    uv_layout: tuple[tuple[int, int, int], ...]  # (set number, first word, words: 1 compressed or 2 float)
    words: int

    @classmethod
    def of(cls, buffer: Buffer) -> VertexFormat:
        d = buffer.descriptor
        tangents = bin(d[3] >> 4).count("1")
        colour_words = bin(d[4] & ~0x02 & 0xFF).count("1")
        flags = d[5] | d[6] << 8
        layout, word = [], tangents + colour_words
        for number in range(5):
            if flags & (0x02 << number):
                width = 1 if flags & (0x40 << number) else 2
                layout.append((number, word, width))
                word += width
        return cls(tangents, colour_words, tuple(layout), buffer.stride // 4)

    @property
    def uv_sets(self) -> int:
        return len(self.uv_layout)

    @property
    def expected_words(self) -> int:
        return self.tangents + self.colour_words + sum(width for _, _, width in self.uv_layout)

    def uv_set(self, number: int) -> tuple[int, int] | None:
        """(first word, words) of UV set ``number``, or None when the format lacks it."""
        for set_number, word, width in self.uv_layout:
            if set_number == number and word + width <= self.words:
                return word, width
        return None

    def has_uv_set(self, number: int) -> bool:
        return self.uv_set(number) is not None


@dataclass(frozen=True)
class Draw:
    """One draw: a vertex range and a triangle range inside its batch."""

    first_vertex: int
    vertex_count: int
    first_triangle: int
    triangle_count: int
    lod_mask: int
    batch: int
    uv_density: tuple[float, float] | None = None  # UV units per metre (u, v) at the draw's first LOD


@dataclass
class GeometryTables:
    section_offset: int
    object_count: int
    buffers: list[Buffer]
    batches: list[tuple[tuple[int, int], ...]]
    draws: list[Draw]
    materials: list[dict[str, object]]
    frames: np.ndarray
    frame_of_draw: dict[int, int] = field(default_factory=dict)


def _field(buf: bytes, p: int) -> tuple[int, int]:
    return struct.unpack_from("<HI", buf, p)


def _children(buf: bytes, start: int, end: int) -> dict[int, tuple[int, int]]:
    """Child fields of a payload, as id -> (payload start, end). Raises if they do not tile it."""
    out, p = {}, start
    while p < end:
        fid, fend = _field(buf, p)
        if not p + 6 <= fend <= end:
            raise SerFormatError("geometry field tree is malformed", offset=p)
        out[fid] = (p + 6, fend)
        p = fend
    return out


def find_section(lg: bytes) -> int:
    """Offset of the counts field (id 0, 26-byte payload) followed by the buffer table (id 2 with children 0, 4, 1, 2)."""
    n = len(lg)
    for alignment in range(4):
        count = (n - alignment) // 4
        values = np.frombuffer(lg, dtype="<u4", offset=alignment, count=count).astype(np.int64)
        offsets = alignment + 4 * np.arange(count, dtype=np.int64)
        for off in offsets[np.flatnonzero(values == offsets + 30)]:
            p = int(off) - 2
            if p < 0 or struct.unpack_from("<H", lg, p)[0] != 0 or p + 38 > n:
                continue
            fid, fend = _field(lg, p + 32)
            if fid != 2 or not p + 38 < fend <= n:
                continue
            try:
                kids = _children(lg, p + 38, fend)
            except SerFormatError:
                continue
            if list(kids) == [0, 4, 1, 2]:
                return p
    raise SerFormatError("no geometry section found in level graph")


def parse_tables(lg: bytes) -> GeometryTables:
    p = find_section(lg)
    object_count, buffer_count, batch_count, draw_count = struct.unpack_from("<IIII", lg, p + 8)

    _, buffers_end = _field(lg, p + 32)
    table = _children(lg, p + 38, buffers_end)
    strides = struct.unpack_from(f"<{buffer_count}H", lg, table[1][0])
    sizes = struct.unpack_from(f"<{buffer_count}I", lg, table[2][0])
    pools = lg[table[4][0]:table[4][1]]
    descriptors = lg[table[0][0]:table[0][1]]
    buffers = [
        Buffer(i, strides[i], sizes[i], pools[3 * i + 2], descriptors[11 * i:11 * i + 11]) for i in range(buffer_count)
    ]

    batch_field = buffers_end
    _, batch_end = _field(lg, batch_field)
    batch_children = _children(lg, batch_field + 6, batch_end)
    q, batches = batch_children[2][0], []
    for _ in range(batch_count):
        n = lg[q]
        batches.append(tuple(struct.unpack_from("<II", lg, q + 1 + 8 * k) for k in range(n)))
        q += 1 + 8 * n

    _, draw_section_end = _field(lg, batch_end)
    per_draw = _children(lg, batch_end + 6, draw_section_end)
    draw_batch = np.frombuffer(lg, dtype="<u4", offset=per_draw[1][0], count=draw_count)
    records, q = [], per_draw[0][0]
    for i in range(draw_count):
        v0, nv, t0, nt, _, _, mask = struct.unpack_from("<6HI", lg, q)
        density = struct.unpack_from("<2f", lg, q + 16) if mask else None
        records.append(Draw(v0, nv, t0, nt, mask, int(draw_batch[i]), density))
        q += 16 + 8 * bin(mask).count("1")
    if q != per_draw[0][1]:
        raise SerFormatError("draw records do not fill their field", offset=q)

    materials, q = [], per_draw[8][0]
    for _ in range(draw_count):
        material, q = read_dict_body(lg, q + 2)  # each material is prefixed by a u16 index
        materials.append(material)

    frames_start, frames_end = per_draw[5]
    frames = np.frombuffer(lg, dtype="<i2", offset=frames_start, count=(frames_end - frames_start) // 2)
    frames = frames.reshape(-1, 6).astype(np.float64)

    tables = GeometryTables(p, object_count, buffers, batches, records, materials, frames)
    quantized = [i for i, d in enumerate(records) if buffers[batches[d.batch][0][0]].position_format == POSITION_QUANTIZED]
    if len(quantized) != len(frames):
        raise SerFormatError(f"{len(quantized)} quantized draws but {len(frames)} frames", offset=frames_start)
    tables.frame_of_draw = {draw: k for k, draw in enumerate(quantized)}
    return tables


# -- decoding -------------------------------------------------------------------------------


def is_quantized(tables: GeometryTables, draw_index: int) -> bool:
    d = tables.draws[draw_index]
    return tables.buffers[tables.batches[d.batch][0][0]].position_format == POSITION_QUANTIZED


def dequantize(quantized: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """``centre + q / 32767 * half_size`` for an (n, 3) array of raw i16 positions."""
    return frame[:3] + quantized / QUANTIZED_RANGE * frame[3:]


def decode_byte_normals(packed: np.ndarray) -> np.ndarray:
    """Unit vectors from integers packing x, y, z as bytes (low byte first), each 0..255 for -1..1."""
    p = packed.astype(np.int64)
    xyz = np.stack([p & 0xFF, (p >> 8) & 0xFF, (p >> 16) & 0xFF], axis=1) / 127.5 - 1.0
    return _normalize(xyz)


def decode_grid_normals(w: np.ndarray) -> np.ndarray:
    """Unit vectors from quantized positions' ``w``: x and z on a 181 x 181 grid, the sign of w is the sign of y."""
    w = w.astype(np.int64)
    a = np.abs(w)
    # Columns 1 and 180 hold x = -1 and +1 (fitted on ny_01_pvp: column 1 measures -0.9986
    # over 4,914 vertices); rows 0 and 180 hold z = -1 and +1.
    x = np.clip(((a % NORMAL_GRID) - NORMAL_COLUMN_CENTRE) / NORMAL_COLUMN_HALF, -1.0, 1.0)
    z = (a // NORMAL_GRID) / NORMAL_GRID_HALF - 1.0
    y = np.sqrt(np.clip(1.0 - x * x - z * z, 0.0, None)) * np.where(w < 0, -1.0, 1.0)
    return _normalize(np.stack([x, y, z], axis=1))


def decode_tangents(words: np.ndarray) -> np.ndarray:
    """(n, 4) uint8 tangent words -> (n, 4) float: unit xyz and the bitangent sign (+1 or -1)."""
    xyz = _normalize(words[:, :3].astype(np.float64) / 127.5 - 1.0)
    sign = np.where(words[:, 3] >= 128, 1.0, -1.0)
    return np.concatenate([xyz, sign[:, None]], axis=1)


def _normalize(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(length, 1e-12)


def _draw_indices(tables: GeometryTables, starts: dict[int, int], data: bytes, d: Draw) -> np.ndarray:
    """The draw's triangles as local vertex numbers, (m, 3) int64."""
    ib_buffer, ib_offset = tables.batches[d.batch][-1]
    ib = tables.buffers[ib_buffer]
    index_at = starts[ib_buffer] + ib_offset + d.first_triangle * INDEX_STRIDE
    if index_at + d.triangle_count * INDEX_STRIDE > starts[ib_buffer] + ib.size:
        raise SerFormatError("draw indices run past their buffer", offset=index_at)
    indices = np.frombuffer(data, dtype="<u2", offset=index_at, count=d.triangle_count * 3).astype(np.int64)
    local = indices - d.first_vertex
    if local.min() < 0 or local.max() >= d.vertex_count:
        raise SerFormatError("draw indices leave the draw's vertex range", offset=index_at)
    return local.reshape(-1, 3)


def _stream_records(tables: GeometryTables, starts: dict[int, int], data: bytes, d: Draw, stream: int) -> np.ndarray:
    """The draw's records in one of its batch's streams, (vertex_count, stride) uint8."""
    buffer_index, offset = tables.batches[d.batch][stream]
    buffer = tables.buffers[buffer_index]
    base = starts[buffer_index] + offset + d.first_vertex * buffer.stride
    if base + d.vertex_count * buffer.stride > starts[buffer_index] + buffer.size:
        raise SerFormatError("draw vertices run past their buffer", offset=base)
    raw = np.frombuffer(data, dtype=np.uint8, offset=base, count=d.vertex_count * buffer.stride)
    return raw.reshape(d.vertex_count, buffer.stride)


def draw_mesh(
    tables: GeometryTables,
    starts: dict[int, int],
    data: bytes,
    draw_index: int,
    frame: np.ndarray | None = None,
    raw: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (vertex_count, 3) float64 and triangles (m, 3) int64 indexing them, for one draw.

    Vertices are in world space. For quantized draws, ``frame`` overrides the
    draw's frame from the table, and ``raw=True`` returns the undequantized i16
    values instead.
    """
    d = tables.draws[draw_index]
    if d.vertex_count == 0 or d.triangle_count == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    triangles = _draw_indices(tables, starts, data, d)
    records = _stream_records(tables, starts, data, d, 0)
    fmt = tables.buffers[tables.batches[d.batch][0][0]].position_format
    if fmt == POSITION_FLOAT:
        with np.errstate(invalid="ignore", over="ignore"):
            vertices = records.view("<f4")[:, :3].astype(np.float64)
    elif fmt == POSITION_QUANTIZED:
        quantized = records.view("<i2")[:, :3].astype(np.float64)
        if raw:
            return quantized, triangles
        chosen = tables.frames[tables.frame_of_draw[draw_index]] if frame is None else frame
        vertices = dequantize(quantized, chosen)
    else:
        raise SerFormatError(f"unknown position format {fmt:#x}")
    return vertices, triangles


def draw_normals(tables: GeometryTables, starts: dict[int, int], data: bytes, draw_index: int) -> np.ndarray:
    """The draw's stored per-vertex normals, (vertex_count, 3)."""
    d = tables.draws[draw_index]
    records = _stream_records(tables, starts, data, d, 0)
    fmt = tables.buffers[tables.batches[d.batch][0][0]].position_format
    if fmt == POSITION_FLOAT:
        with np.errstate(invalid="ignore", over="ignore"):
            packed = np.nan_to_num(records.view("<f4")[:, 3]).clip(0, 2**24 - 1)
        return decode_byte_normals(packed)
    if fmt == POSITION_QUANTIZED:
        return decode_grid_normals(records.view("<i2")[:, 3])
    raise SerFormatError(f"unknown position format {fmt:#x}")


@dataclass(frozen=True)
class DrawAttributes:
    """Decoded attribute stream of one draw; fields are None when the draw has no such data."""

    tangents: np.ndarray | None  # (n, 4): the first tangent frame, xyz and bitangent sign
    uv_raw: np.ndarray | None  # (n, 2) UV set 0's stored i16 values when it is compressed; see uv_scale
    lightmap_uv: np.ndarray | None  # (n, 2) in 0..1, v running down the lightmap
    uv: np.ndarray | None = None  # (n, 2) UV set 0 when it is stored as floats (no scale needed)
    colours: np.ndarray | None = None  # (n, 4) uint8, the first colour word
    uv_sets: dict[int, np.ndarray] = field(default_factory=dict)  # further UV sets asked for, as floats

    @property
    def uv0_samples(self) -> np.ndarray | None:
        """UV set 0 in whatever units it is stored in (for gradient directions only)."""
        return self.uv_raw if self.uv_raw is not None else self.uv


def _read_uv_set(words: np.ndarray, fmt: VertexFormat, number: int) -> tuple[np.ndarray, bool] | None:
    """(values, compressed) of UV set ``number``: raw i16 values when compressed, else the stored floats."""
    found = fmt.uv_set(number)
    if found is None:
        return None
    word, width = found
    if width == 1:
        return words[:, word].copy().view("<i2").reshape(-1, 2).astype(np.float64), True
    return words[:, word:word + 2].copy().reshape(len(words), 8).view("<f4").astype(np.float64), False


def draw_attributes(
    tables: GeometryTables,
    starts: dict[int, int],
    data: bytes,
    draw_index: int,
    lightmap_set: int | None = None,
    extra_sets: tuple[int, ...] = (),
) -> DrawAttributes:
    """The draw's attribute stream: tangents, UV set 0, colours, and the UV sets asked for.

    ``lightmap_set`` (the material's ``lm.uvSetIdx``) is read as lightmap coordinates: vertex-lit
    materials name a compressed set whose u16 values point at texel centres (raw / 32768);
    lightmap-textured materials name one of the regular sets. ``extra_sets`` (the sets material
    layers draw with) come back as floats; compressed ones as ``raw / 32767`` (their scale is not
    stored, and these sets are unwraps within 0..1).
    """
    d = tables.draws[draw_index]
    streams = tables.batches[d.batch]
    if len(streams) < 3:
        return DrawAttributes(None, None, None)
    fmt = VertexFormat.of(tables.buffers[streams[1][0]])
    records = _stream_records(tables, starts, data, d, 1)
    words = records.reshape(d.vertex_count, fmt.words, 4)
    tangents = decode_tangents(words[:, 0]) if fmt.tangents else None
    uv_raw = uv = None
    first = _read_uv_set(words, fmt, 0)
    if first is not None:
        uv_raw, uv = (first[0], None) if first[1] else (None, first[0])
    lightmap = None
    if lightmap_set is not None:
        found = _read_uv_set(words, fmt, lightmap_set)
        if found is not None:
            values, compressed = found
            lightmap = (values % 65536) / LIGHTMAP_UV_RANGE if compressed else values  # u16 texel coordinates
    colours = words[:, fmt.tangents].copy() if fmt.colour_words else None
    extra = {}
    for number in extra_sets:
        found = _read_uv_set(words, fmt, number)
        if found is not None:
            extra[number] = found[0] / UV_RANGE if found[1] else found[0]
    return DrawAttributes(tangents, uv_raw, lightmap, uv, colours, extra)


def uv_scale(
    vertices: np.ndarray, triangles: np.ndarray, uv_raw: np.ndarray, density: tuple[float, float] | None
) -> float:
    """The draw's UV scale ``k`` in ``uv = raw / 32767 * k``.

    ``k = ceil(max |uv|)``, so only integers ``k`` with ``k - 1 < m * k``
    (``m = max |raw| / 32767``) are possible. The stored density (UV per metre)
    divided by the density of ``raw / 32767`` estimates ``k``; it is snapped to
    the nearest possible integer, or kept as it is when no integer is close.
    """
    m = float(np.abs(uv_raw).max()) / UV_RANGE if len(uv_raw) else 0.0
    if m == 0.0 or density is None or len(triangles) == 0:
        return 1.0
    p0, p1, p2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    q = uv_raw / UV_RANGE
    q0, q1, q2 = q[triangles[:, 0]], q[triangles[:, 1]], q[triangles[:, 2]]
    e1, e2, d1, d2 = p1 - p0, p2 - p0, q1 - q0, q2 - q0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    area = np.linalg.norm(np.cross(e1, e2), axis=1) / 2
    ok = (np.abs(det) > 1e-12) & (area > 1e-10)
    if not ok.any():
        return 1.0
    e1, e2, d1, d2, det, area = e1[ok], e2[ok], d1[ok], d2[ok], det[ok, None], area[ok]
    # Metres per (raw / 32767) unit along u and v, per triangle; area-weighted density per axis.
    metres_u = np.linalg.norm((e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / det, axis=1)
    metres_v = np.linalg.norm((e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / det, axis=1)
    raw_density = (area.sum() / np.sum(area * metres_u), area.sum() / np.sum(area * metres_v))
    spans = q.max(axis=0) - q.min(axis=0)
    estimates = [
        (span, DENSITY_BIAS * stored / measured)
        for stored, measured, span in zip(density, raw_density, spans)
        if stored > 0 and measured > 0 and np.isfinite(measured) and span > 0.02
    ]
    if not estimates:
        return 1.0
    estimate = max(estimates)[1]  # the axis with the longer UV span measures best
    limit = np.inf if m >= 1.0 else 1.0 / (1.0 - m)  # k must stay below this
    candidates = [k for k in (np.floor(estimate), np.ceil(estimate)) if 1 <= k < limit]
    if candidates:
        k = min(candidates, key=lambda c: abs(c - estimate))
        if abs(k - estimate) <= max(0.35, 0.15 * estimate):
            return float(k)
    return float(max(estimate, 1.0))


# -- physical layout ----------------------------------------------------------------------


def _roles(tables: GeometryTables) -> dict[int, str]:
    roles: dict[int, str] = {}
    for streams in tables.batches:
        roles[streams[0][0]] = "position"
        roles[streams[-1][0]] = "index"
        for buffer_index, _ in streams[1:-1]:
            roles[buffer_index] = "attribute"
    return roles


def _sample_rows(count: int) -> np.ndarray:
    head = np.arange(min(count, 8))
    spread = np.linspace(0, count - 1, min(count, SAMPLE_RECORDS)).astype(np.int64)
    return np.unique(np.concatenate([head, spread]))


def _is_blocker(material: dict[str, object]) -> bool:
    layer0 = material.get("layer0")
    return isinstance(layer0, dict) and layer0.get("texName") == BLOCKER_TEXTURE


def _is_uninformative(material: dict[str, object]) -> bool:
    """Draws whose vertex data says nothing about placement: invisible blockers (junk attributes)
    and lens flares (camera-facing sprites whose tangents and normals are arbitrary)."""
    layer0 = material.get("layer0")
    name = layer0.get("texName") if isinstance(layer0, dict) else None
    return name == BLOCKER_TEXTURE or (isinstance(name, str) and "flare" in name)


def _draw_samples(tables: GeometryTables) -> dict[int, list[int]]:
    """Per buffer, up to SAMPLE_DRAWS non-empty draws that read it, spread over the draw list.

    Blockers and lens flares are left out (see :func:`_is_uninformative`).
    """
    users: dict[int, list[int]] = {}
    for i, d in enumerate(tables.draws):
        material = tables.materials[i] if i < len(tables.materials) else {}
        if d.triangle_count and not _is_uninformative(material):
            for buffer_index, _ in tables.batches[d.batch]:
                users.setdefault(buffer_index, []).append(i)
    return {
        b: found if len(found) <= SAMPLE_DRAWS else [found[int(k)] for k in np.linspace(0, len(found) - 1, SAMPLE_DRAWS)]
        for b, found in users.items()
    }


def _self_score(buffer: Buffer, role: str, data: bytes, start: int) -> float:
    """Share of sampled records that look like this buffer's kind of data; 0 when the first record does not."""
    count = buffer.size // buffer.stride
    if role == "index" or count == 0:
        return 1.0
    view = np.frombuffer(data, dtype=np.uint8, offset=start, count=count * buffer.stride).reshape(count, buffer.stride)
    rows = _sample_rows(count)
    records = view[rows]
    if role == "attribute":
        fmt = VertexFormat.of(buffer)
        if fmt.tangents == 0:
            return 1.0
        # Only the first tangent: where a later UV set is degenerate, its tangent is not a unit vector.
        words = records[:, :4].astype(np.float64)
        unit = np.abs(np.linalg.norm(words[:, :3] / 127.5 - 1.0, axis=1) - 1.0) < 0.03
        good = unit & ((words[:, 3] == 0) | (words[:, 3] == 255))
    elif buffer.position_format == POSITION_FLOAT:
        with np.errstate(invalid="ignore", over="ignore"):
            packed = records.copy().view("<f4")[:, 3].astype(np.float64)
            good = np.isfinite(packed) & (packed == np.round(packed)) & (packed >= 0) & (packed < 2**24)
            p = np.where(good, packed, 0).astype(np.int64)
            xyz = np.stack([p & 0xFF, (p >> 8) & 0xFF, (p >> 16) & 0xFF], axis=1) / 127.5 - 1.0
            good &= np.abs(np.linalg.norm(xyz, axis=1) - 1.0) < 0.05
    elif buffer.position_format == POSITION_QUANTIZED:
        a = np.abs(records.copy().view("<i2")[:, 3].astype(np.int64))
        x = ((a % NORMAL_GRID) - NORMAL_COLUMN_CENTRE) / NORMAL_COLUMN_HALF
        z = (a // NORMAL_GRID) / NORMAL_GRID_HALF - 1.0
        good = (x * x + z * z <= 1.03) & (a < NORMAL_GRID * NORMAL_GRID)
    else:
        return 1.0
    return float(good.mean()) if good[0] else 0.0


def _geometry_score(
    tables: GeometryTables, starts: dict[int, int], data: bytes, buffer_index: int, role: str, samples: dict[int, list[int]]
) -> float | None:
    """How well the placed buffer fits its draws, or None when the buffers it needs are not placed yet.

    Index buffers: share of real triangles (indices in range and distinct). Position buffers:
    share of triangles whose stored vertex normal agrees with the face. Attribute
    buffers: share of vertices whose stored tangent is orthogonal to their stored normal.
    """
    scores = []
    for i in samples.get(buffer_index, []):
        d = tables.draws[i]
        streams = tables.batches[d.batch]
        needed = {buffer_index} if role == "index" else {streams[0][0], streams[-1][0], buffer_index}
        if not needed <= starts.keys():
            continue
        try:
            if role == "index":
                # Zero padding also reads as in-range indices, but only as degenerate triangles.
                t = _draw_indices(tables, starts, data, d)
                distinct = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
                scores.append(float(distinct.mean()))
                continue
            vertices, triangles = draw_mesh(tables, starts, data, i)
        except (SerFormatError, ValueError, KeyError):
            scores.append(0.0)
            continue
        a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
        face = np.cross(b - a, c - a)
        length = np.linalg.norm(face, axis=1)
        keep = np.isfinite(length) & (length > 1e-9)
        if not keep.any():
            continue
        face = face[keep] / length[keep, None]
        corner = triangles[keep, 0]
        if role == "position":
            normals = draw_normals(tables, starts, data, i)
            scores.append(float(np.mean(np.sum(normals[corner] * face, axis=1) > 0.3)))
        else:
            # Tangents are stored orthogonal to their vertex's normal (up to the normals' quantization),
            # and each triangle's UV gradient points along its tangent. The second test tells apart
            # formats of one stride that share the tangent words but keep UV0 in another word.
            attributes = draw_attributes(tables, starts, data, i)
            if attributes.tangents is None:
                continue
            normals = draw_normals(tables, starts, data, i)
            orthogonal = float(np.mean(np.abs(np.sum(attributes.tangents[:, :3] * normals, axis=1)) < 0.2))
            along = _uv_tangent_agreement(vertices, triangles[keep], attributes)
            scores.append(orthogonal if along is None else min(orthogonal, along))
    return float(np.median(scores)) if scores else None  # a few odd draws must not sink a right placement


def _first_nonzero(data: bytes, position: int, chunk: int = 1 << 20) -> int:
    """Offset of the first non-zero byte at or after ``position``; ``len(data)`` when there is none."""
    while position < len(data):
        block = np.frombuffer(data, dtype=np.uint8, offset=position, count=min(chunk, len(data) - position))
        found = np.flatnonzero(block)
        if len(found):
            return position + int(found[0])
        position += len(block)
    return len(data)


def _uv_tangent_agreement(vertices: np.ndarray, triangles: np.ndarray, attributes: DrawAttributes) -> float | None:
    """Share of triangles whose direction of growing u (from positions and UV0) follows the stored tangent."""
    uv = attributes.uv0_samples
    if uv is None or attributes.tangents is None:
        return None
    p0, p1, p2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    q0, q1, q2 = uv[triangles[:, 0]], uv[triangles[:, 1]], uv[triangles[:, 2]]
    e1, e2, d1, d2 = p1 - p0, p2 - p0, q1 - q0, q2 - q0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    ok = np.abs(det) > (1.0 if attributes.uv_raw is not None else 1e-9)  # skip triangles with (nearly) no UV area
    if ok.sum() < 3:
        return None
    du = (e1[ok] * d2[ok, 1:2] - e2[ok] * d1[ok, 1:2]) / det[ok, None]
    du /= np.maximum(np.linalg.norm(du, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum(du * attributes.tangents[triangles[ok, 0], :3], axis=1) > 0.5))


def _fit(
    tables: GeometryTables,
    data: bytes,
    starts: dict[int, int],
    buffer_index: int,
    role: str,
    position: int,
    samples: dict[int, list[int]],
    trust_records: bool = False,
) -> tuple[int, float, bool] | None:
    """The first start at or after ``position`` (behind zero padding only) where the buffer fits.

    Returns (start, score, fits); when no start fits, the best-scoring one with ``fits`` False.
    With ``trust_records``, a buffer whose own records look right fits even when its draws disagree.
    """
    buffer = tables.buffers[buffer_index]
    content = _first_nonzero(data, position)
    lowest = max(position, content - MAX_LEADING_ZEROS)
    rel = lowest - GUID_SIZE
    first = GUID_SIZE + (rel + buffer.alignment - 1) // buffer.alignment * buffer.alignment
    best: tuple[int, float, bool] | None = None
    for start in range(first, content + 1, buffer.alignment):
        if start + buffer.size > len(data):
            break
        own = _self_score(buffer, role, data, start)
        needed = FLOAT_POSITION_FIT if role == "position" and buffer.position_format == POSITION_FLOAT else GOOD_FIT
        geometric = None
        if own >= needed:
            trial = dict(starts)
            trial[buffer_index] = start
            geometric = _geometry_score(tables, trial, data, buffer_index, role, samples)
        score = own if geometric is None else min(own, geometric)
        if own >= needed and (geometric is None or geometric >= GEOMETRIC_FIT or (trust_records and own >= 0.99)):
            return start, score, True
        if best is None or score > best[1]:
            best = (start, score, False)
    return best


class _SearchExhausted(Exception):
    pass


def solve_layout(tables: GeometryTables, data: bytes, max_trials: int = 4000) -> dict[int, int]:
    """Buffer start offsets in ``.lg_data``.

    Buffers are placed one after another, each behind zero padding. At each
    step the remaining buffers are tried in table order (the usual physical
    order) and the first one whose content fits is placed. When two buffers of
    the same kind both fit at a spot, the wrong guess shows up later as a dead
    end; the search then backs up and tries the next buffer that fits. A
    layout is complete when every buffer is placed and only zeros follow.

    If the search finds no complete layout, the best guess per step is used
    instead (see :func:`_greedy_layout`), so a map still exports.
    """
    roles = _roles(tables)
    samples = _draw_samples(tables)
    failed: set[tuple[int, frozenset[int]]] = set()
    trials = 0
    trust_records = False

    def place(position: int, remaining: list[int], starts: dict[int, int]) -> dict[int, int] | None:
        nonlocal trials
        if not remaining:
            return dict(starts) if position <= len(data) and not any(data[position:]) else None
        key = (position, frozenset(remaining))
        if key in failed:
            return None
        # Buffers whose draws agree come first; in the second pass, then those trusted on their records alone.
        for trusting in (False, True) if trust_records else (False,):
            for buffer_index in remaining:
                trials += 1
                if trials > max_trials:
                    raise _SearchExhausted
                role = roles.get(buffer_index, "attribute")
                found = _fit(tables, data, starts, buffer_index, role, position, samples, trusting)
                if found is None or not found[2]:
                    continue
                if trusting:
                    strict = _fit(tables, data, starts, buffer_index, role, position, samples, False)
                    if strict is not None and strict[2]:
                        continue  # already tried in the strict round
                starts[buffer_index] = found[0]
                rest = [b for b in remaining if b != buffer_index]
                result = place(found[0] + tables.buffers[buffer_index].size, rest, starts)
                if result is not None:
                    return result
                del starts[buffer_index]
        failed.add(key)
        return None

    # First pass: draws must agree. Second pass, for meshes with unusual vertex data:
    # perfect-looking records may stand against their draws.
    for trust_records in (False, True):
        failed.clear()
        trials = 0
        try:
            result = place(GUID_SIZE, list(range(len(tables.buffers))), {})
        except _SearchExhausted:
            result = None
        if result is not None:
            return result
    return _greedy_layout(tables, data, roles, samples)


def _greedy_layout(
    tables: GeometryTables, data: bytes, roles: dict[int, str], samples: dict[int, list[int]]
) -> dict[int, int]:
    """Place buffers in order, taking the first that fits or else the best-scoring one."""
    remaining = list(range(len(tables.buffers)))
    starts: dict[int, int] = {}
    position = GUID_SIZE
    while remaining:
        chosen: tuple[int, int, float, bool] | None = None
        for buffer_index in remaining:
            found = _fit(tables, data, starts, buffer_index, roles.get(buffer_index, "attribute"), position, samples)
            if found is None:
                continue
            if found[2]:
                chosen = (buffer_index, *found)
                break
            if chosen is None or found[1] > chosen[2]:
                chosen = (buffer_index, *found)
        if chosen is None:
            raise SerFormatError("no buffer fits the geometry blob", offset=position)
        buffer_index, start, _, _ = chosen
        starts[buffer_index] = start
        position = start + tables.buffers[buffer_index].size
        remaining.remove(buffer_index)
    if position > len(data):
        raise SerFormatError("buffers run past the end of the geometry blob", offset=position)
    return starts


def layout_quality(tables: GeometryTables, starts: dict[int, int], data: bytes) -> dict[int, float]:
    """Per buffer, how well its content fits the geometry (1.0 = perfectly); for reports and tests."""
    roles = _roles(tables)
    samples = _draw_samples(tables)
    out = {}
    for buffer_index in starts:
        role = roles.get(buffer_index, "attribute")
        geometric = _geometry_score(tables, starts, data, buffer_index, role, samples)
        own = _self_score(tables.buffers[buffer_index], role, data, starts[buffer_index])
        out[buffer_index] = round(min(own, geometric if geometric is not None else own), 4)
    return out


# -- batch index ranges -------------------------------------------------------------------
#
# A batch's triangles occupy its index buffer from its offset up to the next
# batch's offset. Decoding a whole range as one triangle list is a useful
# cross-check. The chained draw records of a batch add up exactly to its vertex
# count and triangle count.


def batch_index_ranges(tables: GeometryTables) -> dict[int, tuple[int, int, int]]:
    """Per batch: (index buffer, start byte, end byte). A range ends where the next batch's begins."""
    by_buffer: dict[int, list[tuple[int, int]]] = {}
    for batch, streams in enumerate(tables.batches):
        index_buffer, offset = streams[-1]
        by_buffer.setdefault(index_buffer, []).append((offset, batch))
    ranges = {}
    for index_buffer, entries in by_buffer.items():
        entries.sort()
        for k, (offset, batch) in enumerate(entries):
            end = entries[k + 1][0] if k + 1 < len(entries) else tables.buffers[index_buffer].size
            ranges[batch] = (index_buffer, offset, end)
    return ranges
