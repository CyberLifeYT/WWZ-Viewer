from __future__ import annotations

import struct

import pytest

from extractor.ser import BinaryReader, SerFormatError


def test_scalars_round_trip() -> None:
    data = struct.pack("<BbHhIiQqfd", 200, -5, 60000, -3000, 4_000_000_000, -7, 2**40, -(2**40), 1.5, -2.25)
    reader = BinaryReader(data)
    assert reader.u8() == 200
    assert reader.i8() == -5
    assert reader.u16() == 60000
    assert reader.i16() == -3000
    assert reader.u32() == 4_000_000_000
    assert reader.i32() == -7
    assert reader.u64() == 2**40
    assert reader.i64() == -(2**40)
    assert reader.f32() == 1.5
    assert reader.f64() == -2.25
    assert reader.at_end()


def test_vectors_and_matrices_keep_storage_order() -> None:
    values = [float(index) for index in range(16)]
    reader = BinaryReader(struct.pack("<3f12f16f", *values[:3], *values[:12], *values))
    assert reader.vec3() == (0.0, 1.0, 2.0)
    assert reader.mat3x4() == tuple(values[:12])
    assert reader.mat4() == tuple(values)


def test_len_string_strips_nul_padding() -> None:
    # "<scene>" is stored as 8 bytes: 7 characters plus one NUL.
    reader = BinaryReader(struct.pack("<I", 8) + b"<scene>\x00" + b"tail")
    assert reader.len_string() == "<scene>"
    assert reader.raw(4) == b"tail"


def test_fixed_string_stops_at_first_nul() -> None:
    reader = BinaryReader(b"obb\x00garbage\x00" + b"next")  # 12-byte field, then the next field
    assert reader.fixed_string(12) == "obb"
    assert reader.raw(4) == b"next"


def test_align_moves_to_next_boundary_only_when_needed() -> None:
    reader = BinaryReader(bytes(16))
    reader.align(4)
    assert reader.tell() == 0
    reader.skip(5)
    reader.align(4)
    assert reader.tell() == 8


def test_peek_does_not_advance() -> None:
    reader = BinaryReader(b"ABCDEF")
    assert reader.peek(3) == b"ABC"
    assert reader.tell() == 0


def test_read_past_end_reports_absolute_offset() -> None:
    reader = BinaryReader(b"\x01\x02\x03", base_offset=0x100)
    reader.skip(2)
    with pytest.raises(SerFormatError) as caught:
        reader.u32()
    assert caught.value.offset == 0x102
    assert "0x102" in str(caught.value)


def test_seek_outside_buffer_is_rejected() -> None:
    reader = BinaryReader(bytes(4))
    reader.seek(4)  # the end itself is a valid position
    with pytest.raises(SerFormatError):
        reader.seek(5)


def test_len_string_guards_against_absurd_lengths() -> None:
    reader = BinaryReader(struct.pack("<I", 0xFFFFFFF0) + b"abc")
    with pytest.raises(SerFormatError, match="exceeds limit"):
        reader.len_string()
