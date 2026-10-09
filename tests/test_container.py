from __future__ import annotations

import struct

import pytest

from extractor.ser import SerFormatError, is_ser, load, parse_header
from extractor.ser.container import HEADER_SIZE

LEVEL_GUID = bytes.fromhex("977ce2c78546de4aa03f2af348eccdd2")


def test_resource_without_dependencies_has_payload_at_0x40(ser_builder) -> None:
    data = ser_builder("obb", payload=b"\x46\x1d\x00\x00payload")
    resource = load(data)
    assert resource.type_name == "obb"
    assert resource.header.resource_guid is None
    assert resource.header.dependencies == ()
    assert resource.header.payload_offset == HEADER_SIZE == 0x40
    assert bytes(resource.payload) == b"\x46\x1d\x00\x00payload"


def test_dependency_layout_matches_observed_cdt_header(ser_builder) -> None:
    # A .cdt referencing its level graph: payload starts at 0x70 in the game file.
    data = ser_builder("cdt", payload=b"DATA", dependencies=[(0x01000001, LEVEL_GUID, "<scene>\\ny_01_pvp.lg")])
    header = parse_header(data)
    assert header.payload_offset == 0x70
    (dependency,) = header.dependencies
    assert dependency.flags == 0x01000001
    assert dependency.guid == LEVEL_GUID and dependency.has_guid
    assert dependency.path == "<scene>\\ny_01_pvp.lg"


def test_guid_signature_and_payload_magic(ser_builder) -> None:
    data = ser_builder(
        "lg",
        payload=b"SCN1" + bytes(8),
        dependencies=[(0, bytes(16), "<scene>\\ny_01_pvp.geom_dbg")],
        resource_guid=LEVEL_GUID,
    )
    resource = load(data)
    assert resource.header.resource_guid == LEVEL_GUID
    assert not resource.header.dependencies[0].has_guid
    assert resource.header.payload_offset == 0x74
    assert resource.payload_magic == "SCN1"


def test_payload_magic_is_none_for_binary_payload(ser_builder) -> None:
    assert load(ser_builder("obb", payload=b"\x46\x1d\x00\x00")).payload_magic is None


def test_version_word_is_exposed(ser_builder) -> None:
    assert parse_header(ser_builder("light_probe", version=4)).version == 4


def test_multiple_dependencies(ser_builder) -> None:
    data = ser_builder(
        "tpl",
        payload=b"X",
        dependencies=[(0, bytes(16), "<scene>\\ny_01_pvp.lg"), (0, bytes(16), "<scene>\\tpl\\some_template.tpl")],
    )
    header = parse_header(data)
    assert [dependency.path for dependency in header.dependencies] == [
        "<scene>\\ny_01_pvp.lg",
        "<scene>\\tpl\\some_template.tpl",
    ]
    assert data[header.payload_offset :] == b"X"


def test_reader_reports_absolute_offsets(ser_builder) -> None:
    resource = load(ser_builder("obb", payload=struct.pack("<I", 7)))
    reader = resource.reader()
    assert reader.u32() == 7
    with pytest.raises(SerFormatError) as caught:
        reader.u32()
    assert caught.value.offset == 0x44


def test_bad_magic_is_rejected(ser_builder) -> None:
    data = b"2SER" + ser_builder("obb")[4:]
    assert not is_ser(data)
    with pytest.raises(SerFormatError, match="bad magic"):
        parse_header(data)


def test_truncated_header_is_rejected(ser_builder) -> None:
    with pytest.raises(SerFormatError, match="smaller than the 1SER header"):
        parse_header(ser_builder("obb")[:0x30])


def test_dependency_block_size_mismatch_is_rejected(ser_builder) -> None:
    data = bytearray(ser_builder("cdt", payload=bytes(8), dependencies=[(1, LEVEL_GUID, "<scene>\\a.lg")]))
    declared = struct.unpack_from("<I", data, 0x3C)[0]
    struct.pack_into("<I", data, 0x3C, declared + 4)
    with pytest.raises(SerFormatError, match="declares"):
        parse_header(data)


def test_dependency_block_past_end_is_rejected(ser_builder) -> None:
    data = bytearray(ser_builder("cdt"))
    struct.pack_into("<II", data, 0x38, 1, 1000)
    with pytest.raises(SerFormatError, match="runs past end"):
        parse_header(data)
