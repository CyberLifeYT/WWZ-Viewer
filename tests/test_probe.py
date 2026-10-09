from __future__ import annotations

import struct

import numpy as np

from extractor.ser.probe import (
    WORD_FLOAT,
    WORD_INT,
    WORD_OTHER,
    WORD_ZERO,
    build_report,
    classify_words,
    find_strings,
    float_runs,
    guess_strides,
    hexdump,
    size_equations,
)


def make_records(count: int = 500, seed: int = 7) -> bytes:
    """Records of 7 words: id, x, y, z, 1.0, constant flags, 0.0 (28 bytes each)."""
    rng = np.random.default_rng(seed)
    out = bytearray()
    for index in range(count):
        x, y, z = rng.uniform(-100.0, 100.0, size=3)
        out += struct.pack("<I3ffIf", index + 1, x, y, z, 1.0, 0xAABBCCDD, 0.0)
    return bytes(out)


def test_word_classes() -> None:
    raw = np.frombuffer(struct.pack("<IIfIfI", 0, 0x80000000, 1.0, 5, 12.5, 0x7FC00000), dtype="<u4")
    assert classify_words(raw).tolist() == [WORD_ZERO, WORD_ZERO, WORD_FLOAT, WORD_INT, WORD_FLOAT, WORD_OTHER]


def test_strings_detect_u32_prefix_padding_and_glued_length_bytes() -> None:
    data = (
        struct.pack("<I", 8) + b"<scene>\x00"  # padded, length includes the NUL
        + struct.pack("<I", 13) + b"light_redlamp"
        + struct.pack("<I", 0x3F) + b"x" * 0x3F  # 0x3F is '?', printable
        + b"\x00\x01loose_text\x00"
        + b"\x0bhk_2014.2.5\x00"
    )
    found = {item.text: item for item in find_strings(data, base_offset=0x40)}
    assert found["<scene>"].prefix == "u32"
    assert found["<scene>"].offset == 0x44
    # The '?' length byte would otherwise glue onto "light_redlamp".
    assert found["light_redlamp"].prefix == "u32"
    assert found["x" * 0x3F].prefix == "u32"
    assert found["loose_text"].prefix is None
    assert found["hk_2014.2.5"].prefix == "u8"


def test_float_runs_find_aligned_and_misaligned_regions() -> None:
    floats = struct.pack("<40f", *[float(index) + 0.5 for index in range(40)])
    ints = struct.pack("<8I", *([0xDEADBEEF] * 8))
    data = ints + floats + ints + b"\x07" + floats + ints
    runs = float_runs(data, base_offset=0x100)
    assert len(runs) == 2
    offsets = sorted(run.offset for run in runs)
    assert offsets == [0x100 + 32, 0x100 + 32 + 160 + 32 + 1]
    assert all(run.word_count == 40 for run in runs)


def test_float_runs_skip_zero_padding() -> None:
    assert float_runs(bytes(4096)) == []


def test_guess_strides_finds_record_size() -> None:
    guesses = guess_strides(make_records())
    assert guesses[0].stride == 28
    assert guesses[0].multiple_of is None
    assert any(guess.stride == 56 and guess.multiple_of == 28 for guess in guess_strides(make_records(), top=64))


def test_size_equations_find_count_and_stride() -> None:
    data = struct.pack("<II", 50, 3) + bytes(50 * 24)
    equations = size_equations(data)
    assert equations[0].count == 50
    assert equations[0].stride == 24
    assert equations[0].array_offset == 8


def test_hexdump_uses_base_offset() -> None:
    line = hexdump(b"1SERobb\x00", base_offset=0x40)
    assert line.startswith("00000040  31 53 45 52 6f 62 62 00")
    assert line.endswith("1SERobb.")


def test_report_covers_header_and_structure(ser_builder) -> None:
    records = make_records(300)
    data = ser_builder("obb", payload=struct.pack("<I", 300) + records)
    report = build_report(data, "synthetic")
    assert "type            obb" in report
    assert "u32 300 at 0x40 -> 300 records of 28 bytes from 0x44" in report
    assert "28B (score" in report
