"""Reverse-engineering helpers for payloads that have no decoder yet.

None of these functions know a specific format. They answer the questions you
ask when looking at an unknown binary blob:

* What do the first words look like as u32 / i32 / f32? (:func:`word_table`)
* Is one of the leading integers an array count whose records fill the rest of
  the file exactly? (:func:`size_equations`)
* Where are the strings, and are they length-prefixed the way the engine
  usually stores them? (:func:`find_strings`)
* Which regions are made of floats (positions, matrices, vertices)?
  (:func:`float_runs`)
* What record size do those regions repeat with? (:func:`guess_strides`)

:func:`build_report` runs all of them and formats a plain-text report.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

import numpy as np

from extractor.ser.container import is_ser, load
from extractor.ser.errors import SerFormatError

# Floats in level data are positions (up to ~1e4 m), unit vectors and rotation
# entries (|v| <= 1, often with noise around 1e-12), scales and colours. A random
# 32-bit word (hash, flags, packed data) decodes to a float with a roughly
# uniform exponent; this window accepts about 29% of those, so a run of 8
# random words passes as "floats" by chance only about 0.005% of the time.
PLAUSIBLE_FLOAT_MIN = 1e-15
PLAUSIBLE_FLOAT_MAX = 1e7

# u32 values below this are probably counts, indices or enums. They never
# collide with plausible floats: as floats they are denormals below 1e-40.
SMALL_INT_LIMIT = 0x10000

WORD_ZERO = 0
WORD_FLOAT = 1
WORD_INT = 2
WORD_OTHER = 3
WORD_CLASS_NAMES = ("zero", "float", "int", "other")

NEGATIVE_ZERO = 0x80000000


@dataclass(frozen=True)
class FloatRun:
    """A stretch of consecutive 4-byte words that all read as plausible floats or zero."""

    offset: int
    """Absolute byte offset of the first word."""

    word_count: int
    float_fraction: float
    """Share of words that are non-zero floats (the rest are zero)."""

    @property
    def byte_length(self) -> int:
        return self.word_count * 4

    @property
    def end(self) -> int:
        return self.offset + self.byte_length


@dataclass(frozen=True)
class FoundString:
    offset: int
    """Absolute offset of the first character."""

    text: str
    prefix: str | None
    """``"u32"`` or ``"u8"`` when a matching length field sits right before the text, else ``None``."""


@dataclass(frozen=True)
class StrideGuess:
    stride: int
    """Candidate record size in bytes."""

    score: float
    purity: float
    """How consistently each column keeps one word class (float / int / zero / other)."""

    repeat: float
    """How often a word equals the same column in the previous record (constants, flags, unit axes)."""

    multiple_of: int | None
    """Set when a smaller stride that divides this one scores about as well."""


@dataclass(frozen=True)
class SizeEquation:
    """``count`` records of ``stride`` bytes starting at ``array_offset`` end exactly at the end of the data."""

    count_offset: int
    count: int
    array_offset: int
    stride: int


# -- word classification -----------------------------------------------------------


def _as_bytes_view(data: bytes | bytearray | memoryview) -> memoryview:
    return memoryview(data).cast("B")


def words(data: bytes | bytearray | memoryview, alignment: int = 0) -> np.ndarray:
    """``data[alignment:]`` as little-endian u32 words; a trailing partial word is dropped."""
    view = _as_bytes_view(data)[alignment:]
    usable = len(view) - len(view) % 4
    return np.frombuffer(view[:usable], dtype="<u4")


def classify_words(raw: np.ndarray) -> np.ndarray:
    """One of ``WORD_ZERO / WORD_FLOAT / WORD_INT / WORD_OTHER`` per u32 word."""
    floats = raw.view("<f4")
    with np.errstate(invalid="ignore", over="ignore"):
        magnitude = np.abs(floats)
        plausible = np.isfinite(floats) & (magnitude >= PLAUSIBLE_FLOAT_MIN) & (magnitude <= PLAUSIBLE_FLOAT_MAX)
    classes = np.full(raw.shape, WORD_OTHER, dtype=np.uint8)
    classes[raw < SMALL_INT_LIMIT] = WORD_INT
    classes[plausible] = WORD_FLOAT
    classes[(raw == 0) | (raw == NEGATIVE_ZERO)] = WORD_ZERO
    return classes


def classify_word(value: int) -> str:
    return WORD_CLASS_NAMES[int(classify_words(np.array([value], dtype="<u4"))[0])]


def _true_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start indices and lengths of every run of ``True`` in a boolean array."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts = edges[0::2]
    return starts, edges[1::2] - starts


# -- text views --------------------------------------------------------------------


def hexdump(
    data: bytes | bytearray | memoryview,
    start: int = 0,
    length: int = 256,
    base_offset: int = 0,
    width: int = 16,
) -> str:
    """Classic hex + ASCII dump of ``data[start:start+length]``; offsets include ``base_offset``."""
    view = _as_bytes_view(data)
    end = min(len(view), start + length)
    lines = []
    for row in range(start, end, width):
        chunk = view[row : min(row + width, end)].tobytes()
        hex_part = " ".join(f"{byte:02x}" for byte in chunk)
        ascii_part = "".join(chr(byte) if 0x20 <= byte < 0x7F else "." for byte in chunk)
        lines.append(f"{base_offset + row:08x}  {hex_part:<{width * 3 - 1}}  {ascii_part}")
    return "\n".join(lines)


def word_table(
    data: bytes | bytearray | memoryview,
    start: int = 0,
    count: int = 32,
    base_offset: int = 0,
) -> str:
    """Each 4-byte word from ``start`` shown as hex, u32, i32 and f32, with a class guess."""
    view = _as_bytes_view(data)
    lines = [f"  {'offset':<10}  {'bytes':<11}  {'u32':>11}  {'i32':>11}  {'f32':>14}  class  ascii"]
    for index in range(count):
        position = start + index * 4
        if position + 4 > len(view):
            break
        raw = view[position : position + 4].tobytes()
        unsigned = int.from_bytes(raw, "little")
        signed = int.from_bytes(raw, "little", signed=True)
        as_float = struct.unpack("<f", raw)[0]
        ascii_part = "".join(chr(byte) if 0x20 <= byte < 0x7F else "." for byte in raw)
        lines.append(
            f"  {base_offset + position:#010x}  {raw.hex(' ')}  {unsigned:>11}  {signed:>11}  "
            f"{as_float:>14.6g}  {classify_word(unsigned):<5}  {ascii_part}"
        )
    return "\n".join(lines)


# -- structure finders -------------------------------------------------------------


def size_equations(
    data: bytes | bytearray | memoryview,
    start: int = 0,
    leading_words: int = 16,
    max_gap_words: int = 16,
    max_stride: int = 4096,
) -> list[SizeEquation]:
    """Leading integers that look like the count of an array running to the end of the data.

    For each of the first ``leading_words`` u32 values at ``start``, try array
    starts from right after the count up to ``max_gap_words`` words later and
    report the first one where the remaining bytes divide evenly into that many
    records of 4 to ``max_stride`` bytes.
    """
    view = _as_bytes_view(data)
    total = len(view)
    equations = []
    for index in range(leading_words):
        position = start + index * 4
        if position + 4 > total:
            break
        count = int.from_bytes(view[position : position + 4], "little")
        if count < 2 or count > total:
            continue
        for gap in range(max_gap_words + 1):
            array_offset = position + 4 + gap * 4
            remaining = total - array_offset
            if remaining < count * 4:
                break
            if remaining % count == 0 and remaining // count <= max_stride:
                equations.append(SizeEquation(position, count, array_offset, remaining // count))
                break
    return equations


_PRINTABLE_RUN_CACHE: dict[int, re.Pattern[bytes]] = {}


def find_strings(
    data: bytes | bytearray | memoryview,
    base_offset: int = 0,
    min_length: int = 4,
) -> list[FoundString]:
    """Printable-ASCII runs, flagged when a length field directly precedes them.

    The engine usually stores strings as a ``u32`` length followed by the bytes,
    sometimes NUL-padded to a multiple of 4 (the length then includes the padding).
    A printable length byte can glue itself onto the end of the previous string
    (``"...light_01?"`` where ``?`` = 0x3F is the next string's length); when the
    preceding ``u32`` is shorter than the run, the text is cut to that length.
    """
    pattern = _PRINTABLE_RUN_CACHE.get(min_length)
    if pattern is None:
        pattern = re.compile(rb"[\x20-\x7e]{%d,}" % min_length)
        _PRINTABLE_RUN_CACHE[min_length] = pattern

    buffer = bytes(data)
    found = []
    for match in pattern.finditer(buffer):
        start = match.start()
        text = match.group()
        prefix = None
        if start >= 4:
            declared = int.from_bytes(buffer[start - 4 : start], "little")
            if min_length <= declared <= len(text):
                text, prefix = text[:declared], "u32"
            elif len(text) < declared <= len(text) + 4 and not buffer[start + len(text) : start + declared].strip(b"\x00"):
                prefix = "u32"
        if prefix is None and start >= 1:
            declared = buffer[start - 1]
            if min_length <= declared <= len(text):
                text, prefix = text[:declared], "u8"
        found.append(FoundString(base_offset + start, text.decode("ascii"), prefix))
    return found


def float_runs(
    data: bytes | bytearray | memoryview,
    base_offset: int = 0,
    min_words: int = 8,
    min_float_fraction: float = 0.5,
) -> list[FloatRun]:
    """Regions of consecutive plausible floats, longest first.

    All four byte alignments are scanned because variable-length strings can
    leave later data unaligned. Overlapping candidates from different
    alignments are resolved in favour of the longer run. Runs where fewer than
    ``min_float_fraction`` of words are non-zero (mostly zero padding) are dropped.
    """
    candidates: list[tuple[int, int, float]] = []  # (byte start, word count, float fraction)
    for alignment in range(4):
        raw = words(data, alignment)
        if raw.size < min_words:
            continue
        classes = classify_words(raw)
        is_float = classes == WORD_FLOAT
        starts, lengths = _true_runs(is_float | (classes == WORD_ZERO))
        long_enough = lengths >= min_words
        float_prefix_sum = np.concatenate(([0], np.cumsum(is_float, dtype=np.int64)))
        for start, length in zip(starts[long_enough].tolist(), lengths[long_enough].tolist()):
            fraction = (float_prefix_sum[start + length] - float_prefix_sum[start]) / length
            if fraction >= min_float_fraction:
                candidates.append((alignment + start * 4, length, float(fraction)))

    candidates.sort()
    accepted: list[tuple[int, int, float]] = []
    for byte_start, length, fraction in candidates:
        if accepted:
            last_start, last_length, _ = accepted[-1]
            if byte_start < last_start + last_length * 4:
                if length > last_length:
                    accepted[-1] = (byte_start, length, fraction)
                continue
        accepted.append((byte_start, length, fraction))

    runs = [FloatRun(base_offset + start, length, fraction) for start, length, fraction in accepted]
    runs.sort(key=lambda run: run.word_count, reverse=True)
    return runs


def guess_strides(
    data: bytes | bytearray | memoryview,
    max_stride_words: int = 64,
    max_words: int = 1 << 18,
    top: int = 6,
) -> list[StrideGuess]:
    """Rank candidate record sizes for a region believed to hold an array of records.

    The region is cut into rows of ``s`` words for every ``s`` up to
    ``max_stride_words``. At the true record size each column holds the same
    field in every row, so it keeps one word class (high *purity*) and constant
    fields repeat row to row (high *repeat*). Wrong sizes smear fields across
    columns and lower both. Multiples of the true size score just as well, so a
    guess is marked ``multiple_of`` when a smaller divisor scores within 0.02.

    Pass data that starts at a record boundary when possible; a misaligned start
    only rotates the columns and does not change the ranking.
    """
    raw = words(data)[:max_words]
    classes = classify_words(raw)
    scored: dict[int, tuple[float, float, float]] = {}
    for stride_words in range(1, max_stride_words + 1):
        rows = raw.size // stride_words
        if rows < 8:
            break
        grid = raw[: rows * stride_words].reshape(rows, stride_words)
        class_grid = classes[: rows * stride_words].reshape(rows, stride_words)
        class_counts = np.stack([(class_grid == word_class).sum(axis=0) for word_class in range(4)])
        purity = float((class_counts.max(axis=0) / rows).mean())
        repeat = float((grid[1:] == grid[:-1]).mean())
        scored[stride_words] = ((purity + repeat) / 2, purity, repeat)

    guesses = []
    for stride_words, (score, purity, repeat) in scored.items():
        divisor = next(
            (
                smaller
                for smaller in range(1, stride_words)
                if stride_words % smaller == 0 and scored[smaller][0] >= score - 0.02
            ),
            None,
        )
        guesses.append(
            StrideGuess(
                stride=stride_words * 4,
                score=score,
                purity=purity,
                repeat=repeat,
                multiple_of=divisor * 4 if divisor is not None else None,
            )
        )
    # Base strides first, then by score; multiples are only informative.
    guesses.sort(key=lambda guess: (guess.multiple_of is not None, -guess.score))
    return guesses[:top]


# -- report ------------------------------------------------------------------------


def _section(title: str, body: str) -> str:
    return f"== {title} ==\n{body}\n"


def _floats_preview(data: memoryview, start: int, count: int) -> str:
    available = min(count, (len(data) - start) // 4)
    values = struct.unpack_from(f"<{available}f", data, start)
    return " ".join(f"{value:.6g}" for value in values)


def build_report(
    data: bytes | bytearray | memoryview,
    name: str,
    base_offset: int = 0,
    parse_container: bool = True,
    max_strings: int = 40,
    max_runs: int = 12,
    table_words: int = 48,
) -> str:
    """Run every probe over ``data`` and return a plain-text report.

    When ``parse_container`` is true and the data starts with ``1SER``, the
    header is decoded and the remaining probes start at the payload. Pass
    ``base_offset`` when ``data`` is a window into a larger resource so offsets
    in the report match the original file.
    """
    view = _as_bytes_view(data)
    parts = [f"Probe report: {name}\nSize: {len(view):,} bytes (window starts at {base_offset:#x})\n"]

    payload_start = 0
    if parse_container and base_offset == 0 and is_ser(view):
        try:
            resource = load(view)
        except SerFormatError as exc:
            parts.append(_section("1SER header", f"  could not parse: {exc}"))
        else:
            header = resource.header
            lines = [
                f"  type            {header.type_name or '(empty)'}",
                f"  own GUID        {header.resource_guid.hex() if header.resource_guid else '(none, S3DRESOURCE signature)'}",
                f"  version         {header.version}",
                f"  payload offset  {header.payload_offset:#x}",
                f"  payload size    {resource.payload_size:,} bytes",
                f"  payload magic   {resource.payload_magic or '(none)'}",
                f"  dependencies    {len(header.dependencies)}",
            ]
            for dependency in header.dependencies:
                guid = dependency.guid_hex if dependency.has_guid else "-" * 32
                lines.append(f"    flags={dependency.flags:#010x}  guid={guid}  {dependency.path}")
            parts.append(_section("1SER header", "\n".join(lines)))
            payload_start = header.payload_offset

    payload = view[payload_start:]
    payload_base = base_offset + payload_start

    parts.append(_section("Leading words", word_table(view, payload_start, table_words, base_offset)))

    equations = size_equations(view, payload_start)
    if equations:
        body = "\n".join(
            f"  u32 {eq.count:,} at {base_offset + eq.count_offset:#x} -> {eq.count:,} records of "
            f"{eq.stride} bytes from {base_offset + eq.array_offset:#x} fill the data exactly"
            for eq in equations
        )
    else:
        body = "  none: no leading integer is the count of an array that runs to the end"
    parts.append(_section("Size equations", body))

    strings = find_strings(payload, base_offset=payload_base)
    prefixed = sum(1 for found in strings if found.prefix)
    lines = [f"  {len(strings):,} strings, {prefixed:,} with a length prefix; first {min(max_strings, len(strings))}:"]
    for found in strings[:max_strings]:
        lines.append(f"  {found.offset:#010x}  [{found.prefix or '-':>3}]  {found.text[:100]!r}")
    parts.append(_section("Strings", "\n".join(lines)))

    runs = float_runs(payload, base_offset=payload_base)
    covered = sum(run.byte_length for run in runs)
    share = covered / len(payload) if len(payload) else 0.0
    lines = [f"  {len(runs):,} runs covering {covered:,} bytes ({share:.1%} of payload); longest {min(max_runs, len(runs))}:"]
    for run in runs[:max_runs]:
        local = run.offset - base_offset
        lines.append(
            f"  {run.offset:#010x}..{run.end:#010x}  {run.word_count:>9,} words  "
            f"{run.float_fraction:5.1%} non-zero  first: {_floats_preview(view, local, 6)}"
        )
    parts.append(_section("Float runs", "\n".join(lines)))

    # Records that mix integers with floats never form long float runs, so also
    # guess over the whole payload and over any array a size equation found.
    regions = [("payload", payload_start, len(view))]
    regions += [(f"array at {base_offset + eq.array_offset:#x}", eq.array_offset, len(view)) for eq in equations[:2]]
    regions += [
        (f"float run at {run.offset:#x}", run.offset - base_offset, run.offset - base_offset + run.byte_length)
        for run in runs[:3]
        if run.word_count >= 64
    ]
    stride_lines = []
    for label, start, end in regions:
        if end - start < 64 * 4:
            continue
        guesses = guess_strides(view[start:end])
        rendered = ", ".join(
            f"{guess.stride}B (score {guess.score:.2f}"
            + (f", x of {guess.multiple_of}B" if guess.multiple_of else "")
            + ")"
            for guess in guesses
        )
        stride_lines.append(f"  {label}: {rendered}")
    parts.append(_section("Record size guesses", "\n".join(stride_lines) or "  no region long enough (64+ words)"))

    return "\n".join(parts)
