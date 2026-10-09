"""Little-endian cursor over an in-memory buffer.

Every Swarm Engine format seen so far is little-endian with 4-byte alignment,
32-bit floats, and strings stored as a ``u32`` byte length followed by the bytes
(sometimes null-padded to a multiple of 4). This reader covers those primitives
and reports the absolute offset whenever a read runs past the end, which is
what you need when decoding an unknown layout.
"""

from __future__ import annotations

import struct

from extractor.ser.errors import SerFormatError

_U8 = struct.Struct("<B")
_I8 = struct.Struct("<b")
_U16 = struct.Struct("<H")
_I16 = struct.Struct("<h")
_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")
_U64 = struct.Struct("<Q")
_I64 = struct.Struct("<q")
_F32 = struct.Struct("<f")
_F64 = struct.Struct("<d")
_VEC2 = struct.Struct("<2f")
_VEC3 = struct.Struct("<3f")
_VEC4 = struct.Struct("<4f")
_MAT3 = struct.Struct("<9f")
_MAT3X4 = struct.Struct("<12f")
_MAT4 = struct.Struct("<16f")

GUID_SIZE = 16


class BinaryReader:
    """Sequential reader with random access (``seek``) over ``bytes`` or ``memoryview``.

    ``base_offset`` is added to positions in error messages and :meth:`absolute`,
    so a reader over a payload slice can still report file offsets.
    """

    def __init__(self, data: bytes | bytearray | memoryview, position: int = 0, base_offset: int = 0) -> None:
        self._data = memoryview(data).cast("B")
        self._size = len(self._data)
        self._pos = 0
        self.base_offset = base_offset
        self.seek(position)

    # -- position --------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    def tell(self) -> int:
        return self._pos

    def absolute(self) -> int:
        """Current position plus ``base_offset``, i.e. the offset in the original file."""
        return self.base_offset + self._pos

    def remaining(self) -> int:
        return self._size - self._pos

    def at_end(self) -> bool:
        return self._pos >= self._size

    def seek(self, position: int) -> None:
        if not 0 <= position <= self._size:
            raise SerFormatError(
                f"seek to {position:#x} outside buffer of {self._size:#x} bytes",
                offset=self.base_offset + position,
            )
        self._pos = position

    def skip(self, count: int) -> None:
        self.seek(self._pos + count)

    def align(self, boundary: int) -> None:
        """Advance to the next multiple of ``boundary`` (relative to the buffer start)."""
        if boundary <= 0:
            raise ValueError("boundary must be positive")
        overshoot = self._pos % boundary
        if overshoot:
            self.skip(boundary - overshoot)

    # -- raw access ------------------------------------------------------------------

    def _require(self, count: int) -> None:
        if count < 0:
            raise SerFormatError(f"negative read length {count}", offset=self.absolute())
        if self._pos + count > self._size:
            raise SerFormatError(
                f"read of {count} bytes runs past end of buffer ({self.remaining()} bytes left)",
                offset=self.absolute(),
            )

    def _unpack(self, packer: struct.Struct) -> tuple:
        self._require(packer.size)
        values = packer.unpack_from(self._data, self._pos)
        self._pos += packer.size
        return values

    def raw(self, count: int) -> bytes:
        self._require(count)
        chunk = self._data[self._pos : self._pos + count].tobytes()
        self._pos += count
        return chunk

    def peek(self, count: int) -> bytes:
        """Up to ``count`` bytes from the current position without advancing."""
        return self._data[self._pos : self._pos + count].tobytes()

    # -- scalars ---------------------------------------------------------------------

    def u8(self) -> int:
        return self._unpack(_U8)[0]

    def i8(self) -> int:
        return self._unpack(_I8)[0]

    def u16(self) -> int:
        return self._unpack(_U16)[0]

    def i16(self) -> int:
        return self._unpack(_I16)[0]

    def u32(self) -> int:
        return self._unpack(_U32)[0]

    def i32(self) -> int:
        return self._unpack(_I32)[0]

    def u64(self) -> int:
        return self._unpack(_U64)[0]

    def i64(self) -> int:
        return self._unpack(_I64)[0]

    def f32(self) -> float:
        return self._unpack(_F32)[0]

    def f64(self) -> float:
        return self._unpack(_F64)[0]

    def bool8(self) -> bool:
        return self.u8() != 0

    # -- vectors and matrices (row-major, as stored) ---------------------------------

    def vec2(self) -> tuple[float, float]:
        return self._unpack(_VEC2)

    def vec3(self) -> tuple[float, float, float]:
        return self._unpack(_VEC3)

    def vec4(self) -> tuple[float, float, float, float]:
        return self._unpack(_VEC4)

    def mat3(self) -> tuple[float, ...]:
        """Nine floats in storage order."""
        return self._unpack(_MAT3)

    def mat3x4(self) -> tuple[float, ...]:
        """Twelve floats in storage order (a 3x3 rotation plus a translation, layout per format)."""
        return self._unpack(_MAT3X4)

    def mat4(self) -> tuple[float, ...]:
        """Sixteen floats in storage order."""
        return self._unpack(_MAT4)

    def f32_array(self, count: int) -> tuple[float, ...]:
        return self._unpack(struct.Struct(f"<{count}f"))

    def u32_array(self, count: int) -> tuple[int, ...]:
        return self._unpack(struct.Struct(f"<{count}I"))

    # -- identifiers and strings -----------------------------------------------------

    def guid(self) -> bytes:
        """16 raw bytes. Kept as bytes because the engine's GUID byte order is not known yet."""
        return self.raw(GUID_SIZE)

    def fixed_string(self, length: int, encoding: str = "utf-8") -> str:
        """A ``length``-byte field holding a string terminated or padded with NULs."""
        raw = self.raw(length)
        return raw.split(b"\x00", 1)[0].decode(encoding, errors="replace")

    def len_string(self, encoding: str = "utf-8", max_length: int = 1 << 20) -> str:
        """``u32`` byte length, then that many bytes; trailing NUL padding is stripped.

        ``max_length`` guards against misreading a non-string field as a huge length.
        """
        start = self.absolute()
        length = self.u32()
        if length > max_length:
            raise SerFormatError(f"string length {length} exceeds limit {max_length}", offset=start)
        raw = self.raw(length)
        return raw.rstrip(b"\x00").decode(encoding, errors="replace")
