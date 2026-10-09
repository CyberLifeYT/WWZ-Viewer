"""The ``1SER`` container that wraps almost every Swarm Engine resource.

Layout (all little-endian), verified against ``.lg``, ``.cdt``, ``.nav``, ``.obb``,
``.tpl``, ``.tbl``, ``.class_list``, ``.dom_list`` and ``.cd_list`` resources::

    0x00  char[4]   magic "1SER"
    0x04  char[32]  resource type, NUL-padded ("lg", "cdt", "obb", "tpl", "sm_tpl_tbl", ...)
    0x24  byte[16]  signature: the ASCII text "S3DRESOURCE     " on most resources;
                    on level graphs (.lg) and some templates (.tpl) it is instead the
                    resource's own GUID, which other resources (.cdt, .nav, .lg_data)
                    use to refer back to it
    0x34  u32       version (inferred): 0 on most types, 2 on .rain, 4 on .light_probe2
    0x38  u32       dependency count
    0x3C  u32       dependency block size in bytes
    0x40  dependency[count], each:
              u32      flags
              byte[16] GUID of the referenced resource (all zero when referenced by path only)
              u32      path length in bytes, including NUL padding to a multiple of 4
              char[n]  path, e.g. "<scene>\\ny_01_pvp.lg"
    0x40 + block size: type-specific payload

Resources that carry their own GUID may leave the type name empty; their
payload then starts with a 4-character sub-magic instead (``TPL1`` for those
templates, ``SCN1`` for level graphs). :attr:`SerResource.payload_magic` exposes it.

The payload layout is different for every resource type; decoding it is the job
of the format modules. This module only splits header from payload.
"""

from __future__ import annotations

from dataclasses import dataclass

from extractor.ser.binary_reader import GUID_SIZE, BinaryReader
from extractor.ser.errors import SerFormatError

SER_MAGIC = b"1SER"
RESOURCE_SIGNATURE = b"S3DRESOURCE     "
TYPE_NAME_SIZE = 32
HEADER_SIZE = 0x40
"""Size of the fixed header that precedes the dependency block."""


@dataclass(frozen=True)
class SerDependency:
    """A resource this one refers to, by GUID, path, or both."""

    flags: int
    guid: bytes
    path: str

    @property
    def has_guid(self) -> bool:
        return any(self.guid)

    @property
    def guid_hex(self) -> str:
        return self.guid.hex()


@dataclass(frozen=True)
class SerHeader:
    type_name: str
    """Resource type from the header, e.g. ``"lg"`` or ``"obb"``."""

    resource_guid: bytes | None
    """The resource's own GUID when the signature slot carries one (level graphs), else ``None``."""

    version: int
    """Header word at 0x34. Named from observation only: 0 on most types, 2 on ``rain``, 4 on ``light_probe``."""

    dependencies: tuple[SerDependency, ...]

    payload_offset: int
    """Absolute offset where the type-specific payload begins."""


@dataclass(frozen=True)
class SerResource:
    """A parsed header plus the full resource buffer."""

    header: SerHeader
    data: memoryview

    @property
    def type_name(self) -> str:
        return self.header.type_name

    @property
    def payload(self) -> memoryview:
        return self.data[self.header.payload_offset :]

    @property
    def payload_size(self) -> int:
        return len(self.data) - self.header.payload_offset

    @property
    def payload_magic(self) -> str | None:
        """The payload's 4-character sub-magic (``"SCN1"``, ``"TPL1"``), if it starts with one."""
        head = bytes(self.payload[:4])
        if len(head) == 4 and all(0x30 <= byte <= 0x39 or 0x41 <= byte <= 0x5A for byte in head):
            return head.decode("ascii")
        return None

    def reader(self) -> BinaryReader:
        """A reader over the payload whose error offsets are absolute file offsets."""
        return BinaryReader(self.payload, base_offset=self.header.payload_offset)


def is_ser(data: bytes | bytearray | memoryview) -> bool:
    """Whether a buffer starts with the ``1SER`` magic."""
    return bytes(data[: len(SER_MAGIC)]) == SER_MAGIC


def parse_header(data: bytes | bytearray | memoryview) -> SerHeader:
    """Parse the fixed header and dependency block.

    Raises:
        SerFormatError: on a wrong magic, a truncated header, or a dependency
            block whose entries do not add up to the declared size.
    """
    reader = BinaryReader(data)
    if reader.size < HEADER_SIZE:
        raise SerFormatError(f"buffer of {reader.size} bytes is smaller than the 1SER header", offset=0)

    magic = reader.raw(len(SER_MAGIC))
    if magic != SER_MAGIC:
        raise SerFormatError(f"bad magic {magic!r}, expected {SER_MAGIC!r}", offset=0)

    type_name = reader.fixed_string(TYPE_NAME_SIZE, encoding="ascii")
    signature = reader.raw(GUID_SIZE)
    resource_guid = None if signature == RESOURCE_SIGNATURE else signature
    version = reader.u32()
    dependency_count = reader.u32()
    dependency_block_size = reader.u32()

    block_start = reader.tell()  # == HEADER_SIZE
    block_end = block_start + dependency_block_size
    if block_end > reader.size:
        raise SerFormatError(
            f"dependency block of {dependency_block_size} bytes runs past end of resource",
            offset=block_start,
        )

    dependencies = []
    for _ in range(dependency_count):
        flags = reader.u32()
        guid = reader.guid()
        path = reader.len_string(max_length=dependency_block_size)
        dependencies.append(SerDependency(flags=flags, guid=guid, path=path))

    if reader.tell() != block_end:
        raise SerFormatError(
            f"{dependency_count} dependencies used {reader.tell() - block_start} bytes "
            f"but the header declares {dependency_block_size}",
            offset=block_start,
        )

    return SerHeader(
        type_name=type_name,
        resource_guid=resource_guid,
        version=version,
        dependencies=tuple(dependencies),
        payload_offset=block_end,
    )


def load(data: bytes | bytearray | memoryview) -> SerResource:
    """Parse a complete ``1SER`` resource held in memory."""
    view = memoryview(data).cast("B")
    return SerResource(header=parse_header(view), data=view)
