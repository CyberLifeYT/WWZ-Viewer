"""Shared fixtures: synthetic resources built in code, so tests never need game files."""

from __future__ import annotations

import struct
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

RESOURCE_SIGNATURE = b"S3DRESOURCE     "


def build_ser(
    type_name: str = "obb",
    payload: bytes = b"",
    dependencies: Iterable[tuple[int, bytes, str]] = (),
    resource_guid: bytes | None = None,
    version: int = 0,
) -> bytes:
    """Assemble a 1SER resource with the layout observed in the game files.

    Dependency paths get at least one NUL and are padded to a multiple of 4,
    which reproduces the real files (e.g. a 20-character path is stored in 24 bytes).
    """
    block = bytearray()
    dependency_list = list(dependencies)
    for flags, guid, path in dependency_list:
        encoded = path.encode("ascii")
        stored_length = (len(encoded) // 4 + 1) * 4
        block += struct.pack("<I", flags) + guid + struct.pack("<I", stored_length) + encoded.ljust(stored_length, b"\x00")
    signature = resource_guid if resource_guid is not None else RESOURCE_SIGNATURE
    header = (
        b"1SER"
        + type_name.encode("ascii").ljust(32, b"\x00")
        + signature
        + struct.pack("<III", version, len(dependency_list), len(block))
    )
    return header + bytes(block) + payload


def build_pak(path: Path, entries: dict[str, bytes], padding: int = 4096) -> Path:
    """Write a ZIP pak like the game's: '.tpl' entries deflated, the rest stored, zero padding after the end record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            method = zipfile.ZIP_DEFLATED if name.endswith(".tpl") else zipfile.ZIP_STORED
            archive.writestr(name, data, compress_type=method)
    with path.open("ab") as handle:
        handle.write(b"\x00" * padding)
    return path


@pytest.fixture
def ser_builder() -> Callable[..., bytes]:
    return build_ser


@pytest.fixture
def pak_builder() -> Callable[..., Path]:
    return build_pak
