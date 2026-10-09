"""Read-only access to Swarm Engine ``.pak`` archives.

A ``.pak`` is a standard ZIP archive (entries are stored or deflated, never
encrypted) followed by zero padding, so Python's ``zipfile`` reads it directly.
Entry names use forward slashes and bracketed mount prefixes, for example
``[scene]/ny_01_pvp.lg`` or ``[ssl_cache]/scenes/ny_01_pvp/ny_01_pvp_release.spdb``.

The game ships most content twice, once under ``Content/client`` (rendering data)
and once under ``Content/server`` (gameplay data). :class:`PakSet` pairs the two
halves of one pak name so callers can search both at once.

Several paks in the shipped build are 0-byte stubs (``client/shared.pak``,
``server/boot.pak``, ...). Those open as empty archives instead of raising.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import IO, Iterator, Literal

Side = Literal["client", "server"]
SIDES: tuple[Side, ...] = ("client", "server")


class PakError(RuntimeError):
    """Raised when a pak cannot be opened or an entry cannot be found."""


@dataclass(frozen=True)
class PakEntry:
    """One file inside a pak archive."""

    name: str
    """Full entry name exactly as stored in the archive, e.g. ``[scene]/ny_01_pvp.obb``."""

    size: int
    """Uncompressed size in bytes."""

    compressed_size: int
    """Size as stored in the archive."""

    is_compressed: bool
    """True for deflated entries, False for stored ones."""

    side: Side | None
    """Which half of the install the entry came from, if known."""

    @property
    def basename(self) -> str:
        return PurePosixPath(self.name).name

    @property
    def extension(self) -> str:
        """Lower-cased extension including the dot, or ``""`` when there is none."""
        return PurePosixPath(self.name).suffix.lower()


class PakArchive:
    """One ``.pak`` file opened read-only.

    Use as a context manager, or call :meth:`close`, to release the file handle.
    """

    def __init__(self, path: str | Path, side: Side | None = None) -> None:
        self.path = Path(path)
        self.side = side
        self._zip: zipfile.ZipFile | None = None
        self._entries: dict[str, PakEntry] = {}

        if not self.path.is_file():
            raise PakError(f"Pak not found: {self.path}")
        if self.path.stat().st_size == 0:
            # 0-byte placeholder shipped with the game: behaves as an empty archive.
            return

        try:
            self._zip = zipfile.ZipFile(self.path, mode="r")
        except zipfile.BadZipFile as exc:
            raise PakError(f"{self.path} is not a readable pak archive: {exc}") from exc

        for info in self._zip.infolist():
            if info.is_dir():
                continue
            self._entries[info.filename] = PakEntry(
                name=info.filename,
                size=info.file_size,
                compressed_size=info.compress_size,
                is_compressed=info.compress_type != zipfile.ZIP_STORED,
                side=side,
            )

    # -- lifecycle -----------------------------------------------------------------

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __enter__(self) -> PakArchive:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- listing -------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Pak name without extension, e.g. ``ny_01_pvp``."""
        return self.path.stem

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, entry_name: object) -> bool:
        return entry_name in self._entries

    def __iter__(self) -> Iterator[PakEntry]:
        return iter(self._entries.values())

    def entries(self) -> list[PakEntry]:
        """All entries in archive order."""
        return list(self._entries.values())

    def get(self, entry_name: str) -> PakEntry:
        try:
            return self._entries[entry_name]
        except KeyError:
            raise PakError(f"{entry_name!r} is not in {self.path.name}") from None

    def find(self, query: str) -> list[PakEntry]:
        """Entries matching ``query``.

        An exact entry name wins outright. Otherwise the match is a
        case-insensitive suffix match, so ``.obb``, ``ny_01_pvp.obb`` and
        ``[scene]/ny_01_pvp.obb`` all find the same entry.
        """
        if query in self._entries:
            return [self._entries[query]]
        needle = query.replace("\\", "/").lower()
        return [entry for entry in self._entries.values() if entry.name.lower().endswith(needle)]

    # -- reading -------------------------------------------------------------------

    def open(self, entry_name: str) -> IO[bytes]:
        """Open an entry as a seekable binary stream."""
        self.get(entry_name)  # raises PakError with a clear message when missing
        if self._zip is None:
            raise PakError(f"{self.path.name} is closed")
        return self._zip.open(entry_name, mode="r")

    def read(self, entry_name: str, offset: int = 0, size: int | None = None) -> bytes:
        """Read an entry, or ``size`` bytes of it starting at ``offset``.

        Seeking inside a stored entry is cheap; inside a deflated entry it
        decompresses from the start, so prefer whole reads for compressed entries.
        """
        if offset < 0:
            raise ValueError("offset must be non-negative")
        with self.open(entry_name) as stream:
            if offset:
                stream.seek(offset)
            return stream.read() if size is None else stream.read(size)


class PakSet:
    """The client and server halves of one pak name, searched together.

    Either half may be absent (for example ``shared_textures`` exists only on the
    client side).
    """

    def __init__(self, name: str, client: PakArchive | None, server: PakArchive | None) -> None:
        if client is None and server is None:
            raise PakError(f"Pak set {name!r} has neither a client nor a server archive")
        self.name = name
        self.client = client
        self.server = server

    def close(self) -> None:
        for archive in self.archives():
            archive.close()

    def __enter__(self) -> PakSet:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def archives(self, side: Side | None = None) -> list[PakArchive]:
        """The archives present, client first, optionally restricted to one side."""
        pairs = (("client", self.client), ("server", self.server))
        return [archive for archive_side, archive in pairs if archive is not None and side in (None, archive_side)]

    def archive(self, side: Side) -> PakArchive:
        archive = self.client if side == "client" else self.server
        if archive is None:
            raise PakError(f"{self.name} has no {side} pak")
        return archive

    def entries(self, side: Side | None = None) -> Iterator[PakEntry]:
        for archive in self.archives(side):
            yield from archive

    def find(self, query: str, side: Side | None = None) -> list[PakEntry]:
        """Matches from every archive in the set (see :meth:`PakArchive.find`)."""
        return [entry for archive in self.archives(side) for entry in archive.find(query)]

    def resolve(self, query: str, side: Side) -> PakEntry:
        """Exactly one entry on ``side`` matching ``query``, or a :class:`PakError` listing the candidates."""
        matches = self.find(query, side)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise PakError(f"No {side} entry in {self.name} matches {query!r}")
        preview = "\n  ".join(entry.name for entry in matches[:15])
        more = f"\n  ... and {len(matches) - 15} more" if len(matches) > 15 else ""
        raise PakError(
            f"{len(matches)} {side} entries in {self.name} match {query!r}; be more specific:\n  {preview}{more}"
        )

    def _archive_for(self, entry: PakEntry) -> PakArchive:
        if entry.side is None:
            raise PakError(f"Entry {entry.name!r} has no side; read it through its PakArchive")
        return self.archive(entry.side)

    def open(self, entry: PakEntry) -> IO[bytes]:
        return self._archive_for(entry).open(entry.name)

    def read(self, entry: PakEntry, offset: int = 0, size: int | None = None) -> bytes:
        return self._archive_for(entry).read(entry.name, offset, size)
