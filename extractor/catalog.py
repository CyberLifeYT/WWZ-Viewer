"""Discover the game's paks and group them by name.

The install has two parallel folders, ``Content/client`` and ``Content/server``,
which mostly hold a pak of the same name each. Pak names fall into three groups:

* **scene**: one playable level or menu, e.g. ``ny_01_pvp``, ``rome_03_pve``,
  ``hm_jer`` (horde mode), ``main_menu``.
* **shared**: resources used by many scenes, e.g. ``shared_tpl`` (object
  templates) and ``shared_textures``.
* **core**: engine start-up data, ``boot`` and ``initial``.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from extractor.pak_reader import PakArchive, PakError, PakSet

PakKind = Literal["scene", "shared", "core"]

CORE_PAK_NAMES = frozenset({"boot", "initial"})
SHARED_PAK_PREFIX = "shared"


class CatalogError(PakError):
    """Raised when a requested pak name does not exist."""


def classify(name: str) -> PakKind:
    """Which group a pak name belongs to (see the module docstring)."""
    if name in CORE_PAK_NAMES:
        return "core"
    if name.startswith(SHARED_PAK_PREFIX):
        return "shared"
    return "scene"


@dataclass(frozen=True)
class PakGroupInfo:
    """Where the client and server paks of one name live on disk."""

    name: str
    kind: PakKind
    client_path: Path | None
    server_path: Path | None
    client_size: int
    server_size: int

    def open(self) -> PakSet:
        """Open both halves read-only. Close the result (or use ``with``) when done."""
        client = PakArchive(self.client_path, side="client") if self.client_path else None
        try:
            server = PakArchive(self.server_path, side="server") if self.server_path else None
        except PakError:
            if client is not None:
                client.close()
            raise
        return PakSet(self.name, client, server)


class Catalog:
    """Index of every pak under a Content folder. Building it only stats files."""

    def __init__(self, content_dir: str | Path) -> None:
        self.content_dir = Path(content_dir)
        client_dir = self.content_dir / "client"
        server_dir = self.content_dir / "server"
        if not client_dir.is_dir():
            raise CatalogError(f"{self.content_dir} has no client/ folder")

        client = {path.stem: path for path in client_dir.glob("*.pak")}
        server = {path.stem: path for path in server_dir.glob("*.pak")} if server_dir.is_dir() else {}

        self._groups: dict[str, PakGroupInfo] = {}
        for name in sorted(client.keys() | server.keys()):
            client_path = client.get(name)
            server_path = server.get(name)
            self._groups[name] = PakGroupInfo(
                name=name,
                kind=classify(name),
                client_path=client_path,
                server_path=server_path,
                client_size=client_path.stat().st_size if client_path else 0,
                server_size=server_path.stat().st_size if server_path else 0,
            )

    def groups(self, kind: PakKind | None = None) -> list[PakGroupInfo]:
        """All pak groups sorted by name, optionally filtered by kind."""
        return [group for group in self._groups.values() if kind is None or group.kind == kind]

    def scenes(self) -> list[PakGroupInfo]:
        return self.groups("scene")

    def get(self, name: str) -> PakGroupInfo:
        """Look up a pak group by exact name, suggesting close matches on a miss."""
        try:
            return self._groups[name]
        except KeyError:
            suggestions = difflib.get_close_matches(name, self._groups.keys(), n=5)
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise CatalogError(f"No pak named {name!r} in {self.content_dir}.{hint}") from None

    def open(self, name: str) -> PakSet:
        return self.get(name).open()
