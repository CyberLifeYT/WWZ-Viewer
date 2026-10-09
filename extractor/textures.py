"""Find and decode textures by name, straight from the game's paks (read-only).

Materials refer to textures by bare name (``bricks_01``, ``wall_plaster_09``).
The files live in ``shared_textures`` as ``[textures]/<folder>/<name>.pct``;
:class:`TextureLibrary` indexes those names once and then reads only the
header plus the single mip it needs from each texture.
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from pathlib import PurePosixPath

import numpy as np
from PIL import Image

from extractor.catalog import Catalog
from extractor.formats import pct
from extractor.pak_reader import PakArchive, PakError

TEXTURE_PAKS = ("shared_textures",)
# When several folders hold the same file name, prefer level art over characters and effects.
FOLDER_PREFERENCE = ("scenes", "objects", "detail", "grass", "plants", "vehicles", "water", "sky", "lighting")


SOLID_EROSION = 2
"""Texels the transparent area is shrunk by to tell solid regions from thin lines (at 256 px)."""


@dataclass(frozen=True)
class AlphaProfile:
    """The shape of a texture's alpha channel, which the game's files never label.

    * ``transparent``: share of texels with alpha below 128.
    * ``binary``: share with alpha near 0 or 255. Cut-out masks are nearly all
      0 or 255; alphas holding heights or gloss have smooth gradients.
    * ``solid``: share of the transparent area that survives shrinking it by
      :data:`SOLID_EROSION` texels. Cut-out shapes and unused atlas space are
      solid regions; alphas marking grout or panel seams (which must not be
      cut) are thin lines and vanish.
    """

    transparent: float
    binary: float
    solid: float

    @classmethod
    def of(cls, alpha: np.ndarray) -> AlphaProfile:
        clear = alpha < 128
        core = clear.copy()
        for _ in range(SOLID_EROSION):  # textures tile, so wrapping at the edges is right
            core &= np.roll(core, 1, 0) & np.roll(core, -1, 0) & np.roll(core, 1, 1) & np.roll(core, -1, 1)
        binary = float(((alpha < 24) | (alpha > 231)).mean())
        return cls(float(clear.mean()), binary, float(core.sum() / clear.sum()) if clear.any() else 0.0)


def _folder_rank(entry_name: str) -> int:
    parts = PurePosixPath(entry_name).parts
    folder = parts[1] if len(parts) > 2 else ""
    return FOLDER_PREFERENCE.index(folder) if folder in FOLDER_PREFERENCE else len(FOLDER_PREFERENCE)


class TextureLibrary:
    """Texture name -> decoded image, reading the game paks on demand. Thread-safe for concurrent reads."""

    def __init__(self, catalog: Catalog) -> None:
        self._archives: list[PakArchive] = []
        self._index: dict[str, tuple[PakArchive, str]] = {}
        self._lock = threading.Lock()
        for pak_name in TEXTURE_PAKS:
            group = catalog.get(pak_name)
            if group.client_path is None:
                continue
            archive = PakArchive(group.client_path, side="client")
            self._archives.append(archive)
            candidates: dict[str, list[str]] = {}
            for entry in archive:
                if entry.extension == ".pct":
                    candidates.setdefault(PurePosixPath(entry.name).stem.lower(), []).append(entry.name)
            for stem, names in candidates.items():
                self._index.setdefault(stem, (archive, min(names, key=lambda n: (_folder_rank(n), n))))

    def close(self) -> None:
        for archive in self._archives:
            archive.close()

    def __enter__(self) -> TextureLibrary:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._index

    def entry_name(self, name: str) -> str | None:
        found = self._index.get(name.lower())
        return found[1] if found else None

    def decode(self, name: str, max_size: int = 1024) -> Image.Image:
        """The largest mip of texture ``name`` that fits in ``max_size`` pixels, as RGBA."""
        found = self._index.get(name.lower())
        if found is None:
            raise PakError(f"no texture named {name!r}")
        archive, entry_name = found
        with self._lock:  # zipfile streams on one archive must not interleave
            header_bytes = archive.read(entry_name, 0, pct.FIXED_HEADER_SIZE)
            header = pct.parse_header(header_bytes)
            fmt = pct.KNOWN_CODES.get(header.format_code)
            if fmt is None:
                # Unknown codes need sample data to detect the format; read the whole (small) file once.
                data = archive.read(entry_name)
                image, _, _ = pct.decode_mip(data, max_size)
                return image
            offset, size, width, height = pct.choose_mip(pct.mip_offsets(header, fmt), max_size)
            payload = archive.read(entry_name, offset, size)
        if len(payload) < size:
            raise PakError(f"texture {name!r} is shorter than its mip table says")
        return pct.fit(pct.decode_payload(fmt, payload, width, height), max_size)

    def size(self, name: str) -> tuple[int, int] | None:
        """Width and height of texture ``name`` in pixels, from its header only."""
        found = self._index.get(name.lower())
        if found is None:
            return None
        archive, entry_name = found
        with self._lock:
            header = pct.parse_header(archive.read(entry_name, 0, pct.FIXED_HEADER_SIZE))
        return header.width, header.height

    def transparent_share(self, name: str) -> float | None:
        """Share of texels that are mostly transparent (alpha below 128), from a small mip."""
        profile = self.alpha_profile(name, max_size=64)
        return profile.transparent if profile else None

    def alpha_profile(self, name: str, max_size: int = 256) -> AlphaProfile | None:
        """How the texture's alpha channel is shaped, from a mip of ``max_size`` (see :class:`AlphaProfile`).

        Smaller mips blur fine masks (lattices) into gradients, so the default is not too small.
        """
        try:
            alpha = np.asarray(self.decode(name, max_size=max_size).convert("RGBA"))[..., 3]
        except (PakError, ValueError, OSError):
            return None
        return AlphaProfile.of(alpha)

    def average_color(self, name: str) -> tuple[int, int, int] | None:
        try:
            return pct.average_color(self.decode(name, max_size=32))
        except (PakError, ValueError, OSError):
            return None

    def png_bytes(self, name: str, max_size: int = 1024) -> bytes:
        buffer = io.BytesIO()
        self.decode(name, max_size).save(buffer, format="PNG", optimize=False)
        return buffer.getvalue()
