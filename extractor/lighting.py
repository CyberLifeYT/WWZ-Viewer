"""A level's baked lighting, read from the map's ``[scene]/lm/`` textures (read-only).

Every material names how it is lit (``lm.source``, ``lm.texName``, ``lm.uvSetIdx``):

* ``vertex`` (``<map>_vc``, 1848 x 1848 in ny_01_pvp): lighting per vertex. Each
  vertex's lightmap UV (the attribute word after the regular UV sets, u16 / 32768)
  points at the centre of a texel of its own, so the texture is really a list
  of vertex colours.
* ``tex`` (``<map>_tex``, 2048 x 2048): an ordinary lightmap atlas, sampled with
  the same UV set across each surface.
* ``none``: not lit by the bake (sky domes, far backdrops).

Each set has five textures:

* ``rnm0``, ``rnm1``, ``rnm2`` (BC6H): incoming light along three tangent-space
  basis directions ("radiosity normal mapping"). For a surface without bumps
  the three weigh equally.
* ``dircolor`` (BC6H) and ``dir`` (BC7): the colour of the strongest direct
  light and its world-space direction (rgb 0..255 for -1..1), on about 9% of
  vertices. On lit vertices the direction points out of the surface: for
  walls facing +z the mean direction is +z, for walls facing -x it is -x.

Light reaching an unbumped surface with normal ``n`` is then
``(rnm0 + rnm1 + rnm2) / 3 + dircolor * max(0, n . dir)``; with a normal map the
viewer weighs the three maps by the bumped normal's alignment with each basis
direction, as radiosity normal mapping does. The values are linear HDR
(1.0 is the bake's white; about a fifth of the texels are brighter, lamps reach
70 and more), so the viewer tone-maps it. The ``low_`` copies of each set are
lower-quality duplicates and are not used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from extractor.export.encoding import rgbm
from extractor.formats import pct
from extractor.pak_reader import PakError, PakSet
from extractor.ser.errors import SerFormatError

LIGHTMAP_FOLDER = "[scene]/lm"


@dataclass(frozen=True)
class LightSet:
    """One decoded set (``vc`` or ``tex``): float32 arrays of (height, width, 3) texels."""

    rnm: np.ndarray  # (3, height, width, 3): light along the three tangent-space basis directions
    direct_color: np.ndarray
    direct_direction: np.ndarray  # unit vectors, or zero where there is no direct light

    @property
    def indirect(self) -> np.ndarray:
        """Light on an unbumped surface: the three basis directions weigh equally."""
        return self.rnm.mean(axis=0)

    @property
    def size(self) -> tuple[int, int]:
        height, width = self.rnm.shape[1:3]
        return width, height

    def texels(self, lightmap_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row and column of the texel each UV points at."""
        width, height = self.size
        column = np.clip((lightmap_uv[:, 0] * width).astype(np.int64), 0, width - 1)
        row = np.clip((lightmap_uv[:, 1] * height).astype(np.int64), 0, height - 1)
        return row, column

    def light_at(self, lightmap_uv: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Light (linear RGB, 1.0 = the bake's white) reaching unbumped vertices with these lightmap UVs and normals."""
        row, column = self.texels(lightmap_uv)
        cosine = np.clip(np.sum(normals * self.direct_direction[row, column], axis=1), 0.0, None)
        return self.rnm[:, row, column].mean(axis=0) + self.direct_color[row, column] * cosine[:, None]

    def rnm_at(self, lightmap_uv: np.ndarray) -> np.ndarray:
        """(3, n, 3): each vertex's light along the three basis directions."""
        row, column = self.texels(lightmap_uv)
        return self.rnm[:, row, column]

    def direct_at(self, lightmap_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each vertex's direct light colour (n, 3) and world-space direction (n, 3, zero when none)."""
        row, column = self.texels(lightmap_uv)
        return self.direct_color[row, column], self.direct_direction[row, column]

    def textures(self) -> dict[str, np.ndarray]:
        """The set as uint8 images for the viewer: three RNM maps and the direct colour as RGBM,
        the direct direction as rgb 0..255 for -1..1."""
        out = {f"rnm{k}": rgbm(self.rnm[k]) for k in range(3)}
        out["direct"] = rgbm(self.direct_color)
        out["direct_dir"] = np.clip(np.round(self.direct_direction * 127.5 + 127.5), 0, 255).astype(np.uint8)
        return out


def _read(paks: PakSet, name: str, side: str) -> bytes:
    return paks.read(paks.resolve(f"{LIGHTMAP_FOLDER}/{name}.pct", side))


def _decode_hdr(paks: PakSet, name: str, side: str) -> np.ndarray:
    """A BC6H light texture as float32 RGB, full range."""
    return pct.decode_hdr(_read(paks, name, side))


def _decode_direction(paks: PakSet, name: str, side: str) -> np.ndarray:
    """The BC7 direction texture as float32 xyz in -1..1."""
    image, _, _ = pct.decode_mip(_read(paks, name, side), max_size=1 << 14)
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0


def load_set(paks: PakSet, base: str, side: str = "client") -> LightSet | None:
    """Decode the set ``base`` (e.g. ``ny_01_pvp_vc``), or None when the map does not ship it."""
    try:
        rnm = [_decode_hdr(paks, f"{base}_rnm{k}", side) for k in range(3)]
        direct_color = _decode_hdr(paks, f"{base}_dircolor", side)
        direction = _decode_direction(paks, f"{base}_dir", side)
    except (PakError, SerFormatError):
        return None
    length = np.linalg.norm(direction, axis=2, keepdims=True)
    lit = np.any(direct_color > 0, axis=2, keepdims=True) & (length > 0.5)
    direction = np.where(lit, direction / np.maximum(length, 1e-6), 0.0).astype(np.float32)
    return LightSet(np.stack(rnm), direct_color, direction)


@dataclass(frozen=True)
class BakedLighting:
    """The map's per-vertex set and lightmap set, keyed by their file names (``<map>_vc``, ``<map>_tex``).

    Materials name the set in ``lm.texName``: in full in 32 of the 44 maps, and
    without the map name (``_vc``, ``_tex``) in the other 12 (camp_02_pve,
    hm_base, horde_mode_02, jap_01/02, kam_01/03, mar_02, mos_02_td/03, veg_03).
    """

    sets: dict[str, LightSet]
    map_name: str = ""

    @classmethod
    def load(cls, paks: PakSet, map_name: str, side: str = "client") -> BakedLighting:
        sets = {}
        for suffix in ("vc", "tex"):
            found = load_set(paks, f"{map_name}_{suffix}", side)
            if found is not None:
                sets[f"{map_name}_{suffix}"] = found
        return cls(sets, map_name)

    def get(self, name: object) -> LightSet | None:
        """The set a material names, in full or without the map name."""
        if not isinstance(name, str) or not name:
            return None
        if name.startswith("_"):
            name = self.map_name + name
        return self.sets.get(name)
