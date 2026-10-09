"""Reflection probes: ``[scene]/lc/lc_prebuild_<map>_<lighting group>__<probe name>_cube.pct``.

Each is a BC6H (HDR) cubemap, 128 x 128 per face, stored face by face (+x, -x,
+y, -y, +z, -z, each with its own mip chain). The names vary by map
(``reflection12``, ``reflection_12``, ``subway_reflection3``, ``outdoor_main_cm_2``,
``global_reflection_probe``, ...), so every cubemap in the map's ``lc``
folder counts. The game hands objects the cubemap of a nearby probe; where
the probes stand is not stored in any file the extractor reads (objects name
their probe only now and then, through a ``reflection_cubemap_name`` tag), so
the viewer gets one representative probe per map: the one that sees the most
sky above it (the brightest upward face), an open outdoor view rather than an
interior. (Some maps have a "global" probe, but several of those are dark.)
"""

from __future__ import annotations


import numpy as np

from extractor.formats import bc6h, pct
from extractor.pak_reader import PakEntry, PakError, PakSet
from extractor.ser.errors import SerFormatError

FACES = 6
ENVIRONMENT_WIDTH = 256  # equirectangular width; height is half
UP_FACE = 2
LC_FOLDER = "[scene]/lc"
_LUMA = np.array([0.2126, 0.7152, 0.0722])


def decode_cubemap(blob: bytes, faces: tuple[int, ...] = tuple(range(FACES))) -> np.ndarray:
    """(len(faces), size, size, 3) float32 HDR faces (mip 0) of a ``.pct`` cubemap."""
    header = pct.parse_header(blob)
    if header.faces != FACES or pct.KNOWN_CODES.get(header.format_code) != "BC6H":
        raise SerFormatError("not a BC6H cubemap", offset=0)
    mips = pct.mip_offsets(header, "BC6H")
    per_face = sum(size for _, size, _, _ in mips)
    first, size, width, height = mips[0]
    decoded = []
    for face in faces:
        start = first + face * per_face
        if start + size > len(blob):
            raise SerFormatError("cubemap face runs past the end of the file", offset=start)
        decoded.append(bc6h.decode(blob[start:start + size], width, height))
    return np.stack(decoded)


def _face_lookup(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direct3D cube addressing: face index and (u, v) in 0..1 (v down) for unit directions."""
    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    face = np.where((ax >= ay) & (ax >= az), np.where(x > 0, 0, 1), np.where(ay >= az, np.where(y > 0, 2, 3), np.where(z > 0, 4, 5)))
    major = np.choose(face, [ax, ax, ay, ay, az, az])
    sc = np.choose(face, [-z, z, x, x, x, -x])
    tc = np.choose(face, [-y, -y, z, -z, -y, -y])
    return face, (sc / major + 1.0) * 0.5, (tc / major + 1.0) * 0.5


def to_equirect(faces: np.ndarray, width: int = ENVIRONMENT_WIDTH) -> np.ndarray:
    """(width / 2, width, 3) equirectangular image of a cubemap; row 0 looks straight down.

    Columns follow three.js's convention: ``u = atan2(dir.z, dir.x) / 2pi + 0.5``,
    ``v = asin(dir.y) / pi + 0.5``.
    """
    height = width // 2
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    phi = (u - 0.5) * 2.0 * np.pi
    theta = (v - 0.5) * np.pi
    cos_t = np.cos(theta)[:, None]
    directions = np.stack(
        [np.cos(phi)[None, :] * cos_t, np.broadcast_to(np.sin(theta)[:, None], (height, width)), np.sin(phi)[None, :] * cos_t],
        axis=-1,
    )
    face, fu, fv = _face_lookup(directions)
    size = faces.shape[1]
    column = np.clip((fu * size).astype(np.int64), 0, size - 1)
    row = np.clip((fv * size).astype(np.int64), 0, size - 1)
    return faces[face, row, column].astype(np.float32)


def probe_entries(paks: PakSet, map_name: str, side: str = "client") -> list[PakEntry]:
    """The map's reflection cubemaps (every ``_cube.pct`` in its ``lc`` folder), by name."""
    prefix = f"{LC_FOLDER}/lc_prebuild_{map_name}_"
    return sorted((entry for entry in paks.find("_cube.pct", side) if entry.name.startswith(prefix)), key=lambda e: e.name)


def representative_environment(paks: PakSet, map_name: str, side: str = "client") -> np.ndarray | None:
    """The map's most open-sky reflection probe as an equirectangular HDR image, or None without probes."""
    best: tuple[float, bytes] | None = None
    for entry in probe_entries(paks, map_name, side):
        try:
            blob = paks.read(entry)
            sky = float(np.median(decode_cubemap(blob, (UP_FACE,))[0] @ _LUMA))
        except (PakError, SerFormatError, ValueError):
            continue
        if best is None or sky > best[0]:
            best = (sky, blob)
    return None if best is None else to_equirect(decode_cubemap(best[1]))
