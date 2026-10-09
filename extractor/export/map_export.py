"""Write decoded map data in the format the viewer loads.

Each exported map gets a folder ``<output>/maps/<map>/`` containing:

* ``map_data.json``: the manifest, described by ``schema/map_data.schema.json``.
* ``obb_boxes.bin``: little-endian float32, 15 values per box: origin xyz,
  the three axis rows (9 values), extents xyz.
* ``obb_groups.bin``: little-endian uint32, the group index of each box.

``<output>/maps/index.json`` (``schema/maps_index.schema.json``) lists every
exported map so the viewer can offer a picker.

Values are written exactly as the engine stores them: metres, +Y up, handedness
not yet verified. Rendering conventions belong to the viewer. The output has no
timestamps, so re-exporting unchanged data produces identical files.

Each manifest records the extractor ``build``: a digest of the extractor's own
source code. :func:`is_current` compares it, so a resumed batch export skips
only maps that the same code already exported and redoes the rest.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from extractor import __version__
from extractor.catalog import Catalog
from extractor.export.geometry_export import GEOMETRY_FORMAT, export_geometry, remove_geometry_files
from extractor.formats import obb
from extractor.pak_reader import Side
from extractor.textures import TextureLibrary

SCHEMA_ID = "wwz-mapviewer/map_data"
SCHEMA_VERSION = 1
INDEX_SCHEMA_ID = "wwz-mapviewer/maps_index"
INDEX_SCHEMA_VERSION = 1

MAPS_FOLDER = "maps"
MANIFEST_NAME = "map_data.json"
INDEX_NAME = "index.json"
BOXES_NAME = "obb_boxes.bin"
GROUPS_NAME = "obb_groups.bin"

BOX_LAYOUT = (
    "origin.x", "origin.y", "origin.z",
    "axis0.x", "axis0.y", "axis0.z",
    "axis1.x", "axis1.y", "axis1.z",
    "axis2.x", "axis2.y", "axis2.z",
    "extent0", "extent1", "extent2",
)  # fmt: skip

BOX_CONVENTION = (
    "A box covers origin + t0*extent0*axis0 + t1*extent1*axis1 + t2*extent2*axis2 for t in [0, 1]. "
    "axisN is row N of the stored 3x3 rotation, in world space."
)


BUILD_DIGEST_LENGTH = 16
_PACKAGE_DIR = Path(__file__).resolve().parents[1]


@functools.cache
def extractor_build() -> str:
    """Digest of the extractor's source files: changes whenever the code that writes exports changes."""
    digest = hashlib.sha256()
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        digest.update(path.relative_to(_PACKAGE_DIR).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()[:BUILD_DIGEST_LENGTH]


def _rounded(values: np.ndarray) -> list[float]:
    return [round(float(value), 3) for value in values]


BOUND_PERCENTILES = {"p1": 1, "p10": 10, "p50": 50, "p90": 90, "p99": 99}


def _bounds(points: np.ndarray) -> dict[str, list[float]]:
    """Extremes and percentiles per axis.

    Maps surround the playable area with kilometres of sparse scenery, so the
    extremes and even the 1st-99th percentile midpoint sit far from where the
    action is. The median and the 10th-90th percentile spread locate it reliably.
    """
    if len(points) == 0:
        return {name: [0.0, 0.0, 0.0] for name in ("min", "max", *BOUND_PERCENTILES)}
    bounds = {"min": _rounded(points.min(axis=0)), "max": _rounded(points.max(axis=0))}
    for name, percentile in BOUND_PERCENTILES.items():
        bounds[name] = _rounded(np.percentile(points, percentile, axis=0))
    return bounds


def build_manifest(map_name: str, side: Side, entry_name: str, boxes: obb.ObbSet) -> dict[str, object]:
    """The ``map_data.json`` contents for one map. Pure function: no I/O."""
    return {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "map": map_name,
        "generator": {"name": "wwz-mapviewer", "version": __version__, "build": extractor_build()},
        "source": {"pak": map_name, "side": side, "entry": entry_name},
        "coordinates": {"units": "metres", "up": "+y", "handedness": "unverified"},
        "layers": {
            "obb": {
                "group_count": boxes.group_count,
                "box_count": boxes.box_count,
                "boxes": {
                    "uri": BOXES_NAME,
                    "dtype": "float32",
                    "byte_order": "little",
                    "components": len(BOX_LAYOUT),
                    "layout": list(BOX_LAYOUT),
                },
                "groups": {"uri": GROUPS_NAME, "dtype": "uint32", "byte_order": "little", "components": 1},
                "convention": BOX_CONVENTION,
                "center_bounds": _bounds(boxes.centers()),
            }
        },
    }


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a temporary file so a viewer reading concurrently never sees a partial file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def map_folder(output_dir: Path, map_name: str) -> Path:
    return output_dir / MAPS_FOLDER / map_name


def write_map(output_dir: Path, manifest: dict[str, object], boxes: obb.ObbSet) -> Path:
    """Write one map's manifest and box buffers; returns the map folder."""
    folder = map_folder(output_dir, str(manifest["map"]))
    folder.mkdir(parents=True, exist_ok=True)

    packed = np.concatenate(
        [boxes.origins, boxes.axes.reshape(-1, 9), boxes.extents],
        axis=1,
    ).astype("<f4", copy=False)
    _write_atomic(folder / BOXES_NAME, packed.tobytes())
    _write_atomic(folder / GROUPS_NAME, boxes.group_ids().astype("<u4", copy=False).tobytes())
    # The manifest goes last: its presence marks the export as complete.
    _write_atomic(folder / MANIFEST_NAME, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return folder


def export_map(
    catalog: Catalog,
    map_name: str,
    output_dir: Path,
    side: Side = "client",
    textures: TextureLibrary | None = None,
) -> Path:
    """Decode a scene's boxes (and, when ``textures`` is given, its real geometry) and write them under ``output_dir``."""
    with catalog.open(map_name) as paks:
        entry = paks.resolve(f"{map_name}.obb", side)
        boxes = obb.decode_bytes(paks.read(entry))
    manifest = build_manifest(map_name, side, entry.name, boxes)
    folder = map_folder(output_dir, map_name)
    folder.mkdir(parents=True, exist_ok=True)
    if textures is not None:
        manifest["layers"]["geometry"] = export_geometry(catalog, map_name, folder, textures, side)
    else:
        # The new manifest has no geometry layer; drop the buffers an earlier export left behind.
        remove_geometry_files(folder)
    return write_map(output_dir, manifest, boxes)


def is_current(output_dir: Path, map_name: str, side: Side = "client", geometry: bool = True) -> bool:
    """True when ``map_name`` already has a complete export from this extractor build.

    The manifest is written last, so its presence marks a finished export. With
    ``geometry``, a boxes-only export does not count.
    """
    try:
        manifest = json.loads((map_folder(output_dir, map_name) / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA_ID:
        return False
    if manifest.get("generator", {}).get("build") != extractor_build():
        return False
    if manifest.get("source", {}).get("side") != side:
        return False
    layer = manifest.get("layers", {}).get("geometry")
    has_geometry = isinstance(layer, dict) and layer.get("format") == GEOMETRY_FORMAT
    return has_geometry or not geometry


def update_index(output_dir: Path) -> Path:
    """Rebuild ``maps/index.json`` from the manifests present on disk."""
    maps_dir = output_dir / MAPS_FOLDER
    maps_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for manifest_path in sorted(maps_dir.glob(f"*/{MANIFEST_NAME}")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA_ID:
            continue
        layer = manifest["layers"]["obb"]
        geometry = manifest["layers"].get("geometry")
        entries.append(
            {
                "map": manifest["map"],
                "path": f"{manifest_path.parent.name}/{MANIFEST_NAME}",
                "box_count": layer["box_count"],
                "group_count": layer["group_count"],
                "triangle_count": geometry["triangle_count"] if geometry else 0,
            }
        )
    index = {"schema": INDEX_SCHEMA_ID, "schema_version": INDEX_SCHEMA_VERSION, "maps": entries}
    index_path = maps_dir / INDEX_NAME
    _write_atomic(index_path, (json.dumps(index, indent=2) + "\n").encode("utf-8"))
    return index_path
