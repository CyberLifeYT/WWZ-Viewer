from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from extractor.catalog import Catalog
from extractor.config import PROJECT_ROOT
from extractor.export import map_export

IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
QUARTER_TURN_Y = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
SCHEMA_DIR = PROJECT_ROOT / "schema"


def check_schema(value: object, schema: dict, root: dict, where: str = "$") -> None:
    """Validate the JSON-Schema subset our schemas use: type, const, enum, required,
    properties, items, minItems, maxItems, minimum and local $ref."""
    if "$ref" in schema:
        name = schema["$ref"].removeprefix("#/$defs/")
        check_schema(value, root["$defs"][name], root, where)
        return
    if "const" in schema:
        assert value == schema["const"], f"{where}: {value!r} != {schema['const']!r}"
    if "enum" in schema:
        assert value in schema["enum"], f"{where}: {value!r} not in {schema['enum']}"
    expected_type = schema.get("type")
    python_types = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float)}
    if expected_type:
        assert isinstance(value, python_types[expected_type]) and not isinstance(value, bool), (
            f"{where}: expected {expected_type}, got {type(value).__name__}"
        )
    if "minimum" in schema:
        assert value >= schema["minimum"], f"{where}: {value} < {schema['minimum']}"
    if isinstance(value, dict):
        for key in schema.get("required", []):
            assert key in value, f"{where}: missing {key!r}"
        for key, child in schema.get("properties", {}).items():
            if key in value:
                check_schema(value[key], child, root, f"{where}.{key}")
    if isinstance(value, list):
        if "minItems" in schema:
            assert len(value) >= schema["minItems"], f"{where}: too few items"
        if "maxItems" in schema:
            assert len(value) <= schema["maxItems"], f"{where}: too many items"
        for index, item in enumerate(value):
            check_schema(item, schema.get("items", {}), root, f"{where}[{index}]")


def load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def obb_payload(groups: list[list[tuple]]) -> bytes:
    out = struct.pack("<I", len(groups))
    for boxes in groups:
        out += struct.pack("<I", len(boxes))
        for origin, axes, extents in boxes:
            out += struct.pack("<15f", *origin, *axes, *extents)
    return out


@pytest.fixture
def catalog(tmp_path: Path, pak_builder, ser_builder) -> Catalog:
    content = tmp_path / "Content"
    payload = obb_payload(
        [
            [((0, 0, 0), IDENTITY, (2, 4, 6))],
            [((10, 0, 0), QUARTER_TURN_Y, (1, 2, 3)), ((10, 2, 0), IDENTITY, (1, 1, 1))],
        ]
    )
    pak_builder(content / "client" / "test_map.pak", {"[scene]/test_map.obb": ser_builder("obb", payload=payload)})
    pak_builder(content / "client" / "shared_tpl.pak", {"x.tpl": b"x"})
    return Catalog(content)


def test_export_writes_manifest_and_buffers(catalog: Catalog, tmp_path: Path) -> None:
    output = tmp_path / "output"
    folder = map_export.export_map(catalog, "test_map", output)
    assert folder == output / "maps" / "test_map"

    manifest = json.loads((folder / "map_data.json").read_text(encoding="utf-8"))
    check_schema(manifest, load_schema("map_data.schema.json"), load_schema("map_data.schema.json"))
    layer = manifest["layers"]["obb"]
    assert layer["box_count"] == 3 and layer["group_count"] == 2
    assert manifest["source"]["entry"] == "[scene]/test_map.obb"
    # Box centres: (1,2,3); rows (0,0,1),(0,1,0),(-1,0,0) from (10,0,0) -> (8.5, 1, 0.5); (10.5, 2.5, 0.5).
    assert layer["center_bounds"]["min"] == [1.0, 1.0, 0.5]
    assert layer["center_bounds"]["max"] == [10.5, 2.5, 3.0]
    assert layer["center_bounds"]["p50"] == [8.5, 2.0, 0.5]

    boxes = np.frombuffer((folder / "obb_boxes.bin").read_bytes(), dtype="<f4").reshape(-1, 15)
    np.testing.assert_array_equal(boxes[1], [10, 0, 0, *QUARTER_TURN_Y, 1, 2, 3])
    groups = np.frombuffer((folder / "obb_groups.bin").read_bytes(), dtype="<u4")
    assert groups.tolist() == [0, 1, 1]
    assert not list(folder.glob("*.tmp"))


def test_export_is_deterministic(catalog: Catalog, tmp_path: Path) -> None:
    first = map_export.export_map(catalog, "test_map", tmp_path / "a")
    second = map_export.export_map(catalog, "test_map", tmp_path / "b")
    for name in ("map_data.json", "obb_boxes.bin", "obb_groups.bin"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_index_lists_exported_maps(catalog: Catalog, tmp_path: Path) -> None:
    output = tmp_path / "output"
    map_export.export_map(catalog, "test_map", output)
    (output / "maps" / "stray_folder").mkdir()  # folders without a manifest are skipped
    index = json.loads(map_export.update_index(output).read_text(encoding="utf-8"))
    check_schema(index, load_schema("maps_index.schema.json"), load_schema("maps_index.schema.json"))
    assert index["maps"] == [
        {"map": "test_map", "path": "test_map/map_data.json", "box_count": 3, "group_count": 2, "triangle_count": 0}
    ]


def exported_manifests() -> list[Path]:
    return sorted((PROJECT_ROOT / "output" / "maps").glob("*/map_data.json"))


@pytest.mark.parametrize("manifest", exported_manifests() or [None], ids=lambda path: path.parent.name if path else "none")
def test_local_exports_match_the_schema(manifest: Path | None) -> None:
    # Real exports need the game files, so this only checks what has been exported on this machine.
    if manifest is None:
        pytest.skip("no maps exported under output/maps")
    schema = load_schema("map_data.schema.json")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    check_schema(document, schema, schema)
    geometry = document["layers"].get("geometry")
    if geometry:
        assert len(geometry["groups"]) == len(geometry["materials"])
        assert sum(group["triangle_count"] for group in geometry["groups"]) == geometry["triangle_count"]
