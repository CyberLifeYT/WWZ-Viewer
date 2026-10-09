"""End-to-end runs of every CLI command against a small synthetic install."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from extractor.cli import main

IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@pytest.fixture
def install(tmp_path: Path, pak_builder, ser_builder) -> tuple[Path, Path]:
    content = tmp_path / "Content"
    obb_payload = struct.pack("<II15f", 1, 1, 0, 0, 0, *IDENTITY, 1, 2, 3)
    lg = ser_builder("lg", payload=b"SCN1" + bytes(64), resource_guid=bytes(range(16)))
    entries = {"[scene]/test_map.obb": ser_builder("obb", payload=obb_payload), "[scene]/test_map.lg": lg}
    pak_builder(content / "client" / "test_map.pak", entries)
    pak_builder(content / "server" / "test_map.pak", entries)
    (content / "server" / "shared.pak").write_bytes(b"")
    return content, tmp_path / "output"


def run(install: tuple[Path, Path], *args: str) -> int:
    content, output = install
    return main(["--content-dir", str(content), "--output-dir", str(output), *args])


def test_listing_commands(install, capsys) -> None:
    assert run(install, "maps", "--all") == 0
    assert run(install, "ls", "test_map", "--types") == 0
    assert run(install, "info", "test_map", ".lg") == 0
    printed = capsys.readouterr().out
    assert "test_map" in printed and "shared" in printed
    assert "lg " in printed  # the --types column
    assert "payload magic   SCN1" in printed


def test_inspection_commands(install, capsys) -> None:
    assert run(install, "hexdump", "test_map", ".obb", "--length", "16") == 0
    assert run(install, "probe", "test_map", ".obb", "--save") == 0
    assert run(install, "obb", "test_map") == 0
    printed = capsys.readouterr().out
    assert "1SERobb" in printed
    assert "Probe report" in printed
    assert "1 groups, 1 boxes" in printed
    assert list((install[1] / "probe").rglob("*.txt"))


def test_dump_and_export(install) -> None:
    assert run(install, "dump", "test_map", ".obb", "--side", "server") == 0
    assert list((install[1] / "raw" / "test_map" / "server").rglob("test_map.obb"))
    assert run(install, "export", "--all") == 0
    index = json.loads((install[1] / "maps" / "index.json").read_text(encoding="utf-8"))
    assert [entry["map"] for entry in index["maps"]] == ["test_map"]


def test_a_failing_map_is_logged_keeps_its_boxes_and_fails_the_run(install, pak_builder, capsys) -> None:
    content, output = install
    # With textures present the real geometry is attempted; this map's .lg has none, so that part fails.
    pak_builder(content / "client" / "shared_textures.pak", {"[textures]/objects/x.pct": b"x"})
    assert run(install, "export", "test_map", "--log") == 1
    printed = capsys.readouterr().out
    assert "FAILED" in printed and "its boxes were written" in printed and "1 failed: test_map" in printed
    manifest = json.loads((output / "maps" / "test_map" / "map_data.json").read_text(encoding="utf-8"))
    assert "geometry" not in manifest["layers"] and manifest["layers"]["obb"]["box_count"] == 1
    log = (output / "export.log").read_text(encoding="utf-8")
    assert "Traceback" in log and "1 failed: test_map" in log


def test_resume_skips_maps_this_build_already_exported(install, capsys, monkeypatch) -> None:
    from extractor.export import map_export

    assert run(install, "export", "--all", "--boxes-only", "--resume") == 0
    assert "[1/1] test_map" in capsys.readouterr().out
    assert run(install, "export", "--all", "--boxes-only", "--resume") == 0
    assert "Skipping 1 map already exported" in capsys.readouterr().out
    assert map_export.is_current(install[1], "test_map", geometry=False)
    assert not map_export.is_current(install[1], "test_map", geometry=True)  # boxes only
    assert not map_export.is_current(install[1], "test_map", side="server", geometry=False)
    monkeypatch.setattr(map_export, "extractor_build", lambda: "a newer build")
    assert not map_export.is_current(install[1], "test_map", geometry=False)
    assert run(install, "export", "--all", "--boxes-only", "--resume") == 0
    assert "[1/1] test_map" in capsys.readouterr().out


def test_verify_reports_stub_and_parses_headers(install, capsys) -> None:
    assert run(install, "verify", "--headers") == 0
    printed = capsys.readouterr().out
    assert "empty (0-byte stub)" in printed
    assert "4 1SER headers parsed" in printed


def test_errors_return_nonzero(install, capsys) -> None:
    assert run(install, "info", "test_map", ".missing") == 1
    assert run(install, "export") == 2
    assert run(install, "export", "shared") == 1
    errors = capsys.readouterr().err
    assert "No client entry" in errors
    assert "not a scene" in errors
