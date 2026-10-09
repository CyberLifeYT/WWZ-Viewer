from __future__ import annotations

from pathlib import Path

import pytest

from extractor.catalog import Catalog, CatalogError, classify
from extractor.pak_reader import PakArchive, PakError, PakSet

ENTRIES = {
    "[scene]/test_map.obb": b"OBB-CLIENT" * 10,
    "[scene]/tpl/crate.tpl/crate.tpl": b"crate template " * 50,
    "[scene]/tpl/barrel.tpl/barrel.tpl": b"barrel template " * 50,
}


@pytest.fixture
def client_pak(tmp_path: Path, pak_builder) -> Path:
    return pak_builder(tmp_path / "client" / "test_map.pak", ENTRIES)


def test_lists_entries_with_sizes_and_compression(client_pak: Path) -> None:
    with PakArchive(client_pak, side="client") as archive:
        assert len(archive) == 3
        obb = archive.get("[scene]/test_map.obb")
        assert obb.size == 100
        assert not obb.is_compressed
        assert obb.extension == ".obb"
        assert obb.basename == "test_map.obb"
        assert obb.side == "client"
        crate = archive.get("[scene]/tpl/crate.tpl/crate.tpl")
        assert crate.is_compressed
        assert crate.compressed_size < crate.size


def test_reads_whole_entries_and_windows(client_pak: Path) -> None:
    with PakArchive(client_pak) as archive:
        assert archive.read("[scene]/test_map.obb") == ENTRIES["[scene]/test_map.obb"]
        assert archive.read("[scene]/test_map.obb", offset=10, size=10) == b"OBB-CLIENT"
        assert archive.read("[scene]/tpl/crate.tpl/crate.tpl", offset=6, size=8) == b"template"


def test_find_matches_exact_suffix_case_and_backslashes(client_pak: Path) -> None:
    with PakArchive(client_pak) as archive:
        assert [entry.name for entry in archive.find("[scene]/test_map.obb")] == ["[scene]/test_map.obb"]
        assert len(archive.find(".tpl")) == 2
        assert [entry.basename for entry in archive.find("CRATE.TPL")] == ["crate.tpl"]
        assert [entry.basename for entry in archive.find("tpl\\barrel.tpl")] == ["barrel.tpl"]
        assert archive.find(".missing") == []


def test_missing_entry_raises(client_pak: Path) -> None:
    with PakArchive(client_pak) as archive, pytest.raises(PakError, match="not in"):
        archive.read("[scene]/nope.obb")


def test_zero_byte_stub_is_an_empty_archive(tmp_path: Path) -> None:
    stub = tmp_path / "shared.pak"
    stub.write_bytes(b"")
    with PakArchive(stub) as archive:
        assert len(archive) == 0
        assert archive.find(".tpl") == []


def test_non_zip_file_raises_pak_error(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.pak"
    bogus.write_bytes(b"not a zip at all" * 10)
    with pytest.raises(PakError, match="not a readable pak"):
        PakArchive(bogus)


def test_pak_set_routes_reads_to_the_entry_side(tmp_path: Path, pak_builder, client_pak: Path) -> None:
    server_pak = pak_builder(tmp_path / "server" / "test_map.pak", {"[scene]/test_map.obb": b"OBB-SERVER"})
    with PakSet("test_map", PakArchive(client_pak, "client"), PakArchive(server_pak, "server")) as paks:
        client_entry = paks.resolve(".obb", "client")
        server_entry = paks.resolve(".obb", "server")
        assert paks.read(client_entry, size=10) == b"OBB-CLIENT"
        assert paks.read(server_entry) == b"OBB-SERVER"
        assert len(paks.find(".obb")) == 2
        assert [entry.side for entry in paks.entries("server")] == ["server"]


def test_pak_set_resolve_errors_list_candidates(client_pak: Path) -> None:
    with PakSet("test_map", PakArchive(client_pak, "client"), None) as paks:
        with pytest.raises(PakError, match="2 client entries"):
            paks.resolve(".tpl", "client")
        with pytest.raises(PakError, match="No client entry"):
            paks.resolve(".nav", "client")
        with pytest.raises(PakError, match="no server pak"):
            paks.archive("server")


def test_classify_pak_names() -> None:
    assert classify("ny_01_pvp") == "scene"
    assert classify("hm_rome") == "scene"
    assert classify("main_menu") == "scene"
    assert classify("shared_tpl") == "shared"
    assert classify("boot") == "core"
    assert classify("initial") == "core"


def test_catalog_pairs_client_and_server(tmp_path: Path, pak_builder) -> None:
    content = tmp_path / "Content"
    pak_builder(content / "client" / "ny_01_pvp.pak", {"[scene]/ny_01_pvp.obb": b"c"})
    pak_builder(content / "server" / "ny_01_pvp.pak", {"[scene]/ny_01_pvp.obb": b"s"})
    pak_builder(content / "client" / "shared_textures.pak", {"a.pct": b"t"})
    (content / "server" / "boot.pak").write_bytes(b"")

    catalog = Catalog(content)
    assert [group.name for group in catalog.groups()] == ["boot", "ny_01_pvp", "shared_textures"]
    assert [group.name for group in catalog.scenes()] == ["ny_01_pvp"]

    textures = catalog.get("shared_textures")
    assert textures.kind == "shared" and textures.server_path is None and textures.server_size == 0

    with catalog.open("ny_01_pvp") as paks:
        assert paks.read(paks.resolve(".obb", "server")) == b"s"

    with pytest.raises(CatalogError, match="Did you mean: ny_01_pvp"):
        catalog.get("ny_01_pv")
