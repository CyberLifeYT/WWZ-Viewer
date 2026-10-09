"""The website build: a self-contained copy of the viewer for static hosting (GitHub Pages)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from extractor.config import PROJECT_ROOT


def load_build_site():
    spec = importlib.util.spec_from_file_location("build_site", PROJECT_ROOT / "build_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_site_without_maps_is_the_viewer_told_it_runs_as_a_website(tmp_path: Path) -> None:
    site = load_build_site()
    out = tmp_path / "site"
    size = site.build(out, [], "https://example.invalid/repo")
    page = (out / "viewer" / "index.html").read_text(encoding="utf-8")
    settings = json.loads(page.split("window.WWZ_SITE = ", 1)[1].split(";</script>", 1)[0])
    assert settings == {"textures": "../textures/", "textureFormat": "webp", "repository": "https://example.invalid/repo"}
    assert page.index("window.WWZ_SITE") < page.index('src="js/main.js"')  # set before the app starts
    assert (out / "viewer" / "js" / "main.js").is_file() and (out / "schema").is_dir()
    index = json.loads((out / "output" / "maps" / "index.json").read_text(encoding="utf-8"))
    assert index["maps"] == []
    assert 'url=viewer/' in (out / "index.html").read_text(encoding="utf-8")
    assert (out / ".nojekyll").is_file() and not (out / "textures").exists()
    assert size > 0
    # A second build replaces the first; a folder that is not a build is never deleted.
    site.build(out, [], "https://example.invalid/repo")
    other = tmp_path / "mine"
    (other / "viewer").mkdir(parents=True)
    (other / "viewer" / "index.html").write_text("<!doctype html>", encoding="utf-8")
    with pytest.raises(site.SiteError, match="not an earlier website build"):
        site.build(other, [], "x")
    assert (other / "viewer" / "index.html").is_file()


def test_a_rebuild_keeps_the_published_history(tmp_path: Path) -> None:
    # The gh-pages history lives in site/.git; keeping it means an update uploads only what changed.
    site = load_build_site()
    out = tmp_path / "site"
    site.build(out, [], "x")
    (out / ".git").mkdir()
    (out / ".git" / "HEAD").write_text("ref: refs/heads/gh-pages\n", encoding="utf-8")
    (out / "stale.txt").write_text("from an older build", encoding="utf-8")
    site.build(out, [], "x")
    assert (out / ".git" / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/gh-pages\n"
    assert not (out / "stale.txt").exists()
    assert not list(tmp_path.glob(".site-git-kept*"))


def test_every_texture_a_map_uses_is_collected() -> None:
    site = load_build_site()
    manifest = {
        "layers": {
            "geometry": {
                "textures": ["a"],
                "materials": [
                    {
                        "texture": "b", "spec_map": "b_spec", "emissive_map": None,
                        "layers": [{"texture": "b", "normal_map": "b_nm", "heightmap": None}, {"texture": "dirt", "heightmap": "b_hdetm"}],
                    }
                ],
            }
        }
    }
    assert site._texture_names(manifest) == {"a", "b", "b_spec", "b_nm", "dirt", "b_hdetm"}
    assert site._texture_names({"layers": {"obb": {}}}) == set()  # boxes-only export


def test_github_limits_are_checked(tmp_path: Path, monkeypatch) -> None:
    site = load_build_site()
    (tmp_path / "big.bin").write_bytes(bytes(2048))
    monkeypatch.setattr(site, "MAX_FILE_BYTES", 1024)
    with pytest.raises(site.SiteError, match="over 100 MB"):
        site._check_sizes(tmp_path)
    monkeypatch.setattr(site, "MAX_FILE_BYTES", 4096)
    monkeypatch.setattr(site, "MAX_SITE_BYTES", 1024)
    with pytest.raises(site.SiteError, match="at most 1 GB"):
        site._check_sizes(tmp_path)


def test_unexported_maps_are_refused(tmp_path: Path) -> None:
    site = load_build_site()
    with pytest.raises(site.SiteError, match="has not been exported"):
        site.build(tmp_path / "site", ["no_such_map"], "x")
