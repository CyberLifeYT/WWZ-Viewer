"""Build the website copy of the viewer (for GitHub Pages), and optionally publish it.

    .venv/Scripts/python build_site.py                          # the viewer, no maps
    .venv/Scripts/python build_site.py --maps ny_01_pvp jer_02_pvp
    .venv/Scripts/python build_site.py --maps ny_01_pvp --publish https://github.com/<user>/<repo>.git

serve.py runs the viewer on this PC: it decodes the game's textures when the
page asks for them, so it needs the game installed. A website has neither a
server nor the game (GitHub Pages only hands out files), so this script builds
a self-contained folder instead (``site/`` by default):

* ``viewer/`` and ``schema/`` as they are, with the page told that it runs as a
  website (``window.WWZ_SITE``): textures come from files, not from serve.py;
* ``output/maps/`` with the chosen maps (none unless ``--maps`` is given) and
  an index of them;
* ``textures/<name>.webp`` for every texture those maps use, decoded once
  from the game files at the size the viewer asks for (WebP: 3-6 times smaller
  than PNG, alpha kept exactly);
* ``index.html`` sending visitors to ``viewer/``, and ``.nojekyll`` so GitHub
  serves every file as it is.

The map buffers stay gzip files; GitHub Pages sends them as they are and the
viewer inflates them itself. GitHub refuses files over 100 MB and serves sites
up to 1 GB, which the build checks. ``--publish`` commits the folder to the
``gh-pages`` branch of the given repository (replacing what was there) and
pushes it with your own git login (the branch history is kept in ``site/.git``,
so an update uploads only what changed); turn on Pages for that branch once, in the
repository's Settings > Pages.

The maps and textures are made from the game's files, which belong to their
publisher: make sure you may publish them before adding maps.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "site"
DEFAULT_REPOSITORY = "https://github.com/CyberLifeYT/WWZ-Viewer"
TEXTURE_SIZE = 512  # what the viewer requests from serve.py (texture_cache.js)
TEXTURE_FORMAT = "webp"
TEXTURE_QUALITY = 82  # lossy colour; WebP keeps alpha lossless, so cut-outs stay exact
MAX_FILE_BYTES = 100 * 1024 * 1024  # GitHub rejects larger files
MAX_SITE_BYTES = 1024 * 1024 * 1024  # GitHub Pages' published size limit
PAGES_BRANCH = "gh-pages"
SITE_MARKER = "<!-- website build -->"


class SiteError(RuntimeError):
    """The site cannot be built or published as asked."""


def _texture_names(manifest: dict) -> set[str]:
    """Every texture a map's materials use: layers, normal maps, heightmaps, specular and emissive maps."""
    geometry = manifest["layers"].get("geometry") or {}
    names = set(geometry.get("textures", []))
    for material in geometry.get("materials", []):
        names.update(filter(None, (material.get("texture"), material.get("spec_map"), material.get("emissive_map"))))
        for layer in material.get("layers", []):
            names.update(filter(None, (layer.get("texture"), layer.get("normal_map"), layer.get("heightmap"))))
    return names


def _copy_viewer(out: Path, repository: str) -> None:
    shutil.copytree(ROOT / "viewer", out / "viewer")
    shutil.copytree(ROOT / "schema", out / "schema")
    page = out / "viewer" / "index.html"
    html = page.read_text(encoding="utf-8")
    settings = json.dumps({"textures": "../textures/", "textureFormat": TEXTURE_FORMAT, "repository": repository})
    marker = '<script type="module" src="js/main.js"></script>'
    if marker not in html:
        raise SiteError(f"{page} no longer loads js/main.js the way build_site.py expects")
    html = html.replace(marker, f"{SITE_MARKER}\n    <script>window.WWZ_SITE = {settings};</script>\n    {marker}", 1)
    page.write_text(html, encoding="utf-8")


def _copy_maps(out: Path, maps: list[str]) -> set[str]:
    """Copy the chosen maps' exports; returns the textures they need."""
    from extractor.export import map_export

    source = ROOT / "output" / map_export.MAPS_FOLDER
    textures: set[str] = set()
    for name in maps:
        folder = source / name
        manifest_path = folder / map_export.MANIFEST_NAME
        if not manifest_path.is_file():
            raise SiteError(f"{name} has not been exported (no {manifest_path}); run export_all_maps.bat first")
        shutil.copytree(folder, out / "output" / map_export.MAPS_FOLDER / name, ignore=shutil.ignore_patterns("*.tmp"))
        textures |= _texture_names(json.loads(manifest_path.read_text(encoding="utf-8")))
    map_export.update_index(out / "output")
    return textures


def _write_textures(out: Path, names: set[str]) -> int:
    """Decode every texture into ``textures/<name>.webp``; returns how many the game lacked."""
    if not names:
        return 0
    from extractor.catalog import Catalog
    from extractor.config import load_paths
    from extractor.textures import TextureLibrary

    folder = out / "textures"
    folder.mkdir(parents=True)
    missing = 0
    with TextureLibrary(Catalog(load_paths().content_dir)) as library:

        def write(name: str) -> bool:
            if name not in library:
                return False
            image = library.decode(name, max_size=TEXTURE_SIZE)
            image.save(folder / f"{name}.{TEXTURE_FORMAT}", "WEBP", quality=TEXTURE_QUALITY, method=4)
            return True

        with ThreadPoolExecutor(max_workers=8) as pool:
            for done, ok in enumerate(pool.map(write, sorted(names)), 1):
                missing += not ok
                if done % 200 == 0 or done == len(names):
                    print(f"  textures {done}/{len(names)}", flush=True)
    return missing


def _check_sizes(out: Path) -> int:
    files = [path for path in out.rglob("*") if path.is_file() and ".git" not in path.relative_to(out).parts]
    too_big = [path for path in files if path.stat().st_size > MAX_FILE_BYTES]
    if too_big:
        listed = ", ".join(f"{path.relative_to(out)} ({path.stat().st_size / 2**20:.0f} MB)" for path in too_big)
        raise SiteError(f"GitHub refuses files over 100 MB: {listed}")
    total = sum(path.stat().st_size for path in files)
    if total > MAX_SITE_BYTES:
        raise SiteError(f"the site is {total / 2**30:.2f} GB; GitHub Pages publishes at most 1 GB: choose fewer maps")
    return total


def build(out: Path, maps: list[str], repository: str) -> int:
    """Build the site into ``out`` (replacing an earlier build there); returns its size in bytes."""
    kept_history = None
    if out.exists():
        page = out / "viewer" / "index.html"
        if not page.is_file() or SITE_MARKER not in page.read_text(encoding="utf-8"):
            raise SiteError(f"{out} exists and is not an earlier website build; choose another --out")
        # Keep the published history (see publish), so an update uploads only what changed.
        if (out / ".git").is_dir():
            kept_history = out.parent / f".{out.name}-git-kept"
            if kept_history.exists():
                raise SiteError(f"{kept_history} is left over from an interrupted build; move it back to {out / '.git'}")
            (out / ".git").rename(kept_history)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if kept_history is not None:
        kept_history.rename(out / ".git")
    _copy_viewer(out, repository)
    textures = _copy_maps(out, maps)
    missing = _write_textures(out, textures)
    if missing:
        print(f"  {missing} textures the game does not ship are left out (the viewer shows them grey)")
    (out / "index.html").write_text(
        '<!doctype html>\n<meta charset="utf-8">\n<title>WWZ Map Viewer</title>\n'
        '<meta http-equiv="refresh" content="0; url=viewer/">\n<a href="viewer/">Open the WWZ Map Viewer</a>\n',
        encoding="utf-8",
    )
    (out / ".nojekyll").write_bytes(b"")
    return _check_sizes(out)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


DEFAULT_MESSAGE = "Website build of the WWZ Map Viewer"


def publish(out: Path, remote: str, message: str = DEFAULT_MESSAGE) -> None:
    """Commit the built site to ``gh-pages`` and push it, replacing the branch on the remote.

    The branch's history lives in ``out/.git`` and is kept from build to build, so
    an update commits on top of the last published site and the push uploads only
    the files that changed (a first publish uploads everything).
    """
    identity = []
    for key in ("user.name", "user.email"):
        value = subprocess.run(["git", "config", key], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        if value:
            identity += ["-c", f"{key}={value}"]
    if not (out / ".git").is_dir():
        _git("init", "--quiet", cwd=out)
        _git("checkout", "--quiet", "-b", PAGES_BRANCH, cwd=out)
    _git("add", "--all", cwd=out)
    unchanged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=out).returncode == 0
    if not unchanged:
        _git(*identity, "commit", "--quiet", "-m", message, cwd=out)
    _git("push", "--force", remote, f"{PAGES_BRANCH}:{PAGES_BRANCH}", cwd=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build (and publish) the website copy of the WWZ Map Viewer.")
    parser.add_argument("--maps", nargs="*", default=[], help="exported maps to include (default: none)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="output folder (default: site/)")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY, help="the tool's repository, linked from the page")
    parser.add_argument("--publish", metavar="REMOTE", help=f"push the site to this repository's {PAGES_BRANCH} branch")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="commit message for the published site")
    args = parser.parse_args(argv)
    try:
        size = build(args.out.resolve(), args.maps, args.repository)
        print(f"Built {args.out} ({size / 2**20:.0f} MB, {len(args.maps)} maps)")
        if args.publish:
            publish(args.out.resolve(), args.publish, args.message)
            print(f"Published to the {PAGES_BRANCH} branch of {args.publish}")
    except (SiteError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
