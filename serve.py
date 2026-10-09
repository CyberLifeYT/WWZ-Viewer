"""Serve the viewer and the exported map data on this machine only.

    .venv/Scripts/python serve.py [--port 8765] [--open]

Then open http://127.0.0.1:8765/viewer/ (``/`` redirects there); ``--open``
does that in the default browser. If the port is taken by another copy of this
server, ``--open`` just opens the page it already serves.

Why not ``python -m http.server``: on Windows it takes MIME types from the
registry, which can label ``.js`` as ``text/plain``, and browsers refuse to run
that as a module. This server sets the types explicitly, disables caching so
re-exports show up on reload, and serves only the folders the viewer needs.

Textures come straight from the game install: ``/api/texture/<name>.png?size=N``
finds ``<name>.pct`` in the game's ``shared_textures`` pak (read-only), decodes
the largest mip that fits in ``N`` pixels, and caches the PNG under
``output/texture_cache/``.

Exported buffers are stored gzip-compressed (``*.bin.gz``); they are sent as
they are with ``Content-Encoding: gzip``, so the browser hands the viewer the
decompressed bytes.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import posixpath
import re
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIEWER_TITLE = b"<title>WWZ Map Viewer</title>"
ALLOWED_PREFIXES = ("/viewer/", "/output/maps/", "/schema/")
DEFAULT_PORT = 8765
TEXTURE_ROUTE = "/api/texture/"
TEXTURE_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")
TEXTURE_SIZES = (64, 128, 256, 512, 1024, 2048)


class TextureService:
    """Decodes game textures on demand; created lazily so the viewer works without a game install."""

    def __init__(self) -> None:
        self._library = None
        self._cache_dir: Path | None = None
        self._lock = threading.Lock()

    def _ensure(self):
        with self._lock:
            if self._library is None:
                from extractor.catalog import Catalog
                from extractor.config import load_paths
                from extractor.textures import TextureLibrary

                paths = load_paths()
                self._library = TextureLibrary(Catalog(paths.content_dir))
                self._cache_dir = paths.output_dir / "texture_cache"
                self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._library, self._cache_dir

    def png(self, name: str, size: int) -> bytes | None:
        """PNG bytes for texture ``name``, or None when the game has no such texture."""
        library, cache_dir = self._ensure()
        cached = cache_dir / f"{name.lower()}_{size}.png"
        if cached.is_file():
            return cached.read_bytes()
        if name not in library:
            return None
        data = library.png_bytes(name, size)
        temporary = cached.with_name(cached.name + f".{threading.get_ident()}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, cached)
        return data


TEXTURES = TextureService()


class ViewerRequestHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".mjs": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json",
        ".bin": "application/octet-stream",
        ".gz": "application/octet-stream",  # compressed buffers, sent with Content-Encoding: gzip
        ".svg": "image/svg+xml",
    }
    _status = 0

    def _route(self) -> bool:
        """Redirect ``/`` to the viewer and refuse anything outside the allowed folders.

        Returns True when the request may be served normally.
        """
        raw_path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
        # Decide the redirect on the raw path: normpath turns "/viewer/" into "/viewer".
        if raw_path in ("/", "/viewer"):
            self.send_response(302)
            self.send_header("Location", "/viewer/")
            self.end_headers()
            return False
        path = posixpath.normpath(raw_path)
        if not any(f"{path}/".startswith(prefix) for prefix in ALLOWED_PREFIXES):
            self.send_error(404, "Not served")
            return False
        return True

    def _serve_texture(self) -> None:
        url = urllib.parse.urlsplit(self.path)
        name = urllib.parse.unquote(url.path[len(TEXTURE_ROUTE):])
        if name.lower().endswith(".png"):
            name = name[:-4]
        if not TEXTURE_NAME.match(name):
            self.send_error(400, "Bad texture name")
            return
        query = urllib.parse.parse_qs(url.query)
        try:
            requested = int(query.get("size", ["512"])[0])
        except ValueError:
            requested = 512
        size = min(TEXTURE_SIZES, key=lambda s: (s < requested, abs(s - requested)))
        try:
            data = TEXTURES.png(name, size)
        except Exception as exc:  # decoding problems must not take the server down
            self.send_error(500, f"Could not decode {name}: {exc}")
            return
        if data is None:
            self.send_error(404, f"No texture named {name}")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        if self.path.startswith(TEXTURE_ROUTE):
            self._serve_texture()
        elif self._route():
            super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        if self._route():
            super().do_HEAD()

    def send_response(self, code: int, message: str | None = None) -> None:
        self._status = code
        super().send_response(code, message)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        if self._status == 200 and urllib.parse.urlsplit(self.path).path.endswith(".gz"):
            self.send_header("Content-Encoding", "gzip")
        super().end_headers()


class ViewerServer(http.server.ThreadingHTTPServer):
    # On Windows, SO_REUSEADDR lets a second server bind a port that is already
    # listening, so two copies would silently share it. Fail instead.
    allow_reuse_address = sys.platform != "win32"


def _viewer_already_serving(url: str) -> bool:
    """True when ``url`` answers with this project's viewer page (another copy of this server)."""
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return VIEWER_TITLE in response.read(4096)
    except (OSError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the WWZ map viewer on 127.0.0.1.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--open", action="store_true", help="open the viewer in the default browser once serving")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}/viewer/"
    handler = functools.partial(ViewerRequestHandler, directory=str(ROOT))
    try:
        server = ViewerServer(("127.0.0.1", args.port), handler)
    except OSError as exc:
        if _viewer_already_serving(url):
            print(f"The viewer is already being served at {url}", flush=True)
            if args.open:
                webbrowser.open(url)
            return 0
        print(f"error: cannot listen on port {args.port} ({exc}); try --port", flush=True)
        return 1
    with server:
        print(f"Serving {ROOT} at {url}  (Ctrl+C to stop)", flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
