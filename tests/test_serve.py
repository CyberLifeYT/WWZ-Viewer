"""The dev server must serve only the viewer, exported maps and schemas, with module-safe MIME types."""

from __future__ import annotations

import functools
import http.server
import importlib.util
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from extractor.config import PROJECT_ROOT


def load_serve_module():
    spec = importlib.util.spec_from_file_location("serve", PROJECT_ROOT / "serve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def base_url(tmp_path: Path) -> Iterator[str]:
    (tmp_path / "viewer" / "js").mkdir(parents=True)
    (tmp_path / "viewer" / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (tmp_path / "viewer" / "js" / "main.js").write_text("export {};", encoding="utf-8")
    (tmp_path / "output" / "maps" / "m").mkdir(parents=True)
    (tmp_path / "output" / "maps" / "m" / "obb_boxes.bin").write_bytes(bytes(8))
    (tmp_path / "output" / "raw").mkdir(parents=True)
    (tmp_path / "output" / "raw" / "secret.bin").write_bytes(b"x")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "paths.toml").write_text("content_dir = 'x'", encoding="utf-8")

    serve = load_serve_module()
    handler = functools.partial(serve.ViewerRequestHandler, directory=str(tmp_path))
    handler.func.log_message = lambda *args: None  # keep test output quiet
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def fetch(url: str) -> tuple[int, str, dict[str, str]]:
    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(url) as response:
            return response.status, response.headers.get("Content-Type", ""), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), dict(error.headers)


def test_serves_viewer_files_with_module_safe_types(base_url: str) -> None:
    status, content_type, headers = fetch(f"{base_url}/viewer/js/main.js")
    assert status == 200
    assert content_type.startswith("text/javascript")
    assert headers["Cache-Control"] == "no-store"
    assert fetch(f"{base_url}/viewer/")[0] == 200
    assert fetch(f"{base_url}/output/maps/m/obb_boxes.bin")[1] == "application/octet-stream"


def test_root_redirects_to_viewer(base_url: str) -> None:
    status, _, headers = fetch(f"{base_url}/")
    assert status == 302
    assert headers["Location"] == "/viewer/"


@pytest.mark.parametrize(
    "path",
    ["/config/paths.toml", "/output/raw/secret.bin", "/viewer/../config/paths.toml", "/viewer/%2e%2e/config/paths.toml"],
)
def test_everything_else_is_refused(base_url: str, path: str) -> None:
    assert fetch(f"{base_url}{path}")[0] == 404


def test_a_second_server_cannot_share_the_port_and_recognises_the_first(tmp_path: Path) -> None:
    serve = load_serve_module()
    (tmp_path / "viewer").mkdir()
    (tmp_path / "viewer" / "index.html").write_bytes(b"<!doctype html><head>" + serve.VIEWER_TITLE + b"</head>")
    handler = functools.partial(serve.ViewerRequestHandler, directory=str(tmp_path))
    handler.func.log_message = lambda *args: None
    first = serve.ViewerServer(("127.0.0.1", 0), handler)
    port = first.server_address[1]
    thread = threading.Thread(target=first.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(OSError):
            serve.ViewerServer(("127.0.0.1", port), handler)
        assert serve._viewer_already_serving(f"http://127.0.0.1:{port}/viewer/")
        assert not serve._viewer_already_serving(f"http://127.0.0.1:{port}/viewer/js/missing.js")
    finally:
        first.shutdown()
        first.server_close()
