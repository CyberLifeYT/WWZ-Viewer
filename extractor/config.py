"""Locate the game's Content folder and the output folder.

Each value is resolved in this order, first hit wins:

1. An explicit argument (the CLI's ``--content-dir`` / ``--output-dir`` flags).
2. The environment variables ``WWZ_CONTENT_DIR`` / ``WWZ_OUTPUT_DIR``.
3. ``config/paths.toml`` at the project root.

The output folder falls back to ``<project root>/output`` when nothing sets it.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "paths.toml"
DEFAULT_OUTPUT_DIR = "output"


class ConfigError(RuntimeError):
    """Raised when the game folder cannot be located or is not a WWZ Content folder."""


@dataclass(frozen=True)
class Paths:
    """Resolved locations used by every command."""

    content_dir: Path
    """The game's ``Content`` folder, which holds ``client/`` and ``server/``."""

    output_dir: Path
    """Where dumps and exports are written. Never inside ``content_dir``."""


def _read_config_file(config_file: Path) -> dict[str, object]:
    if not config_file.is_file():
        return {}
    with config_file.open("rb") as handle:
        return tomllib.load(handle)


def load_paths(
    content_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    config_file: Path = CONFIG_FILE,
) -> Paths:
    """Resolve and validate the content and output folders.

    Raises:
        ConfigError: if no content folder is configured, if it lacks a ``client/``
            subfolder, or if the output folder would sit inside the game folder.
    """
    file_values = _read_config_file(config_file)

    content = content_dir or os.environ.get("WWZ_CONTENT_DIR") or file_values.get("content_dir")
    if not content:
        raise ConfigError(
            "No game Content folder configured. Copy config/paths.example.toml to "
            "config/paths.toml, set WWZ_CONTENT_DIR, or pass --content-dir."
        )
    content_path = Path(str(content)).expanduser().resolve()
    if not (content_path / "client").is_dir():
        raise ConfigError(
            f"{content_path} has no client/ folder. Point content_dir at the game's "
            "Content directory (the one containing client/ and server/)."
        )

    output = (
        output_dir
        or os.environ.get("WWZ_OUTPUT_DIR")
        or file_values.get("output_dir")
        or DEFAULT_OUTPUT_DIR
    )
    output_path = Path(str(output)).expanduser()
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path = output_path.resolve()

    # The game folder is managed by the Xbox app and may be repaired or replaced;
    # refusing to write inside it also guarantees we never touch the original paks.
    if output_path == content_path or content_path in output_path.parents:
        raise ConfigError(f"Output folder {output_path} must not be inside the game folder.")

    return Paths(content_dir=content_path, output_dir=output_path)
