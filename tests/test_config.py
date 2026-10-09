from __future__ import annotations

from pathlib import Path

import pytest

from extractor.config import PROJECT_ROOT, ConfigError, load_paths


@pytest.fixture
def content_dir(tmp_path: Path) -> Path:
    content = tmp_path / "Content"
    (content / "client").mkdir(parents=True)
    return content


@pytest.fixture(autouse=True)
def clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WWZ_CONTENT_DIR", raising=False)
    monkeypatch.delenv("WWZ_OUTPUT_DIR", raising=False)


def write_config(path: Path, content: Path, output: str = "output") -> Path:
    path.write_text(f'content_dir = "{content.as_posix()}"\noutput_dir = "{output}"\n', encoding="utf-8")
    return path


def test_reads_config_file_and_resolves_relative_output(tmp_path: Path, content_dir: Path) -> None:
    paths = load_paths(config_file=write_config(tmp_path / "paths.toml", content_dir))
    assert paths.content_dir == content_dir.resolve()
    assert paths.output_dir == (PROJECT_ROOT / "output").resolve()


def test_environment_overrides_file_and_arguments_override_both(
    tmp_path: Path, content_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_config(tmp_path / "paths.toml", tmp_path / "does-not-exist")
    monkeypatch.setenv("WWZ_CONTENT_DIR", str(content_dir))
    monkeypatch.setenv("WWZ_OUTPUT_DIR", str(tmp_path / "from-env"))
    assert load_paths(config_file=config).output_dir == (tmp_path / "from-env").resolve()
    assert load_paths(output_dir=tmp_path / "from-arg", config_file=config).output_dir == (tmp_path / "from-arg").resolve()


def test_missing_configuration_is_explained(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="No game Content folder configured"):
        load_paths(config_file=tmp_path / "absent.toml")


def test_folder_without_client_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="has no client/ folder"):
        load_paths(content_dir=tmp_path, config_file=tmp_path / "absent.toml")


def test_output_inside_game_folder_is_rejected(tmp_path: Path, content_dir: Path) -> None:
    with pytest.raises(ConfigError, match="must not be inside the game folder"):
        load_paths(content_dir=content_dir, output_dir=content_dir / "out", config_file=tmp_path / "absent.toml")
