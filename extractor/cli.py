"""Command-line entry point: ``wwz-extract`` or ``python -m extractor``.

Commands:

* ``maps``     list paks (scenes by default, ``--all`` for shared and core too)
* ``ls``       list the entries inside a pak, optionally with their 1SER type
* ``info``     show a resource's 1SER header and dependencies
* ``hexdump``  hex + ASCII dump of part of an entry
* ``probe``    full reverse-engineering report for an entry (see ``extractor.ser.probe``)
* ``dump``     copy an entry's raw bytes to the output folder, for use in a hex editor
* ``obb``      decode a scene's oriented bounding boxes and summarise them
* ``export``   write scenes in the viewer's format (``output/maps/``); a map
  that fails is reported and the batch goes on (``--resume`` skips finished maps)
* ``verify``   open every pak, and optionally parse every 1SER header, to check the reader

Entries can be named in full (``[scene]/ny_01_pvp.obb``) or by any unique
suffix (``.obb``, ``ny_01_pvp.obb``). Entry commands read the client pak unless
``--side server`` is given.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
import traceback
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import numpy as np

from extractor.catalog import Catalog, PakGroupInfo
from extractor.config import ConfigError, Paths, load_paths
from extractor.export import map_export
from extractor.formats import obb
from extractor.pak_reader import SIDES, PakEntry, PakError, PakSet
from extractor.ser import SerFormatError, is_ser, load, parse_header
from extractor.ser.container import SER_MAGIC, TYPE_NAME_SIZE
from extractor.ser.probe import build_report, hexdump
from extractor.textures import TextureLibrary

EXPORT_LOG_NAME = "export.log"
"""``export --log`` appends to this file in the output folder."""

HEADER_READ_SIZE = 1 << 16
"""Bytes read when only a 1SER header is needed; covers every dependency block seen so far."""

_WINDOWS_UNSAFE = str.maketrans({character: "_" for character in '<>:"|?*'})


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _ser_type(paks: PakSet, entry: PakEntry) -> str:
    """The 1SER type tag of an entry, ``(empty)`` for an empty tag, or ``-`` when it is not 1SER."""
    head = paks.read(entry, size=len(SER_MAGIC) + TYPE_NAME_SIZE)
    if not is_ser(head):
        return "-"
    type_name = head[len(SER_MAGIC) :].split(b"\x00", 1)[0].decode("ascii", errors="replace")
    return type_name or "(empty)"


def _read_header_bytes(paks: PakSet, entry: PakEntry) -> bytes:
    """Enough of an entry to parse its 1SER header, falling back to the whole entry if needed."""
    head = paks.read(entry, size=HEADER_READ_SIZE)
    if entry.size <= HEADER_READ_SIZE or not is_ser(head):
        return head
    try:
        parse_header(head)
    except SerFormatError:
        return paks.read(entry)
    return head


def _safe_relative_path(entry_name: str) -> Path:
    """An entry name turned into a relative path that is valid on Windows and cannot escape its folder."""
    parts = [part.translate(_WINDOWS_UNSAFE) for part in PurePosixPath(entry_name).parts if part not in ("", ".", "..")]
    if not parts:
        raise PakError(f"Entry name {entry_name!r} has no usable path components")
    return Path(*parts)


# -- commands ----------------------------------------------------------------------


def cmd_maps(args: argparse.Namespace, paths: Paths) -> int:
    catalog = Catalog(paths.content_dir)
    groups = catalog.groups() if args.all else catalog.scenes()
    print(f"{'name':<18} {'kind':<7} {'client':>10} {'server':>10}")
    for group in groups:
        client = _human_size(group.client_size) if group.client_path else "-"
        server = _human_size(group.server_size) if group.server_path else "-"
        print(f"{group.name:<18} {group.kind:<7} {client:>10} {server:>10}")
    print(f"\n{len(groups)} pak groups in {paths.content_dir}")
    return 0


def cmd_ls(args: argparse.Namespace, paths: Paths) -> int:
    with Catalog(paths.content_dir).open(args.pak) as paks:
        shown = 0
        for entry in paks.entries(args.side):
            if args.ext and entry.extension != args.ext.lower():
                continue
            if args.grep and args.grep.lower() not in entry.name.lower():
                continue
            type_column = f"{_ser_type(paks, entry):<14} " if args.types else ""
            packed = "deflate" if entry.is_compressed else "stored"
            print(f"{entry.side:<6} {entry.size:>12,} {packed:<7} {type_column}{entry.name}")
            shown += 1
            if args.limit and shown >= args.limit:
                break
        print(f"\n{shown} entries shown")
    return 0


def cmd_info(args: argparse.Namespace, paths: Paths) -> int:
    with Catalog(paths.content_dir).open(args.pak) as paks:
        entry = paks.resolve(args.entry, args.side)
        data = _read_header_bytes(paks, entry)
        print(f"{entry.side} {entry.name}  ({entry.size:,} bytes, {'deflate' if entry.is_compressed else 'stored'})")
        if not is_ser(data):
            print(f"Not a 1SER resource. First bytes: {data[:16].hex(' ')}")
            return 0
        header = parse_header(data)
        print(f"  type            {header.type_name or '(empty)'}")
        print(f"  own GUID        {header.resource_guid.hex() if header.resource_guid else '(none)'}")
        print(f"  version         {header.version}")
        print(f"  payload offset  {header.payload_offset:#x}  ({entry.size - header.payload_offset:,} payload bytes)")
        print(f"  dependencies    {len(header.dependencies)}")
        for dependency in header.dependencies:
            guid = dependency.guid_hex if dependency.has_guid else "-" * 32
            print(f"    flags={dependency.flags:#010x}  guid={guid}  {dependency.path}")
        magic = load(data).payload_magic
        if magic:
            print(f"  payload magic   {magic}")
    return 0


def cmd_hexdump(args: argparse.Namespace, paths: Paths) -> int:
    with Catalog(paths.content_dir).open(args.pak) as paks:
        entry = paks.resolve(args.entry, args.side)
        data = paks.read(entry, offset=args.offset, size=args.length)
        print(hexdump(data, length=len(data), base_offset=args.offset))
    return 0


def cmd_probe(args: argparse.Namespace, paths: Paths) -> int:
    with Catalog(paths.content_dir).open(args.pak) as paks:
        entry = paks.resolve(args.entry, args.side)
        data = paks.read(entry, offset=args.offset, size=args.length)
    title = f"{args.pak} / {entry.side} / {entry.name}"
    report = build_report(data, title, base_offset=args.offset)
    print(report)
    if args.save:
        target = paths.output_dir / "probe" / args.pak / entry.side / _safe_relative_path(entry.name + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
        print(f"Saved to {target}")
    return 0


def cmd_dump(args: argparse.Namespace, paths: Paths) -> int:
    with Catalog(paths.content_dir).open(args.pak) as paks:
        entry = paks.resolve(args.entry, args.side)
        target = paths.output_dir / "raw" / args.pak / entry.side / _safe_relative_path(entry.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with paks.open(entry) as source, target.open("wb") as destination:
            while chunk := source.read(1 << 20):
                destination.write(chunk)
    print(f"Wrote {entry.size:,} bytes to {target}")
    return 0


def cmd_obb(args: argparse.Namespace, paths: Paths) -> int:
    with Catalog(paths.content_dir).open(args.pak) as paks:
        entry = paks.resolve(".obb", args.side)
        boxes = obb.decode_bytes(paks.read(entry))
    print(f"{entry.side} {entry.name}: {boxes.group_count:,} groups, {boxes.box_count:,} boxes")
    if boxes.box_count == 0:
        return 0
    sizes = np.bincount(boxes.group_sizes())
    print("  boxes per group: " + "  ".join(f"{size}:{count}" for size, count in enumerate(sizes) if count))
    centers = boxes.centers()
    low, high = centers.min(axis=0), centers.max(axis=0)
    p1, p99 = np.percentile(centers, 1, axis=0), np.percentile(centers, 99, axis=0)
    for axis, name in enumerate("xyz"):
        print(
            f"  center.{name}  min {low[axis]:10.1f}  p1 {p1[axis]:10.1f}  "
            f"p99 {p99[axis]:10.1f}  max {high[axis]:10.1f}"
        )
    median = np.median(boxes.extents, axis=0)
    flat = float(np.mean((boxes.extents == 0).any(axis=1)))
    median_text = " / ".join(f"{value:.3f}" for value in median)
    print(f"  extents median {median_text}  max {boxes.extents.max():.1f}  flat (one zero axis) {flat:.1%}")
    return 0


@contextlib.contextmanager
def _keep_awake() -> Iterator[None]:
    """Ask Windows not to sleep while a long export runs.

    The request belongs to this process only and ends with it; no power
    setting is changed. A no-op on other systems.
    """
    if sys.platform != "win32":
        yield
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    set_state(es_continuous | es_system_required)
    try:
        yield
    finally:
        set_state(es_continuous)


def _maps(count: int) -> str:
    return f"{count} map" if count == 1 else f"{count} maps"


def _duration(seconds: float) -> str:
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


class _ExportLog:
    """Progress lines on the console and, optionally, appended to a log file with full tracebacks."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8")

    def line(self, text: str, detail: str = "") -> None:
        print(text, flush=True)
        if self._file is not None:
            self._file.write(text + "\n" + (detail.rstrip("\n") + "\n" if detail else ""))
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def _export_one(
    catalog: Catalog,
    name: str,
    args: argparse.Namespace,
    paths: Paths,
    textures: TextureLibrary | None,
    log: _ExportLog,
    prefix: str,
) -> int | None:
    """Export one map; returns the bytes written, or None when it failed.

    A failure is logged with its traceback and does not stop a batch. When the
    real geometry is what failed, the map's boxes are still written so it opens
    in the viewer.
    """
    try:
        folder = map_export.export_map(catalog, name, paths.output_dir, side=args.side, textures=textures)
    except Exception as exc:  # one broken map must not end a long batch; the traceback goes to the log
        log.line(f"{prefix} FAILED  {type(exc).__name__}: {exc}", traceback.format_exc())
        if textures is not None:
            try:
                map_export.export_map(catalog, name, paths.output_dir, side=args.side, textures=None)
                log.line(f"{' ' * len(prefix)} its boxes were written, so it still opens (without real geometry)")
            except Exception as boxes_exc:  # same reasoning as above
                log.line(f"{' ' * len(prefix)} its boxes failed too: {type(boxes_exc).__name__}: {boxes_exc}", traceback.format_exc())
        return None
    return sum(path.stat().st_size for path in folder.iterdir() if path.is_file())


def cmd_export(args: argparse.Namespace, paths: Paths) -> int:
    catalog = Catalog(paths.content_dir)
    if args.all:
        names = [group.name for group in catalog.scenes()]
    elif args.maps:
        names = args.maps
    else:
        print("error: name one or more maps, or pass --all", file=sys.stderr)
        return 2
    for name in names:
        group = catalog.get(name)
        if group.kind != "scene":
            raise PakError(f"{name} is a {group.kind} pak, not a scene")
    has_textures = "shared_textures" in {group.name for group in catalog.groups()}
    if not args.boxes_only and not has_textures:
        print("warning: the shared_textures pak is missing, so only boxes are exported", file=sys.stderr)
    geometry = not args.boxes_only and has_textures
    pending = [name for name in names if not (args.resume and map_export.is_current(paths.output_dir, name, args.side, geometry))]
    log = _ExportLog(paths.output_dir / EXPORT_LOG_NAME if args.log else None)
    if log.path is not None:
        log.line(f"\n=== export {' '.join(names) if not args.all else '--all'} ({len(pending)} to do, build {map_export.extractor_build()}) ===")
    if len(pending) < len(names):
        log.line(f"Skipping {_maps(len(names) - len(pending))} already exported by this build (--resume).")

    textures = TextureLibrary(catalog) if geometry and pending else None
    total = 0
    failures: list[str] = []
    started = time.monotonic()
    try:
        with _keep_awake():
            for number, name in enumerate(pending, 1):
                prefix = f"  [{number:>{len(str(len(pending)))}}/{len(pending)}] {name:<16}"
                map_started = time.monotonic()
                if sys.stdout.isatty():
                    # Shows which map is being worked on; the result line below overwrites it.
                    print(f"{prefix} exporting...", end="\r", flush=True)
                size = _export_one(catalog, name, args, paths, textures, log, prefix)
                # Keep the index current after every map, so the viewer offers finished maps while a batch runs.
                map_export.update_index(paths.output_dir)
                if size is None:
                    failures.append(name)
                    continue
                total += size
                elapsed = time.monotonic() - started
                remaining = len(pending) - number
                eta = f"  ~{_duration(elapsed / number * remaining)} left" if remaining else ""
                log.line(f"{prefix} {_human_size(size):>10}  {_duration(time.monotonic() - map_started):>12}{eta}")
        index = map_export.update_index(paths.output_dir)
        summary = f"Exported {_maps(len(pending) - len(failures))} ({_human_size(total)}) in {_duration(time.monotonic() - started)}."
        if failures:
            details = f" (details in {log.path})" if log.path else ""
            summary += f" {len(failures)} failed: {', '.join(failures)}{details}"
        log.line(f"\n{summary}\nIndex: {index}")
    finally:
        if textures is not None:
            textures.close()
        log.close()
    return 1 if failures else 0


def _verify_group(group: PakGroupInfo, check_headers: bool) -> tuple[int, int]:
    """Print one line per side and return (problems, headers parsed)."""
    problems = 0
    parsed = 0
    try:
        paks = group.open()
    except PakError as exc:
        print(f"  FAIL {group.name}: {exc}")
        return 1, 0
    with paks:
        for archive in paks.archives():
            side = archive.side
            if len(archive) == 0:
                stub = archive.path.stat().st_size == 0
                print(f"  {group.name:<18} {side:<6} {'empty (0-byte stub)' if stub else '0 entries'}")
                continue
            failures = []
            if check_headers:
                for entry in archive:
                    head = _read_header_bytes(paks, entry)
                    if not is_ser(head):
                        continue
                    try:
                        parse_header(head)
                        parsed += 1
                    except SerFormatError as exc:
                        failures.append(f"{entry.name}: {exc}")
            status = f"{len(failures)} header failures" if failures else "ok"
            print(f"  {group.name:<18} {side:<6} {len(archive):>6,} entries  {status}")
            for failure in failures[:5]:
                print(f"      {failure}")
            problems += len(failures)
    return problems, parsed


def cmd_verify(args: argparse.Namespace, paths: Paths) -> int:
    catalog = Catalog(paths.content_dir)
    groups = [catalog.get(name) for name in args.paks] if args.paks else catalog.groups()
    problems = 0
    parsed = 0
    for group in groups:
        group_problems, group_parsed = _verify_group(group, args.headers)
        problems += group_problems
        parsed += group_parsed
    summary = f"{len(groups)} pak groups checked, {problems} problems"
    if args.headers:
        summary += f", {parsed:,} 1SER headers parsed"
    print(f"\n{summary}")
    return 1 if problems else 0


# -- argument parsing --------------------------------------------------------------


def _add_entry_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pak", help="pak name without extension, e.g. ny_01_pvp or shared_tpl")
    parser.add_argument("entry", help="entry name or unique suffix, e.g. .obb or ny_01_pvp.lg")
    parser.add_argument("--side", choices=SIDES, default="client", help="which half of the install (default: client)")


def _add_window_arguments(parser: argparse.ArgumentParser, default_length: int | None) -> None:
    parser.add_argument("--offset", type=lambda text: int(text, 0), default=0, help="start offset (accepts 0x...)")
    parser.add_argument(
        "--length",
        type=lambda text: int(text, 0),
        default=default_length,
        help="bytes to read" + (f" (default: {default_length})" if default_length else " (default: to end)"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wwz-extract",
        description="Read-only explorer and extractor for World War Z map data.",
    )
    parser.add_argument("--content-dir", help="game Content folder (overrides config/paths.toml)")
    parser.add_argument("--output-dir", help="output folder (overrides config/paths.toml)")
    commands = parser.add_subparsers(dest="command", required=True)

    maps = commands.add_parser("maps", help="list paks")
    maps.add_argument("--all", action="store_true", help="include shared and core paks")
    maps.set_defaults(handler=cmd_maps)

    ls = commands.add_parser("ls", help="list entries in a pak")
    ls.add_argument("pak")
    ls.add_argument("--side", choices=SIDES, help="only one half of the install (default: both)")
    ls.add_argument("--ext", help="only entries with this extension, e.g. .tpl")
    ls.add_argument("--grep", help="only entries whose name contains this text")
    ls.add_argument("--types", action="store_true", help="read and show each entry's 1SER type")
    ls.add_argument("--limit", type=int, default=0, help="stop after this many entries")
    ls.set_defaults(handler=cmd_ls)

    info = commands.add_parser("info", help="show a resource's 1SER header")
    _add_entry_arguments(info)
    info.set_defaults(handler=cmd_info)

    dump_hex = commands.add_parser("hexdump", help="hex dump part of an entry")
    _add_entry_arguments(dump_hex)
    _add_window_arguments(dump_hex, default_length=512)
    dump_hex.set_defaults(handler=cmd_hexdump)

    probe = commands.add_parser("probe", help="reverse-engineering report for an entry")
    _add_entry_arguments(probe)
    _add_window_arguments(probe, default_length=None)
    probe.add_argument("--save", action="store_true", help="also write the report under output/probe/")
    probe.set_defaults(handler=cmd_probe)

    dump = commands.add_parser("dump", help="copy an entry's raw bytes to output/raw/")
    _add_entry_arguments(dump)
    dump.set_defaults(handler=cmd_dump)

    boxes = commands.add_parser("obb", help="decode and summarise a scene's oriented bounding boxes")
    boxes.add_argument("pak", help="scene pak name, e.g. ny_01_pvp")
    boxes.add_argument("--side", choices=SIDES, default="client", help="which half of the install (default: client)")
    boxes.set_defaults(handler=cmd_obb)

    export = commands.add_parser("export", help="write scenes in the viewer's format under output/maps/")
    export.add_argument("maps", nargs="*", help="scene pak names, e.g. ny_01_pvp")
    export.add_argument("--all", action="store_true", help="export every scene")
    export.add_argument("--side", choices=SIDES, default="client", help="which half of the install (default: client)")
    export.add_argument("--boxes-only", action="store_true", help="skip the real geometry (much faster)")
    export.add_argument(
        "--resume",
        action="store_true",
        help="skip maps this exact extractor build already exported (re-run an interrupted batch)",
    )
    export.add_argument("--log", action="store_true", help=f"also append progress and full errors to <output>/{EXPORT_LOG_NAME}")
    export.set_defaults(handler=cmd_export)

    verify = commands.add_parser("verify", help="check that every pak opens")
    verify.add_argument("paks", nargs="*", help="pak names to check (default: all)")
    verify.add_argument("--headers", action="store_true", help="also parse every 1SER header (slower)")
    verify.set_defaults(handler=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Entry names are ASCII, but keep a stray byte from crashing the Windows console.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    args = build_parser().parse_args(argv)
    try:
        paths = load_paths(args.content_dir, args.output_dir)
        return args.handler(args, paths)
    except (ConfigError, PakError, SerFormatError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
