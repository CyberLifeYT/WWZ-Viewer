# WWZ Viewer

Map viewer and freecam tooling for *World War Z* (Saber Interactive's Swarm Engine).

An unofficial fan tool, not affiliated with Saber Interactive or Focus
Entertainment. This repository contains no game files: the maps are made on
your own PC from your own copy of the game.

1. **Step 1, extractor:** read the game's `.pak` archives and decode level data
   into a neutral format. Bounding boxes, textures, real level geometry (with
   the game's normals and UVs) and the baked lighting are decoded.
2. **Step 2, browser viewer:** a Three.js scene with a physics-free freecam. Working.
3. **Step 3, in-game freecam:** camera-matrix hooking in the live process (solo play only).

The extractor is strictly read-only towards the game folder. Everything it
writes goes to `output/` (git-ignored), and it refuses to write inside the game folder.

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
cp config/paths.example.toml config/paths.toml   # then edit content_dir
```

`content_dir` must point at the game's `Content` folder (the one containing
`client/` and `server/`). `WWZ_CONTENT_DIR` / `WWZ_OUTPUT_DIR` or the
`--content-dir` / `--output-dir` flags override the file.

## Viewing the maps

**The easy way (Windows):** double-click `export_all_maps.bat` in the project
folder (a shortcut to it works too). The first time it creates `.venv` and
installs the extractor, and finds the game (asking for its `Content` folder
only if it is not in a usual Game Pass, Steam or Epic location). It then
starts the viewer's server in a minimized "WWZ viewer server" window, so
finished maps can be opened at http://127.0.0.1:8765/viewer/ while the rest
export, exports every map with real geometry, materials and lighting (1 to 3
minutes per map, about 1.5 hours and 11 GB in all; the PC is kept awake; the
line of the map being worked on reads "exporting..."), and finally opens the
viewer in the browser. Keep the server window open while viewing; close it
when done. Running the file again skips maps the same extractor code already
exported, so an interrupted run carries on where it stopped (after an
extractor update every map is redone), and with everything exported it just
starts the viewer. A map that fails is reported (full details in
`output/export.log`) and still gets its boxes; the others carry on.
`--fresh` re-exports everything, `--no-view` skips the viewer.

If the viewer shows a map but no textures, its server has stopped: the panel
says so. Start it again with the `.bat` or `serve.py`.

By hand:

```bash
wwz-extract export --all --resume --log   # every map; skips finished ones, logs to output/export.log
wwz-extract export ny_01_pvp              # one map: ~1-2 min and 120-320 MB
wwz-extract export --all --boxes-only     # boxes only for every map: ~5 s, ~180 MB
.venv/Scripts/python serve.py --open      # http://127.0.0.1:8765/viewer/ in the default browser
```

The page loads Three.js from jsDelivr, so it needs internet access.

## Website (GitHub Pages)

`serve.py` needs the game on the same PC: it decodes textures when the page
asks for them. GitHub Pages only hands out files, so `build_site.py` builds a
self-contained copy of the viewer for it instead: the viewer, the chosen
exported maps, and every texture they use decoded in advance (WebP), with the
page switched to read those files.

```bash
.venv/Scripts/python build_site.py                        # the viewer only, into site/
.venv/Scripts/python build_site.py --maps ny_01_pvp       # with exported maps
.venv/Scripts/python build_site.py --maps ny_01_pvp --publish https://github.com/<user>/<repo>.git
```

`--publish` commits `site/` as the `gh-pages` branch of that repository and
pushes it with your git login; then turn on Pages once (Settings > Pages >
Deploy from a branch > `gh-pages`, `/ (root)`). The build checks GitHub's
limits (100 MB per file, 1 GB per site): one map with its textures is about
130-450 MB. The maps and textures come from the game's files, which belong to
its publisher, so check that you may publish them before adding maps.

**Textures come straight from the game install.** The server answers
`/api/texture/<name>.png?size=N` by reading `<name>.pct` from the game's
`shared_textures` pak (read-only), decoding it, and caching the PNG in
`output/texture_cache/`. The panel's *Textures in this map* gallery shows every
texture a map's materials use; click one for the full-size image.

**Real geometry, textured and lit like the game.** `wwz-extract export <map>`
also decodes the level's actual meshes (about 3.8 million triangles for
ny_01_pvp) with the game's own normals, tangents, texture coordinates and
materials, and its baked lighting. The viewer draws them with the game's
material system:

- up to four texture layers per material, blended by vertex colour (dirt,
  moss, wear), with the game's tints and blend heightmaps;
- per-pixel normal maps, lit with the bake's three "radiosity normal mapping"
  directions plus its direct light, so bumps catch the game's own light;
- reflections of the map's reflection probe, masked by the specular maps;
  emissive maps (lit signs and windows); water;
- doors and other movable objects lit from the game's light probes;
- cut-outs for fences, foliage, signs and debris, blended decals and glass,
  an unlit sky dome.

*Game lighting (baked)* shows the game's light and shadows, tone-mapped. *Auto
exposure* adapts the brightness to what you are looking at, like the game's
eye adaptation (sunlit rooftops and dark interiors differ a hundredfold); the
*Exposure* slider brightens or darkens on top of it, or sets it directly with
auto exposure off. Untick *Game lighting* to light the map with the viewer's
own sun instead. The game's invisible blockers are hidden by default. Exported
buffers are gzip-compressed (the server sends them as they are). See
`docs/formats.md` for what is and is not solved.

Not shown yet: objects the game builds from templates rather than the level
geometry. On some maps that includes the water surface of a lake or the sea,
so its dark bed shows instead (geo_01_pve).

| Input | Action |
|---|---|
| Click the map | capture the mouse (Esc releases it) |
| Mouse | look around |
| W A S D | fly where you look |
| Space / Shift | up / down along world Y |
| Mouse wheel | faster / slower (x1.25 per notch, 0.5 to 5000 m/s) |
| F | identify what is at the crosshair: a box, or a geometry surface with its texture |
| R | reset the view |

The panel switches maps (the URL remembers the map as `#map=<name>`), colours
boxes by object group or by height, hides oversized boxes (world boundaries
and distant scenery), and has a wireframe mode, a ground grid and a "Mirror X"
toggle (the map is not mirrored: poster text reads correctly). The HUD shows the camera
position in map coordinates, which will be the reference for Step 3.

## Extractor commands

```bash
wwz-extract maps                          # scene paks with client/server sizes (--all for shared/core)
wwz-extract ls ny_01_pvp --ext .tpl --types --limit 20
wwz-extract info ny_01_pvp .cdt           # 1SER header and dependencies
wwz-extract hexdump ny_01_pvp .lg --offset 0x70 --length 256
wwz-extract probe ny_01_pvp .lg --save    # reverse-engineering report, also saved under output/probe/
wwz-extract dump ny_01_pvp .lg            # raw bytes to output/raw/ for a hex editor
wwz-extract obb ny_01_pvp                 # decode and summarise the oriented bounding boxes
wwz-extract export ny_01_pvp rome_03_pve  # or --all
wwz-extract verify --headers              # open every pak and parse every 1SER header
```

Entries can be named by any unique suffix (`.obb`, `ny_01_pvp.lg`). Entry
commands read the client pak unless `--side server` is given. `python -m
extractor` works the same as `wwz-extract`.

## Layout

```
extractor/                Step 1 (Python): no rendering code
  config.py               locate the game and output folders
  pak_reader.py           read-only ZIP access; PakSet pairs client + server halves
  catalog.py              find paks and classify them (scene / shared / core)
  ser/
    binary_reader.py      little-endian cursor with absolute-offset errors
    container.py          the 1SER header and dependency block
    probe.py              format-agnostic RE tools: strings, float runs, record-size guesses
  formats/obb.py          oriented bounding boxes
  formats/pct.py          textures: header, mip chain, BC1-BC7 decoding
  formats/bc6h.py         BC6H decoding with the full HDR range (lightmaps, reflections)
  formats/level_geometry.py  geometry section of .lg + .lg_data: layout, positions, normals, UVs
  formats/light_probes.py the light probes that light movable objects
  textures.py             texture name -> decoded image, reading only the needed mip
  lighting.py             the level's baked lighting ([scene]/lm/)
  reflections.py          reflection cubemaps -> the map's environment image
  export/map_export.py    map_data.json + binary buffers for the viewer
  export/geometry_export.py  real geometry, materials and lighting grouped by material
  export/encoding.py      gzip buffers, RGBM light, PNG
  cli.py                  the wwz-extract command
schema/                   JSON Schemas: the contract between extractor and viewer
viewer/                   Step 2 (Three.js, ES modules, no build step)
  js/data/map_loader.js   fetch and validate exported maps; no rendering code
  js/render/              viewport (renderer, camera, lights), box layer, geometry layer,
                          the level-surface shader (surface_material.js) and texture cache
  js/input/freecam.js     pointer-lock fly camera; knows nothing about maps
  js/ui/                  side panel and HUD
  js/main.js              wires the modules together
serve.py                  localhost-only dev server: viewer/, output/maps/, schema/, and /api/texture/
build_site.py             the website copy (GitHub Pages): no server, textures decoded in advance
export_all_maps.bat       one double-click: set up, export every map, open the viewer
docs/formats.md           what is known about each file format, and how it was verified
tests/                    unit and end-to-end tests on synthetic data (no game files needed)
```

## Status

- Pak reading and the `1SER` container are verified across the whole install
  (101 paks, 40,286 headers, 0 failures).
- `.obb` is decoded and verified across all 44 scenes (2,963,534 boxes),
  including the box convention (origin corner, row axes, full extents).
- Textures: BC1/3/5/6H/7 and RGBA8 are decoded; all 11,463 game textures decode.
- Level geometry, materials and lighting are decoded for all 44 maps (2.5 to
  13 million triangles each): the game's normals, tangents, UVs, up to four
  material layers, normal maps, baked radiosity-normal-mapped light, light
  probes, reflection probes, emissive maps, cut-outs and light effects. Every
  map passes an export health check (layout, lighting coverage, placement
  against the map's own bounding boxes).
- The viewer adapts its exposure to the view, like the game's eye adaptation.
- Not yet: objects built from templates (`.tpl`), which on some maps include
  the water surface of a lake or the sea; the in-game freecam (Step 3). See
  `docs/formats.md` for what is known and how it was verified.

Run the tests with `.venv/Scripts/python -m pytest`.
