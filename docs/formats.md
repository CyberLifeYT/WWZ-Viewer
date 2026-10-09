# World War Z data formats: reverse-engineering notes

Everything here was observed in the Game Pass / Xbox app build (`wwzRetail.exe`).
Each claim is marked **verified** (checked across the whole install by code in
this repo) or **observed** (seen in a few files, not yet checked everywhere).

## Install layout

`Content/client/*.pak` holds rendering data and `Content/server/*.pak` holds
gameplay data. Most pak names exist on both sides. There are 54 pak groups:
44 scenes, 8 shared, 2 core (`boot`, `initial`). Four paks are 0-byte stubs:
`client/shared`, `server/shared`, `server/boot`, `server/shared_video`.

## `.pak` archives (verified)

Standard ZIP archives with stored or deflated entries, no encryption, followed
by zero padding. Python's `zipfile` opens all 101 non-empty paks (~90,000 entries).

Entry names use `/` and bracketed mount points:

- `[scene]/<map>.lg`, `.lg_data`, `.cdt`, `.nav`, `.obb`, ...: per-level resources
- `[scene]/tpl/<name>.tpl/<name>.tpl`: object templates (a directory named like the file)
- `[ssl_cache]/scenes/<map>/...`: compiled script bundles (`.sslbundle`, `.spdb`)
- `pak.load_list`, `pak.ssl_active_module_list`: pak-level metadata

## The `1SER` container (verified: 40,286 headers, 0 failures)

| Offset | Type | Meaning |
|---|---|---|
| `0x00` | `char[4]` | magic `1SER` |
| `0x04` | `char[32]` | resource type, NUL-padded (`lg`, `cdt`, `obb`, `tpl`, ...) |
| `0x24` | `byte[16]` | `S3DRESOURCE` + 5 spaces, **or** the resource's own GUID |
| `0x34` | `u32` | version (inferred): 0 on most types, 2 on `rain`, 4 on `light_probe` |
| `0x38` | `u32` | dependency count |
| `0x3C` | `u32` | dependency block size |
| `0x40` | | dependencies, then the payload at `0x40 + block size` |

Each dependency: `u32 flags`, `byte[16] guid` (zero when referenced by path
only), `u32 length`, path bytes NUL-padded to a multiple of 4
(e.g. `<scene>\ny_01_pvp.lg`).

**GUID cross-references (observed).** The level graph `.lg` carries its own
GUID in the signature slot (`977ce2c7…` for ny_01_pvp). The `.cdt` and `.nav`
of the same level list that GUID as a dependency with flags `0x01000001`, and
`.lg_data` (not a 1SER file) begins with the same 16 bytes.

**Sub-magics (observed).** Resources that carry their own GUID may leave the
type name empty and start their payload with a 4-character tag instead:
`SCN1` for level graphs and `TPL1` for some templates (138 in ny_01_pvp, 14 in rome_03_pve).

## Per-scene resources

Inventory of `ny_01_pvp` (client side; rome_03_pve has the same resource types). Descriptions
without "decoded" are inferred from names and strings, not from decoding:

| Entry | 1SER type | Size | Status |
|---|---|---|---|
| `.obb` | `obb` | 3.2 MB | **Decoded**, see below |
| `.lg` | `lg` (`SCN1`) | 27 MB | Level graph: object instances. Next target |
| `.lg_data` | not 1SER | 163 MB | Starts with the `.lg` GUID; data blob for the level graph |
| `.cdt` | `cdt` | 25 MB | Probably collision data; depends on the `.lg` |
| `.nav` | `ai` | 250 KB | Navigation; contains the string `hk_2014.2.5-r1` (Havok 2014.2.5) |
| `.tpl` | `tpl` or empty (`TPL1`) | ~626 files | Object templates |
| `.grs_units` | `grs_units` | ~350 files | Probably grass/scatter placement |
| `.class_list`, `.dom_list`, `.cd_list` | same | small | u32-length-prefixed name lists |
| `sm_obj_tbl.tbl` | `sm_tpl_tbl` | 2 KB | Name table, e.g. `geom_office_01__alarm_light_01_ref1\|actor_redlamp\|…` → `sm_geom_00033` |
| `.light_probe2`, `.rain` | `light_probe`, `rain` | 9 MB, 2 MB | Lighting and weather |
| `.pct` + `.pct_header` | not 1SER | | Textures (by name) |

`shared_tpl` (809 MB) holds the templates shared between scenes: `tpl`, `dsc`
and `cdt` resources (so collision appears to exist per template) plus
`.tpl_data` blobs.

## `.obb`: oriented bounding boxes (verified: 44 scenes, 2,963,534 boxes)

```
u32 group_count
group[group_count]:
    u32 box_count                  1..16
    box[box_count], 60 bytes:
        f32[3] origin              one corner of the box, world space, metres
        f32[9] axes                3x3 rotation; row i is local axis i in world space
        f32[3] extents             full edge lengths, >= 0; 25% of boxes have a zero axis (flat)
```

A box covers `origin + t0*e0*axes[0] + t1*e1*axes[1] + t2*e2*axes[2]` for
`t` in [0, 1]. The client and server copies are byte-identical in every scene.
Every rotation is orthonormal to within 4e-7 with determinant +1. Decoder:
`extractor/formats/obb.py`.

**How the convention was settled (verified).** ny_01_pvp's group 1 is a
five-box world boundary: four edge walls and a 1 m floor whose origins sit
exactly on the corners of a 6,901 m square, each with a 6,901 m extent. Read as
"origin corner + row axes + full extents", every box ends exactly on the
opposite corner and the five close into a square. Read column-major, the far
ends miss by 4-7 km; read as centred boxes, they overshoot by 3.4 km. In the
viewer, facades, fire escapes and rooftop clutter line up, and picking with
the browser's ray caster lands inside the reported box in every test.

**Groups.** A group is one object: the whole boundary fence is a single group,
and groups have 1-16 boxes.

**Coordinate system.** Y is up: in 43 of 44 maps the spread of box centres
along Y is 5-20x smaller than along X or Z, and in ny_01_pvp the 1st
percentile of box-centre height is exactly 0.0 (street level). `ny_02_pve`
spans 233 m vertically, which looks like a genuinely tall level. Maps are
surrounded by kilometres of sparse scenery, so the median box centre locates
the playable area far better than the bounds do (ny_01_pvp: median
(2.5, 17.1, -5.4), bounds out to about 8 km).

**Still open: handedness.** A mirrored map is just as self-consistent, so
boxes alone cannot settle it. Compare against the game itself (a recognisable
landmark, or the camera coordinates found in Step 3); the viewer's
"Mirror X" toggle shows both versions.

## `.pct` textures (verified: 392 of a random 400 decode; viewer shows them)

`shared_textures` holds 11,463 textures (39 GB) as `[textures]/<folder>/<name>.pct`;
materials refer to them by bare name. `shared_textures_base` has low-resolution
`.pct_base` copies. Decoder: `extractor/formats/pct.py`; name lookup and partial
reads: `extractor/textures.py`.

```
0x00  header: 64 fixed bytes + 8 bytes per mip (mip table)
        0x06 "TCIP"   0x10 u32 width, height, depth, faces   0x26 u32 format   0x30 u32 mips
then  mip chain, largest first, tightly packed
end   6 trailing bytes        file size == 64 + 8*mips + chain + 6  (exact)
```

| Code | Format | Notes |
|---|---|---|
| `0x24` | BC5 | normal maps (5,238) |
| `0x33`, `0x34` | BC7 | colour textures (4,065) |
| `0x11` | BC3 | decals with alpha |
| `0x0c`, `0x0d`, `0x25` | BC1 or BC4 | resolved at runtime by decoding both and keeping the smoother |
| `0x16`, `0x00` | RGBA8 | small uncompressed |
| `0x31` | BC6H (HDR) | lightmaps (`[scene]/lm/*_rnm*`, `*_dircolor`); see Baked lighting |

Pitfalls met on the way: data at "file size minus chain" (off by the trailer)
and a fixed 136-byte header (true only for 9 mips) both give structured noise.
The BC1/BC4 test decode must use the sample mip's real size: assuming 64 × 64
failed every non-square texture with those codes (girder_03 is 2048 × 512, its
sample mip 64 × 16). All 11,463 textures decode.

## Level geometry (`.lg` geometry section + `.lg_data`)

Module: `extractor/formats/level_geometry.py`. The `.lg` payload is a tree of
fields (`u16 id, u32 absolute end, payload`); the geometry section is found by
pattern in every map (a 26-byte counts field followed by a buffer table whose
children are ids 0, 4, 1, 2).

**Tables (verified: ny_01_pvp, rome_03_pve, jer_02_pvp):**

- Counts: objects, buffers (18-22), batches (72-190), draws (3,175-15,158).
- Buffer table: per buffer an 11-byte descriptor, a pool byte, stride (6 = u16
  index triangles), byte size. Descriptor byte 2: `0x03` float positions,
  `0x0b` quantized positions. For attribute buffers see *Attributes* below.
- Batches (id 3 › id 2): `u8 n` then `n × (u32 buffer, u32 byte offset)`;
  first stream = positions, last = indices, the one between = attributes.
- Per draw: batch index (id 4 › id 1), material (id 4 › id 8, a typed
  key/value tree), and one quantization frame per quantized draw (id 4 › id 5:
  i16 centre xyz, i16 half-size xyz in metres; the k-th quantized draw uses the
  k-th frame; position = centre + q/32767 × half-size).
- Draw records (id 4 › id 0): `u16 first vertex, u16 vertex count,
  u16 first triangle, u16 triangle count, u16 0, u16 0xffff, u32 LOD mask`,
  then two floats per mask bit: the draw's UV density (UV units per metre,
  along u and v) at that LOD. Ranges are batch-relative.

**Blob layout (verified: all 44 maps end exactly at the end of the file).**
`.lg_data` = 16-byte GUID + buffers. Counted from the end of the GUID, each
buffer starts at a multiple of its own stride, 6 for index buffers, often
behind zero padding (up to 2,960 bytes in rome_03_pve). The physical order is
the table order with a few buffers moved later (ny_01_pvp: …, 20, 2, 21, 11),
so no fixed rule places them. `solve_layout` searches depth-first: at each
point it tries the unplaced buffers whose content fits there, backtracking
when a later buffer finds no home, and accepts the layout only when nothing
but zeros follows the last buffer. "Fits" is judged on the content, sampled
over several draws (blocker and flare draws excluded, the median taken):
index buffers must give real, in-range triangles; stored normals must agree
with the faces they belong to; attribute buffers must hold unit tangents at
right angles to the normals whose direction follows the UV gradient. A strict
pass runs first, a relaxed one if it fails, and a greedy fallback last. The
earlier stride-only solver put quantized position buffers 4 and 1 rows early
(the "stray vertices") and swapped buffers 11 and 21 (why no UV encoding
seemed to fit).

**Positions and normals (verified).**

- Float: `f32 x, y, z, n`, world space; `n` is an integer stored as a float,
  packing the normal as three bytes (x, y, z) for -1..1. Agreement with the
  faces: 0.99.
- Quantized: `i16 x, y, z, w`. `w` packs the normal: `|w| = row × 181 + col`,
  `z = row/90 − 1`, `x = (col − 90.5)/89.5` (fitted: columns 1 and 180 are -x
  and +x), `y = sign(w) × √(1 − x² − z²)`.
- Stored normals agree with `cross(b − a, c − a)`: counter-clockwise front faces.
  Text on posters reads correctly, so the map is not mirrored.

*How exact the normals are.* The game's tangents are at right angles to the
decoded normals: median 0.24° (float positions) and 0.4° (quantized), about
what their 8-bit storage allows. Against the true faces, quantized normals
of walls deviate 1-2° (90th percentile 4°): the grid stores only x and z, so a
half-cell rounding of either turns a horizontal normal's `y = 0` into up to
0.1. That is the game's own precision, and the game shades with it: moving
each normal inside its grid cell toward the mesh's smooth normal made the
game's tangents fit worse (90th percentile 3.8° instead of 2.4°), so the
decoder keeps the stored values.

The stored normals also guard the exporter's safety net against broken
triangles: a triangle much longer than the rest of its draw is dropped only
when its face disagrees with its vertices' normals (a triangle stitched from
the wrong vertices). Dropping long triangles regardless, as first done, cut
real walls and roofs: ~4,400 triangles in geo_01_pve, 25 with the check.

**Attributes (verified: the descriptor predicts the stride of all 30 formats
in the 44 maps).** 4-byte words: the tangent frames, the colour words, then
the UV sets. Descriptor byte 3's high nibble has one bit per tangent frame
(bytes x, y, z for -1..1, then 0 or 255 for the bitangent sign; bitangent =
normal × tangent × sign, which matches +dP/dv on 98% of triangles); byte 4
one bit per colour word (0x02 is always set and adds nothing). Bytes 5-6,
read as a little-endian u16, list the UV sets: bit `0x02 << k` marks set `k`
(0-4) present, bit `0x40 << k` marks it compressed to one word (`i16 u, v`)
instead of two (`f32 u, v`). Sets can be sparse (0 and 2). The UV gradients
of the decoded triangles point along the stored tangents (0.94-0.99). The
lightmap UV is the set the material's `lm.uvSetIdx` names; compressed, it is
`u16 / 32768`.

**UV scale (working).** `uv = raw/32767 × k`, one `k` per draw. The game picks
`k = ceil(max |uv|)`: 1 for atlases (a graffiti quad maps to exactly one
quarter of its atlas), 2 for most walls, 8-25 for long tiling runs. `k` is not
stored; the stored UV density divided by the density of `raw/32767` (area-
weighted, divided by 1.0082) estimates it, and it is snapped to the integer
the ceil rule allows. 91% of draws come out integer; the rest (degenerate or
very uneven UVs) keep the estimate.

**Duplicates:** the level stores some meshes several times (byte-identical,
1,179 of 7,326 LOD-0 draws in ny_01_pvp), probably once per render pass. The
exporter keeps one copy.

**Blockers:** draws whose material is `red` (a texture name the game does not
ship) are invisible blocker geometry; the viewer hides them.

**Material layers (verified on ny_01_pvp by eye).** A material has up to four
layers (`layer0`-`layer3`). Each names a texture (`texName`), a named variant
of it (`mtlName`: `default`, colour variants, `no_transp`, `lights_glass`, …),
tiling, a UV set and a `tint` stored alpha first (A, R, G, B; sRGB: painted
steel reads grey-green this way, not maroon). Layers 1-3 blend over layer 0:
`common` mixes, `dirt` and `dirt_no_spec` multiply by 2× the layer colour in
gamma space. A layer's weight is `weightMultiplier` × a vertex colour channel
(`weights.colorSetIdx`; channels taken as stored B, G, R, A, not verified on
data) × optionally the layer texture's alpha (`useLayerAlpha`), optionally
sharpened by a heightmap (`useHeightmap`, the texture's `_hdetm`) with
`heightmapSoftness`. Companion textures by suffix: `_nm` (BC5 tangent-space
normal map, x and y, green along +v: on the heightmaps that ship with their
normal maps the x and y slopes share a sign), `_spec` (specular mask), `_em`,
`_hdetm` (height). `_em` is two different things: on about 190 of the 1,091
textures it is a light map, black but for the lit parts (window lights, lamp
glass, vending machines, signs); on the other ~900 it is a bright greyscale
mask of unknown use (corpse piles, car bodies, planks), which drawn as light
made those glow white. The viewer lets an `_em` texture glow only when at
least half of it is nearly black (or its mean is below 0.1). Material variants
named `*_off`, `light-off` or `no_emissive` are the lights switched off; their
`_em` is not used.

**Alpha (heuristic, reviewed across all 44 maps).** Nothing marks alpha
testing: not the materials, not the `.pct_header` files (a copy of the
texture header), not the scene files; the game compiles it into its shaders.
Many textures keep gloss, metal or grout masks in alpha (a train texture is
99% "transparent", chrome 100%). So three signals decide, in this order:

1. `blend` when `extraParams.transparency.enabled`; `opaque` for the
   texture's opaque variants (`no_transp`, the misspelt `no_tarnsp`,
   `non_transp`, `opaque`, `no_blend`, `blend_none`, …) and for textures with
   <3% or >97% transparent texels.
2. `cutout` by name when the alpha is a mask (at 256 px, >60% of texels near 0
   or 255): open structures and foliage (lattice, fence, wire, grate, plant,
   leaves, …), signs, letter sheets, posters, flags, banners, scrim, curtains,
   blinds, papers, debris and garbage, backdrop billboards (`pb_*`,
   mountains, backgrounds).
3. `blend` for overlays (decal, blood, graffiti, puddle, road marking, …),
   glass, and the see-through variants of shared atlases (`lights_glass`,
   `glass`, `transparent`, `decals`: a car's headlight covers, a bus's windows).
4. `cutout` for any other texture whose alpha is a crisp mask (≥90% of texels
   near 0 or 255) whose clear part forms solid regions (≥40% of it survives
   shrinking by 2 texels) and covers at most 85%: cut shapes and unused atlas
   space. Thin clear lines are grout and panel seams (`je_tile_01`,
   `wall_metal_siding_02`), and mostly clear alphas mark something else (lit
   window panes, `windows_13`); those stay opaque.

Checked by eye on contact sheets of colour and alpha: about 90 textures, the
ones with the most draws whose class changed. Water textures are normal maps
(the shader colours water); sky domes (`cloud*`, `*panorama*`) are unlit.

**Light effects.** Light cones, ray fans and flares (`lig_*`, the game's
prefix for lighting-effect textures, plus `light_cone*`, `*rays*`,
`*flare*`, `*halo*`) are light on black, added onto the scene. The bake
often lights their meshes like any other, which drew them as black solids,
so these names make glows whatever the bake says (`glow` and `beam` only
when the bake leaves the mesh unlit: steel beams are real). A light cone is
dozens of crossed planes (a line of sight crosses up to 40), and no vertex
colour or alpha fades them, so the viewer adds each plane at a modest
strength and fades it as it turns edge-on to the camera, like the game's
effect shaders; the colour is the layer tint.

## Baked lighting (`[scene]/lm/`, verified on ny_01_pvp)

Module: `extractor/lighting.py`. Each material's `lm` entry names the source
(`vertex`, `tex` or `none`), the set (`<map>_vc`, `<map>_tex`) and the UV set.
12 of the 44 maps name the set without the map (`_vc`, `_tex`: camp_02_pve,
hm_base, horde_mode_02, jap_01/02, kam_01/03, mar_02, mos_02_td/03, veg_03);
the files always carry it.

- `vertex` (`<map>_vc`, 1848 × 1848): per-vertex light stored as a texture.
  Each vertex's lightmap UV (the first extra attribute word, u16/32768) lands on
  the centre of a texel of its own.
- `tex` (`<map>_tex`, 2048 × 2048): a lightmap atlas sampled across surfaces with
  one of the regular UV sets.
- Each set: `rnm0..2` (BC6H, light along three tangent-space basis directions;
  equal weights for an unbumped surface), `dircolor` (BC6H, colour of the
  dominant direct light) and `dir` (BC7, its world-space direction; on lit
  vertices it points out of the surface). Light = mean(rnm) + dircolor ×
  max(0, n·dir), linear HDR (median 0.2, a fifth above 1, lamps beyond 70).
- The basis is Half-Life 2's "radiosity normal mapping" basis in the vertex's
  tangent frame (verified on the data): `(√(2/3), 0, 1/√3)`,
  `(−1/√6, 1/√2, 1/√3)`, `(−1/√6, −1/√2, 1/√3)`. With a normal map the viewer
  weighs the three colours by `saturate(n·bᵢ)²`, normalised.
- `.pct` code `0x31` is BC6H. Pillow clamps it to 0..1; `extractor/formats/bc6h.py`
  decodes the full range (all 14 modes; matches Pillow within rounding inside
  0..1). `low_*` copies are lower-quality duplicates.

The viewer multiplies textures by this light and tone-maps the result (ACES);
the game's own exposure curve is not known. The light spans a huge range:
between maps (what a camera mostly sees, the area-weighted median of light
times surface colour, is 0.0075 on the night map jer_03_pve, 0.47 on
ny_01_pvp, 4.4 on the sunny kam_02_pve) and within one (a sunlit rooftop gets
the sun at 16 and more on top of 2-4 of sky light; interiors a tenth of that).
So the viewer adapts like an eye, as the game does: a few times a second it
reads back a block from the middle of the displayed image and moves the
exposure toward the value that brings its log-average to middle grey, within
0.06-6 so night stays darker than day (`viewer/js/render/eye_adaptation.js`). A
map starts at an estimate from its own light (`auto_exposure.js`). Measuring
by vertex instead of by area is misleading: detailed props and interiors hold
most vertices (geo_03_pve's median vertex light is 0.012, its area median as
bright as ny_01_pvp's).

## Light probes (`[scene]/<map>.light_probe2`, verified on all 44 maps)

Module: `extractor/formats/light_probes.py`. Movable objects (doors,
characters) are left out of the bake and lit from probes: 3 to 811 groups and
112 to 61,801 probes per map (1,055,763 in all). Each group is a tetrahedral
mesh over its probes, then `u32 n`, `n` positions (`f32 x, y, z`), `u32 n`,
the tag `02 00 00 00 01`, then one 370-byte record per probe starting with
`03 00 00 00 02 00 00 00 01` and 27 floats: second-order spherical harmonics
of red, green and blue. The decoder works back from the record markers (a
marker with the tag before it, and the count twice around `n` positions,
opens a group); this claims every record on every map. The first decoder
scanned for "count, positions, count" blocks and was only checked on
ny_01_pvp: on 35 other maps it also matched chance patterns in the index
lists and missed groups of fewer than 4 probes or at one height.

Probe values are in other units than the bake, and their level varies a lot
between maps, so the exporter measures the ratio per map on baked vertices
near probes (within 1 m, widening to 8 m when fewer than 50 are that close;
0.026-0.09 so far) and lights the bake's left-out draws with probe
irradiance × ratio, blending each vertex's four nearest probes (found with a
k-d tree: maps have up to 62,000 probes and millions of probe-lit vertices,
which a brute-force search turns into gigabytes of temporaries and hours of
paging).

## Reflection probes (`[scene]/lc/*_cube.pct`)

Module: `extractor/reflections.py`. BC6H cubemaps, 128 × 128 per face, stored
face by face (+x, −x, +y, −y, +z, −z, each with its own mip chain), addressed
the Direct3D way (seams between faces line up only with that orientation:
2-7× smaller mismatch than with flipped faces). 4 to 178 per map, named
`lc_prebuild_<map>_<lighting group>__<probe>_cube.pct` with names that vary
from map to map (`reflection12`, `reflection_12`, `subway_reflection3`,
`outdoor_main_cm_2`, `global_reflection_probe`, …), so every cubemap in the
map's `lc` folder counts. Where each probe stands is not in any file read so
far (objects name theirs only now and then), so the viewer gets one per map:
the probe whose upward face is brightest, an open view of the sky. (Some maps
have a "global" probe, but several of those are dark.) It is exported as an
equirectangular float16 image. On a night map the brightest probe stands
under a street lamp (jer_03_pve: 120 times the median baked light, against
0.3-2.2 on day maps), and glossy surfaces reflected it as daylight, so the
exporter scales the probe down to at most 3 times the map's median baked light.

**Not in the level geometry:** objects built from templates (`[scene]/tpl/*.tpl`
plus `shared_tpl`), placed by the scene description (`.cdt`). Some maps keep
their lake or sea surface there (geo_01_pve: `water_surf_01_nm` appears only in
"final_viewzone" templates), so the viewer shows the black lake bed
(`sc_black`) where the game draws water. Decoding templates and their
placement is not done yet.

**Open:** template objects (above); the vertex colour channel order of layer
weights (B, G, R, A is assumed); reflection probe positions (one probe per map
is used); the game's exposure curve; how the game lights water.
