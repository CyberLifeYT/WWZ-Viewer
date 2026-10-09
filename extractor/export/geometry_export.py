"""Export a map's real level geometry, materials and baked lighting for the viewer.

Every buffer is written gzip-compressed as ``<name>.bin.gz`` next to the
manifest (the dev server sends them with ``Content-Encoding: gzip``), one
value set per vertex, vertices grouped by material:

=====================  =======  ====  ==========================================================
buffer                 type     n     contents
=====================  =======  ====  ==========================================================
``positions``          f32      3     world space, metres
``normals``            i8       3     the game's normals, -127..127
``tangents``           i8       4     first tangent frame xyz and bitangent sign (zero if none)
``uvs``                f32      2     UV set 0 (no tiling; materials carry it); v runs down
``colours``            u8       4     first vertex colour word: layer blend weights
``uv_set1..3``         f16      2     UV sets 1-3, for material layers that draw with them
``rnm0..2``            u8       4     RGBM: light along the three bake basis directions
``direct``             u8       4     RGBM: the dominant direct light's colour
``direct_dir``         i8       4     its world-space direction (zero when there is none)
``lightmap_uvs``       u16      2     where lightmap-lit vertices sample the lightmap textures
``indices``            u32      3     triangles, sorted by material
=====================  =======  ====  ==========================================================

plus ``lightmap_<name>.png`` (the lightmap atlas: three RGBM basis maps, direct
colour, direct direction) and ``environment.bin.gz`` (the map's reflection
probe as an equirectangular float16 RGBA image).

Materials group draws that render alike: same layers (textures, tiling, UV
sets, blending), alpha use, lighting source. Each lists its layers with their
normal maps and heightmaps, plus a specular and an emissive map where the
game ships them.

Lighting per vertex (see ``extractor.lighting``): vertex-lit draws carry the
bake's three basis colours and the direct light; lightmap-lit draws carry
lightmap UVs; draws the bake leaves out (doors and other movable objects) get
the light probes' irradiance, scaled to the bake's units by comparing probes
with baked vertices near them. Skies, glows and blockers are unlit.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from extractor import reflections
from extractor.catalog import Catalog
from extractor.export.box_oracle import BoxOracle
from extractor.export.encoding import RGBM_RANGE, png_bytes, rgbm, write_atomic, write_buffer
from extractor.formats import level_geometry, light_probes, obb
from extractor.lighting import BakedLighting
from extractor.pak_reader import PakError
from extractor.ser.errors import SerFormatError
from extractor.textures import AlphaProfile, TextureLibrary

GEOMETRY_FORMAT = 2
BLOCKER_TEXTURE = level_geometry.BLOCKER_TEXTURE
UNTEXTURED = "(untextured)"
FALLBACK_COLOR = [190, 185, 176]
EXTRA_UV_SETS = (1, 2, 3)  # UV sets exported for layers; set 0 is ``uvs``
MAX_LAYERS = 4
QUALITY_SAMPLE = 20_000  # vertices checked against the map's boxes for the quality figure
# Safety net against broken draws: within a draw, a triangle more than SPIKE_FACTOR times
# longer than the draw's median triangle and over SPIKE_MIN_EDGE is suspect, and dropped
# unless it faces the way its vertices' stored normals say (SPIKE_FACING, mean cosine).
# Long real triangles (walls, roofs, girders) pass; a triangle stitched from the wrong
# vertices does not. Length alone dropped real geometry: ~4,000 triangles per map.
SPIKE_FACTOR = 12.0
SPIKE_MIN_EDGE = 8.0
SPIKE_FACING = 0.5
UNLIT = 1.0  # light for surfaces shown as they are (skies, glows) or lit by the lightmap texture
# Probes and the bake use different units, and the ratio differs per map (probe brightness
# spans 0.7 to 300 across the maps), so each map measures its own on baked vertices near
# probes, looking further out when too few are close. The default (ny_01_pvp's ratio) is
# only for maps with no baked vertices near any probe.
DEFAULT_PROBE_SCALE = 0.03
PROBE_CALIBRATION_RADII = (1.0, 2.0, 4.0, 8.0)
PROBE_CALIBRATION_MIN_SAMPLES = 50
PROBE_CALIBRATION_SAMPLES = 20_000
CALIBRATION_VERTICES_PER_DRAW = 8
# Reflection probe level relative to the map's median baked light (see _balance_environment).
ENVIRONMENT_MAX_RATIO = 3.0
ENVIRONMENT_SAMPLE_STEP = 16
LUMA = np.array([0.2126, 0.7152, 0.0722])

# Alpha. Neither the materials nor the texture headers flag alpha testing (the
# game compiles it into its shaders), and many textures keep gloss, metal or
# grout masks in alpha (a train texture is 99% "transparent"). So the texture's
# name, its material's name and the shape of its alpha decide together, checked
# against every map's textures by eye:
# * a material that enables transparency blends; the texture's opaque-variant
#   materials (``no_transp``, and the misspelt ``no_tarnsp``) stay opaque;
# * textures with almost no (or almost only) transparent texels stay opaque;
# * open structures, foliage, signs, posters and flags cut out when their alpha
#   is a mask (nearly all texels fully opaque or fully transparent);
# * overlays (decals, blood, graffiti, road paint) and glass blend, and so do
#   the see-through parts of shared atlases, which the game gives their own
#   material (``lights_glass``, ``transparent``, ``decals``);
# * any other texture cuts out when its alpha is a crisp mask whose clear part
#   forms solid regions (cut shapes, unused atlas space) and is the smaller part;
#   thin clear lines are grout or seam masks and a mostly clear alpha marks
#   something else (lit window panes), so those stay opaque;
# * sky domes blend, unlit, behind everything.
TRANSPARENT_SHARE = (0.03, 0.97)
CUTOUT_BINARY = 0.6
GENERIC_CUTOUT_BINARY = 0.9
GENERIC_CUTOUT_SOLID = 0.4
GENERIC_CUTOUT_MAX_SHARE = 0.85
BLEND_NAMES = re.compile(
    r"decal|dirt|blood|leak|stain|puddle|graffiti|crack|moss|soot|scorch|splat|drip|footprint|tyre_track|oil_|"
    r"glass|transp(?!ort)|sticker|marking|stripe|snowflake|^ef_paint|sos_paint"
)
CUTOUT_NAMES = re.compile(
    r"lattice|fence|fenece|net_|_net|grate|grid|wire|cable|chain|railing|plant|leaves|leaf|foliage|grass|branch|"
    r"(?<!s)tree|bush|ivy|vine|farplane|debris|glass_broken|glass_garbage|garbage|cardboard|barbed|mesh_|palm|reed|"
    r"hedge|weed|flower|cactus|shrub|ficus|olive|icicle|scrim|cage|balustrade|curtain|blinds|umbrella|markiz|"
    r"alphabet|font|poster|sign|sighn|sing_|flag(?!stone)|banner|newspaper|(?<!wall)paper|pages|cart_|^pb_|mountains|"
    r"background|_bg_"
)
SKY_NAMES = re.compile(r"^(cloud|sky)|panorama")
# Light effects (cones, rays, flares) are dark textures added onto the scene. The bake often lights
# them like any mesh, which would draw them as dark solids, so these names make glows regardless.
# ``lig_`` is the game's own prefix for lighting-effect textures.
EFFECT_NAMES = re.compile(r"^lig_|light_cone|(^|_)rays?(_|$)|flare|halo(?!gen)|godray|light_shaft")
# Ambiguous names (steel beams) are glows only when the bake leaves them unlit.
GLOW_NAMES = re.compile(r"glow|beam")
# Water surfaces carry only a normal map; the game colours them in the shader.
WATER_NAMES = re.compile(r"water")
OPAQUE_MATERIALS = frozenset(
    {"no_transp", "no_tarnsp", "non_transp", "no_transp_dec", "no_transp_anim", "opaque", "opaque_flat", "no_blend", "blend_none", "em_no_transluc"}
)
BLEND_MATERIALS = re.compile(r"transp(?!ort)|transluc|glass|decal|^blend")
# Material variants with the lights switched off: their emissive map stays dark.
EMISSIVE_OFF_MATERIALS = re.compile(r"(^|[_-])off($|[_-])|no_emissive")
LIT_BY_STAT = {"vertex": "vertex_lit_draws", "lightmap": "lightmap_lit_draws", "probe": "probe_lit_draws", "none": "unlit_draws"}


def _layer(material: dict[str, object], key: str) -> dict[str, object]:
    layer = material.get(key)
    return layer if isinstance(layer, dict) else {}


def _layer_textures(material: dict[str, object]) -> list[str]:
    names = []
    for key in sorted(k for k in material if k.startswith("layer")):
        name = _layer(material, key).get("texName")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _float(value: object, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) and value else default


def _int(value: object, default: int) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _looks_like_normal_map(color: tuple[int, int, int] | None) -> bool:
    """Tangent-space normal maps average to a light blue-violet: x and y near 0, z near 1."""
    return color is not None and color[2] > 160 and abs(color[0] - 128) < 50 and abs(color[1] - 128) < 50


def _kind(name: str, color: tuple[int, int, int] | None, lit_by: str = "vertex") -> str:
    """'blocker' (invisible in-game), 'sky' (unlit dome), 'glow' (additive light effect: cones,
    rays, flares), 'water' (its texture is a normal map) or 'surface'."""
    if name == BLOCKER_TEXTURE:
        return "blocker"
    if SKY_NAMES.search(name):
        return "sky"
    if EFFECT_NAMES.search(name) or (lit_by == "none" and GLOW_NAMES.search(name)):
        return "glow"
    if WATER_NAMES.search(name) and _looks_like_normal_map(color):
        return "water"
    return "surface"


def _material_name(material: dict[str, object]) -> str:
    """The texture's named material variant this draw uses (``default``, ``no_transp``, ``lights_glass``...)."""
    name = _layer(material, "layer0").get("mtlName")
    return name if isinstance(name, str) else ""


def _alpha_mode(material: dict[str, object], name: str, profile: AlphaProfile | None) -> str:
    """'blend', 'cutout' or 'opaque': how the viewer should use the texture's alpha (rules above)."""
    extra = material.get("extraParams")
    transparency = extra.get("transparency") if isinstance(extra, dict) else None
    if isinstance(transparency, dict) and transparency.get("enabled"):
        return "blend"
    variant = _material_name(material)
    if variant in OPAQUE_MATERIALS or profile is None:
        return "opaque"
    if not TRANSPARENT_SHARE[0] < profile.transparent < TRANSPARENT_SHARE[1]:
        return "opaque"
    if SKY_NAMES.search(name):
        return "blend"
    if CUTOUT_NAMES.search(name) and profile.binary > CUTOUT_BINARY:
        return "cutout"
    if BLEND_NAMES.search(name) or BLEND_MATERIALS.search(variant):
        return "blend"
    if (
        profile.binary >= GENERIC_CUTOUT_BINARY
        and profile.solid >= GENERIC_CUTOUT_SOLID
        and profile.transparent <= GENERIC_CUTOUT_MAX_SHARE
    ):
        return "cutout"
    return "opaque"


def _planar_uv(vertices: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Fallback texture coordinates for a draw without UVs: project along each vertex's main normal axis."""
    axis = np.argmax(np.abs(normals), axis=1)
    u = np.where(axis == 0, vertices[:, 2], vertices[:, 0])
    v = np.where(axis == 1, vertices[:, 2], vertices[:, 1])
    return np.stack([u, -v], axis=1) / 2.0


class _TextureFacts:
    """Per-texture lookups (existence, companions, colour, alpha), each computed once."""

    def __init__(self, textures: TextureLibrary) -> None:
        self.textures = textures
        self._color: dict[str, tuple[int, int, int] | None] = {}
        self._profile: dict[str, tuple[float, float] | None] = {}

    def known(self, name: object) -> str | None:
        return name if isinstance(name, str) and name and name in self.textures else None

    def companion(self, name: str | None, suffix: str) -> str | None:
        return self.known(f"{name}{suffix}") if name else None

    def color(self, name: str | None) -> tuple[int, int, int] | None:
        if not name:
            return None
        if name not in self._color:
            self._color[name] = self.textures.average_color(name) if self.known(name) else None
        return self._color[name]

    def profile(self, name: str) -> tuple[float, float] | None:
        if name not in self._profile:
            self._profile[name] = self.textures.alpha_profile(name) if self.known(name) else None
        return self._profile[name]


def _layer_record(layer: dict[str, object], facts: _TextureFacts, base: bool) -> dict[str, object]:
    """What the viewer needs to draw one material layer."""
    blending = _layer(layer, "blending")
    texture = facts.known(layer.get("texName"))
    heightmap = None
    if blending.get("useHeightmap"):
        heightmap = facts.known(blending.get("heightmapOverride")) or facts.companion(texture, "_hdetm")
    # Tints are stored alpha first (A, R, G, B; alpha is 255): painted steel reads grey-green this way.
    tint = layer.get("tint")
    if isinstance(tint, list) and len(tint) == 4:
        tint = [float(c) / 255.0 for c in (tint[1], tint[2], tint[3], tint[0])]
    else:
        tint = [1.0, 1.0, 1.0, 1.0]
    uv_set = _int(layer.get("uvSetIdx"), 0)
    return {
        "texture": texture,
        "normal_map": facts.companion(texture, "_nm"),
        "tiling": [_float(layer.get("tilingU"), 1.0), _float(layer.get("tilingV"), 1.0)],
        "uv_set": uv_set if uv_set in EXTRA_UV_SETS else 0,
        "method": "base" if base else str(blending.get("method") or "common"),
        "channel": _int(_layer(blending, "weights").get("colorSetIdx"), -1),
        "use_alpha": bool(blending.get("useLayerAlpha")),
        "multiplier": _float(blending.get("weightMultiplier"), 1.0),
        "heightmap": heightmap,
        "softness": _float(blending.get("heightmapSoftness"), 1.0),
        "tint": [round(c, 4) for c in tint] + [1.0] * (4 - len(tint)),
    }


@dataclass
class _Part:
    """One draw's vertex data, in export types."""

    positions: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    uvs: np.ndarray
    colours: np.ndarray
    uv_sets: dict[int, np.ndarray]
    rnm: np.ndarray  # (3, n, 3) linear light along the three basis directions
    direct: np.ndarray
    direct_dir: np.ndarray
    lightmap_uvs: np.ndarray
    triangles: np.ndarray
    probe_lit: bool = False


@dataclass
class _Group:
    record: dict[str, object]
    parts: list[_Part] = field(default_factory=list)


def _baked_light(
    lm: dict[str, object], lighting: BakedLighting, lightmap_uv: np.ndarray | None, n: int
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """How the bake lights a draw of ``n`` vertices, from its material's ``lm`` block.

    Returns ``(lit_by, rnm (3, n, 3), direct (n, 3), direct_dir (n, 3), lightmap_uvs (n, 2))``.
    ``lit_by`` is 'vertex' (the bake's light per vertex fills rnm and direct),
    'lightmap' (the viewer samples the lightmap images at lightmap_uvs) or 'none'
    (rnm stays at UNLIT; the caller may light it from the probes instead).
    """
    rnm = np.full((3, n, 3), UNLIT)
    direct = np.zeros((n, 3))
    direct_dir = np.zeros((n, 3))
    lightmap_uvs = np.zeros((n, 2))
    light_set = lighting.get(lm.get("texName"))
    if light_set is None or lightmap_uv is None:
        return "none", rnm, direct, direct_dir, lightmap_uvs
    if lm.get("source") == "vertex":
        direct, direct_dir = light_set.direct_at(lightmap_uv)
        return "vertex", light_set.rnm_at(lightmap_uv).astype(np.float64), direct, direct_dir, lightmap_uvs
    if lm.get("source") == "tex":
        return "lightmap", rnm, direct, direct_dir, lightmap_uv
    return "none", rnm, direct, direct_dir, lightmap_uvs


def _balance_environment(environment: np.ndarray | None, groups: dict[str, _Group]) -> tuple[np.ndarray | None, float]:
    """The reflection probe scaled down to the map's own light level, and the scale used.

    The game reflects a nearby probe, which matches its surroundings; the viewer
    has one probe per map, the most open-sky one, and on a night map that one
    sits under a street lamp (jer_03_pve: 120 times brighter than the median
    baked surface, against 0.3-2.2 on day maps). Capping the probe's median at
    ENVIRONMENT_MAX_RATIO times the median baked light leaves day maps as they
    are and keeps night maps from reflecting daylight.
    """
    if environment is None:
        return None, 1.0
    samples = [
        part.rnm.mean(axis=0)[::ENVIRONMENT_SAMPLE_STEP]
        for group in groups.values()
        if group.record["lighting"] == "vertex"
        for part in group.parts
    ]
    if not samples:
        return environment, 1.0
    baked = float(np.median(np.concatenate(samples) @ LUMA))
    probe = float(np.median(environment.reshape(-1, 3) @ LUMA))
    if baked <= 0.0 or probe <= 0.0:
        return environment, 1.0
    scale = min(1.0, ENVIRONMENT_MAX_RATIO * baked / probe)
    return environment * scale, round(scale, 5)


def _probe_scale(
    probes: light_probes.LightProbes, points: list[np.ndarray], normals: list[np.ndarray], light: list[np.ndarray]
) -> float:
    """Ratio of baked light to probe irradiance at baked vertices near probes (log-median of luminances)."""
    if len(probes) == 0 or not points:
        return DEFAULT_PROBE_SCALE
    p, n, baked = np.concatenate(points), np.concatenate(normals), np.concatenate(light)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(p), size=min(PROBE_CALIBRATION_SAMPLES, len(p)), replace=False)
    p, n, baked = p[pick], n[pick], baked[pick]
    distance = probes.nearest(p)[0][:, 0]
    for radius in PROBE_CALIBRATION_RADII:
        near = distance < radius
        if near.sum() < PROBE_CALIBRATION_MIN_SAMPLES:
            continue
        predicted = probes.irradiance(p[near], n[near], neighbours=1) @ LUMA
        measured = baked[near] @ LUMA
        ok = (predicted > 1e-4) & (measured > 1e-4)
        if ok.sum() >= PROBE_CALIBRATION_MIN_SAMPLES:
            return float(np.exp(np.median(np.log(measured[ok] / predicted[ok]))))
    return DEFAULT_PROBE_SCALE


def export_geometry(
    catalog: Catalog,
    map_name: str,
    folder: Path,
    textures: TextureLibrary,
    side: str = "client",
) -> dict[str, object]:
    """Decode and write one map's geometry, materials and lighting; returns the manifest layer."""
    with catalog.open(map_name) as paks:
        lg = paks.read(paks.resolve(f"{map_name}.lg", side))
        data = paks.read(paks.resolve(f"{map_name}.lg_data", side))
        boxes = obb.decode_bytes(paks.read(paks.resolve(f"{map_name}.obb", side)))
        lighting = BakedLighting.load(paks, map_name, side)
        try:
            probes = light_probes.decode(paks.read(paks.resolve(f"{map_name}.light_probe2", side)))
        except (PakError, SerFormatError):
            probes = light_probes.LightProbes(np.zeros((0, 3)), np.zeros((0, 3, 9)))
        environment = reflections.representative_environment(paks, map_name, side)
    tables = level_geometry.parse_tables(lg)
    starts = level_geometry.solve_layout(tables, data)
    facts = _TextureFacts(textures)
    remove_geometry_files(folder)

    groups: dict[str, _Group] = {}
    seen: set[bytes] = set()
    stats: dict[str, float] = {"draws": 0, "skipped_draws": 0, "lod_draws_skipped": 0, "duplicate_draws": 0}
    stats.update({"spike_triangles": 0, "draws_without_uvs": 0} | {name: 0 for name in LIT_BY_STAT.values()})
    scales: list[float] = []
    calibration: tuple[list, list, list] = ([], [], [])
    for index, draw in enumerate(tables.draws):
        if not draw.lod_mask & 1:
            stats["lod_draws_skipped"] += 1
            continue
        material = tables.materials[index]
        lm = _layer(material, "lm")
        layer_keys = sorted(k for k in material if re.fullmatch(r"layer\d", k))[:MAX_LAYERS]
        layers = [_layer(material, key) for key in layer_keys] or [{}]
        extra_sets = tuple(sorted({_int(layer.get("uvSetIdx"), 0) for layer in layers[1:]} & set(EXTRA_UV_SETS)))
        lightmap_set = lm.get("uvSetIdx")
        try:
            vertices, triangles = level_geometry.draw_mesh(tables, starts, data, index)
            normals = level_geometry.draw_normals(tables, starts, data, index)
            attributes = level_geometry.draw_attributes(
                tables, starts, data, index, lightmap_set if isinstance(lightmap_set, int) else None, extra_sets
            )
        except (SerFormatError, KeyError, ValueError):
            stats["skipped_draws"] += 1
            continue
        if len(triangles) == 0 or not np.isfinite(vertices).all():
            stats["skipped_draws"] += 1
            continue

        # UV set 0: compressed sets need the per-draw scale, float sets are exact.
        if attributes.uv_raw is not None:
            scale = level_geometry.uv_scale(vertices, triangles, attributes.uv_raw, draw.uv_density)
            scales.append(scale)
            uv = attributes.uv_raw / level_geometry.UV_RANGE * scale
        elif attributes.uv is not None:
            uv = attributes.uv
        else:
            stats["draws_without_uvs"] += 1
            uv = _planar_uv(vertices, normals)

        a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
        longest = np.maximum.reduce([np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1)])
        spikes = longest > max(SPIKE_MIN_EDGE, SPIKE_FACTOR * float(np.median(longest)))
        if spikes.any():
            face = np.cross(b[spikes] - a[spikes], c[spikes] - a[spikes])
            face /= np.maximum(np.linalg.norm(face, axis=1), 1e-12)[:, None]
            stored = normals[triangles[spikes]].mean(axis=1)
            spikes[spikes] = np.einsum("ij,ij->i", face, stored) < SPIKE_FACING
        if spikes.any():
            stats["spike_triangles"] += int(spikes.sum())
            triangles = triangles[~spikes]
            if len(triangles) == 0:
                continue

        n = len(vertices)
        lit_by, rnm, direct, direct_dir, lightmap_uvs = _baked_light(lm, lighting, attributes.lightmap_uv, n)
        if lit_by == "vertex":
            # A few vertices from every baked draw, so calibration sees the whole map.
            step = max(1, n // CALIBRATION_VERTICES_PER_DRAW)
            calibration[0].append(vertices[::step])
            calibration[1].append(normals[::step])
            calibration[2].append(lighting.get(lm.get("texName")).light_at(attributes.lightmap_uv[::step], normals[::step]))

        texture_name = layers[0].get("texName")
        name = texture_name if isinstance(texture_name, str) and texture_name else UNTEXTURED
        kind = _kind(name, facts.color(facts.known(name)), lit_by)
        if lit_by == "none" and kind == "surface" and len(probes):
            lit_by = "probe"

        layer_records = [_layer_record(layer, facts, base=k == 0) for k, layer in enumerate(layers)]
        base_texture = layer_records[0]["texture"]
        lights_off = EMISSIVE_OFF_MATERIALS.search(_material_name(material))
        record = {
            "name": name,
            "kind": kind,
            "alpha": _alpha_mode(material, name, facts.profile(name)),
            "lighting": lit_by,
            "layers": layer_records,
            "spec_map": facts.companion(base_texture, "_spec"),
            "emissive_map": None if lights_off else facts.companion(base_texture, "_em"),
        }
        key = json.dumps(record, sort_keys=True)
        # The level stores some meshes several times over (same vertices, same place, other
        # batches; probably one copy per render pass). Draw each only once.
        fingerprint = hashlib.blake2b(
            key.encode() + vertices.astype("<f4").tobytes() + uv.astype("<f4").tobytes() + triangles.astype("<u4").tobytes(),
            digest_size=16,
        ).digest()
        if fingerprint in seen:
            stats["duplicate_draws"] += 1
            continue
        seen.add(fingerprint)

        tangents = attributes.tangents
        part = _Part(
            positions=vertices.astype("<f4"),
            normals=np.clip(np.round(normals * 127.0), -127, 127).astype(np.int8),
            tangents=(
                np.clip(np.round(tangents * 127.0), -127, 127).astype(np.int8)
                if tangents is not None
                else np.zeros((n, 4), dtype=np.int8)
            ),
            uvs=uv.astype("<f4"),
            colours=attributes.colours if attributes.colours is not None else np.zeros((n, 4), dtype=np.uint8),
            uv_sets={s: np.clip(values, -65504.0, 65504.0).astype("<f2") for s, values in attributes.uv_sets.items()},
            rnm=rnm,
            direct=direct,
            direct_dir=direct_dir,
            lightmap_uvs=lightmap_uvs,
            triangles=triangles,
            probe_lit=lit_by == "probe",
        )
        groups.setdefault(key, _Group(record)).parts.append(part)
        stats["draws"] += 1
        stats[LIT_BY_STAT[lit_by]] += 1

    # Probe lighting, in the bake's units.
    probe_scale = _probe_scale(probes, *calibration)
    stats["probe_scale"] = round(probe_scale, 5)
    for group in groups.values():
        for part in group.parts:
            if part.probe_lit:
                points = part.positions.astype(np.float64)
                facing = part.normals.astype(np.float64) / 127.0
                light = probes.irradiance(points, facing) * probe_scale
                part.rnm = np.broadcast_to(light, (3,) + light.shape).copy()

    environment, stats["environment_scale"] = _balance_environment(environment, groups)
    layer = _write_buffers(folder, groups, facts, textures)
    layer.update(_write_lighting_images(folder, lighting, map_name, environment))

    # Quality figures: share of vertices inside the map's own bounding boxes (97% when
    # placement is right) and how well each buffer's content fits the geometry.
    positions = np.concatenate([part.positions for group in groups.values() for part in group.parts]) if groups else np.zeros((0, 3))
    rng = np.random.default_rng(0)
    sample = positions[rng.choice(len(positions), size=min(QUALITY_SAMPLE, len(positions)), replace=False)]
    stats["vertices_inside_boxes"] = round(BoxOracle(boxes).inside_fraction(sample), 4) if len(sample) else 0.0
    quality = level_geometry.layout_quality(tables, starts, data)
    stats["buffer_fit_min"] = min(quality.values()) if quality else 0.0
    if scales:
        values = np.array(scales)
        stats["uv_scale_integer_share"] = round(float(np.mean(values == np.round(values))), 4)
    stats["probes"] = len(probes)
    layer["textures"] = sorted({n for m in tables.materials for n in _layer_textures(m) if n in textures})
    layer["stats"] = stats
    return layer


def remove_geometry_files(folder: Path) -> None:
    """Delete the geometry files an earlier export left in this map's output folder."""
    for pattern in ("geometry_*", "lightmap*", "environment*"):
        for path in folder.glob(pattern):
            if path.is_file():
                path.unlink()


def _write_buffers(
    folder: Path, groups: dict[str, _Group], facts: _TextureFacts, textures: TextureLibrary
) -> dict[str, object]:
    """Concatenate the parts in material order and write every vertex buffer and the index buffer."""
    order = sorted(groups, key=lambda key: (groups[key].record["name"], key))
    columns: dict[str, list[np.ndarray]] = {}

    def add(name: str, values: np.ndarray) -> None:
        columns.setdefault(name, []).append(values)

    groups_out, materials = [], []
    vertex_base = triangle_base = 0
    for material_index, key in enumerate(order):
        group = groups[key]
        count = 0
        for part in group.parts:
            n = len(part.positions)
            add("positions", part.positions)
            add("normals", part.normals)
            add("tangents", part.tangents)
            add("uvs", part.uvs)
            add("colours", part.colours)
            for s in EXTRA_UV_SETS:
                add(f"uv_set{s}", part.uv_sets.get(s, np.zeros((n, 2), dtype="<f2")))
            for k in range(3):
                add(f"rnm{k}", rgbm(part.rnm[k]))
            add("direct", rgbm(part.direct))
            direction = np.clip(np.round(part.direct_dir * 127.0), -127, 127).astype(np.int8)
            add("direct_dir", np.concatenate([direction, np.zeros((n, 1), dtype=np.int8)], axis=1))
            add("lightmap_uvs", np.clip(np.round(part.lightmap_uvs * 65535.0), 0, 65535).astype("<u2"))
            add("indices", (part.triangles + vertex_base).astype("<u4"))
            vertex_base += n
            count += len(part.triangles)
        groups_out.append({"material": material_index, "first_triangle": triangle_base, "triangle_count": count})
        triangle_base += count
        record = dict(group.record)
        texture = record["layers"][0]["texture"]
        color = facts.color(texture)
        size = textures.size(texture) if texture else None
        record.update(
            texture=texture,
            color=list(color) if color else FALLBACK_COLOR,
            texture_size=list(size) if size else None,
            lightmap=record["lighting"] == "lightmap",
            draws=len(group.parts),
        )
        materials.append(record)

    def joined(name: str, dtype: str, width: int) -> np.ndarray:
        parts = columns.get(name, [])
        return np.concatenate(parts) if parts else np.zeros((0, width), dtype=dtype)

    attributes = {
        "positions": write_buffer(folder, "geometry_positions", joined("positions", "<f4", 3), "<f4", 3),
        "normals": write_buffer(folder, "geometry_normals", joined("normals", "i1", 3), "i1", 3, normalized=True),
        "tangents": write_buffer(folder, "geometry_tangents", joined("tangents", "i1", 4), "i1", 4, normalized=True),
        "uvs": write_buffer(folder, "geometry_uvs", joined("uvs", "<f4", 2), "<f4", 2, v_down=True),
        "colours": write_buffer(folder, "geometry_colours", joined("colours", "u1", 4), "u1", 4, normalized=True),
    }
    for s in EXTRA_UV_SETS:
        attributes[f"uv_set{s}"] = write_buffer(folder, f"geometry_uv_set{s}", joined(f"uv_set{s}", "<f2", 2), "<f2", 2)
    for name in ("rnm0", "rnm1", "rnm2", "direct"):
        attributes[name] = write_buffer(folder, f"geometry_{name}", joined(name, "u1", 4), "u1", 4, normalized=True, encoding="rgbm")
    attributes["direct_dir"] = write_buffer(folder, "geometry_direct_dir", joined("direct_dir", "i1", 4), "i1", 4, normalized=True)
    attributes["lightmap_uvs"] = write_buffer(
        folder, "geometry_lightmap_uvs", joined("lightmap_uvs", "<u2", 2), "<u2", 2, normalized=True
    )
    indices = joined("indices", "<u4", 3)
    return {
        "format": GEOMETRY_FORMAT,
        "vertex_count": int(vertex_base),
        "triangle_count": int(len(indices)),
        "attributes": attributes,
        "indices": write_buffer(folder, "geometry_indices", indices, "<u4", 3),
        "rgbm_range": RGBM_RANGE,
        "groups": groups_out,
        "materials": materials,
    }


def _write_lighting_images(
    folder: Path, lighting: BakedLighting, map_name: str, environment: np.ndarray | None
) -> dict[str, object]:
    """The lightmap atlas as PNGs and the reflection environment as float16 RGBA."""
    out: dict[str, object] = {"lightmap": None, "environment": None}
    lightmap_set = lighting.get(f"{map_name}_tex")
    if lightmap_set is not None:
        images = {}
        for name, pixels in lightmap_set.textures().items():
            write_atomic(folder / f"lightmap_{name}.png", png_bytes(pixels))
            images[name] = f"lightmap_{name}.png"
        width, height = lightmap_set.size
        out["lightmap"] = {"width": width, "height": height, "images": images}
    if environment is not None:
        rgba = np.concatenate([environment, np.ones(environment.shape[:2] + (1,), dtype=environment.dtype)], axis=2)
        entry = write_buffer(folder, "environment", np.clip(rgba, 0.0, 65504.0), "<f2", 4)
        entry.update(width=int(environment.shape[1]), height=int(environment.shape[0]), projection="equirectangular")
        out["environment"] = entry
    return out
