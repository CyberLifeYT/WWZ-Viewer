from __future__ import annotations

import numpy as np
import pytest

from extractor.export import geometry_export
from extractor.export.encoding import unrgbm
from extractor.formats import level_geometry
from extractor.lighting import BakedLighting, LightSet
from extractor.textures import AlphaProfile


def two_by_two() -> LightSet:
    """Texel (0, 0) has direct light from +y, the others only indirect light; (1, 1) is HDR."""
    rnm = np.full((3, 2, 2, 3), 0.25, dtype=np.float32)
    rnm[:, 1, 1] = 3.0  # brighter than the bake's white
    rnm[0, 0, 1] = 0.55  # one basis direction brighter: an unbumped surface sees the mean
    direct_color = np.zeros((2, 2, 3), dtype=np.float32)
    direct_color[0, 0] = (2.0, 1.0, 0.5)
    direction = np.zeros((2, 2, 3), dtype=np.float32)
    direction[0, 0] = (0.0, 1.0, 0.0)
    return LightSet(rnm, direct_color, direction)


def test_lightmap_uvs_pick_the_texel_they_point_into() -> None:
    light_set = two_by_two()
    # u is the column, v the row from the top; u16 / 32768 lands inside the texel.
    uv = np.array([[8, 8], [16392, 8], [8, 16392], [32758, 32758]]) / level_geometry.LIGHTMAP_UV_RANGE
    rows, columns = light_set.texels(uv)
    assert rows.tolist() == [0, 0, 1, 1] and columns.tolist() == [0, 1, 0, 1]


def test_unbumped_light_is_the_basis_mean_plus_direct_light_by_facing() -> None:
    light_set = two_by_two()
    uv = np.zeros((3, 2))
    normals = np.array([(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)])
    light = light_set.light_at(uv, normals)
    assert np.allclose(light[0], (2.25, 1.25, 0.75))  # facing the light
    assert np.allclose(light[1], 0.25)  # edge-on: indirect light only
    assert np.allclose(light[2], 0.25)  # facing away
    beside = light_set.light_at(np.array([[0.75, 0.25]]), np.array([(0.0, 1.0, 0.0)]))
    assert np.allclose(beside, (0.55 + 0.25 + 0.25) / 3)


def test_basis_and_direct_light_per_vertex() -> None:
    light_set = two_by_two()
    uv = np.array([[0.25, 0.25], [0.75, 0.75]])
    rnm = light_set.rnm_at(uv)
    assert rnm.shape == (3, 2, 3) and np.allclose(rnm[:, 1], 3.0)
    color, direction = light_set.direct_at(uv)
    assert np.allclose(color[0], (2.0, 1.0, 0.5)) and np.allclose(direction[0], (0, 1, 0))
    assert np.allclose(color[1], 0) and np.allclose(direction[1], 0)


def test_light_textures_keep_hdr_through_rgbm() -> None:
    images = two_by_two().textures()
    assert set(images) == {"rnm0", "rnm1", "rnm2", "direct", "direct_dir"}
    assert np.allclose(unrgbm(images["rnm0"])[1, 1], 3.0, rtol=0.03)
    assert np.allclose(unrgbm(images["direct"])[0, 0], (2.0, 1.0, 0.5), rtol=0.03)
    assert images["direct_dir"][0, 0].tolist() == [128, 255, 128]  # +y as rgb 0..255 for -1..1


def test_draws_are_lit_per_vertex_by_lightmap_or_not_at_all() -> None:
    lighting = BakedLighting({"map_vc": two_by_two(), "map_tex": two_by_two()})
    uv = np.zeros((2, 2))

    lit_by, rnm, direct, direct_dir, lightmap_uvs = geometry_export._baked_light(
        {"source": "vertex", "texName": "map_vc", "uvSetIdx": 1}, lighting, uv, 2
    )
    assert lit_by == "vertex" and np.allclose(rnm, 0.25) and np.allclose(direct[0], (2.0, 1.0, 0.5))
    assert np.allclose(direct_dir[0], (0, 1, 0)) and np.allclose(lightmap_uvs, 0)

    lit_by, rnm, direct, _, lightmap_uvs = geometry_export._baked_light(
        {"source": "tex", "texName": "map_tex", "uvSetIdx": 1}, lighting, uv, 2
    )
    assert lit_by == "lightmap" and np.allclose(rnm, geometry_export.UNLIT) and lightmap_uvs is uv
    assert np.allclose(direct, 0)

    lit_by, rnm, *_ = geometry_export._baked_light({"source": "none", "texName": ""}, lighting, uv, 2)
    assert lit_by == "none" and np.allclose(rnm, geometry_export.UNLIT)

    lit_by, *_ = geometry_export._baked_light({"source": "vertex", "texName": "map_vc"}, lighting, None, 2)
    assert lit_by == "none"  # no lightmap UVs in the draw


def baked_group(light: float, lighting: str = "vertex") -> geometry_export._Group:
    part = geometry_export._Part(
        positions=np.zeros((64, 3)), normals=np.zeros((64, 3)), tangents=np.zeros((64, 4)), uvs=np.zeros((64, 2)),
        colours=np.zeros((64, 4)), uv_sets={}, rnm=np.full((3, 64, 3), light), direct=np.zeros((64, 3)),
        direct_dir=np.zeros((64, 3)), lightmap_uvs=np.zeros((64, 2)), triangles=np.zeros((0, 3)),
    )
    return geometry_export._Group({"lighting": lighting}, [part])


def test_reflection_probe_is_dimmed_to_the_maps_light_level_only_when_brighter() -> None:
    probe = np.full((8, 16, 3), 0.6)
    night, scale = geometry_export._balance_environment(probe, {"a": baked_group(0.005)})
    assert np.isclose(scale, geometry_export.ENVIRONMENT_MAX_RATIO * 0.005 / 0.6, rtol=1e-3)
    assert np.allclose(night, 0.6 * scale)
    day, scale = geometry_export._balance_environment(probe, {"a": baked_group(0.5)})
    assert scale == 1.0 and np.allclose(day, 0.6)
    # Only the bake counts: probe-lit and unlit draws do not set the level.
    _, scale = geometry_export._balance_environment(probe, {"a": baked_group(0.5), "b": baked_group(9.0, "none")})
    assert scale == 1.0
    assert geometry_export._balance_environment(None, {})[0] is None


def test_sets_are_found_with_or_without_the_map_name() -> None:
    # 12 maps name their sets "_vc" / "_tex"; the files carry the map name.
    lighting = BakedLighting({"camp_02_pve_vc": two_by_two()}, "camp_02_pve")
    assert lighting.get("_vc") is lighting.get("camp_02_pve_vc") is not None
    lit_by, *_ = geometry_export._baked_light({"source": "vertex", "texName": "_vc"}, lighting, np.zeros((1, 2)), 1)
    assert lit_by == "vertex"
    assert lighting.get("_tex") is None and lighting.get("") is None and lighting.get(None) is None


# -- materials ------------------------------------------------------------------------------

PLAIN = {"layer0": {"mtlName": "default"}}
MASK = AlphaProfile(transparent=0.4, binary=0.95, solid=0.7)  # crisp mask with solid clear regions
GRADUAL = AlphaProfile(transparent=0.4, binary=0.2, solid=0.7)  # gloss or height in alpha
SEAMS = AlphaProfile(transparent=0.05, binary=0.98, solid=0.0)  # thin clear lines: grout, panel seams
MOSTLY_CLEAR = AlphaProfile(transparent=0.94, binary=0.96, solid=0.99)  # marks something else (lit panes)


def variant(name: str) -> dict:
    return {"layer0": {"mtlName": name}}


@pytest.mark.parametrize(
    ("material", "name", "profile", "expected"),
    [
        (PLAIN, "lattice_05", MASK, "cutout"),
        (PLAIN, "lattice_05", GRADUAL, "opaque"),  # alpha holds gloss
        (PLAIN, "blood_04", GRADUAL, "blend"),
        (PLAIN, "glass_03", GRADUAL, "blend"),
        (PLAIN, "road_marking_01", MASK, "blend"),  # road paint is an overlay
        (PLAIN, "glass_garbage_01", MASK, "cutout"),  # shards on the ground cut out, not blend
        (PLAIN, "poster_05", AlphaProfile(0.15, 0.79, 0.39), "cutout"),  # torn edges, by name
        (PLAIN, "box_wood_01", MASK, "cutout"),  # any crisp, solid mask: cut shapes or unused atlas space
        (PLAIN, "je_tile_01", SEAMS, "opaque"),  # grout lines must not become holes
        (PLAIN, "windows_13", MOSTLY_CLEAR, "opaque"),
        (PLAIN, "street_lamp_01", AlphaProfile(0.2, 0.79, 0.39), "opaque"),  # "street" is not a tree
        (PLAIN, "wallpaper_03", AlphaProfile(0.3, 0.8, 0.2), "opaque"),  # "paper", but not a sheet of paper
        (PLAIN, "transport_box_01", GRADUAL, "opaque"),  # "transp", but not transparent
        (PLAIN, "lattice_05", AlphaProfile(0.99, 0.99, 0.99), "opaque"),  # nearly all transparent
        (variant("no_transp"), "lattice_05", MASK, "opaque"),
        (variant("no_tarnsp"), "glass_02", GRADUAL, "opaque"),  # the game's own misspelling
        (variant("lights_glass"), "car_sedan_03", GRADUAL, "blend"),  # the atlas's glass part
        (variant("lights"), "car_sedan_03", GRADUAL, "opaque"),
        (variant("wet_blend"), "garbage_01", GRADUAL, "opaque"),  # wetness blending, not alpha
        ({"layer0": {}, "extraParams": {"transparency": {"enabled": 1}}}, "anything", None, "blend"),
    ],
)
def test_alpha_modes(material: dict, name: str, profile: AlphaProfile | None, expected: str) -> None:
    assert geometry_export._alpha_mode(material, name, profile) == expected


def test_alpha_profile_tells_solid_regions_from_thin_lines() -> None:
    alpha = np.full((64, 64), 255, dtype=np.uint8)
    alpha[::8, :] = 0  # one-texel grout lines
    lines = AlphaProfile.of(alpha)
    assert lines.binary == 1.0 and lines.solid == 0.0
    alpha = np.full((64, 64), 255, dtype=np.uint8)
    alpha[16:48, 16:48] = 0  # a cut-out square
    square = AlphaProfile.of(alpha)
    assert square.transparent == 0.25 and square.solid > 0.7


@pytest.mark.parametrize(
    ("name", "off"),
    [("light_off", True), ("lamp_ny_off", True), ("lights-off", True), ("no_emissive", True), ("lightbar_off", True),
     ("lights", False), ("light_on", False), ("office", False), ("default", False)],
)
def test_switched_off_light_variants(name: str, off: bool) -> None:
    assert bool(geometry_export.EMISSIVE_OFF_MATERIALS.search(name)) is off


def test_material_kinds() -> None:
    assert geometry_export._kind("red", None) == "blocker"
    assert geometry_export._kind("cloudspanorama1", (200, 200, 200)) == "sky"
    assert geometry_export._kind("new_graph_water_new_01", (132, 137, 224)) == "water"
    assert geometry_export._kind("tank_water_01", (90, 80, 70)) == "surface"  # a coloured texture, not a normal map
    assert geometry_export._kind("bricks_01", (150, 90, 70)) == "surface"
    # Light effects are glows even when the bake lights them; "beam" and "glow" only when it does not.
    for effect in ("lig_ray_cone1", "lig_rays_down", "light_cone_01", "geo_lamp_rays_01", "jer_03_flare", "moon_halo2"):
        assert geometry_export._kind(effect, (30, 30, 30), "vertex") == "glow", effect
    assert geometry_export._kind("steel_beam_01", (90, 90, 90), "vertex") == "surface"
    assert geometry_export._kind("steel_beam_01", (90, 90, 90), "none") == "glow"
    for solid in ("road_cone_01", "food_trays_01", "spotlight_01", "halogen_lamp_01", "spray_can_01"):
        assert geometry_export._kind(solid, (90, 90, 90), "vertex") == "surface", solid


def test_layer_tints_are_stored_alpha_first() -> None:
    facts = geometry_export._TextureFacts(textures=set())  # no textures: names resolve to None
    record = geometry_export._layer_record({"tint": [255, 64, 128, 32]}, facts, base=True)
    assert record["tint"] == [round(64 / 255, 4), round(128 / 255, 4), round(32 / 255, 4), 1.0]
    assert record["method"] == "base"
