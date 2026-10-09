from __future__ import annotations

import numpy as np
import pytest

from extractor import reflections


@pytest.mark.parametrize(
    ("direction", "face"),
    [((1, 0, 0), 0), ((-1, 0, 0), 1), ((0, 1, 0), 2), ((0, -1, 0), 3), ((0, 0, 1), 4), ((0, 0, -1), 5)],
)
def test_axis_directions_hit_the_centre_of_their_face(direction, face) -> None:
    found, u, v = reflections._face_lookup(np.array([direction], dtype=float))
    assert found[0] == face and np.isclose(u[0], 0.5) and np.isclose(v[0], 0.5)


def test_face_v_runs_down_and_faces_meet_at_their_edges() -> None:
    # On a side face, looking slightly up lands in the upper half (v < 0.5).
    _, _, v = reflections._face_lookup(np.array([(1.0, 0.2, 0.0)]))
    assert v[0] < 0.5
    # Just either side of the +x / +z edge, the two faces sample the texels along their shared edge.
    face, u, _ = reflections._face_lookup(np.array([(1.0, 0.0, 0.999), (0.999, 0.0, 1.0)]))
    assert face.tolist() == [0, 4] and u[0] < 0.01 and u[1] > 0.99


def test_every_cubemap_in_the_maps_lc_folder_is_a_probe(tmp_path, pak_builder) -> None:
    from extractor.catalog import Catalog

    names = [
        "[scene]/lc/lc_prebuild_test_map_lighting_test_map__reflection12_cube.pct",
        "[scene]/lc/lc_prebuild_test_map_lighting__reflection_3_cube.pct",
        "[scene]/lc/lc_prebuild_test_map_reflections__global_reflection_probe_cube.pct",
        "[scene]/lc/lc_prebuild_other_map_lighting__reflection_1_cube.pct",  # not this map
        "[scene]/lm/test_map_vc_cube.pct",  # not a reflection probe
    ]
    pak_builder(tmp_path / "Content" / "client" / "test_map.pak", {name: b"x" for name in names})
    with Catalog(tmp_path / "Content").open("test_map") as paks:
        found = [entry.name for entry in reflections.probe_entries(paks, "test_map")]
    assert found == sorted(names[:3])


def test_equirect_rows_run_from_down_to_up_and_columns_follow_threejs() -> None:
    faces = np.zeros((6, 4, 4, 3), dtype=np.float32)
    for index in range(6):
        faces[index] = index  # each face filled with its own number
    image = reflections.to_equirect(faces, width=64)
    assert image.shape == (32, 64, 3)
    assert np.all(image[0] == 3) and np.all(image[-1] == 2)  # row 0 looks down, the last row up
    middle = image[16]
    # three.js: u = atan2(dir.z, dir.x) / 2pi + 0.5, so +x is at u = 0.5 and +z at u = 0.75.
    assert middle[32, 0] == 0 and middle[48, 0] == 4 and middle[16, 0] == 5 and middle[0, 0] == 1
