from __future__ import annotations

import struct

import numpy as np
import pytest

from extractor.formats import light_probes
from extractor.ser import SerFormatError

# Radiance SH with only the constant term set so that radiance is 1 in every direction.
UNIT_RADIANCE = 1.0 / 0.282095


RECORD_EXTRAS = bytes(370 - 9 - 108)  # the record's further fields, up to the game's 370 bytes


def group_bytes(positions: list[tuple[float, float, float]], coefficients: list[np.ndarray], records: int | None = None) -> bytes:
    """One probe group laid out like the game's: mesh, positions bracketed by their count, tag, records."""
    count = len(positions)
    # The tetrahedral mesh: "u32 4" and four probe numbers per tetrahedron. Index lists like
    # these can contain the record marker's bytes by chance (3, 2, 1...), as in rome_03_pve.
    out = struct.pack("<5I", 4, 3, 2, 1, 0) + light_probes.RECORD_MARKER + bytes(3)
    out += struct.pack("<I", count) + b"".join(struct.pack("<3f", *p) for p in positions) + struct.pack("<I", count)
    out += light_probes.GROUP_TAG
    for sh in coefficients[: count if records is None else records]:
        out += light_probes.RECORD_MARKER + np.asarray(sh, dtype="<f4").tobytes() + RECORD_EXTRAS
    return out


def probe_file(*groups: bytes) -> bytes:
    return b"1SER" + bytes(0x40 - 4) + b"".join(groups)


def corners() -> list[tuple[float, float, float]]:
    return [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)]


def ambient(level: float) -> np.ndarray:
    sh = np.zeros((3, 9))
    sh[:, 0] = level * UNIT_RADIANCE
    return sh


def test_decode_reads_positions_and_coefficients() -> None:
    coefficients = [ambient(k + 1.0) for k in range(4)]
    probes = light_probes.decode(probe_file(group_bytes(corners(), coefficients)))
    assert len(probes) == 4
    assert np.allclose(probes.positions, corners())
    assert np.allclose(probes.sh, np.stack(coefficients), rtol=1e-6)


def test_decode_reads_every_group_including_single_and_flat_ones() -> None:
    single = group_bytes([(5.0, 1.0, 5.0)], [ambient(2.0)])
    flat = group_bytes([(0.0, 1.5, 0.0), (3.0, 1.5, 0.0), (0.0, 1.5, 3.0)], [ambient(3.0)] * 3)  # all at one height
    probes = light_probes.decode(probe_file(group_bytes(corners(), [ambient(1.0)] * 4), single, flat))
    assert len(probes) == 8
    assert np.allclose(probes.positions[4], (5.0, 1.0, 5.0)) and np.allclose(probes.positions[5:, 1], 1.5)


def test_decode_skips_a_group_whose_records_do_not_match() -> None:
    broken = group_bytes(corners(), [ambient(1.0)] * 4, records=3)
    probes = light_probes.decode(probe_file(broken, group_bytes([(9.0, 0.0, 9.0)], [ambient(1.0)])))
    assert len(probes) == 1 and np.allclose(probes.positions, [(9.0, 0.0, 9.0)])
    with pytest.raises(SerFormatError, match="not a 1SER"):
        light_probes.decode(b"junk" * 40)


def test_uniform_light_gives_pi_times_radiance_whichever_way_a_surface_faces() -> None:
    probes = light_probes.LightProbes(np.array(corners(), dtype=float), np.stack([ambient(1.0)] * 4))
    normals = np.array([(0, 1, 0), (0, -1, 0), (1, 0, 0), (0, 0, -1)], dtype=float)
    assert np.allclose(probes.irradiance(np.ones((4, 3)), normals), np.pi, rtol=1e-4)


def test_light_from_above_brightens_upward_faces() -> None:
    sh = ambient(1.0)
    sh[:, 1] = 1.0  # band 1, y: more light arriving from +y
    probes = light_probes.LightProbes(np.array(corners(), dtype=float), np.stack([sh] * 4))
    up, down = probes.irradiance(np.zeros((2, 3)), np.array([(0, 1, 0), (0, -1, 0)], dtype=float))
    assert np.all(up > np.pi) and np.all(down < np.pi)


def test_nearer_probes_weigh_more() -> None:
    probes = light_probes.LightProbes(np.array(corners(), dtype=float), np.stack([ambient(k + 1.0) for k in range(4)]))
    near_second = probes.irradiance(np.array([(3.9, 0.0, 0.0)]), np.array([(0.0, 1.0, 0.0)]))
    assert np.allclose(near_second, 2.0 * np.pi, rtol=0.05)


def test_nearest_probes_match_a_brute_force_search() -> None:
    rng = np.random.default_rng(3)
    positions = rng.uniform(-50, 50, size=(500, 3))
    probes = light_probes.LightProbes(positions, np.zeros((500, 3, 9)))
    points = rng.uniform(-60, 60, size=(200, 3))
    distance, index = probes.nearest(points, neighbours=4)
    brute = np.linalg.norm(points[:, None, :] - positions[None, :, :], axis=2)
    assert np.array_equal(np.sort(index, axis=1), np.sort(np.argsort(brute, axis=1)[:, :4], axis=1))
    assert np.allclose(distance, np.sort(brute, axis=1)[:, :4])
    one, _ = probes.nearest(points)
    assert one.shape == (200, 1) and np.allclose(one[:, 0], brute.min(axis=1))


def test_no_probes_means_unlit_by_probes() -> None:
    empty = light_probes.LightProbes(np.zeros((0, 3)), np.zeros((0, 3, 9)))
    assert np.allclose(empty.irradiance(np.zeros((2, 3)), np.array([(0, 1, 0)] * 2, dtype=float)), 1.0)
