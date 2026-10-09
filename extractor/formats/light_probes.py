"""Light probes: ``[scene]/<map>.light_probe2`` (1SER ``light_probe``).

The game lights moving objects (doors, characters) from a cloud of probes,
each holding second-order spherical harmonics of the incoming light. Static
level geometry is lit by the bake instead (see ``extractor.lighting``).

Layout (verified on all 44 maps: 3 to 811 groups, 112 to 61,801 probes):
after a short header the file holds groups, each a tetrahedral mesh over its
probes (records of ``u32 4`` and four probe numbers), then ``u32 n``, ``n``
probe positions (``f32 x, y, z``), ``u32 n`` again, the tag ``02 00 00 00 01``,
then one 370-byte record per probe. Each record starts with the marker
``03 00 00 00 02 00 00 00 01`` followed by 27 floats: the SH coefficients of
red, green and blue (9 each, in the usual real SH order), then further fields
(dominant direction, colours).

Decoding works back from the markers: a marker with the tag before it, and
the same count before the tag and before ``n`` positions further back, opens a
group. On every map this claims all records; the few markers left over are the
same 9 bytes occurring by chance inside index lists and floats. (Scanning for
"count, positions, count" blocks instead also finds such chance patterns.)
"""

from __future__ import annotations

import functools
import re
import struct
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from extractor.ser.errors import SerFormatError

RECORD_MARKER = b"\x03\x00\x00\x00\x02\x00\x00\x00\x01"
GROUP_TAG = b"\x02\x00\x00\x00\x01"  # between a group's second count and its first record
SH_COEFFICIENTS = 9
MAX_GROUP_PROBES = 200_000
MAX_COORDINATE = 1e5  # metres; anything larger is not a probe position
# Real SH basis constants (bands 0-2), and the cosine-lobe convolution per band, so that
# irradiance(n) = sum_l A_l * sum_m c_lm * Y_lm(n). Calibrated against the bake (see tests and docs).
_Y = (0.282095, 0.488603, 1.092548, 0.315392, 0.546274)
_A = (np.pi, 2.0 * np.pi / 3.0, np.pi / 4.0)


@dataclass(frozen=True)
class LightProbes:
    positions: np.ndarray  # (n, 3) world space
    sh: np.ndarray  # (n, 3, 9) RGB coefficients

    def __len__(self) -> int:
        return len(self.positions)

    @functools.cached_property
    def _tree(self) -> cKDTree:
        # Maps hold tens of thousands of probes and light millions of vertices from them:
        # comparing every vertex with every probe would need gigabytes and hours.
        return cKDTree(self.positions)

    def nearest(self, points: np.ndarray, neighbours: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """(distances, probe numbers) of each point's ``neighbours`` nearest probes, each (n, neighbours)."""
        distance, index = self._tree.query(points, k=neighbours)
        if neighbours == 1:
            return distance[:, None], index[:, None]
        return distance, index

    def irradiance(self, points: np.ndarray, normals: np.ndarray, neighbours: int = 4) -> np.ndarray:
        """Light (linear RGB) reaching surfaces at ``points`` facing ``normals``, from the nearest probes.

        Each point blends its ``neighbours`` nearest probes by inverse distance; the
        game interpolates inside the tetrahedral mesh instead, which differs only
        slightly at these probe densities.
        """
        if len(self) == 0 or len(points) == 0:
            return np.ones((len(points), 3))
        distance, nearest = self.nearest(points, min(neighbours, len(self)))
        weight = 1.0 / np.maximum(distance, 0.05) ** 2
        weight /= weight.sum(axis=1, keepdims=True)
        out = np.zeros((len(points), 3))
        for start in range(0, len(points), 65536):  # bounds the (chunk, k, 3, 9) temporary
            rows = slice(start, start + 65536)
            coefficients = np.einsum("pk,pkcj->pcj", weight[rows], self.sh[nearest[rows]])
            out[rows] = evaluate_irradiance(coefficients, normals[rows])
        return np.clip(out, 0.0, None)


def sh_basis(normals: np.ndarray) -> np.ndarray:
    """(n, 9) real SH basis values for unit vectors (x, y, z), y up."""
    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
    return np.stack(
        [
            np.full_like(x, _Y[0]),
            _Y[1] * y,
            _Y[1] * z,
            _Y[1] * x,
            _Y[2] * x * y,
            _Y[2] * y * z,
            _Y[3] * (3.0 * z * z - 1.0),
            _Y[2] * x * z,
            _Y[4] * (x * x - y * y),
        ],
        axis=1,
    )


def evaluate_irradiance(coefficients: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """(n, 3) irradiance from (n, 3, 9) radiance SH coefficients."""
    band = np.array([_A[0]] + [_A[1]] * 3 + [_A[2]] * 5)
    return np.einsum("ncj,nj->nc", coefficients * band, sh_basis(normals))


def _group_before(blob: bytes, marker: int) -> tuple[int, int] | None:
    """(count, positions offset) when the record marker at ``marker`` opens a group, else None.

    A group's records follow ``u32 n, n positions, u32 n, GROUP_TAG``, so the
    first record's marker has the tag right before it and the count, twice,
    around ``n`` positions before that.
    """
    tag_at = marker - len(GROUP_TAG)
    if tag_at < 4 or blob[tag_at:marker] != GROUP_TAG:
        return None
    (count,) = struct.unpack_from("<I", blob, tag_at - 4)
    start = tag_at - 4 - 12 * count - 4
    if not 1 <= count <= MAX_GROUP_PROBES or start < 0 or struct.unpack_from("<I", blob, start)[0] != count:
        return None
    return count, start + 4


def decode(blob: bytes) -> LightProbes:
    """Decode a ``.light_probe2`` file (with its 1SER header).

    Groups whose positions are not finite or whose records are not evenly
    spaced are skipped rather than trusted.
    """
    if blob[:4] != b"1SER":
        raise SerFormatError("not a 1SER resource", offset=0)
    # The marker can also occur by chance inside other data (index lists, floats);
    # only markers inside a group's run of records are records.
    markers = [m.start() for m in re.finditer(re.escape(RECORD_MARKER), blob)]
    positions, sh = [], []
    for k, marker in enumerate(markers):
        group = _group_before(blob, marker)
        if group is None:
            continue
        count, start = group
        records = np.array(markers[k:k + count])
        if len(records) != count or (count > 1 and len(set(np.diff(records).tolist())) != 1):
            continue
        if records[-1] + len(RECORD_MARKER) + 4 * 3 * SH_COEFFICIENTS > len(blob):
            continue
        block = np.frombuffer(blob, dtype="<f4", offset=start, count=3 * count).reshape(count, 3).astype(np.float64)
        if not np.isfinite(block).all() or np.abs(block).max() > MAX_COORDINATE:
            continue
        coefficients = np.stack(
            [np.frombuffer(blob, dtype="<f4", offset=int(m) + len(RECORD_MARKER), count=3 * SH_COEFFICIENTS) for m in records]
        ).astype(np.float64)
        positions.append(block)
        sh.append(coefficients.reshape(count, 3, SH_COEFFICIENTS))
    if not positions:
        return LightProbes(np.zeros((0, 3)), np.zeros((0, 3, SH_COEFFICIENTS)))
    return LightProbes(np.concatenate(positions), np.concatenate(sh))
