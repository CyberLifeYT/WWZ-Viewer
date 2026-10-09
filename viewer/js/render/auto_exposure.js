/**
 * A starting exposure for each map, from its own baked light.
 *
 * The game's exposure curve is not known, and the maps differ enormously:
 * what a camera mostly sees (the area-weighted median of baked light times
 * surface colour) is 0.0075 on the night map jer_03_pve, 0.47 on ny_01_pvp and
 * 4.4 on the sunny kam_02_pve, so one fixed exposure washes out the sunny maps
 * and blacks out the night ones.
 *
 * Measured per surface area, not per vertex: detailed props and interiors
 * have most of the vertices, while the large walls and ground fill most of
 * the view (geo_03_pve's vertices are mostly in dark interiors; by area it is
 * as bright as ny_01_pvp). Like an eye adapting part of the way, the exposure
 * scales with the square root of how far that median is from ny_01_pvp's,
 * where exposure 1 looks right, so night still reads darker than day.
 */

/** Area-weighted median of baked light x surface colour (luminance) on ny_01_pvp. */
export const REFERENCE_BRIGHTNESS = 0.47;
/** 0: no adaptation, 1: every map's median brightness ends up the same. */
const ADAPTATION = 0.5;
export const EXPOSURE_MIN = 0.25;
export const EXPOSURE_MAX = 8;
/** Triangles sampled to measure a map. */
const SAMPLES = 200000;
/** Triangles larger than this (m²) are distant backdrops, not what the camera mostly sees. */
const MAX_TRIANGLE_AREA = 2000;
const LUMA = [0.2126, 0.7152, 0.0722];

/**
 * @param {import('../data/map_loader.js').GeometryData} data
 * @returns {number} exposure for the tone-mapped baked light (1 when the map has no per-vertex bake)
 */
export function suggestExposure(data) {
  const median = medianBrightness(data);
  if (!(median > 0)) return 1;
  const exposure = (REFERENCE_BRIGHTNESS / median) ** ADAPTATION;
  return Math.min(EXPOSURE_MAX, Math.max(EXPOSURE_MIN, exposure));
}

/** sRGB 0..255 average texture colour -> linear luminance. */
function albedo(color) {
  return color.reduce((sum, channel, i) => sum + LUMA[i] * (channel / 255) ** 2.2, 0);
}

/**
 * Area-weighted median of the light reaching vertex-lit surfaces (the three
 * basis colours' mean plus the direct light on the normal) times their
 * texture's average colour, or NaN when the map has no per-vertex bake.
 * @param {import('../data/map_loader.js').GeometryData} data
 */
export function medianBrightness(data) {
  const a = data.attributes;
  if (!a.rnm0 || !a.rnm1 || !a.rnm2 || !a.direct || !a.direct_dir || !a.normals) return NaN;
  const groups = data.groups.filter((group) => {
    const material = data.materials[group.material];
    return material.lighting === 'vertex' && material.kind === 'surface';
  });
  const triangles = groups.reduce((sum, group) => sum + group.triangle_count, 0);
  if (triangles === 0) return NaN;
  const step = Math.max(1, Math.floor(triangles / SAMPLES));
  const p = a.positions;
  const rgbm = (array, vertex, channel) =>
    (array[vertex * 4 + channel] / 255) * (array[vertex * 4 + 3] / 255) * data.rgbmRange;
  const samples = [];
  for (const group of groups) {
    const surface = albedo(data.materials[group.material].color);
    for (let t = group.first_triangle; t < group.first_triangle + group.triangle_count; t += step) {
      const [i, j, k] = [data.indices[t * 3], data.indices[t * 3 + 1], data.indices[t * 3 + 2]];
      const e1 = [0, 1, 2].map((c) => p[j * 3 + c] - p[i * 3 + c]);
      const e2 = [0, 1, 2].map((c) => p[k * 3 + c] - p[i * 3 + c]);
      const area =
        0.5 *
        Math.hypot(e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]);
      if (!(area > 0) || area > MAX_TRIANGLE_AREA) continue;
      const cosine = Math.max(
        0,
        (a.normals[i * 3] * a.direct_dir[i * 4] +
          a.normals[i * 3 + 1] * a.direct_dir[i * 4 + 1] +
          a.normals[i * 3 + 2] * a.direct_dir[i * 4 + 2]) /
          (127 * 127),
      );
      let light = 0;
      for (let c = 0; c < 3; c += 1) {
        const basis = (rgbm(a.rnm0, i, c) + rgbm(a.rnm1, i, c) + rgbm(a.rnm2, i, c)) / 3;
        light += LUMA[c] * (basis + rgbm(a.direct, i, c) * cosine);
      }
      samples.push([light * surface, area]);
    }
  }
  if (samples.length === 0) return NaN;
  samples.sort((x, y) => x[0] - y[0]);
  const half = samples.reduce((sum, sample) => sum + sample[1], 0) / 2;
  let running = 0;
  for (const [value, area] of samples) {
    running += area;
    if (running >= half) return value;
  }
  return samples[samples.length - 1][0];
}
