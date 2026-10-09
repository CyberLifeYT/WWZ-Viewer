/**
 * Load exported map data (see schema/map_data.schema.json and schema/maps_index.schema.json).
 *
 * Pure data access: returns typed arrays and plain objects, with no rendering code,
 * so the same loader can feed other front ends later.
 */

const INDEX_SCHEMA = 'wwz-mapviewer/maps_index';
const MAP_SCHEMA = 'wwz-mapviewer/map_data';
const SUPPORTED_VERSION = 1;

/** Buffers are little-endian on disk; typed arrays use the platform's byte order. */
const PLATFORM_IS_LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/** @param {string} dataRoot  folder holding index.json, relative to the page or absolute */
function folderUrl(dataRoot) {
  return new URL(dataRoot.endsWith('/') ? dataRoot : `${dataRoot}/`, window.location.href);
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${url.pathname}: HTTP ${response.status}`);
  return response.json();
}

/** True when ``buffer`` is still a gzip file (it starts with the gzip signature 1f 8b). */
function isGzip(buffer) {
  const head = new Uint8Array(buffer, 0, Math.min(2, buffer.byteLength));
  return head.length === 2 && head[0] === 0x1f && head[1] === 0x8b;
}

async function gunzip(buffer) {
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

async function fetchBuffer(url, expectedBytes) {
  // Default caching: serve.py forbids it (re-exports show up at once); a website allows it.
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url.pathname}: HTTP ${response.status}`);
  let buffer = await response.arrayBuffer();
  // serve.py sends the *.bin.gz buffers with Content-Encoding: gzip, so the browser hands
  // them over inflated. Static hosts (GitHub Pages) send the file as it is: inflate it here.
  if (buffer.byteLength !== expectedBytes && isGzip(buffer)) buffer = await gunzip(buffer);
  if (buffer.byteLength !== expectedBytes) {
    throw new Error(`${url.pathname}: expected ${expectedBytes} bytes, got ${buffer.byteLength}`);
  }
  return buffer;
}

function checkSchema(document, schema, url) {
  if (document.schema !== schema || document.schema_version !== SUPPORTED_VERSION) {
    throw new Error(
      `${url.pathname}: unsupported format ${document.schema} v${document.schema_version}; ` +
        `this viewer reads ${schema} v${SUPPORTED_VERSION}. Re-export with the current extractor.`,
    );
  }
}

/**
 * @typedef {object} MapEntry
 * @property {string} map  scene name, e.g. "ny_01_pvp"
 * @property {string} path  manifest path relative to the data root
 * @property {number} box_count
 * @property {number} group_count
 */

/**
 * List the exported maps.
 * @param {string} dataRoot
 * @returns {Promise<MapEntry[]>}
 */
export async function loadIndex(dataRoot) {
  const url = new URL('index.json', folderUrl(dataRoot));
  const index = await fetchJson(url);
  checkSchema(index, INDEX_SCHEMA, url);
  return index.maps;
}

/**
 * @typedef {object} BoxData
 * @property {number} count  number of boxes
 * @property {number} groupCount
 * @property {number} stride  floats per box (15)
 * @property {Float32Array} boxes  per box: origin xyz, axis rows (9), extents xyz
 * @property {Uint32Array} groups  group index per box
 * @property {{min: number[], max: number[], p1: number[], p10: number[], p50: number[], p90: number[], p99: number[]}} bounds
 *   per-axis extremes and percentiles of box centres
 */

/**
 * @typedef {object} MaterialLayer
 * @property {string | null} texture  colour texture (null when the game does not ship it)
 * @property {string | null} normal_map  its "_nm" companion
 * @property {number[]} tiling  u, v repeat
 * @property {number} uv_set  0 = the main UVs, 1-3 = uvSet1..3
 * @property {'base' | 'common' | 'dirt' | 'dirt_no_spec'} method  how the layer blends over those below
 * @property {number} channel  vertex colour channel (0-3) weighting the layer, -1 for none
 * @property {boolean} use_alpha  the layer texture's alpha also weighs it
 * @property {number} multiplier  weight multiplier
 * @property {string | null} heightmap  heightmap sharpening the blend edge
 * @property {number} softness  heightmap blend softness
 * @property {number[]} tint  rgba multiplier, 0..1
 */

/**
 * @typedef {object} GeometryMaterial
 * @property {string} name  the material's main texture name, e.g. "bricks_01"
 * @property {'surface' | 'blocker' | 'sky' | 'glow' | 'water'} kind  blockers are invisible in-game
 * @property {string | null} texture  main texture when the game ships it
 * @property {MaterialLayer[]} layers  base layer first
 * @property {string | null} spec_map  specular mask (reflections)
 * @property {string | null} emissive_map  glow texture
 * @property {number[]} color  the main texture's average sRGB colour, 0..255
 * @property {'opaque' | 'cutout' | 'blend'} alpha  how the main texture's alpha is used
 * @property {'vertex' | 'lightmap' | 'probe' | 'none'} lighting  where the baked light comes from
 */

/**
 * @typedef {object} GeometryData
 * @property {number} vertexCount
 * @property {number} triangleCount
 * @property {Record<string, ArrayBufferView>} attributes  per-vertex buffers by name (see the extractor)
 * @property {Record<string, {components: number, normalized: boolean}>} layouts  how to read each buffer
 * @property {Uint32Array} indices  three per triangle, sorted by material
 * @property {{material: number, first_triangle: number, triangle_count: number}[]} groups
 * @property {GeometryMaterial[]} materials
 * @property {string[]} textures  every texture the map's materials use, for the gallery
 * @property {number} rgbmRange  multiplier of RGBM-encoded light
 * @property {{width: number, height: number, urls: Record<string, string>} | null} lightmap  lightmap images
 * @property {{data: Uint16Array, width: number, height: number} | null} environment
 *   the reflection probe, equirectangular float16 rgba, row 0 looking down
 * @property {Record<string, number | boolean>} stats
 */

/** Geometry layer format this viewer reads; older exports need a re-export. */
export const GEOMETRY_FORMAT = 2;

const TYPED_ARRAYS = {
  float32: Float32Array,
  float16: Uint16Array, // raw half-float bits, read by THREE.Float16BufferAttribute
  int8: Int8Array,
  uint8: Uint8Array,
  uint16: Uint16Array,
  uint32: Uint32Array,
};

function typedArray(entry, buffer) {
  const Type = TYPED_ARRAYS[entry.dtype];
  if (!Type) throw new Error(`unknown buffer type ${entry.dtype}`);
  return new Type(buffer);
}

/**
 * Load the real-geometry buffers described by a manifest's geometry layer.
 * @param {object} layer  manifest.layers.geometry
 * @param {URL} manifestUrl
 * @returns {Promise<GeometryData>}
 */
async function loadGeometry(layer, manifestUrl) {
  if (layer.format !== GEOMETRY_FORMAT) {
    throw new Error('this map was exported by an older extractor; re-export it (export_all_maps.bat)');
  }
  const names = Object.keys(layer.attributes);
  const [indexBuffer, environmentBuffer, ...buffers] = await Promise.all([
    fetchBuffer(new URL(layer.indices.uri, manifestUrl), layer.triangle_count * 3 * 4),
    layer.environment
      ? fetchBuffer(new URL(layer.environment.uri, manifestUrl), layer.environment.width * layer.environment.height * 4 * 2)
      : Promise.resolve(null),
    ...names.map((name) => {
      const entry = layer.attributes[name];
      const itemBytes = TYPED_ARRAYS[entry.dtype].BYTES_PER_ELEMENT;
      return fetchBuffer(new URL(entry.uri, manifestUrl), layer.vertex_count * entry.components * itemBytes);
    }),
  ]);
  const attributes = {};
  const layouts = {};
  names.forEach((name, index) => {
    const entry = layer.attributes[name];
    attributes[name] = typedArray(entry, buffers[index]);
    layouts[name] = { components: entry.components, normalized: Boolean(entry.normalized), dtype: entry.dtype };
  });
  const lightmap = layer.lightmap
    ? {
        width: layer.lightmap.width,
        height: layer.lightmap.height,
        urls: Object.fromEntries(
          Object.entries(layer.lightmap.images).map(([key, uri]) => [key, new URL(uri, manifestUrl).href]),
        ),
      }
    : null;
  return {
    vertexCount: layer.vertex_count,
    triangleCount: layer.triangle_count,
    attributes,
    layouts,
    indices: new Uint32Array(indexBuffer),
    groups: layer.groups,
    materials: layer.materials,
    textures: layer.textures ?? [],
    rgbmRange: layer.rgbm_range ?? 16,
    lightmap,
    environment: environmentBuffer
      ? { data: new Uint16Array(environmentBuffer), width: layer.environment.width, height: layer.environment.height }
      : null,
    stats: layer.stats ?? {},
  };
}

/**
 * Load one map's manifest and buffers.
 * @param {string} dataRoot
 * @param {MapEntry} entry
 * @returns {Promise<{manifest: object, obb: BoxData, geometry: GeometryData | null, geometryError: string | null}>}
 */
export async function loadMap(dataRoot, entry) {
  if (!PLATFORM_IS_LITTLE_ENDIAN) {
    throw new Error('This viewer needs a little-endian platform to read the exported buffers.');
  }
  const manifestUrl = new URL(entry.path, folderUrl(dataRoot));
  const manifest = await fetchJson(manifestUrl);
  checkSchema(manifest, MAP_SCHEMA, manifestUrl);

  const layer = manifest.layers.obb;
  const stride = layer.boxes.components;
  const geometryLayer = manifest.layers.geometry;
  const [boxBuffer, groupBuffer, geometryResult] = await Promise.all([
    fetchBuffer(new URL(layer.boxes.uri, manifestUrl), layer.box_count * stride * 4),
    fetchBuffer(new URL(layer.groups.uri, manifestUrl), layer.box_count * 4),
    // A geometry problem must not hide the boxes: report it and carry on without geometry.
    geometryLayer
      ? loadGeometry(geometryLayer, manifestUrl).then(
          (geometry) => ({ geometry, error: null }),
          (error) => ({ geometry: null, error: error.message }),
        )
      : Promise.resolve({ geometry: null, error: null }),
  ]);

  return {
    manifest,
    geometry: geometryResult.geometry,
    geometryError: geometryResult.error,
    obb: {
      count: layer.box_count,
      groupCount: layer.group_count,
      stride,
      boxes: new Float32Array(boxBuffer),
      groups: new Uint32Array(groupBuffer),
      bounds: layer.center_bounds,
    },
  };
}
