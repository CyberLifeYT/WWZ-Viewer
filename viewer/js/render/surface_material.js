/**
 * The level-surface shader: the game's material layers, normal maps and baked
 * light, rebuilt from the exported data.
 *
 * Per pixel:
 * 1. Layers. Layer 0 is the base; each further layer blends over it with a
 *    weight from a vertex colour channel, the layer's alpha, a multiplier, and
 *    optionally a heightmap that sharpens the blend edge. "dirt" layers are
 *    grey grime maps applied as a multiply-by-two overlay (mid grey = no change).
 * 2. Normal. The base layer's normal map (BC5: x and y; z rebuilt), in the
 *    vertex's tangent frame (bitangent = normal x tangent x sign).
 * 3. Light.
 *    - baked: the bake's three "radiosity normal mapping" colours, weighted by
 *      how the bumped normal faces each basis direction (Half-Life 2 basis,
 *      verified on the data), plus the direct light on the bumped normal. The
 *      colours come per vertex or from the lightmap textures. Tone-mapped.
 *    - lit: the viewer's own sky and sun, as three.js's Lambert material does.
 * 4. Reflection of the map's reflection probe, masked by the specular map,
 *    and the emissive map on top.
 */
import * as THREE from 'three';

/** Channel (0 R, 1 G, 2 B, 3 A) -> component of the vertex colour attribute (stored B, G, R, A). */
const CHANNEL_COMPONENT = ['z', 'y', 'x', 'w'];
const METHODS = { base: 0, common: 0, dirt: 1, dirt_no_spec: 1 };
const REFLECTION_STRENGTH = 0.6;
const EMISSIVE_STRENGTH = 1.5;
/** An "_em" texture is a light map (window lights, lamp glass) when this share of it is dark... */
const LIGHT_MAP_DARK_SHARE = 0.5;
/** ...or when it is this dark on average (linear luminance). */
const LIGHT_MAP_MEAN = 0.1;

/** @param {{mean: number, darkShare: number}} stats  brightness of an "_em" texture (see TextureCache) */
function isLightMap(stats) {
  return stats.darkShare >= LIGHT_MAP_DARK_SHARE || stats.mean < LIGHT_MAP_MEAN;
}
const WATER_COLOR = new THREE.Color(0x2f4654);

const VERTEX_SHADER = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_vertex>

attribute vec4 aTangent;
attribute vec4 aColours;
attribute vec2 aUvSet1;
attribute vec2 aUvSet2;
attribute vec2 aUvSet3;
attribute vec4 aRnm0;
attribute vec4 aRnm1;
attribute vec4 aRnm2;
attribute vec4 aDirect;
attribute vec4 aDirectDir;
attribute vec2 aLightmapUv;

uniform float rgbmRange;

varying vec2 vUv0;
varying vec2 vUv1;
varying vec2 vUv2;
varying vec2 vUv3;
varying vec2 vLightmapUv;
varying vec4 vColours;
varying vec3 vNormalW;
varying vec4 vTangentW;
varying vec3 vPositionW;
varying vec3 vRnm0;
varying vec3 vRnm1;
varying vec3 vRnm2;
varying vec3 vDirect;
varying vec3 vDirectDir;

vec3 decodeRgbm(vec4 c) { return c.rgb * c.a * rgbmRange; }

void main() {
  vUv0 = uv;
  vUv1 = aUvSet1;
  vUv2 = aUvSet2;
  vUv3 = aUvSet3;
  vLightmapUv = aLightmapUv;
  vColours = aColours;
  vec4 worldPosition = modelMatrix * vec4(position, 1.0);
  vPositionW = worldPosition.xyz;
  vNormalW = normalize(mat3(modelMatrix) * normal);
  vTangentW = vec4(normalize(mat3(modelMatrix) * aTangent.xyz + 1e-6), aTangent.w);
  vRnm0 = decodeRgbm(aRnm0);
  vRnm1 = decodeRgbm(aRnm1);
  vRnm2 = decodeRgbm(aRnm2);
  vDirect = decodeRgbm(aDirect);
  vDirectDir = aDirectDir.xyz;
  gl_Position = projectionMatrix * viewMatrix * worldPosition;
  #include <logdepthbuf_vertex>
}
`;

const FRAGMENT_SHADER = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_fragment>

uniform sampler2D layerMaps[LAYER_COUNT];
uniform vec2 layerTiling[LAYER_COUNT];
uniform vec4 layerTint[LAYER_COUNT];
uniform float layerMultiplier[LAYER_COUNT];
uniform float layerSoftness[LAYER_COUNT];
#if HEIGHTMAP_COUNT > 0
uniform sampler2D heightMaps[HEIGHTMAP_COUNT];
#endif
uniform sampler2D normalMap;
uniform sampler2D specMap;
uniform sampler2D emissiveMap;
uniform sampler2D lightmapRnm0;
uniform sampler2D lightmapRnm1;
uniform sampler2D lightmapRnm2;
uniform sampler2D lightmapDirect;
uniform sampler2D lightmapDirectDir;
uniform sampler2D environmentMap;
uniform float rgbmRange;
uniform float textured;
uniform vec3 averageColor;
uniform vec3 waterColor;
uniform vec3 skyColor;
uniform vec3 groundColor;
uniform vec3 sunColor;
uniform vec3 sunDirection;
uniform float reflectionStrength;
uniform float emissiveStrength;

varying vec2 vUv0;
varying vec2 vUv1;
varying vec2 vUv2;
varying vec2 vUv3;
varying vec2 vLightmapUv;
varying vec4 vColours;
varying vec3 vNormalW;
varying vec4 vTangentW;
varying vec3 vPositionW;
varying vec3 vRnm0;
varying vec3 vRnm1;
varying vec3 vRnm2;
varying vec3 vDirect;
varying vec3 vDirectDir;

const vec3 RNM_BASIS_0 = vec3(0.81649658, 0.0, 0.57735027);
const vec3 RNM_BASIS_1 = vec3(-0.40824829, 0.70710678, 0.57735027);
const vec3 RNM_BASIS_2 = vec3(-0.40824829, -0.70710678, 0.57735027);

vec3 decodeRgbmTexel(vec4 c) { return c.rgb * c.a * rgbmRange; }

vec3 sampleEnvironment(vec3 direction) {
  vec2 uv = vec2(atan(direction.z, direction.x) * RECIPROCAL_PI2 + 0.5, asin(clamp(direction.y, -1.0, 1.0)) * RECIPROCAL_PI + 0.5);
  return texture2D(environmentMap, uv).rgb;
}

float heightBlend(float weight, float height, float softness) {
  float edge = max(softness, 0.05) * 0.5;
  return smoothstep(1.0 - weight - edge, 1.0 - weight + edge, height);
}

void main() {
  #include <logdepthbuf_fragment>

  // 1. Layers.
  vec4 base = texture2D(layerMaps[0], vUv0 * layerTiling[0]) * layerTint[0];
  ${'' /* further layers are generated per material: LAYER_CODE */}
  LAYER_CODE
  if (textured < 0.5) base.rgb = averageColor;
  #ifdef WATER
    base = vec4(waterColor, 1.0);
  #endif
  #ifdef ALPHA_CUTOUT
    if (base.a < 0.5) discard;
  #endif

  // 2. Normal.
  vec3 normal = normalize(vNormalW) * (gl_FrontFacing ? 1.0 : -1.0);
  vec3 bumped = vec3(0.0, 0.0, 1.0);
  #if defined(USE_NORMAL_MAP) || defined(WATER)
    #ifdef WATER
      vec2 xy = texture2D(layerMaps[0], vUv0 * layerTiling[0]).xy * 2.0 - 1.0;
    #else
      vec2 xy = texture2D(normalMap, vUv0 * layerTiling[0]).xy * 2.0 - 1.0;
    #endif
    bumped = vec3(xy, sqrt(max(0.0, 1.0 - dot(xy, xy))));
  #endif
  vec3 tangent = vTangentW.xyz - normal * dot(normal, vTangentW.xyz);
  float tangentLength = length(tangent);
  vec3 normalW = normal;
  if (tangentLength > 1e-3) {
    tangent /= tangentLength;
    vec3 bitangent = cross(normal, tangent) * (vTangentW.w < 0.0 ? -1.0 : 1.0);
    normalW = normalize(tangent * bumped.x + bitangent * bumped.y + normal * bumped.z);
  } else {
    bumped = vec3(0.0, 0.0, 1.0);
  }

  // 3. Light.
  vec3 light;
  #if defined(SHADING_LIT)
    float up = 0.5 * normalW.y + 0.5;
    light = (mix(groundColor, skyColor, up) + sunColor * max(dot(normalW, sunDirection), 0.0)) * RECIPROCAL_PI;
  #elif defined(LIGHTING_VERTEX) || defined(LIGHTING_LIGHTMAP)
    #ifdef LIGHTING_VERTEX
      vec3 rnm0 = vRnm0, rnm1 = vRnm1, rnm2 = vRnm2, direct = vDirect, directDir = vDirectDir;
    #else
      vec3 rnm0 = decodeRgbmTexel(texture2D(lightmapRnm0, vLightmapUv));
      vec3 rnm1 = decodeRgbmTexel(texture2D(lightmapRnm1, vLightmapUv));
      vec3 rnm2 = decodeRgbmTexel(texture2D(lightmapRnm2, vLightmapUv));
      vec3 direct = decodeRgbmTexel(texture2D(lightmapDirect, vLightmapUv));
      vec3 directDir = texture2D(lightmapDirectDir, vLightmapUv).xyz * 2.0 - 1.0;
    #endif
    vec3 dp = clamp(vec3(dot(bumped, RNM_BASIS_0), dot(bumped, RNM_BASIS_1), dot(bumped, RNM_BASIS_2)), 0.0, 1.0);
    dp *= dp;
    light = (dp.x * rnm0 + dp.y * rnm1 + dp.z * rnm2) / max(dp.x + dp.y + dp.z, 1e-4);
    float directLength = length(directDir);
    if (directLength > 0.5) light += direct * max(dot(normalW, directDir / directLength), 0.0);
  #else
    light = vec3(1.0);
  #endif
  vec3 color = base.rgb * light;

  // 4. Reflection and glow.
  #if defined(USE_ENVIRONMENT) && (defined(USE_SPEC) || defined(WATER))
    vec3 view = normalize(vPositionW - cameraPosition);
    vec3 reflected = reflect(view, normalW);
    float fresnel = 0.04 + 0.96 * pow(1.0 - max(dot(-view, normalW), 0.0), 5.0);
    #ifdef WATER
      color = mix(color, sampleEnvironment(reflected), clamp(0.2 + fresnel, 0.0, 1.0));
    #else
      float mask = texture2D(specMap, vUv0 * layerTiling[0]).r;
      color += sampleEnvironment(reflected) * mask * fresnel * reflectionStrength;
    #endif
  #endif
  #ifdef USE_EMISSIVE
    color += texture2D(emissiveMap, vUv0 * layerTiling[0]).rgb * emissiveStrength;
  #endif

  #ifdef ALPHA_BLEND
    gl_FragColor = vec4(color, base.a);
  #else
    gl_FragColor = vec4(color, 1.0);
  #endif
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

/** GLSL for layer ``index`` (1-3) of ``layer``, blending into ``base``. */
function layerCode(layer, index, heightmapSlot) {
  const uv = ['vUv0', 'vUv1', 'vUv2', 'vUv3'][layer.uv_set] ?? 'vUv0';
  const channel = layer.channel >= 0 && layer.channel < 4 ? `vColours.${CHANNEL_COMPONENT[layer.channel]}` : '1.0';
  const lines = [
    `{`,
    `  vec4 texel = texture2D(layerMaps[${index}], ${uv} * layerTiling[${index}]) * layerTint[${index}];`,
    `  float weight = layerMultiplier[${index}] * ${channel};`,
  ];
  if (layer.use_alpha) lines.push(`  weight *= texel.a;`);
  if (heightmapSlot !== null) {
    lines.push(
      `  weight = heightBlend(weight, texture2D(heightMaps[${heightmapSlot}], ${uv} * layerTiling[${index}]).r, layerSoftness[${index}]);`,
    );
  }
  lines.push(`  weight = clamp(weight, 0.0, 1.0);`);
  if (METHODS[layer.method] === 1) {
    // Grime maps: mid grey leaves the base unchanged (multiply by two in gamma space).
    lines.push(`  base.rgb *= mix(vec3(1.0), pow(texel.rgb, vec3(1.0 / 2.2)) * 2.0, weight);`);
  } else {
    lines.push(`  base.rgb = mix(base.rgb, texel.rgb, weight);`);
  }
  lines.push(`}`);
  return lines.join('\n  ');
}

/**
 * Build the shader material of one exported material.
 * @param {import('../data/map_loader.js').GeometryMaterial} source
 * @param {object} context
 * @param {'baked' | 'lit'} context.shading
 * @param {(name: string | null, kind: 'color' | 'data') => THREE.Texture} context.texture  shared texture lookup
 * @param {Record<string, THREE.Texture> | null} context.lightmap  lightmap textures by name
 * @param {THREE.Texture | null} context.environment
 * @param {number} context.rgbmRange
 * @param {{sky: THREE.Color, ground: THREE.Color, sun: THREE.Color, sunDirection: THREE.Vector3}} context.viewerLights
 * @param {THREE.Texture} context.blank  1x1 white
 * @param {THREE.Texture} context.flatNormal  1x1 (0.5, 0.5, 1)
 * @param {(texture: THREE.Texture) => Promise<{mean: number, darkShare: number} | null>} [context.whenLoaded]
 *   brightness of a texture from the cache once its image has arrived (null if it failed)
 */
export function createSurfaceMaterial(source, context) {
  const layers = source.layers.length ? source.layers : [{ texture: source.texture, tiling: [1, 1], tint: [1, 1, 1, 1], uv_set: 0 }];
  const defines = { LAYER_COUNT: layers.length };
  const heightmaps = [];
  const code = layers
    .slice(1)
    .map((layer, k) => {
      let slot = null;
      if (layer.heightmap) {
        slot = heightmaps.length;
        heightmaps.push(layer.heightmap);
      }
      return layerCode(layer, k + 1, slot);
    })
    .join('\n  ');
  defines.HEIGHTMAP_COUNT = heightmaps.length;

  const isWater = source.kind === 'water';
  if (isWater) defines.WATER = '';
  if (context.shading === 'lit') defines.SHADING_LIT = '';
  else if (source.lighting === 'vertex' || source.lighting === 'probe') defines.LIGHTING_VERTEX = '';
  else if (source.lighting === 'lightmap' && context.lightmap) defines.LIGHTING_LIGHTMAP = '';
  const normalMap = layers[0].normal_map;
  if (normalMap && !isWater) defines.USE_NORMAL_MAP = '';
  if (source.spec_map) defines.USE_SPEC = '';
  if (source.emissive_map) defines.USE_EMISSIVE = '';
  if (context.environment) defines.USE_ENVIRONMENT = '';
  if (source.alpha === 'cutout') defines.ALPHA_CUTOUT = '';
  if (source.alpha === 'blend') defines.ALPHA_BLEND = '';

  const average = new THREE.Color().setRGB(source.color[0] / 255, source.color[1] / 255, source.color[2] / 255, THREE.SRGBColorSpace);
  const blank = context.blank;
  const lightmap = context.lightmap ?? {};
  const uniforms = {
    layerMaps: { value: layers.map((layer) => (layer.texture ? context.texture(layer.texture, isWater ? 'data' : 'color') : blank)) },
    layerTiling: { value: layers.map((layer) => new THREE.Vector2(...(layer.tiling ?? [1, 1]))) },
    // Tints are sRGB colours; the shader works in linear light.
    layerTint: {
      value: layers.map((layer) => {
        const [r, g, b, a] = layer.tint ?? [1, 1, 1, 1];
        const linear = new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
        return new THREE.Vector4(linear.r, linear.g, linear.b, a);
      }),
    },
    layerMultiplier: { value: layers.map((layer) => layer.multiplier ?? 1) },
    layerSoftness: { value: layers.map((layer) => layer.softness ?? 1) },
    normalMap: { value: normalMap ? context.texture(normalMap, 'data') : context.flatNormal },
    specMap: { value: source.spec_map ? context.texture(source.spec_map, 'data') : blank },
    emissiveMap: { value: source.emissive_map ? context.texture(source.emissive_map, 'color') : blank },
    lightmapRnm0: { value: lightmap.rnm0 ?? blank },
    lightmapRnm1: { value: lightmap.rnm1 ?? blank },
    lightmapRnm2: { value: lightmap.rnm2 ?? blank },
    lightmapDirect: { value: lightmap.direct ?? blank },
    lightmapDirectDir: { value: lightmap.direct_dir ?? blank },
    environmentMap: { value: context.environment ?? blank },
    rgbmRange: { value: context.rgbmRange },
    textured: { value: 1 },
    averageColor: { value: average },
    waterColor: { value: WATER_COLOR.clone() },
    skyColor: { value: context.viewerLights.sky },
    groundColor: { value: context.viewerLights.ground },
    sunColor: { value: context.viewerLights.sun },
    sunDirection: { value: context.viewerLights.sunDirection },
    reflectionStrength: { value: REFLECTION_STRENGTH },
    emissiveStrength: { value: EMISSIVE_STRENGTH },
  };
  if (heightmaps.length) uniforms.heightMaps = { value: heightmaps.map((name) => context.texture(name, 'data')) };

  const material = new THREE.ShaderMaterial({
    name: source.name,
    defines,
    uniforms,
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER.replace('LAYER_CODE', code),
    // Stored normals follow the triangles' winding, so back faces are culled like in
    // the game; thin cut-out cards (fences, foliage) and water show both sides.
    side: source.alpha === 'cutout' || isWater ? THREE.DoubleSide : THREE.FrontSide,
    transparent: source.alpha === 'blend',
    depthWrite: source.alpha !== 'blend',
  });
  if (source.alpha === 'blend') {
    // Overlays (decals, blood, graffiti) lie on other surfaces; pull them forward so they win the depth test.
    material.polygonOffset = true;
    material.polygonOffsetFactor = -1;
    material.polygonOffsetUnits = -4;
  }
  // Only the baked light is HDR; the viewer's own lighting is shown as it is.
  material.toneMapped = context.shading === 'baked';
  if (source.emissive_map && context.whenLoaded) {
    // Most "_em" textures (900 of the game's 1,091) are bright greyscale masks, not light:
    // corpse piles and car bodies glowed white. Light maps are mostly black around the lit parts.
    context.whenLoaded(uniforms.emissiveMap.value).then((stats) => {
      if (!stats || !isLightMap(stats)) uniforms.emissiveStrength.value = 0;
    });
  }
  return material;
}
