/**
 * Light effects (light cones, ray fans, flares): textures of light on black,
 * added onto the scene.
 *
 * The game builds a light cone from dozens of crossed planes, so a line of
 * sight crosses up to ~40 of them; added at full strength they burn to white.
 * Like the game's effect shaders, each plane here fades as it turns edge-on
 * to the camera and when the camera is right at it, and the layers add at a
 * modest strength. Colour comes from the material's tint (sRGB).
 */
import * as THREE from 'three';

/** Share of the texture's light each plane adds. */
export const GLOW_STRENGTH = 0.35;
/** Planes fade out between these camera distances (metres), so flying through a cone is not a flash. */
const NEAR_FADE = [0.5, 3.0];

const VERTEX_SHADER = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_vertex>

uniform vec2 tiling;

varying vec2 vUv;
varying vec3 vNormalW;
varying vec3 vPositionW;

void main() {
  vUv = uv * tiling;
  vec4 worldPosition = modelMatrix * vec4(position, 1.0);
  vPositionW = worldPosition.xyz;
  vNormalW = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * viewMatrix * worldPosition;
  #include <logdepthbuf_vertex>
}
`;

const FRAGMENT_SHADER = /* glsl */ `
#include <common>
#include <logdepthbuf_pars_fragment>

uniform sampler2D map;
uniform vec3 tint;
uniform float strength;
uniform float textured;
uniform vec2 nearFade;

varying vec2 vUv;
varying vec3 vNormalW;
varying vec3 vPositionW;

void main() {
  #include <logdepthbuf_fragment>
  vec3 toCamera = cameraPosition - vPositionW;
  float distance = length(toCamera);
  // Face-on planes show their light, edge-on planes (seen as thin bright lines) vanish.
  float facing = abs(dot(normalize(vNormalW), toCamera / max(distance, 1e-4)));
  float fade = facing * facing * smoothstep(nearFade.x, nearFade.y, distance);
  vec3 light = textured > 0.5 ? texture2D(map, vUv).rgb : vec3(0.25);
  gl_FragColor = vec4(light * tint * strength * fade, 1.0);
  #include <colorspace_fragment>
}
`;

/**
 * @param {import('../data/map_loader.js').GeometryMaterial} source  a material of kind 'glow'
 * @param {THREE.Texture} texture  its texture (or a blank one)
 */
export function createGlowMaterial(source, texture) {
  const layer = source.layers?.[0];
  const [r, g, b] = layer?.tint ?? [1, 1, 1];
  const material = new THREE.ShaderMaterial({
    uniforms: {
      map: { value: texture },
      tint: { value: new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace) },
      tiling: { value: new THREE.Vector2(...(layer?.tiling ?? [1, 1])) },
      strength: { value: GLOW_STRENGTH },
      textured: { value: 1 },
      nearFade: { value: new THREE.Vector2(...NEAR_FADE) },
    },
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
  });
  material.toneMapped = false; // light added on top of the tone-mapped scene
  return material;
}
