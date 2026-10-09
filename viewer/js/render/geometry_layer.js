/**
 * Draws the level's real geometry: one shared vertex buffer, one material per
 * exported material (a combination of layers, lighting source and alpha use).
 *
 * Surfaces use the level-surface shader (see surface_material.js): the game's
 * material layers, normal maps, baked light and reflections. Sky domes are
 * unlit and blended; glows (light shafts) add their light.
 *
 * Two shadings, switchable at any time:
 * - baked: the game's own baked light, tone-mapped with an adjustable exposure;
 * - lit: the viewer's sun and sky on the game's normals.
 * Every surface material exists once per shading; both share the textures.
 */
import * as THREE from 'three';

import { createGlowMaterial } from './glow_material.js';
import { createSurfaceMaterial } from './surface_material.js';
import { TextureCache } from './texture_cache.js';

/** Exposure for the baked light; the game's own exposure is not known. */
export const DEFAULT_EXPOSURE = 1.0;

/** Geometry attribute name for each exported buffer. */
const ATTRIBUTE_NAMES = {
  positions: 'position',
  normals: 'normal',
  uvs: 'uv',
  tangents: 'aTangent',
  colours: 'aColours',
  uv_set1: 'aUvSet1',
  uv_set2: 'aUvSet2',
  uv_set3: 'aUvSet3',
  rnm0: 'aRnm0',
  rnm1: 'aRnm1',
  rnm2: 'aRnm2',
  direct: 'aDirect',
  direct_dir: 'aDirectDir',
  lightmap_uvs: 'aLightmapUv',
};

function solidTexture(r, g, b, a = 255) {
  const texture = new THREE.DataTexture(new Uint8Array([r, g, b, a]), 1, 1);
  texture.needsUpdate = true;
  return texture;
}

const BLANK = solidTexture(255, 255, 255);

/**
 * Surfaces the game does not show: its invisible blockers ("red") and developer
 * placeholders with a checkerboard texture (checker_01 is a 3 km plane under the
 * sea in jap_01_pve, which shows only because the sea surface is not exported).
 */
function isPlaceholder(source) {
  return source.kind === 'blocker' || /^checker/.test(source.name);
}
const FLAT_NORMAL = solidTexture(128, 128, 255);

export class GeometryLayer {
  /**
   * @param {THREE.Object3D} parent
   * @param {(texture: string, size: number) => string} textureUrl  where to fetch a game texture as PNG
   * @param {THREE.WebGLRenderer} renderer
   * @param {{sky: THREE.Color, ground: THREE.Color, sun: THREE.Color, sunDirection: THREE.Vector3}} viewerLights
   *   the viewer's own lighting, for the "lit" shading
   */
  constructor(parent, textureUrl, renderer, viewerLights) {
    this.parent = parent;
    this.renderer = renderer;
    this.viewerLights = viewerLights;
    this.textures = new TextureCache(textureUrl, renderer);
    /** @type {THREE.Mesh | null} */
    this.mesh = null;
    /** @type {import('../data/map_loader.js').GeometryData | null} */
    this.data = null;
    this.wireframe = false;
    this.showPlaceholders = false;
    this.textured = true;
    this.baked = true;
    this.exposure = DEFAULT_EXPOSURE;
    /** One material per source material, for each shading. @type {{lit: THREE.Material[], baked: THREE.Material[]}} */
    this.materials = { lit: [], baked: [] };
    /** @type {THREE.Texture | null} */
    this.environment = null;
    /** @type {((loaded: number, total: number, failed: number) => void) | null} */
    this.onTextureProgress = null;
    this.textures.onProgress = (settled, requested, failed) => this.onTextureProgress?.(settled, requested, failed);
  }

  /** True when the loaded export has the game's baked light. */
  get hasBakedLight() {
    return Boolean(this.data && this.data.attributes.rnm0);
  }

  /** @param {import('../data/map_loader.js').GeometryData | null} data */
  setData(data) {
    this._dispose();
    this.data = data;
    if (!data) return;

    const geometry = new THREE.BufferGeometry();
    for (const [name, values] of Object.entries(data.attributes)) {
      const attributeName = ATTRIBUTE_NAMES[name];
      if (!attributeName) continue;
      const layout = data.layouts[name];
      const attribute =
        layout.dtype === 'float16'
          ? new THREE.Float16BufferAttribute(values, layout.components)
          : new THREE.BufferAttribute(values, layout.components, layout.normalized);
      geometry.setAttribute(attributeName, attribute);
    }
    geometry.setIndex(new THREE.BufferAttribute(data.indices, 1));
    for (const group of data.groups) {
      geometry.addGroup(group.first_triangle * 3, group.triangle_count * 3, group.material);
    }
    geometry.computeBoundingSphere();

    const lightmap = data.lightmap
      ? Object.fromEntries(Object.entries(data.lightmap.urls).map(([key, url]) => [key, this.textures.image(url)]))
      : null;
    this.environment = data.environment ? this._environmentTexture(data.environment) : null;
    const context = (shading) => ({
      shading,
      texture: (name, kind) => this.textures.get(name, kind),
      whenLoaded: (texture) => this.textures.whenLoaded(texture),
      lightmap,
      environment: this.environment,
      rgbmRange: data.rgbmRange,
      viewerLights: this.viewerLights,
      blank: BLANK,
      flatNormal: FLAT_NORMAL,
    });
    const baked = context('baked');
    const lit = context('lit');
    this.materials = { lit: [], baked: [] };
    for (const source of data.materials) {
      if (source.kind === 'sky' || source.kind === 'glow') {
        const material = this._createUnlit(source);
        this.materials.lit.push(material);
        this.materials.baked.push(material);
        continue;
      }
      this.materials.baked.push(this._finish(createSurfaceMaterial(source, baked), source));
      this.materials.lit.push(this._finish(createSurfaceMaterial(source, lit), source));
    }
    this.mesh = new THREE.Mesh(geometry, this._activeMaterials());
    this.mesh.frustumCulled = false;
    this.parent.add(this.mesh);
    this.renderer.toneMappingExposure = this.exposure;
  }

  /** Apply the view settings every material shares. */
  _finish(material, source) {
    material.userData.source = source;
    material.wireframe = this.wireframe;
    material.visible = !isPlaceholder(source) || this.showPlaceholders;
    if (material.uniforms?.textured) material.uniforms.textured.value = this.textured ? 1 : 0;
    return material;
  }

  /** Sky domes (blended over the background, unlit, in the layer's tint) and glows (added light). */
  _createUnlit(source) {
    const texture = source.texture ? this.textures.get(source.texture, 'color') : null;
    if (source.kind === 'glow') return this._finish(createGlowMaterial(source, texture ?? BLANK), source);
    const material = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const [r, g, b] = source.layers?.[0]?.tint ?? [1, 1, 1];
    material.color.setRGB(r, g, b, THREE.SRGBColorSpace); // tints are sRGB
    material.userData.tint = material.color.clone();
    material.toneMapped = false;
    return this._finish(material, source);
  }

  /** The reflection probe as a float16 equirectangular texture (row 0 looks down). */
  _environmentTexture(environment) {
    const texture = new THREE.DataTexture(
      environment.data,
      environment.width,
      environment.height,
      THREE.RGBAFormat,
      THREE.HalfFloatType,
    );
    texture.wrapS = THREE.RepeatWrapping;
    texture.magFilter = THREE.LinearFilter;
    texture.minFilter = THREE.LinearFilter;
    texture.colorSpace = THREE.NoColorSpace;
    texture.needsUpdate = true;
    return texture;
  }

  _activeMaterials() {
    return this.baked && this.hasBakedLight ? this.materials.baked : this.materials.lit;
  }

  /** Every distinct material of both shadings. */
  _allMaterials() {
    return [...new Set([...this.materials.lit, ...this.materials.baked])];
  }

  /** @param {boolean} textured  show the game textures (true) or their average colours */
  setTextured(textured) {
    this.textured = textured;
    for (const material of this._allMaterials()) {
      if (material.uniforms?.textured) material.uniforms.textured.value = textured ? 1 : 0;
      else if (material.isMeshBasicMaterial) {
        if (textured) material.color.copy(material.userData.tint);
        else material.color.set(0x9db3c8);
      }
    }
  }

  /** @param {boolean} baked  show the game's baked light (true) or light the map with the viewer's sun */
  setBaked(baked) {
    this.baked = baked;
    if (this.mesh) this.mesh.material = this._activeMaterials();
  }

  /** @param {number} exposure  exposure for the tone-mapped baked light */
  setExposure(exposure) {
    this.exposure = exposure;
    this.renderer.toneMappingExposure = exposure;
  }

  /** @param {boolean} visible */
  setVisible(visible) {
    if (this.mesh) this.mesh.visible = visible;
  }

  get visible() {
    return Boolean(this.mesh && this.mesh.visible);
  }

  /** @param {boolean} visible  draw the game's invisible blocker geometry */
  setPlaceholdersVisible(visible) {
    this.showPlaceholders = visible;
    for (const material of this._allMaterials()) {
      material.visible = !isPlaceholder(material.userData.source) || visible;
    }
  }

  /** @param {boolean} enabled */
  setWireframe(enabled) {
    this.wireframe = enabled;
    for (const material of this._allMaterials()) material.wireframe = enabled;
  }

  /**
   * The nearest visible surface hit by a ray, with its material, or null.
   * Brute force over every triangle: fine on demand, too slow per frame.
   * @param {THREE.Raycaster} raycaster
   */
  pick(raycaster) {
    if (!this.mesh || !this.mesh.visible || !this.data) return null;
    const hit = raycaster.intersectObject(this.mesh, false).find((candidate) => {
      const group = this._groupOfTriangle(candidate.faceIndex);
      return group && this.mesh.material[group.material].visible;
    });
    if (!hit) return null;
    const group = this._groupOfTriangle(hit.faceIndex);
    return { distance: hit.distance, point: hit.point, material: this.data.materials[group.material] };
  }

  _groupOfTriangle(triangle) {
    const groups = this.data.groups;
    let low = 0;
    let high = groups.length - 1;
    while (low <= high) {
      const middle = (low + high) >> 1;
      const group = groups[middle];
      if (triangle < group.first_triangle) high = middle - 1;
      else if (triangle >= group.first_triangle + group.triangle_count) low = middle + 1;
      else return group;
    }
    return null;
  }

  _dispose() {
    if (!this.mesh) return;
    this.parent.remove(this.mesh);
    this.mesh.geometry.dispose();
    for (const material of this._allMaterials()) material.dispose();
    this.textures.dispose();
    this.environment?.dispose();
    this.environment = null;
    this.materials = { lit: [], baked: [] };
    this.mesh = null;
  }
}
