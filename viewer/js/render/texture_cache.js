/**
 * Game textures fetched from the local server, shared by every material that
 * uses them. A texture is handed out at once and fills in when its image
 * arrives, so materials never wait.
 */
import * as THREE from 'three';

/** Long side of the textures requested from the server. */
const REQUESTED_TEXTURE_SIZE = 512;
/** Stand-ins for textures that fail to load: neutral grey, and a flat normal (0.5, 0.5, 1). */
const FALLBACK_COLOR = [180, 180, 180, 255];
const FALLBACK_DATA = [128, 128, 255, 255];
/** Texels sampled (per side) to measure an image's brightness. */
const STATS_SIZE = 64;
/** Linear luminance below which a texel counts as dark. */
const DARK_TEXEL = 0.02;
const SRGB_TO_LINEAR = Float32Array.from({ length: 256 }, (_, i) => {
  const c = i / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
});

/**
 * Mean linear luminance and share of dark texels of an image, from a small copy of it.
 * @param {CanvasImageSource} image
 */
export function imageStats(image) {
  const canvas =
    typeof OffscreenCanvas === 'undefined'
      ? Object.assign(document.createElement('canvas'), { width: STATS_SIZE, height: STATS_SIZE })
      : new OffscreenCanvas(STATS_SIZE, STATS_SIZE);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(image, 0, 0, STATS_SIZE, STATS_SIZE);
  const { data } = context.getImageData(0, 0, STATS_SIZE, STATS_SIZE);
  let sum = 0;
  let dark = 0;
  for (let i = 0; i < data.length; i += 4) {
    const luminance = 0.2126 * SRGB_TO_LINEAR[data[i]] + 0.7152 * SRGB_TO_LINEAR[data[i + 1]] + 0.0722 * SRGB_TO_LINEAR[data[i + 2]];
    sum += luminance;
    if (luminance < DARK_TEXEL) dark += 1;
  }
  const texels = data.length / 4;
  return { mean: sum / texels, darkShare: dark / texels };
}

export class TextureCache {
  /**
   * @param {(texture: string, size: number) => string} textureUrl  where to fetch a game texture as PNG
   * @param {THREE.WebGLRenderer} renderer  for the anisotropic filtering limit
   */
  constructor(textureUrl, renderer) {
    this.textureUrl = textureUrl;
    this.maxAnisotropy = renderer.capabilities.getMaxAnisotropy();
    this.loader = new THREE.TextureLoader();
    /** @type {Map<string, THREE.Texture>} */
    this.textures = new Map();
    /** Resolves true when a texture's image has arrived, false when it failed. @type {WeakMap<THREE.Texture, Promise<boolean>>} */
    this.loaded = new WeakMap();
    this.requested = 0;
    this.settled = 0;
    this.failed = 0;
    /** @type {((settled: number, requested: number, failed: number) => void) | null} */
    this.onProgress = null;
  }

  /**
   * The texture called ``name``: 'color' textures are sRGB, 'data' textures (normal maps,
   * heightmaps, specular masks) are read as stored.
   * @param {string} name
   * @param {'color' | 'data'} kind
   */
  get(name, kind) {
    const key = `${kind}:${name}`;
    let texture = this.textures.get(key);
    if (texture) return texture;
    this.requested += 1;
    let arrived;
    const loaded = new Promise((resolve) => {
      arrived = resolve;
    });
    texture = this.loader.load(
      this.textureUrl(name, REQUESTED_TEXTURE_SIZE),
      () => {
        arrived(true);
        this._settle();
      },
      undefined,
      () => {
        // An image that never arrives samples as black in WebGL, which would paint every
        // surface black when the texture server is unreachable. Show a neutral stand-in.
        this._useFallback(texture, kind);
        this.failed += 1;
        arrived(false);
        this._settle();
      },
    );
    this.loaded.set(texture, loaded);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.colorSpace = kind === 'color' ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    texture.anisotropy = this.maxAnisotropy;
    texture.flipY = false; // the game's v runs down the image, as stored
    this.textures.set(key, texture);
    return texture;
  }

  /**
   * A data image from an export (lightmaps): linear, clamped at the edges.
   * @param {string} url
   */
  image(url) {
    const key = `image:${url}`;
    let texture = this.textures.get(key);
    if (texture) return texture;
    this.requested += 1;
    texture = this.loader.load(url, () => this._settle(), undefined, () => {
      this.failed += 1;
      this._settle();
    });
    texture.colorSpace = THREE.NoColorSpace;
    texture.flipY = false;
    this.textures.set(key, texture);
    return texture;
  }

  /**
   * How bright a texture from ``get`` is, once its image has arrived: mean linear luminance
   * and the share of nearly black texels; null when the image failed to load.
   * @param {THREE.Texture} texture
   * @returns {Promise<{mean: number, darkShare: number} | null>}
   */
  async whenLoaded(texture) {
    const ok = await (this.loaded.get(texture) ?? Promise.resolve(false));
    return ok ? imageStats(texture.image) : null;
  }

  /** Replace a texture whose image failed to load: light grey for colour, a flat normal for data. */
  _useFallback(texture, kind) {
    const pixel = kind === 'color' ? FALLBACK_COLOR : FALLBACK_DATA;
    texture.image = { data: new Uint8Array(pixel), width: 1, height: 1 };
    texture.isDataTexture = true; // uploaded from the pixel array above
    texture.generateMipmaps = false;
    texture.minFilter = THREE.NearestFilter;
    texture.magFilter = THREE.NearestFilter;
    texture.needsUpdate = true;
  }

  _settle() {
    this.settled += 1;
    this.onProgress?.(this.settled, this.requested, this.failed);
  }

  dispose() {
    for (const texture of this.textures.values()) texture.dispose();
    this.textures.clear();
    this.requested = 0;
    this.settled = 0;
    this.failed = 0;
  }
}
