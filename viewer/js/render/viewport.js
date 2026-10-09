/**
 * Renderer, scene, camera and lights. Knows nothing about input or map formats.
 */
import * as THREE from 'three';

const BACKGROUND = 0x1b2026;
const DAYLIGHT_BACKGROUND = 0x9db3c8; // hazy daylight, seen through the game's cloud dome
const FAR_PLANE = 60000; // metres: the largest maps reach ~9 km from the origin

export class Viewport {
  /** @param {HTMLCanvasElement} canvas */
  constructor(canvas) {
    this.canvas = canvas;
    // A logarithmic depth buffer keeps 0.1 m and 60 km both free of z-fighting.
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    // Applies only to materials marked toneMapped (the game's HDR baked light).
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(BACKGROUND);

    this.camera = new THREE.PerspectiveCamera(70, 1, 0.1, FAR_PLANE);

    // All map content goes into `world`, so the map can be mirrored to test handedness
    // without touching the data or the camera controls.
    this.world = new THREE.Group();
    this.scene.add(this.world);

    const sky = new THREE.HemisphereLight(0xdfe8f2, 0x3b342b, 1.7);
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(0.45, 1, 0.3); // a direction: directional lights aim at the origin
    this.scene.add(sky, sun);
    // The same lights for custom shaders (colours times intensity, linear; sun direction normalised).
    this.viewerLights = {
      sky: sky.color.clone().multiplyScalar(sky.intensity),
      ground: sky.groundColor.clone().multiplyScalar(sky.intensity),
      sun: sun.color.clone().multiplyScalar(sun.intensity),
      sunDirection: sun.position.clone().normalize(),
    };

    this.grid = new THREE.GridHelper(4000, 400, 0x5a6573, 0x2e353e);
    this.grid.visible = false;
    this.grid.material.toneMapped = false;
    this.scene.add(this.grid);

    this._resize();
    new ResizeObserver(() => this._resize()).observe(canvas);
  }

  /** @param {boolean} daylight  a sky colour behind the real geometry, dark behind the boxes */
  setDaylight(daylight) {
    this.scene.background.set(daylight ? DAYLIGHT_BACKGROUND : BACKGROUND);
  }

  /** @param {boolean} mirrored  flip the map along X (swaps handedness) */
  setMirrored(mirrored) {
    this.world.scale.x = mirrored ? -1 : 1;
  }

  get mirrored() {
    return this.world.scale.x < 0;
  }

  /**
   * Map data coordinates for a scene-space point (undoes the mirror).
   * @param {THREE.Vector3} point
   */
  toData(point) {
    return new THREE.Vector3(point.x * this.world.scale.x, point.y, point.z);
  }

  /**
   * Scene-space point for map data coordinates (applies the mirror).
   * @param {THREE.Vector3} point
   */
  fromData(point) {
    return this.toData(point); // mirroring is its own inverse
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  _resize() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (width === 0 || height === 0) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }
}
