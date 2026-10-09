/**
 * On-screen readouts: camera position (in map data coordinates), view angles,
 * speed, frame rate, the click-to-fly prompt and the last identified box.
 */

const RADIANS_TO_DEGREES = 180 / Math.PI;

function formatVector(values, digits = 1) {
  return values.map((value) => value.toFixed(digits).padStart(9)).join(' ');
}

export class Hud {
  /** @param {Document} root */
  constructor(root) {
    this.position = root.querySelector('#hud-position');
    this.view = root.querySelector('#hud-view');
    this.speed = root.querySelector('#hud-speed');
    this.fps = root.querySelector('#hud-fps');
    this.pick = root.querySelector('#hud-pick');
    this.pickText = root.querySelector('#hud-pick-text');
    this.pickTexture = root.querySelector('#hud-texture');
    this.prompt = root.querySelector('#prompt');
    this.crosshair = root.querySelector('#crosshair');
  }

  /**
   * @param {{position: {x: number, y: number, z: number}, yaw: number, pitch: number, speed: number, fps: number}} state
   *   position in map data coordinates; yaw and pitch in radians
   */
  update({ position, yaw, pitch, speed, fps }) {
    const heading = (((yaw * RADIANS_TO_DEGREES) % 360) + 360) % 360;
    this.position.textContent = formatVector([position.x, position.y, position.z]);
    this.view.textContent = `yaw ${heading.toFixed(0).padStart(3)}°  pitch ${(pitch * RADIANS_TO_DEGREES).toFixed(0).padStart(3)}°`;
    this.speed.textContent = `${speed.toFixed(speed < 10 ? 1 : 0)} m/s`;
    this.fps.textContent = Number.isFinite(fps) ? fps.toFixed(0) : '-';
  }

  /** @param {boolean} locked  whether the pointer is captured for flying */
  setLocked(locked) {
    this.prompt.hidden = locked;
    this.crosshair.hidden = !locked;
  }

  /**
   * Show what is at the crosshair: a geometry surface (with its texture), a box, or nothing.
   * @param {{kind: 'surface', distance: number, point: {x: number, y: number, z: number},
   *          material: import('../data/map_loader.js').GeometryMaterial | null, textureUrl: string | null}
   *       | ({kind: 'box'} & ReturnType<import('../render/box_layer.js').BoxLayer['pick']>)
   *       | null} info
   */
  showPick(info) {
    this.pick.hidden = false;
    this.pickTexture.hidden = true;
    if (!info) {
      this.pickText.textContent = 'nothing at the crosshair';
      return;
    }
    if (info.kind === 'surface') {
      const material = info.material;
      const layers = material && material.layers.length > 1 ? `\nlayers  ${material.layers.join(', ')}` : '';
      this.pickText.textContent =
        `surface ${material ? material.name : 'unknown'}  ${info.distance.toFixed(1)} m away\n` +
        `at     ${formatVector([info.point.x, info.point.y, info.point.z], 2)}` +
        layers;
      if (info.textureUrl) {
        this.pickTexture.src = info.textureUrl;
        this.pickTexture.hidden = false;
      }
      return;
    }
    this.pickText.textContent =
      `box ${info.box}  group ${info.group}  ${info.distance.toFixed(1)} m away\n` +
      `origin  ${formatVector(info.origin, 2)}\n` +
      `extents ${formatVector(info.extents, 2)}`;
  }

  /** @param {string} text  a short status, e.g. while a slow pick runs */
  showPickStatus(text) {
    this.pick.hidden = false;
    this.pickTexture.hidden = true;
    this.pickText.textContent = text;
  }

  clearPick() {
    this.pick.hidden = true;
    this.pickText.textContent = '';
    this.pickTexture.hidden = true;
  }
}
