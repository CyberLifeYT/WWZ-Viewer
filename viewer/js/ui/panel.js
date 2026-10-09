/**
 * Side panel: map picker and display settings. Reports changes through callbacks
 * and holds no rendering or data-loading logic.
 */

// The size filter slider runs 0..100 on a log scale from 10 m to 10 km; 100 means no limit.
const SIZE_MIN = 10;
const SIZE_MAX = 10000;
const SLIDER_MAX = 100;

/** @param {number} position  slider value 0..100 */
export function sliderToSize(position) {
  if (position >= SLIDER_MAX) return Infinity;
  return SIZE_MIN * (SIZE_MAX / SIZE_MIN) ** (position / SLIDER_MAX);
}

function formatExposure(value) {
  return `${value.toFixed(2)}x`;
}

function formatSize(metres) {
  if (!Number.isFinite(metres)) return 'no limit';
  return metres >= 1000 ? `${(metres / 1000).toFixed(1)} km` : `${Math.round(metres)} m`;
}

/**
 * @typedef {object} DisplaySettings
 * @property {'group' | 'height'} colorMode
 * @property {number} maxSize  metres; Infinity shows every box
 * @property {boolean} wireframe
 * @property {boolean} grid
 * @property {boolean} mirrored
 * @property {boolean} showGeometry  draw the real level geometry
 * @property {boolean} showTextures  texture the geometry with the game textures
 * @property {boolean} showBaked  show the game's baked lighting instead of the viewer's sun
 * @property {boolean} autoExposure  adapt the exposure to the view, as the game does
 * @property {number} exposure  exposure for the tone-mapped baked lighting (a factor on the
 *   adapted exposure when autoExposure is on)
 * @property {boolean} showBoxes  draw the oriented bounding boxes
 * @property {boolean} showPlaceholders  draw surfaces whose texture the game does not ship
 */

export class Panel {
  /** @param {HTMLElement} root */
  constructor(root) {
    this.mapSelect = root.querySelector('#map-select');
    this.colorSelect = root.querySelector('#color-mode');
    this.sizeSlider = root.querySelector('#max-size');
    this.sizeLabel = root.querySelector('#max-size-value');
    this.wireframeBox = root.querySelector('#wireframe');
    this.gridBox = root.querySelector('#grid');
    this.mirrorBox = root.querySelector('#mirror');
    this.geometryBox = root.querySelector('#show-geometry');
    this.texturesBox = root.querySelector('#show-textures');
    this.bakedBox = root.querySelector('#show-baked');
    this.autoExposureBox = root.querySelector('#auto-exposure');
    this.exposureSlider = root.querySelector('#exposure');
    this.exposureLabel = root.querySelector('#exposure-value');
    this.boxesBox = root.querySelector('#show-boxes');
    this.placeholdersBox = root.querySelector('#show-placeholders');
    this.statsLine = root.querySelector('#stats');
    this.statusLine = root.querySelector('#status');

    /** @type {((map: string) => void) | null} */
    this.onMapChange = null;
    /** @type {((setting: keyof DisplaySettings) => void) | null} */
    this.onSettingsChange = null;

    this.mapSelect.addEventListener('change', () => this.onMapChange?.(this.mapSelect.value));
    this.colorSelect.addEventListener('change', () => this.onSettingsChange?.('colorMode'));
    this.sizeSlider.addEventListener('input', () => {
      this._updateSizeLabel();
      this.onSettingsChange?.('maxSize');
    });
    this.wireframeBox.addEventListener('change', () => this.onSettingsChange?.('wireframe'));
    this.gridBox.addEventListener('change', () => this.onSettingsChange?.('grid'));
    this.mirrorBox.addEventListener('change', () => this.onSettingsChange?.('mirrored'));
    this.geometryBox.addEventListener('change', () => this.onSettingsChange?.('showGeometry'));
    this.texturesBox.addEventListener('change', () => this.onSettingsChange?.('showTextures'));
    this.bakedBox.addEventListener('change', () => this.onSettingsChange?.('showBaked'));
    this.autoExposureBox.addEventListener('change', () => {
      this._updateExposureLabel();
      this.onSettingsChange?.('autoExposure');
    });
    this.exposureSlider.addEventListener('input', () => {
      this._updateExposureLabel();
      this.onSettingsChange?.('exposure');
    });
    this.boxesBox.addEventListener('change', () => this.onSettingsChange?.('showBoxes'));
    this.placeholdersBox.addEventListener('change', () => this.onSettingsChange?.('showPlaceholders'));
    this._updateSizeLabel();
    this._updateExposureLabel();
  }

  _updateExposureLabel() {
    const value = formatExposure(Number(this.exposureSlider.value));
    // With auto exposure on, the slider brightens or darkens what the adaptation chose.
    this.exposureLabel.value = this.autoExposureBox.checked ? `${value} of auto` : value;
  }

  /** @returns {DisplaySettings} */
  get settings() {
    return {
      colorMode: this.colorSelect.value === 'height' ? 'height' : 'group',
      maxSize: sliderToSize(Number(this.sizeSlider.value)),
      wireframe: this.wireframeBox.checked,
      grid: this.gridBox.checked,
      mirrored: this.mirrorBox.checked,
      showGeometry: this.geometryBox.checked,
      showTextures: this.texturesBox.checked,
      showBaked: this.bakedBox.checked,
      autoExposure: this.autoExposureBox.checked,
      exposure: Number(this.exposureSlider.value),
      showBoxes: this.boxesBox.checked,
      showPlaceholders: this.placeholdersBox.checked,
    };
  }

  /** @param {import('../data/map_loader.js').MapEntry[]} maps */
  setMaps(maps) {
    this.mapSelect.replaceChildren(
      ...maps.map((entry) => new Option(`${entry.map}  (${entry.box_count.toLocaleString()} boxes)`, entry.map)),
    );
  }

  /** @param {string} map */
  selectMap(map) {
    this.mapSelect.value = map;
  }

  /**
   * @param {number} drawn  boxes currently drawn
   * @param {number} total  boxes in the map
   * @param {number} groups  object groups in the map
   * @param {{triangleCount: number, materials: unknown[]} | null} [geometry]  real geometry, if exported
   */
  setStats(drawn, total, groups, geometry = null) {
    const hidden = total - drawn;
    const geometryLine = geometry
      ? `${geometry.triangleCount.toLocaleString()} triangles, ${geometry.materials.length} materials`
      : 'no real geometry exported (boxes only)';
    this.statsLine.textContent =
      `${geometryLine}\n` +
      `${drawn.toLocaleString()} boxes drawn` +
      (hidden ? `, ${hidden.toLocaleString()} oversized hidden` : '') +
      `\n${groups.toLocaleString()} object groups`;
  }

  /**
   * Enable the geometry toggle only for maps that have geometry; fall back to boxes otherwise.
   * @param {boolean} available  the map has real geometry
   * @param {boolean} [bakedLight]  the geometry has the game's baked lighting
   */
  setGeometryAvailable(available, bakedLight = available) {
    const wasAvailable = !this.geometryBox.disabled;
    this.geometryBox.disabled = !available;
    this.bakedBox.disabled = !bakedLight;
    this.autoExposureBox.disabled = !bakedLight;
    this.exposureSlider.disabled = !bakedLight;
    if (!available) {
      // Boxes-only export: show the boxes instead.
      this.geometryBox.checked = false;
      this.boxesBox.checked = true;
    } else if (!wasAvailable) {
      // Coming from a boxes-only map: switch back to the real geometry.
      this.geometryBox.checked = true;
      this.boxesBox.checked = false;
    }
  }

  /**
   * List the map's textures as thumbnails; each links to the full-size image.
   * Images load lazily, only when the gallery is opened and scrolled.
   * @param {string[]} names  texture names
   * @param {(texture: string, size: number) => string} textureUrl
   */
  setGallery(names, textureUrl) {
    const grid = document.querySelector('#gallery-grid');
    const count = document.querySelector('#gallery-count');
    const textures = [...new Set(names)].sort();
    count.textContent = textures.length ? `(${textures.length})` : '';
    if (!textures.length) {
      grid.replaceChildren(
        Object.assign(document.createElement('p'), {
          className: 'gallery-note',
          textContent: 'Export this map with geometry ("wwz-extract export <map>") to list its textures.',
        }),
      );
      return;
    }
    grid.replaceChildren(
      ...textures.map((texture) => {
        const link = Object.assign(document.createElement('a'), {
          className: 'gallery-item',
          href: textureUrl(texture, 1024),
          target: '_blank',
          rel: 'noopener',
          title: texture,
        });
        const image = Object.assign(document.createElement('img'), {
          src: textureUrl(texture, 128),
          alt: texture,
          loading: 'lazy',
        });
        link.append(image, texture);
        return link;
      }),
    );
  }

  /**
   * @param {string} text
   * @param {boolean} [isError=false]
   */
  setStatus(text, isError = false) {
    this.statusLine.textContent = text;
    this.statusLine.classList.toggle('error', isError);
  }

  _updateSizeLabel() {
    this.sizeLabel.textContent = formatSize(sliderToSize(Number(this.sizeSlider.value)));
  }
}
