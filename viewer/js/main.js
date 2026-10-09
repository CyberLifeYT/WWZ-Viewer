/**
 * Entry point: wires data loading, rendering, the freecam and the UI together.
 * Each of those modules is independent; this file is the only one that knows them all.
 */
import * as THREE from 'three';
import { loadIndex, loadMap } from './data/map_loader.js';
import { Freecam } from './input/freecam.js';
import { suggestExposure } from './render/auto_exposure.js';
import { BoxLayer } from './render/box_layer.js';
import { EyeAdaptation } from './render/eye_adaptation.js';
import { GeometryLayer } from './render/geometry_layer.js';
import { Viewport } from './render/viewport.js';
import { Hud } from './ui/hud.js';
import { Panel } from './ui/panel.js';

/** Folder with index.json, relative to this page. Override with ?data=... */
const DATA_ROOT = new URLSearchParams(window.location.search).get('data') ?? '../output/maps';
const STORAGE_KEY = 'wwz-mapviewer:last-map';
/** Decodes textures from the game's paks on demand (see serve.py). */
const TEXTURE_API = '/api/texture/';
/**
 * Set by build_site.py in the website copy of this page. A website has no texture
 * server: its textures are PNG files decoded in advance, and it offers a selection
 * of maps (possibly none).
 * @type {{textures: string, textureFormat: string, repository: string} | undefined}
 */
const SITE = window.WWZ_SITE;
const TEXTURE_PREVIEW_SIZE = 256;
const HUD_INTERVAL_SECONDS = 0.1;
const REBUILD_DELAY_MS = 120;

const canvas = /** @type {HTMLCanvasElement} */ (document.querySelector('#viewport'));
const viewport = new Viewport(canvas);
const boxes = new BoxLayer(viewport.world);
const geometryLayer = new GeometryLayer(viewport.world, textureUrl, viewport.renderer, viewport.viewerLights);
const eyeAdaptation = new EyeAdaptation(viewport.renderer);
const freecam = new Freecam(viewport.camera, canvas);
const panel = new Panel(document.querySelector('#panel'));
const hud = new Hud(document);
const raycaster = new THREE.Raycaster();
const screenCenter = new THREE.Vector2(0, 0);

/** @type {import('./data/map_loader.js').MapEntry[]} */
let maps = [];
/** @type {{entry: import('./data/map_loader.js').MapEntry, obb: import('./data/map_loader.js').BoxData,
 *            geometry: import('./data/map_loader.js').GeometryData | null} | null} */
let current = null;
let loadToken = 0;
let rebuildTimer = 0;

// -- persistence (best effort: storage can be unavailable) ----------------------------

function readStoredMap() {
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function storeMap(name) {
  try {
    window.localStorage.setItem(STORAGE_KEY, name);
  } catch {
    // Not important enough to report.
  }
}

function mapFromHash() {
  const match = /(?:^|&)map=([^&]+)/.exec(window.location.hash.slice(1));
  return match ? decodeURIComponent(match[1]) : null;
}

function isFormControl(target) {
  return target instanceof HTMLElement && ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName);
}

// -- map handling -------------------------------------------------------------------

function rebuildBoxes() {
  if (!current) return;
  const { maxSize, colorMode } = panel.settings;
  const drawn = boxes.build({ maxSize, colorMode });
  if (boxes.mesh) boxes.mesh.visible = panel.settings.showBoxes; // build() creates a new mesh
  panel.setStats(drawn, current.obb.count, current.obb.groupCount, current.geometry);
}

/** The panel's settings as last applied (read once per change, not every frame). */
let displaySettings = panel.settings;

function applyDisplaySettings() {
  const settings = panel.settings;
  displaySettings = settings;
  boxes.setWireframe(settings.wireframe);
  geometryLayer.setWireframe(settings.wireframe);
  geometryLayer.setVisible(settings.showGeometry);
  viewport.setDaylight(geometryLayer.visible);
  geometryLayer.setPlaceholdersVisible(settings.showPlaceholders);
  geometryLayer.setTextured(settings.showTextures);
  geometryLayer.setBaked(settings.showBaked);
  if (!autoExposureActive()) geometryLayer.setExposure(settings.exposure);
  if (boxes.mesh) boxes.mesh.visible = settings.showBoxes;
  viewport.grid.visible = settings.grid;
  if (settings.mirrored !== viewport.mirrored) {
    viewport.setMirrored(settings.mirrored);
    // Mirror the camera too, so it stays over the same spot of the map.
    viewport.camera.position.x = -viewport.camera.position.x;
    freecam.yaw = -freecam.yaw;
    freecam.velocity.set(0, 0, 0);
  }
}

/**
 * Look at the playable area from above and to the +Z side. Its centre is the
 * median box centre and its size the 10th-90th percentile spread: the extremes
 * include kilometres of distant scenery.
 */
function frameMap(obb) {
  const { p10, p50, p90 } = obb.bounds;
  const center = new THREE.Vector3(p50[0], p50[1], p50[2]);
  const span = Math.max(p90[0] - p10[0], p90[2] - p10[2], 50);
  const position = center.clone().add(new THREE.Vector3(0, span * 0.45, span * 0.6));
  freecam.lookAt(viewport.fromData(position), viewport.fromData(center));
  // Crossing the playable area takes about ten seconds at the starting speed.
  freecam.setSpeed(Math.min(Math.max(span / 10, 5), 400));
}

async function openMap(name) {
  const entry = maps.find((candidate) => candidate.map === name) ?? maps[0];
  const token = ++loadToken;
  panel.selectMap(entry.map);
  panel.setStatus(`Loading ${entry.map}…`);
  try {
    const { obb, geometry, geometryError } = await loadMap(DATA_ROOT, entry);
    if (token !== loadToken) return; // a newer request replaced this one
    current = { entry, obb, geometry };
    boxes.setData(obb);
    geometryLayer.setData(geometry);
    panel.setGeometryAvailable(Boolean(geometry), geometryLayer.hasBakedLight);
    // Each map starts at an exposure suited to its own light (night maps are ~150x darker than
    // sunny ones); eye adaptation takes over from there.
    if (geometry && geometryLayer.hasBakedLight) eyeAdaptation.reset(suggestExposure(geometry));
    panel.setGallery(geometry ? geometry.textures : [], textureUrl);
    rebuildBoxes();
    applyDisplaySettings();
    frameMap(obb);
    hud.clearPick();
    panel.setStatus(geometryError ? `Showing ${entry.map} as boxes: ${geometryError}` : `Showing ${entry.map}`);
    storeMap(entry.map);
    if (mapFromHash() !== entry.map) {
      window.history.replaceState(null, '', `#map=${encodeURIComponent(entry.map)}`);
    }
  } catch (error) {
    if (token === loadToken) panel.setStatus(`Could not load ${entry.map}: ${error.message}`, true);
    console.error(error);
  }
}

/** URL of a game texture decoded by the local server, at most ``size`` pixels on its long side. */
function textureUrl(texture, size) {
  if (SITE) return `${SITE.textures}${encodeURIComponent(texture)}.${SITE.textureFormat}`; // one size, decoded in advance
  return `${TEXTURE_API}${encodeURIComponent(texture)}.png?size=${size}`;
}

/** Report the surface (or, with geometry hidden, the box) under the crosshair. */
function identify() {
  // World matrices are normally refreshed by render(); refresh them here so a
  // pick right after a camera move does not cast from the previous pose.
  viewport.scene.updateMatrixWorld();
  viewport.camera.updateMatrixWorld();
  raycaster.setFromCamera(screenCenter, viewport.camera);
  if (geometryLayer.visible) {
    // Testing millions of triangles takes a moment; let the status paint first.
    hud.showPickStatus('identifying…');
    window.setTimeout(() => {
      const hit = geometryLayer.pick(raycaster);
      if (!hit) {
        hud.showPick(null);
        return;
      }
      const texture = hit.material && hit.material.texture;
      hud.showPick({
        kind: 'surface',
        distance: hit.distance,
        point: viewport.toData(hit.point),
        material: hit.material,
        textureUrl: texture ? textureUrl(texture, TEXTURE_PREVIEW_SIZE) : null,
      });
    }, 30);
    return;
  }
  const box = boxes.mesh && boxes.mesh.visible ? boxes.pick(raycaster) : null;
  hud.showPick(box ? { kind: 'box', ...box } : null);
}

// -- wiring -------------------------------------------------------------------------

panel.onMapChange = (name) => openMap(name);
panel.onSettingsChange = (setting) => {
  applyDisplaySettings();
  if (setting === 'colorMode' || setting === 'maxSize') {
    window.clearTimeout(rebuildTimer);
    rebuildTimer = window.setTimeout(rebuildBoxes, REBUILD_DELAY_MS);
  }
};
freecam.onLockChange = (locked) => hud.setLocked(locked);
geometryLayer.onTextureProgress = (loaded, total, failed) => {
  if (!current) return;
  if (loaded < total) panel.setStatus(`Loading textures ${loaded}/${total}…`);
  else if (failed === total) {
    panel.setStatus(`Showing ${current.entry.map}. No texture could be loaded: is serve.py still running?`);
  } else if (failed) panel.setStatus(`Showing ${current.entry.map} (${failed} of ${total} textures missing, shown grey)`);
  else panel.setStatus(`Showing ${current.entry.map}`);
};

window.addEventListener('hashchange', () => {
  const wanted = mapFromHash();
  if (wanted && wanted !== current?.entry.map) openMap(wanted);
});

window.addEventListener('keydown', (event) => {
  if (event.repeat || isFormControl(event.target)) return;
  if (event.code === 'KeyF') identify();
  else if (event.code === 'KeyR' && current) frameMap(current.obb);
});

const clock = new THREE.Clock();
// Start "due" so the HUD fills in on the very first frame instead of after the first interval.
let hudElapsed = HUD_INTERVAL_SECONDS;
let hudFrames = 0;

/** True when the exposure follows the view: auto exposure on, showing the game's baked light. */
function autoExposureActive() {
  return displaySettings.autoExposure && geometryLayer.visible && geometryLayer.baked && geometryLayer.hasBakedLight;
}

function frame() {
  const delta = clock.getDelta();
  freecam.update(delta);
  viewport.render();
  if (autoExposureActive()) {
    // Measures the frame just drawn (it must be read before the browser presents it).
    const adapted = eyeAdaptation.update(delta, displaySettings.exposure, viewport.scene.background);
    geometryLayer.setExposure(adapted * displaySettings.exposure);
  }

  hudFrames += 1;
  hudElapsed += delta;
  if (hudElapsed >= HUD_INTERVAL_SECONDS) {
    hud.update({
      position: viewport.toData(viewport.camera.position),
      yaw: viewport.mirrored ? -freecam.yaw : freecam.yaw,
      pitch: freecam.pitch,
      speed: freecam.speed,
      fps: hudFrames > 1 ? hudFrames / hudElapsed : NaN, // not measurable from a single frame
    });
    hudFrames = 0;
    hudElapsed = 0;
  }
}

async function start() {
  hud.setLocked(false);
  viewport.renderer.setAnimationLoop(frame);
  try {
    maps = await loadIndex(DATA_ROOT);
  } catch (error) {
    panel.setStatus(`No exported maps found (${error.message}). Run "wwz-extract export --all", then reload.`, true);
    return;
  }
  if (maps.length === 0) {
    panel.setStatus(
      SITE
        ? `This website hosts no maps: they are made from the game's own files. To fly through the maps, ` +
            `download the tool (${SITE.repository}) and run export_all_maps.bat with your own copy of World War Z.`
        : 'The map index is empty. Run "wwz-extract export --all", then reload.',
      true,
    );
    return;
  }
  panel.setMaps(maps);
  applyDisplaySettings();
  const wanted = mapFromHash() ?? readStoredMap();
  await openMap(maps.some((entry) => entry.map === wanted) ? wanted : maps[0].map);
}

// Debugging handle for the browser console; nothing in the app depends on it.
window.wwzViewer = { viewport, freecam, boxes, geometryLayer, eyeAdaptation, openMap, identify };

start();
