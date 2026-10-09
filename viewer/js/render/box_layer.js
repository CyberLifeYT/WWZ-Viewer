/**
 * Draws oriented boxes as one instanced mesh (a single draw call for ~100k boxes).
 *
 * Box convention, from the exporter: a box covers
 *   origin + t0*e0*axis0 + t1*e1*axis1 + t2*e2*axis2,  t in [0, 1]
 * where axisN is row N of the stored rotation. A unit cube spanning [0, 1]^3 is
 * therefore mapped by the matrix whose columns are e0*axis0, e1*axis1, e2*axis2
 * and whose translation is the origin.
 */
import * as THREE from 'three';

/** Flat boxes get this much thickness (metres): a zero scale would give their faces NaN normals. */
const MIN_THICKNESS = 0.02;
const GOLDEN_RATIO_CONJUGATE = 0.618033988749895;

// Offsets of each value within one 15-float box record.
const ORIGIN = 0;
const AXIS0 = 3;
const AXIS1 = 6;
const AXIS2 = 9;
const EXTENTS = 12;

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

export class BoxLayer {
  /** @param {THREE.Object3D} parent */
  constructor(parent) {
    this.parent = parent;
    this.geometry = new THREE.BoxGeometry(1, 1, 1).translate(0.5, 0.5, 0.5);
    this.material = new THREE.MeshLambertMaterial({ color: 0xffffff, toneMapped: false });
    /** @type {THREE.InstancedMesh | null} */
    this.mesh = null;
    /** @type {import('../data/map_loader.js').BoxData | null} */
    this.data = null;
    /** Box index of each drawn instance. */
    this.instanceToBox = new Uint32Array(0);
  }

  /** @param {import('../data/map_loader.js').BoxData} data */
  setData(data) {
    this.data = data;
  }

  /** @param {boolean} enabled */
  setWireframe(enabled) {
    this.material.wireframe = enabled;
  }

  /**
   * (Re)build the instanced mesh.
   * @param {{maxSize: number, colorMode: 'group' | 'height'}} options
   *   maxSize hides boxes whose longest edge exceeds it (world boundaries, sky planes)
   * @returns {number} number of boxes drawn
   */
  build({ maxSize, colorMode }) {
    this._disposeMesh();
    if (!this.data) return 0;
    const { boxes, groups, count, stride, bounds } = this.data;

    const selected = new Uint32Array(count);
    let drawn = 0;
    for (let i = 0; i < count; i++) {
      const o = i * stride + EXTENTS;
      if (Math.max(boxes[o], boxes[o + 1], boxes[o + 2]) <= maxSize) selected[drawn++] = i;
    }

    const mesh = new THREE.InstancedMesh(this.geometry, this.material, Math.max(drawn, 1));
    mesh.count = drawn;
    // One draw call either way; culling it as a whole would only risk popping.
    mesh.frustumCulled = false;

    const matrix = new THREE.Matrix4();
    const color = new THREE.Color();
    const lowY = bounds.p1[1];
    const rangeY = Math.max(bounds.p99[1] - lowY, 1e-3);

    for (let k = 0; k < drawn; k++) {
      const i = selected[k];
      const o = i * stride;
      const e0 = Math.max(boxes[o + EXTENTS], MIN_THICKNESS);
      const e1 = Math.max(boxes[o + EXTENTS + 1], MIN_THICKNESS);
      const e2 = Math.max(boxes[o + EXTENTS + 2], MIN_THICKNESS);

      // Matrix4.set takes row-major arguments; column N is eN * axisN.
      matrix.set(
        boxes[o + AXIS0] * e0, boxes[o + AXIS1] * e1, boxes[o + AXIS2] * e2, boxes[o + ORIGIN],
        boxes[o + AXIS0 + 1] * e0, boxes[o + AXIS1 + 1] * e1, boxes[o + AXIS2 + 1] * e2, boxes[o + ORIGIN + 1],
        boxes[o + AXIS0 + 2] * e0, boxes[o + AXIS1 + 2] * e1, boxes[o + AXIS2 + 2] * e2, boxes[o + ORIGIN + 2],
        0, 0, 0, 1,
      ); // prettier-ignore
      mesh.setMatrixAt(k, matrix);

      if (colorMode === 'height') {
        const centerY =
          boxes[o + ORIGIN + 1] +
          0.5 * (boxes[o + AXIS0 + 1] * e0 + boxes[o + AXIS1 + 1] * e1 + boxes[o + AXIS2 + 1] * e2);
        const t = clamp01((centerY - lowY) / rangeY);
        color.setHSL(0.6 - 0.52 * t, 0.55, 0.36 + 0.26 * t); // low: blue, high: orange
      } else {
        // Golden-ratio hue steps keep neighbouring group ids visually distinct.
        color.setHSL((groups[i] * GOLDEN_RATIO_CONJUGATE) % 1, 0.42, 0.56);
      }
      mesh.setColorAt(k, color);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    this.instanceToBox = selected.subarray(0, drawn);
    this.mesh = mesh;
    this.parent.add(mesh);
    return drawn;
  }

  /**
   * The nearest drawn box hit by a ray, or null.
   * @param {THREE.Raycaster} raycaster
   */
  pick(raycaster) {
    if (!this.mesh || !this.data || this.mesh.count === 0) return null;
    if (this.mesh.boundingSphere === null) this.mesh.computeBoundingSphere();
    const hit = raycaster.intersectObject(this.mesh, false)[0];
    if (!hit || hit.instanceId === undefined) return null;

    const box = this.instanceToBox[hit.instanceId];
    const o = box * this.data.stride;
    const values = this.data.boxes;
    return {
      box,
      group: this.data.groups[box],
      distance: hit.distance,
      origin: [values[o], values[o + 1], values[o + 2]],
      extents: [values[o + EXTENTS], values[o + EXTENTS + 1], values[o + EXTENTS + 2]],
    };
  }

  _disposeMesh() {
    if (!this.mesh) return;
    this.parent.remove(this.mesh);
    this.mesh.dispose();
    this.mesh = null;
    this.instanceToBox = new Uint32Array(0);
  }
}
