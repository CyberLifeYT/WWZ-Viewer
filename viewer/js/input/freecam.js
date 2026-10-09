/**
 * Physics-free fly camera.
 *
 *   Mouse (pointer lock)   look around (yaw and pitch)
 *   W / A / S / D          fly forward / left / back / right along the view direction
 *   Space / Shift          rise / sink along world +Y
 *   Mouse wheel            multiply or divide the movement speed
 *
 * Movement is frame-rate independent and eased: the velocity approaches the
 * target velocity exponentially, so starts and stops are smooth without drifting.
 * There is no collision, gravity or bounds of any kind.
 *
 * This module only touches the camera's position and rotation and listens to
 * DOM events. It knows nothing about maps or rendering.
 */
import * as THREE from 'three';

/** KeyboardEvent.code -> movement action. Codes are layout independent (W is "KeyW" on AZERTY too). */
const MOVE_KEYS = {
  KeyW: 'forward',
  KeyS: 'back',
  KeyA: 'left',
  KeyD: 'right',
  Space: 'up',
  ShiftLeft: 'down',
  ShiftRight: 'down',
};

/** Stop just short of straight up/down so yaw stays well defined. */
const PITCH_LIMIT = Math.PI / 2 - 0.001;

/** Longest frame step applied, so a stalled tab does not teleport the camera on resume. */
const MAX_STEP_SECONDS = 0.1;

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

/** Keys typed into form controls must not move the camera. */
function isFormControl(target) {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable || ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName))
  );
}

export class Freecam {
  /**
   * @param {THREE.PerspectiveCamera} camera  its rotation order is set to 'YXZ' (yaw, then pitch)
   * @param {HTMLElement} element  the element that captures the pointer, normally the canvas
   * @param {object} [options]
   * @param {number} [options.speed=25]  metres per second
   * @param {number} [options.minSpeed=0.5]
   * @param {number} [options.maxSpeed=5000]
   * @param {number} [options.wheelFactor=1.25]  speed multiplier per wheel notch
   * @param {number} [options.lookSensitivity=0.0022]  radians per pixel of mouse movement
   * @param {number} [options.responsiveness=10]  1/s; how fast velocity reaches its target
   */
  constructor(camera, element, options = {}) {
    this.camera = camera;
    this.camera.rotation.order = 'YXZ';
    this.element = element;

    this.speed = options.speed ?? 25;
    this.minSpeed = options.minSpeed ?? 0.5;
    this.maxSpeed = options.maxSpeed ?? 5000;
    this.wheelFactor = options.wheelFactor ?? 1.25;
    this.lookSensitivity = options.lookSensitivity ?? 0.0022;
    this.responsiveness = options.responsiveness ?? 10;

    /** Radians; 0 looks down -Z, positive turns left. */
    this.yaw = 0;
    /** Radians; positive looks up. */
    this.pitch = 0;
    this.velocity = new THREE.Vector3();
    this.locked = false;

    /** @type {((speed: number) => void) | null} */
    this.onSpeedChange = null;
    /** @type {((locked: boolean) => void) | null} */
    this.onLockChange = null;

    this._held = new Set();
    this._target = new THREE.Vector3();
    this._forward = new THREE.Vector3();
    this._right = new THREE.Vector3();
    this._listeners = [];
    this._bindEvents();
  }

  /** Remove every event listener. */
  dispose() {
    for (const [target, type, handler, options] of this._listeners) {
      target.removeEventListener(type, handler, options);
    }
    this._listeners = [];
  }

  /** @param {number} value  metres per second, clamped to [minSpeed, maxSpeed] */
  setSpeed(value) {
    this.speed = clamp(value, this.minSpeed, this.maxSpeed);
    this.onSpeedChange?.(this.speed);
  }

  /**
   * Put the camera at `position` looking at `target`, and stop any motion.
   * @param {THREE.Vector3} position
   * @param {THREE.Vector3} target
   */
  lookAt(position, target) {
    this.camera.position.copy(position);
    const direction = target.clone().sub(position);
    this.yaw = Math.atan2(-direction.x, -direction.z);
    this.pitch = clamp(Math.atan2(direction.y, Math.hypot(direction.x, direction.z)), -PITCH_LIMIT, PITCH_LIMIT);
    this.velocity.set(0, 0, 0);
    this._applyRotation();
  }

  /**
   * Advance by one frame.
   * @param {number} deltaSeconds  time since the previous frame
   */
  update(deltaSeconds) {
    const dt = Math.min(Math.max(deltaSeconds, 0), MAX_STEP_SECONDS);
    this._applyRotation();

    // Unit vectors for the current view: forward includes pitch, so W flies where you look.
    const cosPitch = Math.cos(this.pitch);
    this._forward.set(-Math.sin(this.yaw) * cosPitch, Math.sin(this.pitch), -Math.cos(this.yaw) * cosPitch);
    this._right.set(Math.cos(this.yaw), 0, -Math.sin(this.yaw));

    const actions = new Set([...this._held].map((code) => MOVE_KEYS[code]));
    const along = (positive, negative) => (actions.has(positive) ? 1 : 0) - (actions.has(negative) ? 1 : 0);

    this._target
      .set(0, along('up', 'down'), 0)
      .addScaledVector(this._forward, along('forward', 'back'))
      .addScaledVector(this._right, along('right', 'left'));
    if (this._target.lengthSq() > 0) {
      this._target.normalize().multiplyScalar(this.speed);
    }

    // Exponential approach: identical feel at any frame rate.
    this.velocity.lerp(this._target, 1 - Math.exp(-this.responsiveness * dt));
    if (this._target.lengthSq() === 0 && this.velocity.lengthSq() < 1e-6) {
      this.velocity.set(0, 0, 0);
    }
    this.camera.position.addScaledVector(this.velocity, dt);
  }

  _applyRotation() {
    this.camera.rotation.set(this.pitch, this.yaw, 0, 'YXZ');
  }

  _listen(target, type, handler, options) {
    target.addEventListener(type, handler, options);
    this._listeners.push([target, type, handler, options]);
  }

  _bindEvents() {
    this._listen(this.element, 'click', () => {
      // A focused <select> or slider would otherwise keep swallowing WASD.
      if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
      if (!this.locked) this.element.requestPointerLock?.();
    });

    this._listen(document, 'pointerlockchange', () => {
      this.locked = document.pointerLockElement === this.element;
      this.onLockChange?.(this.locked);
    });

    this._listen(document, 'mousemove', (event) => {
      if (!this.locked) return;
      this.yaw -= event.movementX * this.lookSensitivity;
      this.pitch = clamp(this.pitch - event.movementY * this.lookSensitivity, -PITCH_LIMIT, PITCH_LIMIT);
    });

    this._listen(window, 'keydown', (event) => {
      if (!(event.code in MOVE_KEYS) || isFormControl(event.target)) return;
      this._held.add(event.code);
      event.preventDefault(); // Space would scroll the page
    });

    this._listen(window, 'keyup', (event) => {
      this._held.delete(event.code);
    });

    // Key-up events are lost when the window loses focus; forget held keys then.
    this._listen(window, 'blur', () => this._held.clear());

    this._listen(
      this.element,
      'wheel',
      (event) => {
        event.preventDefault();
        if (event.deltaY === 0) return;
        this.setSpeed(this.speed * (event.deltaY < 0 ? this.wheelFactor : 1 / this.wheelFactor));
      },
      { passive: false },
    );
  }
}
