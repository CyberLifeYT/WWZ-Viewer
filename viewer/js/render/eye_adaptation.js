/**
 * Eye adaptation: the exposure follows how bright the view is, as the game's does.
 *
 * The baked light spans a huge range within one map (a sunlit rooftop gets
 * the sun at 16 and more on top of the sky's 2 to 4; an interior a tenth of
 * that), so no fixed exposure suits both. A few times a second, right after a
 * frame is drawn, the middle of the displayed image is read back, and the
 * exposure is nudged so that its log-average brightness moves toward middle
 * grey: a feedback loop on what is actually on screen, needing no extra render
 * pass. Pixels showing the plain background (the sky colour) are left out, and
 * when more than a few percent of the pixels are already clipped to white the
 * exposure backs off, as a camera protects highlights (a dark patch in the
 * middle, such as tarmac under a plane, would otherwise wash out the rest).
 * The exposure eases toward its target, within limits that keep night maps
 * darker than day.
 */

/** Share of the image (per side) metered, centred, and the largest block read back (device pixels). */
const REGION_SHARE = 0.6;
const REGION_MAX_WIDTH = 640;
const REGION_MAX_HEIGHT = 360;
/** Every SAMPLE_STEP-th pixel of the block (per side) is measured. */
const SAMPLE_STEP = 2;
/** A pixel is clipped when its brightest channel reaches this (8-bit). */
const CLIPPED = 250;
/** More clipped pixels than this share makes the exposure back off by HIGHLIGHT_BACKOFF per measurement. */
const MAX_CLIPPED_SHARE = 0.04;
const HIGHLIGHT_BACKOFF = 0.8;
/** Seconds between measurements. */
const MEASURE_INTERVAL = 0.25;
/** Seconds for the exposure to cover ~63% of the way to a new target. */
const ADAPT_TIME = 0.6;
/** Target log-average brightness of the view, linear: middle grey. */
const KEY = 0.18;
/** Fraction of the measured error corrected per measurement (the tone curve is not linear). */
const FEEDBACK = 0.6;
export const EXPOSURE_MIN = 0.06;
export const EXPOSURE_MAX = 6;
const LUMA = [0.2126, 0.7152, 0.0722];

/** 8-bit sRGB -> linear, as a table. */
const SRGB_TO_LINEAR = Float32Array.from({ length: 256 }, (_, i) => {
  const c = i / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
});

export class EyeAdaptation {
  /** @param {import('three').WebGLRenderer} renderer */
  constructor(renderer) {
    this.gl = renderer.getContext();
    this.pixels = new Uint8Array(REGION_MAX_WIDTH * REGION_MAX_HEIGHT * 4);
    this.exposure = 1;
    this.targetExposure = 1;
    this.elapsed = 0;
  }

  /**
   * Start over from ``exposure`` (a new map).
   * @param {number} exposure
   */
  reset(exposure) {
    this.exposure = Math.min(EXPOSURE_MAX, Math.max(EXPOSURE_MIN, exposure));
    this.targetExposure = this.exposure;
    this.elapsed = 0;
  }

  /**
   * Call right after a frame is drawn at ``this.exposure * compensation``: measures the
   * frame when due, eases toward the target, and returns the exposure for the next frame.
   * @param {number} delta  seconds since the last frame
   * @param {number} compensation  the user's factor on top of the adapted exposure
   * @param {import('three').Color} background  the clear colour, whose pixels are skipped
   */
  update(delta, compensation, background) {
    this.elapsed += delta;
    if (this.elapsed >= MEASURE_INTERVAL) {
      this.elapsed = 0;
      this._measure(compensation, background);
    }
    // Ease in log space: going from 0.1 to 1 feels as fast as from 1 to 10.
    const step = 1 - Math.exp(-delta / ADAPT_TIME);
    this.exposure = Math.exp(Math.log(this.exposure) + (Math.log(this.targetExposure) - Math.log(this.exposure)) * step);
    return this.exposure;
  }

  _measure(compensation, background) {
    const gl = this.gl;
    const width = Math.min(REGION_MAX_WIDTH, Math.round(gl.drawingBufferWidth * REGION_SHARE));
    const height = Math.min(REGION_MAX_HEIGHT, Math.round(gl.drawingBufferHeight * REGION_SHARE));
    if (width < 8 || height < 8) return;
    const x = (gl.drawingBufferWidth - width) >> 1;
    const y = (gl.drawingBufferHeight - height) >> 1;
    gl.readPixels(x, y, width, height, gl.RGBA, gl.UNSIGNED_BYTE, this.pixels);

    // The background colour as the canvas shows it (sRGB bytes).
    const bg = [background.r, background.g, background.b].map((c) => Math.round(linearToSrgb(c) * 255));
    let sum = 0;
    let count = 0;
    let clipped = 0;
    let sampled = 0;
    for (let row = 0; row < height; row += SAMPLE_STEP) {
      for (let column = 0; column < width; column += SAMPLE_STEP) {
        const i = (row * width + column) * 4;
        const [r, g, b] = [this.pixels[i], this.pixels[i + 1], this.pixels[i + 2]];
        sampled += 1;
        if (Math.abs(r - bg[0]) <= 2 && Math.abs(g - bg[1]) <= 2 && Math.abs(b - bg[2]) <= 2) continue;
        const luminance = LUMA[0] * SRGB_TO_LINEAR[r] + LUMA[1] * SRGB_TO_LINEAR[g] + LUMA[2] * SRGB_TO_LINEAR[b];
        sum += Math.log(Math.max(luminance, 1e-3));
        count += 1;
        if (Math.max(r, g, b) >= CLIPPED) clipped += 1;
      }
    }
    if (count < sampled / 10) return; // mostly sky: keep the current exposure
    const shown = Math.exp(sum / count);
    // The frame was drawn at exposure * compensation; adapt the exposure as if compensation were 1.
    let correction = ((KEY * compensation) / shown) ** FEEDBACK;
    if (clipped / count > MAX_CLIPPED_SHARE) correction = Math.min(correction, HIGHLIGHT_BACKOFF);
    this.targetExposure = Math.min(EXPOSURE_MAX, Math.max(EXPOSURE_MIN, this.exposure * correction));
  }
}

function linearToSrgb(c) {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
}
