/**
 * Color scales for weather variable visualization.
 * Each scale maps a normalized value [0, 1] to RGBA.
 */

import {
  DISPLAY_N_LAT, DISPLAY_N_LON, DISPLAY_STEP,
  DISPLAY_LAT_MAX, DISPLAY_LAT_MIN, DISPLAY_LON_MIN,
} from '../geo/mask';

export interface ColorStop {
  pos: number;  // 0–1
  r: number;
  g: number;
  b: number;
}

function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

export function buildLUT(stops: ColorStop[], size = 256): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(size * 4);
  for (let i = 0; i < size; i++) {
    const t = i / (size - 1);
    // Find surrounding stops
    let lo = stops[0];
    let hi = stops[stops.length - 1];
    for (let j = 0; j < stops.length - 1; j++) {
      if (t >= stops[j].pos && t <= stops[j + 1].pos) {
        lo = stops[j];
        hi = stops[j + 1];
        break;
      }
    }
    const span = hi.pos - lo.pos;
    const u = span > 0 ? (t - lo.pos) / span : 0;
    lut[i * 4 + 0] = lerp(lo.r, hi.r, u);
    lut[i * 4 + 1] = lerp(lo.g, hi.g, u);
    lut[i * 4 + 2] = lerp(lo.b, hi.b, u);
    lut[i * 4 + 3] = 200; // global alpha (out of 255)
  }
  return lut;
}

// Temperature: blue -> white -> orange -> red (15°C to 40°C)
export const TEMP_MIN = 15;
export const TEMP_MAX = 40;
export const TEMP_LUT = buildLUT([
  { pos: 0.00, r: 49,  g: 54,  b: 149 }, // 15°C  dark blue
  { pos: 0.20, r: 116, g: 173, b: 209 }, // 20°C  light blue
  { pos: 0.40, r: 224, g: 243, b: 248 }, // 25°C  near white
  { pos: 0.60, r: 254, g: 224, b: 144 }, // 30°C  pale yellow
  { pos: 0.80, r: 252, g: 141, b: 89  }, // 35°C  orange
  { pos: 1.00, r: 165, g: 0,   b: 38  }, // 40°C  dark red
]);

// Precipitation: transparent-white -> cyan -> blue -> green -> yellow -> red (0 to 100+ mm / 6h accumulation)
export const PRECIP_MIN = 0;
export const PRECIP_MAX = 100;
export const PRECIP_LUT = buildLUT([
  { pos: 0.00, r: 255, g: 255, b: 255 }, // 0       white (near transparent)
  { pos: 0.03, r: 200, g: 230, b: 255 }, // 3       pale blue
  { pos: 0.10, r: 100, g: 190, b: 255 }, // 10      sky blue
  { pos: 0.20, r: 30,  g: 130, b: 255 }, // 20      blue
  { pos: 0.35, r: 0,   g: 200, b: 150 }, // 35      teal-green
  { pos: 0.55, r: 50,  g: 220, b: 50  }, // 55      green
  { pos: 0.75, r: 255, g: 220, b: 0   }, // 75      yellow
  { pos: 0.90, r: 255, g: 120, b: 0   }, // 90      orange
  { pos: 1.00, r: 200, g: 0,   b: 0   }, // 100+    red
]);

// Modify PRECIP_LUT so values near 0 are mostly transparent
export const PRECIP_LUT_ALPHA = (() => {
  const lut = new Uint8ClampedArray(PRECIP_LUT);
  const size = lut.length / 4;
  for (let i = 0; i < size; i++) {
    const norm = i / (size - 1);
    // Fade transparency: 0 at norm=0, full at norm=0.05+
    const alpha = norm < 0.02 ? 0 : norm < 0.08 ? Math.round((norm - 0.02) / 0.06 * 200) : 200;
    lut[i * 4 + 3] = alpha;
  }
  return lut;
})();

export function applyColorscale(
  frame: Float32Array,
  lut: Uint8ClampedArray,
  vmin: number,
  vmax: number,
  nLat: number,
  nLon: number,
): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(nLat * nLon * 4);
  const range = vmax - vmin;
  for (let i = 0; i < nLat * nLon; i++) {
    const v = frame[i];
    if (isNaN(v) || v <= -9000) {
      // Transparent for NaN/fill
      rgba[i * 4 + 3] = 0;
      continue;
    }
    const norm = Math.max(0, Math.min(1, (v - vmin) / range));
    const idx = Math.round(norm * 255);
    rgba[i * 4 + 0] = lut[idx * 4 + 0];
    rgba[i * 4 + 1] = lut[idx * 4 + 1];
    rgba[i * 4 + 2] = lut[idx * 4 + 2];
    rgba[i * 4 + 3] = lut[idx * 4 + 3];
  }
  return rgba;
}

export const TEMP_TICKS = [15, 20, 25, 30, 35, 40];
export const PRECIP_TICKS = [0, 5, 10, 20, 40, 60, 100];

const MODEL_STEP = 1.0;

/**
 * Render weather data at display resolution (0.05°) using bilinear interpolation
 * from the model grid (1.0°), with optional Myanmar boundary masking.
 *
 * Model data layout: frame[latIdx * nLonSrc + lonIdx]
 *   latIdx 0 = 9°N (south), latIdx nLatSrc-1 = 29°N (north)
 *
 * Output: RGBA Uint8ClampedArray at DISPLAY_N_LAT × DISPLAY_N_LON.
 *   row 0 = 29°N (north top), row DISPLAY_N_LAT-1 = 9°N (south bottom)
 */
export function renderWithInterpolation(
  frame: Float32Array,
  lut: Uint8ClampedArray,
  vmin: number,
  vmax: number,
  nLatSrc: number,
  nLonSrc: number,
  mask: Uint8Array | null,
): Uint8ClampedArray {
  const nLatDst = DISPLAY_N_LAT;
  const nLonDst = DISPLAY_N_LON;
  const rgba = new Uint8ClampedArray(nLatDst * nLonDst * 4);
  const range = vmax - vmin;

  for (let iLat = 0; iLat < nLatDst; iLat++) {
    // Display lat decreases northward as row index increases
    const lat = DISPLAY_LAT_MAX - iLat * DISPLAY_STEP;
    // Fractional model lat index (model lat_i=0 = DISPLAY_LAT_MIN = 9°N)
    const fi = (lat - DISPLAY_LAT_MIN) / MODEL_STEP;
    const i0 = Math.max(0, Math.min(nLatSrc - 2, Math.floor(fi)));
    const i1 = i0 + 1;
    const ty = fi - i0;

    for (let iLon = 0; iLon < nLonDst; iLon++) {
      const pix = iLat * nLonDst + iLon;

      if (mask && mask[pix] === 0) {
        // Outside Myanmar boundary — leave transparent (rgba default = 0)
        continue;
      }

      const lon = DISPLAY_LON_MIN + iLon * DISPLAY_STEP;
      const fj = (lon - DISPLAY_LON_MIN) / MODEL_STEP;
      const j0 = Math.max(0, Math.min(nLonSrc - 2, Math.floor(fj)));
      const j1 = j0 + 1;
      const tx = fj - j0;

      // Bilinear interpolation across four model grid corners
      const v00 = frame[i0 * nLonSrc + j0];
      const v01 = frame[i0 * nLonSrc + j1];
      const v10 = frame[i1 * nLonSrc + j0];
      const v11 = frame[i1 * nLonSrc + j1];

      if (isNaN(v00) || v00 <= -9000) continue;

      const v = v00 * (1 - ty) * (1 - tx)
              + v01 * (1 - ty) * tx
              + v10 * ty * (1 - tx)
              + v11 * ty * tx;

      const norm = Math.max(0, Math.min(1, (v - vmin) / range));
      const idx = Math.round(norm * 255);
      rgba[pix * 4 + 0] = lut[idx * 4 + 0];
      rgba[pix * 4 + 1] = lut[idx * 4 + 1];
      rgba[pix * 4 + 2] = lut[idx * 4 + 2];
      rgba[pix * 4 + 3] = lut[idx * 4 + 3];
    }
  }

  return rgba;
}
