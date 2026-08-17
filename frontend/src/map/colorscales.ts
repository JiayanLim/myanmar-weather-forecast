/**
 * Color scales for weather variable visualization.
 * Each scale maps a normalized value [0, 1] to RGBA.
 */

import {
  DISPLAY_N_LAT, DISPLAY_N_LON, DISPLAY_STEP,
  DISPLAY_LAT_MAX, DISPLAY_LAT_MIN, DISPLAY_LON_MIN, DISPLAY_LON_MAX,
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
    lut[i * 4 + 3] = 200;
  }
  return lut;
}

// Temperature: blue → white → orange → red (15°C to 40°C)
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

// Precipitation: near-transparent → cyan → blue → green → yellow → red (0 to 2 mm/hr)
// v4.0 unit: mm/hr (estimated average rainfall rate)
// Scale calibrated for January 2021 dry-season dataset (FR-N23a):
//   max observed = 1.62 mm/hr (no saturation); P95 = 0.084 mm/hr (visible above cutoff).
//   Monsoon-season data may saturate above 2 mm/hr — recalibration required for wet-season use.
export const PRECIP_MIN = 0;
export const PRECIP_MAX = 2;
export const PRECIP_LUT = buildLUT([
  { pos: 0.00, r: 255, g: 255, b: 255 }, // 0       white
  { pos: 0.05, r: 200, g: 230, b: 255 }, // 0.5     pale blue
  { pos: 0.10, r: 100, g: 190, b: 255 }, // 1       sky blue
  { pos: 0.20, r: 30,  g: 130, b: 255 }, // 2       blue
  { pos: 0.35, r: 0,   g: 200, b: 150 }, // 3.5     teal-green
  { pos: 0.55, r: 50,  g: 220, b: 50  }, // 5.5     green
  { pos: 0.75, r: 255, g: 220, b: 0   }, // 7.5     yellow
  { pos: 0.90, r: 255, g: 120, b: 0   }, // 9       orange
  { pos: 1.00, r: 200, g: 0,   b: 0   }, // 10+     red
]);

// Near-zero precipitation rendered as fully transparent (noise suppression).
// Cutoff: norm < 0.01 → transparent (= 0.02 mm/hr at PRECIP_MAX=2).
// Ramp: norm 0.01–0.05 → partial opacity (= 0.02–0.10 mm/hr). Above 0.05 → full opacity.
export const PRECIP_LUT_ALPHA = (() => {
  const lut = new Uint8ClampedArray(PRECIP_LUT);
  const size = lut.length / 4;
  for (let i = 0; i < size; i++) {
    const norm = i / (size - 1);
    const alpha = norm < 0.01 ? 0 : norm < 0.05 ? Math.round((norm - 0.01) / 0.04 * 200) : 200;
    lut[i * 4 + 3] = alpha;
  }
  return lut;
})();

// Wind speed: calm/transparent → light blue → green → yellow → red (0 to 30 kt)
export const WIND_MIN = 0;
export const WIND_MAX = 30;
export const WIND_LUT = buildLUT([
  { pos: 0.00, r: 255, g: 255, b: 255 }, // 0      white
  { pos: 0.10, r: 180, g: 220, b: 255 }, // 3 kt   pale blue
  { pos: 0.20, r: 80,  g: 200, b: 255 }, // 6 kt   sky blue
  { pos: 0.33, r: 0,   g: 200, b: 100 }, // 10 kt  green
  { pos: 0.50, r: 100, g: 220, b: 50  }, // 15 kt  yellow-green
  { pos: 0.67, r: 255, g: 220, b: 0   }, // 20 kt  yellow
  { pos: 0.83, r: 255, g: 120, b: 0   }, // 25 kt  orange
  { pos: 1.00, r: 200, g: 0,   b: 0   }, // 30+ kt red
]);

export const WIND_LUT_ALPHA = (() => {
  const lut = new Uint8ClampedArray(WIND_LUT);
  const size = lut.length / 4;
  for (let i = 0; i < size; i++) {
    const norm = i / (size - 1);
    const alpha = norm < 0.015 ? 0 : norm < 0.06 ? Math.round((norm - 0.015) / 0.045 * 200) : 200;
    lut[i * 4 + 3] = alpha;
  }
  return lut;
})();

export const TEMP_TICKS   = [15, 20, 25, 30, 35, 40];
export const PRECIP_TICKS = [0, 0.1, 0.25, 0.5, 1.0, 2.0]; // calibrated for Jan 2021 dry season
export const WIND_TICKS   = [0, 5, 10, 15, 20, 25, 30];

/** HSL to RGB conversion. h in [0,360], s and l in [0,1]. */
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, Math.min(9 - k, 1)));
  };
  return [Math.round(f(0) * 255), Math.round(f(8) * 255), Math.round(f(4) * 255)];
}

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

/**
 * Render scalar weather data at display resolution (0.05°) using bilinear
 * interpolation from the model grid, with optional Myanmar boundary masking.
 *
 * modelStep is derived at runtime from metadata.native_resolution_deg (e.g. 0.25).
 * MODEL_STEP is NOT hardcoded — FR-N11 (ADR-019).
 *
 * Model data layout: frame[latIdx * nLonSrc + lonIdx]
 *   latIdx 0 = lat_min (south), latIdx nLatSrc-1 = lat_max (north)
 *
 * Output: RGBA Uint8ClampedArray at DISPLAY_N_LAT × DISPLAY_N_LON.
 *   row 0 = lat_max (north top), row DISPLAY_N_LAT-1 = lat_min (south bottom)
 */
export function renderWithInterpolation(
  frame: Float32Array,
  lut: Uint8ClampedArray,
  vmin: number,
  vmax: number,
  modelStep: number,
  nLatSrc: number,
  nLonSrc: number,
  mask: Uint8Array | null,
): Uint8ClampedArray {
  const nLatDst = DISPLAY_N_LAT;
  const nLonDst = DISPLAY_N_LON;
  const rgba = new Uint8ClampedArray(nLatDst * nLonDst * 4);
  const range = vmax - vmin;

  for (let iLat = 0; iLat < nLatDst; iLat++) {
    const lat = DISPLAY_LAT_MAX - iLat * DISPLAY_STEP;
    const fi = (lat - DISPLAY_LAT_MIN) / modelStep;
    const i0 = Math.max(0, Math.min(nLatSrc - 2, Math.floor(fi)));
    const i1 = i0 + 1;
    const ty = fi - i0;

    for (let iLon = 0; iLon < nLonDst; iLon++) {
      const pix = iLat * nLonDst + iLon;

      if (mask && mask[pix] === 0) continue;

      const lon = DISPLAY_LON_MIN + iLon * DISPLAY_STEP;
      const fj = (lon - DISPLAY_LON_MIN) / modelStep;
      const j0 = Math.max(0, Math.min(nLonSrc - 2, Math.floor(fj)));
      const j1 = j0 + 1;
      const tx = fj - j0;

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

/**
 * Sampling step for wind arrow overlay (FR-W02).
 * Every WIND_ARROW_GRID_STEP-th model grid point (lat and lon) is considered.
 * Start at 3; adjust to 2 (denser) or 4 (sparser) based on visual testing.
 */
export const WIND_ARROW_GRID_STEP = 3;

/** Calm-wind threshold below which no arrow is drawn (FR-W03, matches verification calm threshold). */
const WIND_ARROW_CALM_KT = 2.0;

/**
 * Draw wind arrows onto a canvas 2D context (FR-W01–FR-W05).
 *
 * Arrows represent the TO direction — the direction the wind blows toward —
 * which is opposite to the stored meteorological FROM convention (FR-W04).
 *   Stored: direction FROM which wind blows.
 *   Arrow head points: (stored_direction + 180) mod 360°.
 *   Sanity: 90°FROM east → arrow points west; 180°FROM south → arrow points north.
 *
 * Arrow length is proportional to wind speed (FR-W05).
 * Calm points (speed < 2 kt) produce no arrow (FR-W03).
 *
 * @param ctx           Canvas 2D rendering context
 * @param wsFrame       Wind speed frame [nLatSrc × nLonSrc] in knots
 * @param wdFrame       Wind direction frame [nLatSrc × nLonSrc] in degrees FROM
 * @param modelStep     Native resolution in degrees (e.g. 0.25)
 * @param nLatSrc       Number of source latitude points
 * @param nLonSrc       Number of source longitude points
 * @param canvasPxW     Rendered canvas CSS pixel width (from map projection)
 * @param canvasPxH     Rendered canvas CSS pixel height (from map projection)
 */
export function drawWindArrows(
  ctx: CanvasRenderingContext2D,
  wsFrame: Float32Array,
  wdFrame: Float32Array,
  modelStep: number,
  nLatSrc: number,
  nLonSrc: number,
  canvasPxW: number,
  canvasPxH: number,
): void {
  // How many display pixels correspond to one model grid step
  const pxPerDegLon = canvasPxW / (DISPLAY_LON_MAX - DISPLAY_LON_MIN);
  const pxPerDegLat = canvasPxH / (DISPLAY_LAT_MAX - DISPLAY_LAT_MIN);
  const arrowSpacingPx = modelStep * WIND_ARROW_GRID_STEP * Math.min(pxPerDegLon, pxPerDegLat);
  // Max arrow body length: half the inter-arrow spacing
  const maxSpeedKt = WIND_MAX; // 30 kt
  const maxLen = arrowSpacingPx * 0.45;

  const DEG2RAD = Math.PI / 180;

  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.85)';
  ctx.fillStyle   = 'rgba(255,255,255,0.85)';
  ctx.lineWidth   = 1.2;
  ctx.lineCap     = 'round';

  for (let latIdx = 0; latIdx < nLatSrc; latIdx += WIND_ARROW_GRID_STEP) {
    for (let lonIdx = 0; lonIdx < nLonSrc; lonIdx += WIND_ARROW_GRID_STEP) {
      const i = latIdx * nLonSrc + lonIdx;
      const speed = wsFrame[i];
      const fromDir = wdFrame[i];

      // Skip calm or invalid points (FR-W03)
      if (!isFinite(speed) || !isFinite(fromDir) || speed < WIND_ARROW_CALM_KT) continue;

      // Compute canvas pixel position for this grid point
      // lat[latIdx] = DISPLAY_LAT_MIN + latIdx * modelStep (ascending south→north)
      const lat = DISPLAY_LAT_MIN + latIdx * modelStep;
      const lon = DISPLAY_LON_MIN + lonIdx * modelStep;
      // Canvas row 0 = DISPLAY_LAT_MAX (north top)
      const cx = (lon - DISPLAY_LON_MIN) / (DISPLAY_LON_MAX - DISPLAY_LON_MIN) * canvasPxW;
      const cy = (DISPLAY_LAT_MAX - lat) / (DISPLAY_LAT_MAX - DISPLAY_LAT_MIN) * canvasPxH;

      // Arrow TO direction = FROM + 180° (FR-W04)
      const toDir = (fromDir + 180) % 360;
      // Convert meteorological angle to math angle:
      // Met: 0°=N, 90°=E, clockwise. Math: 0°=East, counterclockwise.
      // Math angle = 90° - toDir (in degrees), then to radians
      const mathRad = (90 - toDir) * DEG2RAD;

      // Arrow length proportional to speed
      const len = Math.min(maxLen, (speed / maxSpeedKt) * maxLen);
      if (len < 2) continue; // too short to draw

      const dx = Math.cos(mathRad) * len;
      const dy = -Math.sin(mathRad) * len; // canvas y is flipped

      // Tail point, head point
      const tx = cx - dx * 0.5;
      const ty = cy - dy * 0.5;
      const hx = cx + dx * 0.5;
      const hy = cy + dy * 0.5;

      // Draw shaft
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(hx, hy);
      ctx.stroke();

      // Draw arrowhead (filled triangle)
      const headLen = Math.max(3, len * 0.35);
      const headAngle = 0.45; // radians (~26°)
      const ax1 = hx - headLen * Math.cos(mathRad - headAngle);
      const ay1 = hy + headLen * Math.sin(mathRad - headAngle);
      const ax2 = hx - headLen * Math.cos(mathRad + headAngle);
      const ay2 = hy + headLen * Math.sin(mathRad + headAngle);

      ctx.beginPath();
      ctx.moveTo(hx, hy);
      ctx.lineTo(ax1, ay1);
      ctx.lineTo(ax2, ay2);
      ctx.closePath();
      ctx.fill();
    }
  }

  ctx.restore();
}

/**
 * Render wind direction at display resolution using vector-component bilinear
 * interpolation (ADR-020). Raw degree values are NEVER interpolated directly.
 *
 * Algorithm per display pixel:
 *   1. Identify four surrounding model grid corners.
 *   2. Convert each corner's direction (degrees FROM, 0–360) to (sin, cos).
 *   3. Bilinear-interpolate sin and cos separately.
 *   4. Reconstruct: dir = atan2(sinInterp, cosInterp) → degrees [0, 360).
 *   5. Map direction to HSL hue (direction → hue in degrees) for circular coloring.
 *
 * Color convention: N(0°)→red  E(90°)→yellow-green  S(180°)→cyan  W(270°)→blue
 */
export function renderWindDirectionWithInterpolation(
  frame: Float32Array,
  modelStep: number,
  nLatSrc: number,
  nLonSrc: number,
  mask: Uint8Array | null,
): Uint8ClampedArray {
  const nLatDst = DISPLAY_N_LAT;
  const nLonDst = DISPLAY_N_LON;
  const rgba = new Uint8ClampedArray(nLatDst * nLonDst * 4);
  const DEG2RAD = Math.PI / 180;

  for (let iLat = 0; iLat < nLatDst; iLat++) {
    const lat = DISPLAY_LAT_MAX - iLat * DISPLAY_STEP;
    const fi = (lat - DISPLAY_LAT_MIN) / modelStep;
    const i0 = Math.max(0, Math.min(nLatSrc - 2, Math.floor(fi)));
    const i1 = i0 + 1;
    const ty = fi - i0;

    for (let iLon = 0; iLon < nLonDst; iLon++) {
      const pix = iLat * nLonDst + iLon;

      if (mask && mask[pix] === 0) continue;

      const lon = DISPLAY_LON_MIN + iLon * DISPLAY_STEP;
      const fj = (lon - DISPLAY_LON_MIN) / modelStep;
      const j0 = Math.max(0, Math.min(nLonSrc - 2, Math.floor(fj)));
      const j1 = j0 + 1;
      const tx = fj - j0;

      // Raw degrees FROM at the four grid corners
      const d00 = frame[i0 * nLonSrc + j0];
      const d01 = frame[i0 * nLonSrc + j1];
      const d10 = frame[i1 * nLonSrc + j0];
      const d11 = frame[i1 * nLonSrc + j1];

      if (isNaN(d00) || d00 < 0) continue;

      // Convert each corner to unit vector components (sin, cos in radians)
      const r00 = d00 * DEG2RAD;
      const r01 = d01 * DEG2RAD;
      const r10 = d10 * DEG2RAD;
      const r11 = d11 * DEG2RAD;

      // Bilinear interpolate sin and cos separately (ADR-020)
      const w00 = (1 - ty) * (1 - tx);
      const w01 = (1 - ty) * tx;
      const w10 = ty * (1 - tx);
      const w11 = ty * tx;

      const sinInterp = Math.sin(r00) * w00 + Math.sin(r01) * w01
                      + Math.sin(r10) * w10 + Math.sin(r11) * w11;
      const cosInterp = Math.cos(r00) * w00 + Math.cos(r01) * w01
                      + Math.cos(r10) * w10 + Math.cos(r11) * w11;

      // Reconstruct interpolated direction [0, 360)
      const dirDeg = ((Math.atan2(sinInterp, cosInterp) / DEG2RAD) + 360) % 360;

      // Map direction to HSL hue for circular visualization
      const [r, g, b] = hslToRgb(dirDeg, 0.85, 0.5);
      rgba[pix * 4 + 0] = r;
      rgba[pix * 4 + 1] = g;
      rgba[pix * 4 + 2] = b;
      rgba[pix * 4 + 3] = 200;
    }
  }

  return rgba;
}
