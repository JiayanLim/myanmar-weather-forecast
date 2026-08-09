import type { ForecastMetadata } from './types';

export interface ForecastData {
  metadata: ForecastMetadata;
  temperature: Float32Array;
  precipitation: Float32Array;
}

/**
 * Load forecast artifacts from the given base URL.
 * Returns metadata + flat Float32Array for each variable.
 * Layout: [n_times × n_lat × n_lon] C-order (row-major).
 */
export async function loadForecast(baseUrl: string): Promise<ForecastData> {
  const url = (path: string) => {
    const base = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/';
    return base + path;
  };

  // Load metadata first
  const metaRes = await fetch(url('forecast.json'));
  if (!metaRes.ok) throw new Error(`Failed to load forecast.json: ${metaRes.statusText}`);
  const metadata: ForecastMetadata = await metaRes.json();

  // Load binary arrays in parallel
  const [tempBuf, precipBuf] = await Promise.all([
    fetch(url('temperature.bin')).then((r) => {
      if (!r.ok) throw new Error(`Failed to load temperature.bin: ${r.statusText}`);
      return r.arrayBuffer();
    }),
    fetch(url('precipitation.bin')).then((r) => {
      if (!r.ok) throw new Error(`Failed to load precipitation.bin: ${r.statusText}`);
      return r.arrayBuffer();
    }),
  ]);

  return {
    metadata,
    temperature: new Float32Array(tempBuf),
    precipitation: new Float32Array(precipBuf),
  };
}

/**
 * Extract a single [n_lat × n_lon] frame from a flat Float32Array.
 * @param data   Full array [n_times × n_lat × n_lon]
 * @param hour   Time index (0 = init, 1..168 = forecast)
 * @param nLat   Number of latitude points
 * @param nLon   Number of longitude points
 */
export function getFrame(
  data: Float32Array,
  hour: number,
  nLat: number,
  nLon: number,
): Float32Array {
  const start = hour * nLat * nLon;
  return data.subarray(start, start + nLat * nLon);
}

/**
 * Get value at a specific lat/lon grid point.
 * Returns NaN if out of bounds.
 */
export function getPointValue(
  data: Float32Array,
  hour: number,
  latIdx: number,
  lonIdx: number,
  nLat: number,
  nLon: number,
): number {
  if (latIdx < 0 || latIdx >= nLat || lonIdx < 0 || lonIdx >= nLon) return NaN;
  return data[hour * nLat * nLon + latIdx * nLon + lonIdx];
}

/**
 * Find nearest grid indices for a lat/lon point.
 * Returns null if outside the grid bbox.
 */
export function nearestGridPoint(
  lat: number,
  lon: number,
  lats: number[],
  lons: number[],
): { latIdx: number; lonIdx: number } | null {
  if (
    lat < lats[0] - 0.125 ||
    lat > lats[lats.length - 1] + 0.125 ||
    lon < lons[0] - 0.125 ||
    lon > lons[lons.length - 1] + 0.125
  ) {
    return null;
  }
  // Find nearest index
  const step = lats[1] - lats[0]; // 0.25
  const latIdx = Math.round((lat - lats[0]) / step);
  const lonIdx = Math.round((lon - lons[0]) / step);
  return {
    latIdx: Math.max(0, Math.min(lats.length - 1, latIdx)),
    lonIdx: Math.max(0, Math.min(lons.length - 1, lonIdx)),
  };
}
