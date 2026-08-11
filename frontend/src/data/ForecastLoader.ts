import type { ForecastMetadata } from './types';

export interface ForecastData {
  metadata: ForecastMetadata;
  precipitation: Float32Array;
  temperature: Float32Array;
}

/**
 * Load forecast artifacts from the given base URL.
 * Returns metadata + flat Float32Arrays for precipitation and temperature.
 * Layout: [n_times × n_lat × n_lon] C-order (row-major).
 *
 * Binary file names are taken from metadata.variables.*.file.
 */
export async function loadForecast(baseUrl: string): Promise<ForecastData> {
  const url = (path: string) => {
    const base = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/';
    return base + path;
  };

  const metaRes = await fetch(url('forecast.json'));
  if (!metaRes.ok) throw new Error(`Failed to load forecast.json: ${metaRes.statusText}`);
  const metadata: ForecastMetadata = await metaRes.json();

  const precipFile = metadata.variables.precipitation.file;
  const tempFile = metadata.variables.temperature.file;

  const [precipBuf, tempBuf] = await Promise.all([
    fetch(url(precipFile)).then((r) => {
      if (!r.ok) throw new Error(`Failed to load ${precipFile}: ${r.statusText}`);
      return r.arrayBuffer();
    }),
    fetch(url(tempFile)).then((r) => {
      if (!r.ok) throw new Error(`Failed to load ${tempFile}: ${r.statusText}`);
      return r.arrayBuffer();
    }),
  ]);

  return {
    metadata,
    precipitation: new Float32Array(precipBuf),
    temperature: new Float32Array(tempBuf),
  };
}

/**
 * Extract a single [n_lat × n_lon] frame from a flat Float32Array.
 * @param data   Full array [n_times × n_lat × n_lon]
 * @param hour   Frame index (0 = init, 1..n_times-1 = forecast)
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
  const step = lats[1] - lats[0];
  const half = step / 2;
  if (
    lat < lats[0] - half ||
    lat > lats[lats.length - 1] + half ||
    lon < lons[0] - half ||
    lon > lons[lons.length - 1] + half
  ) {
    return null;
  }
  const latIdx = Math.round((lat - lats[0]) / step);
  const lonIdx = Math.round((lon - lons[0]) / step);
  return {
    latIdx: Math.max(0, Math.min(lats.length - 1, latIdx)),
    lonIdx: Math.max(0, Math.min(lons.length - 1, lonIdx)),
  };
}
