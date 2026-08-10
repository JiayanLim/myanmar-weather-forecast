/**
 * Myanmar geographic mask at 0.05° display resolution.
 * Rasterizes the GeoJSON boundary via scanline fill (even-odd rule).
 */

export const DISPLAY_N_LAT = 401;   // (29 - 9) / 0.05 + 1
export const DISPLAY_N_LON = 201;   // (102 - 92) / 0.05 + 1
export const DISPLAY_STEP = 0.05;
export const DISPLAY_LAT_MAX = 29.0;
export const DISPLAY_LAT_MIN = 9.0;
export const DISPLAY_LON_MIN = 92.0;
export const DISPLAY_LON_MAX = 102.0;

type GeoRing = [number, number][]; // [lon, lat] per GeoJSON spec

/** Longitude values where a ring's edges cross the given latitude. */
function ringIntersections(ring: GeoRing, lat: number): number[] {
  const xs: number[] = [];
  for (let i = 0; i < ring.length - 1; i++) {
    const [lon0, lat0] = ring[i];
    const [lon1, lat1] = ring[i + 1];
    // Only count upward crossings (<) to avoid double-counting horizontal edges
    if ((lat0 <= lat && lat < lat1) || (lat1 <= lat && lat < lat0)) {
      const t = (lat - lat0) / (lat1 - lat0);
      xs.push(lon0 + t * (lon1 - lon0));
    }
  }
  return xs;
}

/**
 * Fetch a GeoJSON boundary file and rasterize it into a Uint8Array mask
 * at DISPLAY_N_LAT × DISPLAY_N_LON resolution.
 * mask[iLat * DISPLAY_N_LON + iLon] = 1 if inside Myanmar, 0 otherwise.
 * Row 0 = lat 29°N (north), row 400 = lat 9°N (south).
 */
export async function buildMaskFromGeojson(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  const gj = await res.json() as Record<string, unknown>;

  const rings: GeoRing[] = [];
  const features: Record<string, unknown>[] =
    gj['type'] === 'FeatureCollection'
      ? (gj['features'] as Record<string, unknown>[])
      : [gj];

  for (const feat of features) {
    const geom = (feat['geometry'] ?? feat) as { type: string; coordinates: unknown };
    if (geom.type === 'Polygon') {
      for (const ring of geom.coordinates as GeoRing[]) rings.push(ring);
    } else if (geom.type === 'MultiPolygon') {
      for (const poly of geom.coordinates as GeoRing[][]) {
        for (const ring of poly) rings.push(ring);
      }
    }
  }

  const mask = new Uint8Array(DISPLAY_N_LAT * DISPLAY_N_LON);

  for (let iLat = 0; iLat < DISPLAY_N_LAT; iLat++) {
    const lat = DISPLAY_LAT_MAX - iLat * DISPLAY_STEP;

    // Collect all edge intersections across all rings, then sort
    const allXs: number[] = [];
    for (const ring of rings) allXs.push(...ringIntersections(ring, lat));
    allXs.sort((a, b) => a - b);

    // Even-odd fill: fill between alternating pairs of intersections
    for (let k = 0; k + 1 < allXs.length; k += 2) {
      const iLonStart = Math.max(0, Math.ceil((allXs[k] - DISPLAY_LON_MIN) / DISPLAY_STEP));
      const iLonEnd = Math.min(DISPLAY_N_LON - 1, Math.floor((allXs[k + 1] - DISPLAY_LON_MIN) / DISPLAY_STEP));
      for (let iLon = iLonStart; iLon <= iLonEnd; iLon++) {
        mask[iLat * DISPLAY_N_LON + iLon] = 1;
      }
    }
  }

  return mask;
}
