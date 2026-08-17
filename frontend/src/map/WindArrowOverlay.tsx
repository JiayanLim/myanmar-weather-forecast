/**
 * SVG wind arrow overlay (Phase R12, FR-W01–FR-W07, ADR-025).
 *
 * Replaces the R11 canvas-drawn arrows which had a distribution defect:
 * the minimum-length filter (len < 2px) suppressed ~P75+ points, causing arrows
 * to appear only in southern Myanmar. This component uses map.project([lon,lat])
 * for Mercator-correct screen positioning and guarantees a minimum visible arrow
 * length for any non-calm wind point.
 *
 * Direction convention (FR-W04):
 *   Binary stores FROM direction (meteorological convention).
 *   Arrow renders TO direction = (fromDir + 180) % 360.
 *   SVG rotate(toDeg) with 0°=north, clockwise matches screen map convention.
 *   Sanity: 90°FROM east  → rotate(270°) → arrow points west  ✓
 *           180°FROM south → rotate(0°)   → arrow points north ✓
 */

import { useEffect, useState, useCallback } from 'react';
import type maplibregl from 'maplibre-gl';
import { useForecastStore } from '../data/ForecastStore';
import { getFrame } from '../data/ForecastLoader';

const WIND_ARROW_GRID_STEP = 3; // every 3rd model grid point (FR-W02)
const CALM_KT = 2.0;            // calm threshold — no arrow below this (FR-W03)
const MIN_ARROW_PX = 8;         // guaranteed minimum length for any wind ≥ CALM_KT (FR-W05)
const MAX_ARROW_PX = 22;        // maximum length at 30 kt

interface ArrowData {
  x: number;
  y: number;
  toDeg: number;
  len: number;
}

interface Props {
  map: maplibregl.Map | null;
}

export function WindArrowOverlay({ map }: Props) {
  const [arrows, setArrows] = useState<ArrowData[]>([]);

  const windSpeed      = useForecastStore((s) => s.windSpeed);
  const windDirection  = useForecastStore((s) => s.windDirection);
  const currentHour    = useForecastStore((s) => s.currentHour);
  const metadata       = useForecastStore((s) => s.metadata);
  const activeVariable = useForecastStore((s) => s.activeVariable);
  const isLoaded       = useForecastStore((s) => s.isLoaded);

  const rebuild = useCallback(() => {
    if (
      !map ||
      !isLoaded ||
      !metadata ||
      !windSpeed ||
      !windDirection ||
      (activeVariable !== 'wind_speed' && activeVariable !== 'wind_direction')
    ) {
      setArrows([]);
      return;
    }

    const { n_lat, n_lon } = metadata.grid;
    const modelStep = metadata.native_resolution_deg;
    const latMin = metadata.bbox.lat_min;
    const lonMin = metadata.bbox.lon_min;

    const wsFrame = getFrame(windSpeed,    currentHour, n_lat, n_lon);
    const wdFrame = getFrame(windDirection, currentHour, n_lat, n_lon);

    const result: ArrowData[] = [];

    for (let latIdx = 0; latIdx < n_lat; latIdx += WIND_ARROW_GRID_STEP) {
      for (let lonIdx = 0; lonIdx < n_lon; lonIdx += WIND_ARROW_GRID_STEP) {
        const i = latIdx * n_lon + lonIdx;
        const speed   = wsFrame[i];
        const fromDir = wdFrame[i];

        // Skip calm or invalid points (FR-W03)
        if (!isFinite(speed) || !isFinite(fromDir) || speed < CALM_KT) continue;

        const lat = latMin + latIdx * modelStep;
        const lon = lonMin + lonIdx * modelStep;

        // Mercator-correct screen position via map.project (FR-W01, ADR-025)
        const pt = map.project([lon, lat] as [number, number]);

        // TO direction = FROM + 180° mod 360 (FR-W04)
        const toDeg = (fromDir + 180) % 360;

        // Arrow length: guaranteed minimum for any non-calm point (FR-W05)
        // Linear scale: CALM_KT → MIN_ARROW_PX, 30 kt → MAX_ARROW_PX
        const t = Math.min(1, Math.max(0, (speed - CALM_KT) / (30 - CALM_KT)));
        const len = MIN_ARROW_PX + t * (MAX_ARROW_PX - MIN_ARROW_PX);

        result.push({ x: pt.x, y: pt.y, toDeg, len });
      }
    }

    setArrows(result);
  }, [map, isLoaded, metadata, windSpeed, windDirection, currentHour, activeVariable]);

  // Recompute when data or frame changes
  useEffect(() => {
    rebuild();
  }, [rebuild]);

  // Recompute on map pan / zoom / resize (FR-W01)
  useEffect(() => {
    if (!map) return;
    map.on('move', rebuild);
    map.on('resize', rebuild);
    return () => {
      map.off('move', rebuild);
      map.off('resize', rebuild);
    };
  }, [map, rebuild]);

  if (activeVariable !== 'wind_speed' && activeVariable !== 'wind_direction') return null;
  if (arrows.length === 0) return null;

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        overflow: 'hidden',
      }}
    >
      <svg
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          overflow: 'visible',
        }}
        aria-hidden="true"
      >
        {arrows.map((a, idx) => {
          // Arrow geometry in local (unrotated) space, pointing up = north:
          //   shaft: from tailY (below centre) to headY (above centre)
          //   head:  filled triangle at the tip (headY)
          const tailY = Math.round(a.len * 0.30);   // tail offset below centre
          const headY = -Math.round(a.len * 0.70);  // head above centre
          const hw    = Math.max(3, Math.round(a.len * 0.28)); // arrowhead half-width
          const hb    = Math.round(a.len * 0.35);              // arrowhead base height

          return (
            <g
              key={idx}
              transform={`translate(${a.x.toFixed(1)},${a.y.toFixed(1)}) rotate(${a.toDeg.toFixed(1)})`}
            >
              {/* Shaft */}
              <line
                x1={0} y1={tailY}
                x2={0} y2={headY}
                stroke="white"
                strokeOpacity={0.88}
                strokeWidth={1.6}
                strokeLinecap="round"
              />
              {/* Arrowhead: triangle with tip at (0, headY) */}
              <polygon
                points={`0,${headY} ${-hw},${headY + hb} ${hw},${headY + hb}`}
                fill="white"
                fillOpacity={0.88}
              />
            </g>
          );
        })}
      </svg>
    </div>
  );
}
