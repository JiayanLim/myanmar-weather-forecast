import { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import { useForecastStore } from '../data/ForecastStore';
import { getFrame, getPointValue, nearestGridPoint } from '../data/ForecastLoader';
import {
  renderWithInterpolation,
  renderWindDirectionWithInterpolation,
  PRECIP_LUT_ALPHA, PRECIP_MIN, PRECIP_MAX,
  TEMP_LUT,        TEMP_MIN,   TEMP_MAX,
  WIND_LUT_ALPHA,  WIND_MIN,   WIND_MAX,
} from './colorscales';
import { DISPLAY_N_LAT, DISPLAY_N_LON } from '../geo/mask';

const MYANMAR_CENTER: [number, number] = [96.5, 19.0];

/** Compass bearing label for a FROM direction in degrees. */
function compassLabel(deg: number): string {
  const dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];
  return dirs[Math.round(deg / 22.5) % 16];
}

export function WeatherMap() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const overlayCanvas = useRef<HTMLCanvasElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);

  const {
    metadata,
    precipitation, temperature, windSpeed, windDirection,
    activeVariable,
    currentHour,
    isLoaded, setInspectorPoint, inspectorPoint,
    mask,
  } = useForecastStore();

  const positionOverlay = (map: maplibregl.Map, meta: typeof metadata) => {
    const canvas = overlayCanvas.current;
    if (!canvas || !meta) return;
    const { lon_min, lat_max, lon_max, lat_min } = meta.bbox;
    const nw = map.project([lon_min, lat_max]);
    const se = map.project([lon_max, lat_min]);
    canvas.style.left   = `${nw.x}px`;
    canvas.style.top    = `${nw.y}px`;
    canvas.style.width  = `${Math.max(1, se.x - nw.x)}px`;
    canvas.style.height = `${Math.max(1, se.y - nw.y)}px`;
  };

  // Initialise map
  useEffect(() => {
    if (!mapContainer.current) return;

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxzoom: 19,
          },
        },
        layers: [
          {
            id: 'osm',
            type: 'raster',
            source: 'osm',
            paint: { 'raster-opacity': 0.55, 'raster-saturation': -0.4, 'raster-brightness-min': 0.05 },
          },
        ],
      },
      center: MYANMAR_CENTER,
      zoom: 5.2,
      maxBounds: [[88, 5], [108, 32]],
      minZoom: 4,
      maxZoom: 10,
    });

    mapRef.current = map;

    map.on('load', () => {
      map.addSource('myanmar-boundary', {
        type: 'geojson',
        data: './geo/myanmar-boundary.geojson',
      });

      map.addLayer({
        id: 'myanmar-fill',
        type: 'fill',
        source: 'myanmar-boundary',
        paint: {
          'fill-color': 'rgba(255,255,255,0.03)',
          'fill-outline-color': 'rgba(255,255,255,0.0)',
        },
      });

      map.addLayer({
        id: 'myanmar-outline',
        type: 'line',
        source: 'myanmar-boundary',
        paint: {
          'line-color': 'rgba(255, 255, 255, 0.7)',
          'line-width': 1.5,
        },
      });

      positionOverlay(map, useForecastStore.getState().metadata);
    });

    const onMove = () => positionOverlay(map, useForecastStore.getState().metadata);
    map.on('move', onMove);
    map.on('resize', onMove);

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');

    map.on('click', (e) => {
      const { lng, lat } = e.lngLat;
      useForecastStore.getState().setInspectorPoint({ lat, lon: lng });
    });

    map.on('mousemove', () => {
      map.getCanvas().style.cursor = 'crosshair';
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Draw weather overlay onto the canvas
  useEffect(() => {
    const canvas = overlayCanvas.current;
    const map = mapRef.current;
    if (!canvas || !isLoaded || !metadata) return;

    const { n_lat, n_lon } = metadata.grid;
    // MODEL_STEP derived at runtime from metadata (ADR-019 / FR-N11)
    const modelStep = metadata.native_resolution_deg;

    let rgba: Uint8ClampedArray;

    if (activeVariable === 'wind_direction' && windDirection) {
      const frame = getFrame(windDirection, currentHour, n_lat, n_lon);
      // Vector-component bilinear interpolation (ADR-020) — raw degrees never interpolated
      rgba = renderWindDirectionWithInterpolation(frame, modelStep, n_lat, n_lon, mask);
    } else {
      const [data, lut, vmin, vmax] =
        activeVariable === 'temperature' && temperature
          ? [temperature, TEMP_LUT,       TEMP_MIN, TEMP_MAX]
          : activeVariable === 'wind_speed' && windSpeed
          ? [windSpeed,   WIND_LUT_ALPHA, WIND_MIN, WIND_MAX]
          : precipitation
          ? [precipitation, PRECIP_LUT_ALPHA, PRECIP_MIN, PRECIP_MAX]
          : [null, null, 0, 1];

      if (!data || !lut) return;

      const frame = getFrame(data, currentHour, n_lat, n_lon);
      rgba = renderWithInterpolation(frame, lut, vmin, vmax, modelStep, n_lat, n_lon, mask);
    }

    canvas.width  = DISPLAY_N_LON;
    canvas.height = DISPLAY_N_LAT;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.putImageData(
        new ImageData(new Uint8ClampedArray(rgba.buffer as ArrayBuffer), DISPLAY_N_LON, DISPLAY_N_LAT),
        0, 0,
      );
    }

    if (map) positionOverlay(map, metadata);
  }, [isLoaded, metadata, currentHour, precipitation, temperature, windSpeed, windDirection, activeVariable, mask]);

  // Point inspector popup
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !isLoaded || !metadata) return;

    if (popupRef.current) {
      popupRef.current.remove();
      popupRef.current = null;
    }

    if (!inspectorPoint) return;

    const { lat, lon } = inspectorPoint;
    const { n_lat, n_lon } = metadata.grid;
    const grid = nearestGridPoint(lat, lon, metadata.lat, metadata.lon);

    let content: string;

    if (!grid) {
      content = `
        <div class="wx-popup">
          <div class="wx-popup-title">Outside forecast domain</div>
          <p class="wx-popup-note">No forecast data for this location.</p>
        </div>`;
    } else {
      const precipVal = precipitation
        ? getPointValue(precipitation, currentHour, grid.latIdx, grid.lonIdx, n_lat, n_lon)
        : NaN;
      const tempVal = temperature
        ? getPointValue(temperature, currentHour, grid.latIdx, grid.lonIdx, n_lat, n_lon)
        : NaN;
      const wsVal = windSpeed
        ? getPointValue(windSpeed, currentHour, grid.latIdx, grid.lonIdx, n_lat, n_lon)
        : NaN;
      const wdVal = windDirection
        ? getPointValue(windDirection, currentHour, grid.latIdx, grid.lonIdx, n_lat, n_lon)
        : NaN;

      const validTime = metadata.times_utc[currentHour] ?? '';
      const dt = new Date(validTime);
      const timeStr = dt.toUTCString().replace(' GMT', ' UTC');
      const stepHours = metadata.native_timestep_hours ?? 6;
      const leadH = currentHour * stepHours;

      const fmt = (v: number, dp: number, unit: string) =>
        isNaN(v) ? '—' : `${v.toFixed(dp)} ${unit}`;

      const wdStr = isNaN(wdVal)
        ? '—'
        : `${Math.round(wdVal)}° FROM ${compassLabel(wdVal)}`;

      // Determine active variable label and value for primary display
      const [activeLabel, activeValStr] =
        activeVariable === 'temperature'
          ? ['Temperature',  fmt(tempVal,   1, '°C')]
          : activeVariable === 'wind_speed'
          ? ['Wind Speed',   fmt(wsVal,     1, 'kt')]
          : activeVariable === 'wind_direction'
          ? ['Wind Direction', wdStr]
          : ['Precipitation', fmt(precipVal, 2, 'mm/hr')];

      content = `
        <div class="wx-popup">
          <div class="wx-popup-title">Myanmar</div>
          <div class="wx-popup-time">${timeStr}</div>
          <div class="wx-popup-row">
            <span class="wx-popup-label">${activeLabel}</span>
            <span class="wx-popup-value">${activeValStr}</span>
          </div>
          ${activeVariable === 'precipitation'
            ? `<div class="wx-popup-note">Estimated avg. rainfall rate (6h window)</div>`
            : activeVariable === 'wind_direction'
            ? `<div class="wx-popup-note">Meteorological FROM direction</div>`
            : ''
          }
          <hr style="border:none;border-top:1px solid rgba(255,255,255,0.1);margin:6px 0"/>
          <div class="wx-popup-row" style="opacity:0.65">
            <span class="wx-popup-label">Temp</span>
            <span class="wx-popup-value">${fmt(tempVal, 1, '°C')}</span>
          </div>
          <div class="wx-popup-row" style="opacity:0.65">
            <span class="wx-popup-label">Precip</span>
            <span class="wx-popup-value">${fmt(precipVal, 2, 'mm/hr')}</span>
          </div>
          <div class="wx-popup-row" style="opacity:0.65">
            <span class="wx-popup-label">Wind</span>
            <span class="wx-popup-value">${fmt(wsVal, 1, 'kt')} ${isNaN(wdVal) ? '' : compassLabel(wdVal)}</span>
          </div>
          <div class="wx-popup-coords">${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E · +${leadH}h lead</div>
        </div>`;
    }

    const popup = new maplibregl.Popup({
      closeButton: true,
      closeOnClick: false,
      maxWidth: '260px',
      className: 'wx-maplibre-popup',
    })
      .setLngLat([lon, lat])
      .setHTML(content)
      .addTo(map);

    popup.on('close', () => {
      useForecastStore.getState().setInspectorPoint(null);
    });

    popupRef.current = popup;
  }, [inspectorPoint, currentHour, isLoaded, metadata, precipitation, temperature, windSpeed, windDirection, activeVariable]);

  return (
    <div style={{ position: 'absolute', inset: 0, background: '#0f1117' }}>
      <div ref={mapContainer} style={{ position: 'absolute', inset: 0 }} />
      <canvas
        ref={overlayCanvas}
        style={{
          position: 'absolute',
          pointerEvents: 'none',
          opacity: 0.85,
          imageRendering: 'pixelated',
          display: isLoaded ? 'block' : 'none',
        }}
      />
    </div>
  );
}
