import { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import { useForecastStore } from '../data/ForecastStore';
import { getFrame, getPointValue, nearestGridPoint } from '../data/ForecastLoader';
import { renderWithInterpolation, PRECIP_LUT_ALPHA, PRECIP_MIN, PRECIP_MAX } from './colorscales';
import { DISPLAY_N_LAT, DISPLAY_N_LON } from '../geo/mask';

const MYANMAR_CENTER: [number, number] = [96.5, 19.0];

export function WeatherMap() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const overlayCanvas = useRef<HTMLCanvasElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);

  const {
    metadata, precipitation,
    currentHour,
    isLoaded, setInspectorPoint, inspectorPoint,
    mask,
  } = useForecastStore();

  // Position the canvas overlay to match the forecast bbox on the current map view
  const positionOverlay = (map: maplibregl.Map, meta: typeof metadata) => {
    const canvas = overlayCanvas.current;
    if (!canvas || !meta) return;
    const { lon_min, lat_max, lon_max, lat_min } = meta.bbox;
    const nw = map.project([lon_min, lat_max]);
    const se = map.project([lon_max, lat_min]);
    canvas.style.left = `${nw.x}px`;
    canvas.style.top = `${nw.y}px`;
    canvas.style.width = `${Math.max(1, se.x - nw.x)}px`;
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

      // Reposition overlay on initial load
      positionOverlay(map, useForecastStore.getState().metadata);
    });

    // Reposition overlay when map pans/zooms
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

  // Draw weather data onto the HTML canvas overlay
  useEffect(() => {
    const canvas = overlayCanvas.current;
    const map = mapRef.current;
    if (!canvas || !isLoaded || !metadata) return;

    const { n_lat, n_lon } = metadata.grid;
    if (!precipitation) return;

    const frame = getFrame(precipitation, currentHour, n_lat, n_lon);
    const [lut, vmin, vmax] = [PRECIP_LUT_ALPHA, PRECIP_MIN, PRECIP_MAX];

    // Render at display resolution (0.05°) with bilinear interpolation + Myanmar mask
    const rgba = renderWithInterpolation(frame, lut, vmin, vmax, n_lat, n_lon, mask);

    // Update canvas at display resolution
    canvas.width = DISPLAY_N_LON;
    canvas.height = DISPLAY_N_LAT;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.putImageData(
        new ImageData(new Uint8ClampedArray(rgba.buffer as ArrayBuffer), DISPLAY_N_LON, DISPLAY_N_LAT),
        0, 0,
      );
    }

    // Position canvas to match map projection
    if (map) positionOverlay(map, metadata);
  }, [isLoaded, metadata, currentHour, precipitation, mask]);

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
      const precip = precipitation
        ? getPointValue(precipitation, currentHour, grid.latIdx, grid.lonIdx, n_lat, n_lon)
        : NaN;

      const validTime = metadata.times_utc[currentHour] ?? '';
      const dt = new Date(validTime);
      const timeStr = dt.toUTCString().replace(' GMT', ' UTC');

      const stepHours = metadata.native_timestep_hours ?? 6;
      const leadH = currentHour * stepHours;
      const endDt = new Date(dt.getTime() + stepHours * 3_600_000);
      const startStr = `${dt.getUTCHours().toString().padStart(2, '0')}:00 UTC`;
      const endStr = `${endDt.getUTCHours().toString().padStart(2, '0')}:00 UTC`;

      content = `
        <div class="wx-popup">
          <div class="wx-popup-title">Myanmar</div>
          <div class="wx-popup-time">${timeStr}</div>
          <div class="wx-popup-row">
            <span class="wx-popup-label">Precipitation</span>
            <span class="wx-popup-value">${isNaN(precip) ? '—' : precip.toFixed(2)} mm</span>
          </div>
          <div class="wx-popup-sub">Accumulation: ${startStr}–${endStr}</div>
          <div class="wx-popup-note">6-hour accumulated total · not an instantaneous rate</div>
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
  }, [inspectorPoint, currentHour, isLoaded, metadata, precipitation]);

  return (
    <div style={{ position: 'absolute', inset: 0, background: '#0f1117' }}>
      {/* MapLibre map fills the container */}
      <div ref={mapContainer} style={{ position: 'absolute', inset: 0 }} />
      {/* Weather overlay — drawn via HTML canvas, repositioned on map move */}
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
