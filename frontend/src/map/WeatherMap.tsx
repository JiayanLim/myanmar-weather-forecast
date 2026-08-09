import { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import { useForecastStore } from '../data/ForecastStore';
import { getFrame, getPointValue, nearestGridPoint } from '../data/ForecastLoader';
import { applyColorscale, TEMP_LUT, TEMP_MIN, TEMP_MAX, PRECIP_LUT_ALPHA, PRECIP_MIN, PRECIP_MAX } from './colorscales';

const MYANMAR_BOUNDS: maplibregl.LngLatBoundsLike = [
  [91.5, 8.5],   // SW
  [102.5, 29.5], // NE
];
const MYANMAR_CENTER: [number, number] = [96.5, 19.0];
const IMAGE_ID = 'wx-overlay';
const LAYER_ID = 'wx-layer';
const SOURCE_ID = 'wx-source';

export function WeatherMap() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);
  const overlayReadyRef = useRef(false);

  const {
    metadata, temperature, precipitation,
    activeVariable, currentHour,
    isLoaded, setInspectorPoint, inspectorPoint,
  } = useForecastStore();


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
      // Myanmar boundary GeoJSON
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

      // Weather overlay canvas layer
      const canvas = document.createElement('canvas');
      canvas.width = 1;
      canvas.height = 1;

      // Add a canvas source we can update
      map.addSource(SOURCE_ID, {
        type: 'canvas',
        canvas: canvas,
        coordinates: [
          [metadata?.bbox.lon_min ?? 92, metadata?.bbox.lat_max ?? 29],
          [metadata?.bbox.lon_max ?? 102, metadata?.bbox.lat_max ?? 29],
          [metadata?.bbox.lon_max ?? 102, metadata?.bbox.lat_min ?? 9],
          [metadata?.bbox.lon_min ?? 92, metadata?.bbox.lat_min ?? 9],
        ],
        animate: false,
      } as unknown as maplibregl.CanvasSourceSpecification);

      map.addLayer({
        id: LAYER_ID,
        type: 'raster',
        source: SOURCE_ID,
        paint: { 'raster-opacity': 0.85 },
      }, 'myanmar-outline');

      overlayReadyRef.current = false;
    });

    // Navigation controls
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');

    // Click handler for point inspector
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

  // Update overlay when data or hour changes
  useEffect(() => {
    if (!mapRef.current || !isLoaded || !metadata) return;
    const map = mapRef.current;

    const updateCanvas = () => {
      const { n_lat, n_lon } = metadata.grid;
      const data = activeVariable === 'temperature_2m' ? temperature : precipitation;
      if (!data) return;

      const frame = getFrame(data, currentHour, n_lat, n_lon);
      const [lut, vmin, vmax] =
        activeVariable === 'temperature_2m'
          ? [TEMP_LUT, TEMP_MIN, TEMP_MAX]
          : [PRECIP_LUT_ALPHA, PRECIP_MIN, PRECIP_MAX];

      const rgba = applyColorscale(frame, lut, vmin, vmax, n_lat, n_lon);

      const source = map.getSource(SOURCE_ID) as maplibregl.CanvasSource | undefined;
      if (!source) return;

      // Get the canvas from the source
      const canvasEl = (source as unknown as { _canvas: HTMLCanvasElement })._canvas;
      if (!canvasEl) return;

      canvasEl.width = n_lon;
      canvasEl.height = n_lat;
      const ctx = canvasEl.getContext('2d');
      if (!ctx) return;

      const imgData = new ImageData(new Uint8ClampedArray(rgba.buffer as ArrayBuffer), n_lon, n_lat);
      ctx.putImageData(imgData, 0, 0);
      (source as unknown as { play: () => void }).play?.();
    };

    if (map.isStyleLoaded()) {
      updateCanvas();
    } else {
      map.once('idle', updateCanvas);
    }
  }, [isLoaded, metadata, activeVariable, currentHour, temperature, precipitation]);

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
      const temp = temperature
        ? getPointValue(temperature, currentHour, grid.latIdx, grid.lonIdx, n_lat, n_lon)
        : NaN;
      const precip = precipitation
        ? getPointValue(precipitation, currentHour, grid.latIdx, grid.lonIdx, n_lat, n_lon)
        : NaN;

      const validTime = metadata.times_utc[currentHour] ?? '';
      const dt = new Date(validTime);
      const timeStr = dt.toUTCString().replace(' GMT', ' UTC');

      const precipDisclosure = metadata.variables.precipitation.temporal_semantics;
      const leadH = currentHour;
      const endDt = new Date(dt.getTime() + 3600_000);
      const endStr = `${endDt.getUTCHours().toString().padStart(2, '0')}:00 UTC`;
      const startStr = `${dt.getUTCHours().toString().padStart(2, '0')}:00 UTC`;

      content = `
        <div class="wx-popup">
          <div class="wx-popup-title">Myanmar</div>
          <div class="wx-popup-time">${timeStr}</div>
          <div class="wx-popup-row">
            <span class="wx-popup-label">Temperature</span>
            <span class="wx-popup-value">${isNaN(temp) ? '—' : temp.toFixed(1)} °C</span>
          </div>
          <div class="wx-popup-row">
            <span class="wx-popup-label">Precipitation</span>
            <span class="wx-popup-value">${isNaN(precip) ? '—' : precip.toFixed(2)} mm</span>
          </div>
          <div class="wx-popup-sub">Accumulation: ${startStr}–${endStr}</div>
          <div class="wx-popup-note">1-hour accumulated total · not an instantaneous rate</div>
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
  }, [inspectorPoint, currentHour, isLoaded, metadata, temperature, precipitation]);

  return (
    <div
      ref={mapContainer}
      className="absolute inset-0 w-full h-full"
      style={{ background: '#0f1117' }}
    />
  );
}
