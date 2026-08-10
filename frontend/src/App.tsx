import { useEffect } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { WeatherMap } from './map/WeatherMap';
import { Header } from './components/Header';
import { DemoBanner } from './components/DemoBanner';
import { Legend } from './components/Legend';
import { Timeline } from './components/Timeline';
import { InfoPanel } from './components/InfoPanel';
import { useForecastStore } from './data/ForecastStore';
import { loadForecast } from './data/ForecastLoader';
import { buildMaskFromGeojson } from './geo/mask';

const DATA_URL = import.meta.env.VITE_DATA_URL ?? './data';

export function App() {
  const { setData, setError, setLoading, setMask, setMaskError, isLoaded, error, maskError, isDemo } = useForecastStore(
    useShallow((s) => ({
      setData: s.setData,
      setError: s.setError,
      setLoading: s.setLoading,
      setMask: s.setMask,
      setMaskError: s.setMaskError,
      isLoaded: s.isLoaded,
      error: s.error,
      maskError: s.maskError,
      isDemo: s.metadata?.is_demo ?? true, // default true = demo until proven otherwise
    })),
  );

  useEffect(() => {
    setLoading(true);
    loadForecast(DATA_URL)
      .then((d) => setData(d.metadata, d.precipitation))
      .catch((err) => setError(String(err)));
  }, [setData, setError, setLoading]);

  useEffect(() => {
    // BASE_URL is set by Vite from VITE_BASE_PATH (e.g. /myanmar-weather-forecast/)
    const geoUrl = `${import.meta.env.BASE_URL}geo/myanmar-boundary.geojson`;
    buildMaskFromGeojson(geoUrl)
      .then((m) => setMask(m))
      .catch((err) => {
        const msg = `Myanmar boundary unavailable: ${String(err)}`;
        setMaskError(msg);
        // In debug mode (VITE_DEBUG=true) we log and continue; otherwise record error
        if (import.meta.env.VITE_DEBUG === 'true') {
          console.warn('[mask] Debug mode: proceeding without boundary mask.', err);
        } else {
          console.error('[mask] Boundary load failed. Overlay will be shown unmasked.', err);
        }
      });
  }, [setMask, setMaskError]);

  return (
    <div className="flex flex-col h-full bg-wx-dark">
      <DemoBanner />
      <Header />

      {/* Map area — flex-1 so it takes all available space */}
      <div className="relative flex-1 overflow-hidden">
        {/* Legend — floating over map, bottom-left */}
        <div className="absolute bottom-3 left-3 z-20 bg-wx-panel/90 border border-wx-border rounded p-3 min-w-[220px]">
          {isLoaded && <Legend />}
        </div>

        {!isLoaded && !error && (
          <div className="absolute inset-0 flex items-center justify-center z-30">
            <div className="flex flex-col items-center gap-3 text-slate-400">
              <div className="w-8 h-8 border-2 border-wx-accent border-t-transparent rounded-full animate-spin" />
              <span className="text-sm">Loading forecast data…</span>
            </div>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center z-30">
            <div className="bg-wx-panel border border-red-800 rounded-lg p-6 max-w-sm text-center">
              <p className="text-red-400 font-semibold mb-2">Failed to load forecast</p>
              <p className="text-slate-400 text-sm">{error}</p>
              <p className="text-slate-500 text-xs mt-3">Try refreshing the page.</p>
            </div>
          </div>
        )}

        {/* Mask error warning — shown for production forecasts when boundary fails to load.
            Demo mode and debug mode allow unmasked rendering silently. */}
        {maskError && !isDemo && import.meta.env.VITE_DEBUG !== 'true' && (
          <div className="absolute bottom-16 left-1/2 -translate-x-1/2 z-30 pointer-events-none">
            <div className="bg-amber-900/90 border border-amber-600 rounded px-3 py-1.5 text-xs text-amber-300 flex items-center gap-2 max-w-xs">
              <span>⚠</span>
              <span>Weather boundary unavailable — overlay shown without Myanmar mask</span>
            </div>
          </div>
        )}

        <WeatherMap />
      </div>

      <Timeline />
      <InfoPanel />
    </div>
  );
}
