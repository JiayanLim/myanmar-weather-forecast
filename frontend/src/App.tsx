import { useEffect } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { WeatherMap } from './map/WeatherMap';
import { Header } from './components/Header';
import { DemoBanner } from './components/DemoBanner';
import { VariableSwitcher } from './components/VariableSwitcher';
import { Legend } from './components/Legend';
import { Timeline } from './components/Timeline';
import { InfoPanel } from './components/InfoPanel';
import { useForecastStore } from './data/ForecastStore';
import { loadForecast } from './data/ForecastLoader';

const DATA_URL = import.meta.env.VITE_DATA_URL ?? './data';

export function App() {
  const { setData, setError, setLoading, isLoaded, error } = useForecastStore(
    useShallow((s) => ({
      setData: s.setData,
      setError: s.setError,
      setLoading: s.setLoading,
      isLoaded: s.isLoaded,
      error: s.error,
    })),
  );

  useEffect(() => {
    setLoading(true);
    loadForecast(DATA_URL)
      .then((d) => setData(d.metadata, d.temperature, d.precipitation))
      .catch((err) => setError(String(err)));
  }, [setData, setError, setLoading]);

  return (
    <div className="flex flex-col h-full bg-wx-dark">
      <DemoBanner />
      <Header />

      {/* Map area — flex-1 so it takes all available space */}
      <div className="relative flex-1 overflow-hidden">
        {/* Variable switcher — floating over map */}
        <div className="absolute top-3 left-3 z-20">
          {isLoaded && <VariableSwitcher />}
        </div>

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

        <WeatherMap />
      </div>

      <Timeline />
      <InfoPanel />
    </div>
  );
}
