import { useShallow } from 'zustand/react/shallow';
import { useForecastStore } from '../data/ForecastStore';

function formatInitTime(iso: string): string {
  const d = new Date(iso);
  return d.toUTCString().replace(':00 GMT', ' UTC').replace(/\d{2}:\d{2}:\d{2}/, (m) => m.slice(0, 5));
}

function isForecastStale(generatedAtIso: string, horizonHours: number): boolean {
  const ageMs = Date.now() - new Date(generatedAtIso).getTime();
  return ageMs > horizonHours * 3_600_000;
}

export function Header() {
  const { metadata, isLoaded, isDemo, toggleInfoPanel, toggleModelEvaluation } = useForecastStore(
    useShallow((s) => ({
      metadata: s.metadata,
      isLoaded: s.isLoaded,
      isDemo: s.metadata?.is_demo ?? false,
      toggleInfoPanel: s.toggleInfoPanel,
      toggleModelEvaluation: s.toggleModelEvaluation,
    })),
  );

  const horizonH = metadata?.forecast_horizon_hours ?? 168;
  const horizonLabel = horizonH % 24 === 0 ? `${horizonH / 24}-Day` : `${horizonH}h`;

  return (
    <header className="flex items-center justify-between px-4 py-2 bg-wx-panel border-b border-wx-border shrink-0 z-10">
      <div className="flex flex-col">
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold tracking-widest text-white uppercase">
            Myanmar {horizonLabel} Weather Forecast
          </span>
          {isDemo && (
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-amber-500 text-black">
              DEMO DATA
            </span>
          )}
        </div>
        {isLoaded && metadata && (
          <div className="flex items-center gap-3 text-[10px] text-slate-400 mt-0.5">
            <span>{metadata.model}{metadata.model_version ? ` ${metadata.model_version}` : ''}</span>
            <span>·</span>
            <span title="Model resolution · Display resolution (bilinear interpolation)">
              {metadata.native_resolution_deg}° model
              {metadata.display_resolution_deg != null && (
                <> · {metadata.display_resolution_deg}° display</>
              )}
            </span>
            {!isDemo && isForecastStale(metadata.forecast_generated_at, metadata.forecast_horizon_hours) && (
              <>
                <span>·</span>
                <span className="text-amber-400 font-semibold" title={`Forecast was generated more than ${metadata.forecast_horizon_hours} hours ago`}>
                  ⚠ Forecast may be outdated
                </span>
              </>
            )}
          </div>
        )}
      </div>
      <div className="flex items-center gap-2">
        <button
          onClick={toggleModelEvaluation}
          className="text-slate-400 hover:text-white transition-colors text-xs px-3 py-1.5 border border-wx-border rounded hover:border-slate-500"
          title="Historical model evaluation vs ERA5"
        >
          ⓘ Model Eval
        </button>
        <button
          onClick={toggleInfoPanel}
          className="text-slate-400 hover:text-white transition-colors text-xs px-3 py-1.5 border border-wx-border rounded hover:border-slate-500"
          title="About this forecast"
        >
          INFO
        </button>
      </div>
    </header>
  );
}
