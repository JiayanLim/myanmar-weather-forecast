import { useShallow } from 'zustand/react/shallow';
import { useForecastStore } from '../data/ForecastStore';

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</span>
      <span className="text-sm text-slate-200">{value}</span>
    </div>
  );
}

function formatDt(iso: string): string {
  try {
    return new Date(iso).toUTCString().replace(' GMT', ' UTC');
  } catch { return iso; }
}

export function InfoPanel() {
  const { showInfoPanel, toggleInfoPanel, metadata } = useForecastStore(
    useShallow((s) => ({
      showInfoPanel:   s.showInfoPanel,
      toggleInfoPanel: s.toggleInfoPanel,
      metadata:        s.metadata,
    })),
  );

  if (!showInfoPanel) return null;

  return (
    <div className="absolute inset-0 z-50 flex items-end sm:items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/60" onClick={toggleInfoPanel} />
      <div className="relative bg-wx-panel border border-wx-border rounded-lg p-5 w-full max-w-md max-h-[80vh] overflow-y-auto shadow-2xl">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-bold text-white">Forecast Information</h2>
          <button
            onClick={toggleInfoPanel}
            className="text-slate-500 hover:text-white transition-colors text-lg leading-none"
          >
            ×
          </button>
        </div>

        {metadata ? (
          <div className="flex flex-col gap-4">
            {!metadata.is_demo && (
              <div className="text-xs font-semibold text-wx-accent uppercase tracking-wider">
                Latest Forecast
              </div>
            )}

            <div className="grid grid-cols-2 gap-3">
              <Row
                label="Model"
                value={`${metadata.model}${metadata.model_version ? ` ${metadata.model_version}` : ''}`}
              />
              <Row
                label="Model resolution"
                value={`${metadata.native_resolution_deg}° (~${Math.round(metadata.native_resolution_deg * 111)} km)`}
              />
              {metadata.display_resolution_deg != null && (
                <Row label="Display resolution" value={`${metadata.display_resolution_deg}° — interpolated`} />
              )}
              <Row label="Initialization" value={metadata.initialization_source} />
              <Row label="Horizon" value={`${metadata.forecast_horizon_hours}h`} />
              <Row label="Timestep" value={`${metadata.native_timestep_hours}h`} />
              <Row label="Init time" value={formatDt(metadata.initialization_time)} />
              <Row label="Generated" value={formatDt(metadata.forecast_generated_at)} />
              {metadata.inference_config && (
                <Row
                  label="Generated on"
                  value={String(metadata.inference_config.hardware ?? metadata.inference_config.device ?? '—')}
                />
              )}
            </div>

            <hr className="border-wx-border" />

            <div className="flex flex-col gap-2">
              <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Variables</h3>
              <div className="flex flex-col gap-3 text-sm">

                <div>
                  <span className="text-white font-medium">
                    Temperature · {metadata.variables.temperature.display_unit}
                  </span>
                  <p className="text-slate-500 text-[10px] mt-0.5 font-mono">
                    {metadata.variables.temperature.conversion}
                  </p>
                </div>

                <div>
                  <span className="text-white font-medium">
                    Precipitation · {metadata.variables.precipitation.display_unit}
                  </span>
                  <p className="text-slate-400 text-xs mt-0.5">
                    Estimated average rainfall rate for each 6-hour forecast period.
                    Not an instantaneous rate.
                  </p>
                  <p className="text-slate-500 text-[10px] mt-0.5 font-mono">
                    {metadata.variables.precipitation.conversion}
                  </p>
                </div>

                <div>
                  <span className="text-white font-medium">
                    Wind Speed · {metadata.variables.wind_speed.display_unit}
                  </span>
                  <p className="text-slate-500 text-[10px] mt-0.5 font-mono">
                    {metadata.variables.wind_speed.conversion}
                  </p>
                </div>

                <div>
                  <span className="text-white font-medium">
                    Wind Direction · {metadata.variables.wind_direction.display_unit}
                  </span>
                  <p className="text-slate-400 text-xs mt-0.5">
                    Meteorological FROM direction. Display uses vector-component bilinear interpolation.
                  </p>
                  <p className="text-slate-500 text-[10px] mt-0.5 font-mono">
                    {metadata.variables.wind_direction.conversion}
                  </p>
                </div>

              </div>
            </div>

            <hr className="border-wx-border" />

            {metadata.data_source_attribution && (
              <>
                <div className="flex flex-col gap-2">
                  <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Initialization Data</h3>
                  <p className="text-xs text-slate-400">{metadata.data_source_attribution}</p>
                </div>
                <hr className="border-wx-border" />
              </>
            )}

            <div className="flex flex-col gap-2">
              <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Limitations</h3>
              <ul className="text-[11px] text-slate-400 list-disc list-inside space-y-1">
                <li>
                  The {metadata.native_resolution_deg}° model grid (~{Math.round(metadata.native_resolution_deg * 111)} km)
                  represents large-scale atmospheric conditions and should not be interpreted as
                  neighborhood-scale weather prediction.
                </li>
                <li>
                  The display uses bilinear interpolation to 0.05° for visual clarity. This does not
                  add forecast information beyond the native {metadata.native_resolution_deg}° model resolution.
                  Wind direction uses vector-component interpolation (sin/cos, not raw degrees).
                </li>
                <li>
                  Forecast skill degrades over the {metadata.forecast_horizon_hours}-hour window,
                  particularly for rapidly evolving convective systems.
                </li>
                <li>
                  Precipitation represents estimated average rainfall rate (mm/hr) for each 6-hour
                  forecast period, not an instantaneous rate.
                </li>
                {metadata.is_demo && (
                  <li className="text-amber-400">
                    This is <strong>synthetic demo data</strong> — not a real weather forecast.
                    Do not use for any decision-making.
                  </li>
                )}
              </ul>
            </div>

            <hr className="border-wx-border" />

            <div className="flex flex-col gap-1">
              <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Attribution</h3>
              <p className="text-[10px] text-slate-500">{metadata.model_attribution}</p>
              {metadata.earth2studio_version && (
                <p className="text-[10px] text-slate-500">
                  Framework: NVIDIA Earth2Studio ≥ {metadata.earth2studio_version}
                </p>
              )}
              <p className="text-[10px] text-slate-500">Basemap: © OpenStreetMap contributors</p>
            </div>
          </div>
        ) : (
          <p className="text-slate-400 text-sm">No forecast data loaded.</p>
        )}
      </div>
    </div>
  );
}
