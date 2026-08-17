import { useEffect, useState } from 'react';
import { useForecastStore } from '../data/ForecastStore';

const DATA_URL = import.meta.env.VITE_DATA_URL ?? './data';

// verification.json schema v2.0 types
interface TempLeadResult {
  lead_time_hours: number;
  valid_time: string;
  mae: number;
  rmse: number;
  bias: number;
  n_points: number;
}

interface PrecipLeadResult {
  lead_time_hours: number;
  valid_time: string;
  mae: number;
  rmse: number;
  bias: number;
  pod: number;
  far: number;
  csi: number;
  n_hits: number;
  n_misses: number;
  n_false_alarms: number;
  n_points: number;
}

interface WindSpeedLeadResult {
  lead_time_hours: number;
  valid_time: string;
  mae: number;
  rmse: number;
  bias: number;
  n_points: number;
}

interface WindDirLeadResult {
  lead_time_hours: number;
  valid_time: string;
  circular_mae: number;
  n_points_active: number;
  n_points_calm_excluded: number;
}

interface ModelMetadata {
  model: string;
  init_time: string;
  n_times: number;
  native_timestep_hours: number;
  forecast_horizon_hours: number;
  spatial_resolution_deg: number;
}

interface ReferenceData {
  source: string;
  dataset: string;
  caveats: string[];
}

interface VerificationData {
  schema_version: string;
  generated_at: string;
  model_metadata: ModelMetadata;
  reference_data: ReferenceData;
  precipitation_threshold_mm_hr: number;
  wind_direction_calm_threshold_kt: number;
  variables: {
    temperature: {
      summary: { mae: number; rmse: number; bias: number; n_frames: number };
      by_lead_time: TempLeadResult[];
    };
    precipitation: {
      summary: { mae: number; rmse: number; bias: number; pod: number; far: number; csi: number; n_frames: number };
      by_lead_time: PrecipLeadResult[];
    };
    wind_speed: {
      summary: { mae: number; rmse: number; bias: number; n_frames: number };
      by_lead_time: WindSpeedLeadResult[];
    };
    wind_direction: {
      summary: { circular_mae: number; n_frames: number };
      by_lead_time: WindDirLeadResult[];
    };
  };
}

function sign(n: number): string {
  return n >= 0 ? '+' : '';
}

export function ModelEvaluation() {
  const showModelEvaluation = useForecastStore((s) => s.showModelEvaluation);
  const toggleModelEvaluation = useForecastStore((s) => s.toggleModelEvaluation);

  const [data, setData] = useState<VerificationData | null>(null);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    if (!showModelEvaluation || data !== null || loading) return;
    setLoading(true);
    fetch(`${DATA_URL}/verification.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<VerificationData>;
      })
      .then((d) => { setData(d); setLoading(false); })
      .catch((err) => { setFetchError(String(err)); setLoading(false); });
  }, [showModelEvaluation, data, loading]);

  if (!showModelEvaluation) return null;

  const mm = data?.model_metadata;
  const vars = data?.variables;

  return (
    <div
      className="absolute inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) toggleModelEvaluation(); }}
    >
      <div className="bg-wx-panel border border-wx-border rounded-lg max-w-3xl w-full mx-4 max-h-[85vh] flex flex-col shadow-2xl">

        {/* Modal header */}
        <div className="flex items-start justify-between px-5 py-3 border-b border-wx-border shrink-0">
          <div>
            <h2 className="text-sm font-bold text-white">Historical Model Evaluation</h2>
            <p className="text-[10px] text-slate-500 mt-0.5">
              GraphCastOperational vs ERA5 reanalysis — measures historical consistency, not this forecast's accuracy
            </p>
          </div>
          <button
            onClick={toggleModelEvaluation}
            className="text-slate-400 hover:text-white transition-colors text-xl leading-none px-1 ml-3 shrink-0"
            title="Close"
          >
            ×
          </button>
        </div>

        {/* Modal body */}
        <div className="overflow-y-auto flex-1 px-5 py-4 space-y-5 text-[11px] text-slate-300">

          {loading && (
            <div className="flex items-center gap-2 text-slate-400 py-4">
              <div className="w-4 h-4 border border-wx-accent border-t-transparent rounded-full animate-spin" />
              Loading evaluation data…
            </div>
          )}

          {fetchError && (
            <div className="space-y-2">
              <p className="text-amber-400 font-semibold">No historical evaluation data available</p>
              <p className="text-slate-500 text-[10px]">
                Generate verification metrics by running:
              </p>
              <pre className="bg-slate-800 rounded px-3 py-2 text-[10px] text-slate-300 overflow-x-auto">
                uv run python scripts/verify_forecast.py --forecast-dir data/forecast_v4
              </pre>
              <p className="text-slate-600 text-[10px]">
                Requires ARCO ERA5 access and a real (non-demo) forecast artifact.
              </p>
            </div>
          )}

          {data && mm && vars && (
            <>
              {/* Single-cycle qualification — R4 */}
              <div className="bg-amber-950/40 border border-amber-800/50 rounded p-3 text-[10px] leading-relaxed">
                <span className="text-amber-300 font-semibold">Single-cycle evaluation — </span>
                <span className="text-amber-200/70">
                  based on one initialization ({mm.init_time}, {mm.forecast_horizon_hours / 24}-day cycle).
                  A single cycle is insufficient to characterize general model performance.
                  Treat all metrics below as an initial validation only.
                </span>
              </div>

              {/* ERA5 methodology note */}
              <div className="bg-slate-800/60 border border-slate-700 rounded p-3 text-[10px] text-slate-400 leading-relaxed">
                <span className="text-slate-300 font-semibold">Methodology: </span>
                Verified against ERA5 reanalysis, not direct station observations. GCOp was trained on ERA5;
                agreement may be optimistic relative to independent observations.
              </div>

              {/* Forecast provenance */}
              <div className="text-[10px] text-slate-500 grid grid-cols-2 gap-x-4 gap-y-0.5">
                <span>Model: {mm.model}</span>
                <span>Resolution: {mm.spatial_resolution_deg}°</span>
                <span>Init: {mm.init_time}</span>
                <span>Step: {mm.native_timestep_hours}h · Frames: {mm.n_times} · Horizon: {mm.forecast_horizon_hours}h</span>
                <span className="col-span-2">Generated: {data.generated_at}</span>
                <span className="col-span-2">Reference: {data.reference_data.source}</span>
                <span>Precip threshold: {data.precipitation_threshold_mm_hr} mm/hr</span>
                <span>Calm threshold: {data.wind_direction_calm_threshold_kt} kt</span>
              </div>

              {/* Temperature */}
              <div>
                <h3 className="text-xs font-semibold text-white mb-1">2m Temperature vs ERA5</h3>
                <p className="text-[10px] text-slate-400 mb-1">
                  MAE: {vars.temperature.summary.mae.toFixed(4)}°C &nbsp;
                  RMSE: {vars.temperature.summary.rmse.toFixed(4)}°C &nbsp;
                  Bias: {sign(vars.temperature.summary.bias)}{vars.temperature.summary.bias.toFixed(4)}°C &nbsp;
                  ({vars.temperature.summary.n_frames} frames incl. t+0h init state)
                </p>
                <p className="text-[10px] text-slate-500 mb-2 italic">
                  Table: forecast leads +6h–+{mm.forecast_horizon_hours}h only. t+0h (init state = ERA5 analysis; error = 0 by construction) excluded from table.
                </p>
                <table className="w-full text-[10px] border-collapse">
                  <thead>
                    <tr className="text-slate-500 border-b border-slate-700">
                      <th className="text-left py-1 pr-3">Lead</th>
                      <th className="text-right py-1 pr-3">MAE (°C)</th>
                      <th className="text-right py-1 pr-3">RMSE (°C)</th>
                      <th className="text-right py-1">Bias (°C)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {vars.temperature.by_lead_time
                      .filter((r) => r.lead_time_hours !== 0)
                      .filter((_, i, arr) => i % 4 === 0 || i === arr.length - 1)
                      .map((r) => (
                        <tr key={r.lead_time_hours} className="border-b border-slate-800/60">
                          <td className="py-1 pr-3 text-slate-300">+{r.lead_time_hours}h</td>
                          <td className="py-1 pr-3 text-right">{r.mae.toFixed(4)}</td>
                          <td className="py-1 pr-3 text-right">{r.rmse.toFixed(4)}</td>
                          <td className={`py-1 text-right ${r.bias > 0.05 ? 'text-orange-400' : r.bias < -0.05 ? 'text-sky-400' : ''}`}>
                            {sign(r.bias)}{r.bias.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>

              {/* Precipitation */}
              <div>
                <h3 className="text-xs font-semibold text-white mb-1">
                  Precipitation vs ERA5 (threshold: {data.precipitation_threshold_mm_hr} mm/hr)
                </h3>
                <p className="text-[10px] text-slate-400 mb-1">
                  MAE: {vars.precipitation.summary.mae.toFixed(4)} mm/hr &nbsp;
                  POD: {vars.precipitation.summary.pod.toFixed(4)} &nbsp;
                  FAR: {vars.precipitation.summary.far.toFixed(4)} &nbsp;
                  CSI: {vars.precipitation.summary.csi.toFixed(4)} &nbsp;
                  ({vars.precipitation.summary.n_frames} frames, t+6h–t+{mm.forecast_horizon_hours}h)
                </p>
                <p className="text-[10px] text-slate-500 mb-1 italic">
                  t+0h excluded: GCOp precipitation is 0 at initialization by convention — {vars.precipitation.summary.n_frames} frames verified vs {vars.temperature.summary.n_frames} for temperature/wind.
                </p>
                <p className="text-[10px] text-amber-600/80 mb-2">
                  Jan 2021 (dry season): rain events are infrequent in Myanmar, which inflates FAR and suppresses CSI relative to wet-season conditions. Categorical scores from a single dry-season cycle require multi-cycle evaluation before generalization.
                </p>
                <table className="w-full text-[10px] border-collapse">
                  <thead>
                    <tr className="text-slate-500 border-b border-slate-700">
                      <th className="text-left py-1 pr-3">Lead</th>
                      <th className="text-right py-1 pr-3">MAE</th>
                      <th className="text-right py-1 pr-3">POD</th>
                      <th className="text-right py-1 pr-3">FAR</th>
                      <th className="text-right py-1">CSI</th>
                    </tr>
                  </thead>
                  <tbody>
                    {vars.precipitation.by_lead_time.filter((_, i) => i % 4 === 0 || i === vars.precipitation.by_lead_time.length - 1).map((r) => (
                      <tr key={r.lead_time_hours} className="border-b border-slate-800/60">
                        <td className="py-1 pr-3 text-slate-300">+{r.lead_time_hours}h</td>
                        <td className="py-1 pr-3 text-right">{r.mae.toFixed(4)}</td>
                        <td className="py-1 pr-3 text-right">{r.pod.toFixed(4)}</td>
                        <td className="py-1 pr-3 text-right">{r.far.toFixed(4)}</td>
                        <td className="py-1 text-right">{r.csi.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Wind Speed */}
              <div>
                <h3 className="text-xs font-semibold text-white mb-1">10m Wind Speed vs ERA5</h3>
                <p className="text-[10px] text-slate-400 mb-1">
                  MAE: {vars.wind_speed.summary.mae.toFixed(4)} kt &nbsp;
                  RMSE: {vars.wind_speed.summary.rmse.toFixed(4)} kt &nbsp;
                  Bias: {sign(vars.wind_speed.summary.bias)}{vars.wind_speed.summary.bias.toFixed(4)} kt &nbsp;
                  ({vars.wind_speed.summary.n_frames} frames incl. t+0h init state)
                </p>
                <p className="text-[10px] text-slate-500 mb-2 italic">
                  Table: forecast leads +6h–+{mm.forecast_horizon_hours}h only. t+0h (init state = ERA5 analysis; error = 0 by construction) excluded from table.
                </p>
                <table className="w-full text-[10px] border-collapse">
                  <thead>
                    <tr className="text-slate-500 border-b border-slate-700">
                      <th className="text-left py-1 pr-3">Lead</th>
                      <th className="text-right py-1 pr-3">MAE (kt)</th>
                      <th className="text-right py-1 pr-3">RMSE (kt)</th>
                      <th className="text-right py-1">Bias (kt)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {vars.wind_speed.by_lead_time
                      .filter((r) => r.lead_time_hours !== 0)
                      .filter((_, i, arr) => i % 4 === 0 || i === arr.length - 1)
                      .map((r) => (
                        <tr key={r.lead_time_hours} className="border-b border-slate-800/60">
                          <td className="py-1 pr-3 text-slate-300">+{r.lead_time_hours}h</td>
                          <td className="py-1 pr-3 text-right">{r.mae.toFixed(4)}</td>
                          <td className="py-1 pr-3 text-right">{r.rmse.toFixed(4)}</td>
                          <td className={`py-1 text-right ${r.bias > 0.5 ? 'text-orange-400' : r.bias < -0.5 ? 'text-sky-400' : ''}`}>
                            {sign(r.bias)}{r.bias.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>

              {/* Wind Direction */}
              <div>
                <h3 className="text-xs font-semibold text-white mb-1">
                  10m Wind Direction vs ERA5 (calm excluded: &lt;{data.wind_direction_calm_threshold_kt} kt)
                </h3>
                <p className="text-[10px] text-slate-400 mb-1">
                  Circular MAE: {vars.wind_direction.summary.circular_mae.toFixed(4)}° &nbsp;
                  ({vars.wind_direction.summary.n_frames} frames incl. t+0h init state)
                </p>
                <p className="text-[10px] text-slate-500 mb-2 italic">
                  Table: forecast leads +6h–+{mm.forecast_horizon_hours}h only. t+0h (init state; circular MAE = 0 by construction) excluded from table.
                </p>
                <table className="w-full text-[10px] border-collapse">
                  <thead>
                    <tr className="text-slate-500 border-b border-slate-700">
                      <th className="text-left py-1 pr-3">Lead</th>
                      <th className="text-right py-1 pr-3">Circ. MAE (°)</th>
                      <th className="text-right py-1 pr-3">Active pts</th>
                      <th className="text-right py-1">Calm excl.</th>
                    </tr>
                  </thead>
                  <tbody>
                    {vars.wind_direction.by_lead_time
                      .filter((r) => r.lead_time_hours !== 0)
                      .filter((_, i, arr) => i % 4 === 0 || i === arr.length - 1)
                      .map((r) => (
                        <tr key={r.lead_time_hours} className="border-b border-slate-800/60">
                          <td className="py-1 pr-3 text-slate-300">+{r.lead_time_hours}h</td>
                          <td className="py-1 pr-3 text-right">{r.circular_mae.toFixed(4)}</td>
                          <td className="py-1 pr-3 text-right">{r.n_points_active}</td>
                          <td className="py-1 text-right">{r.n_points_calm_excluded}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>

              {/* Limitations */}
              <div>
                <h3 className="text-xs font-semibold text-white mb-1">Caveats</h3>
                <ul className="space-y-1 text-[10px] text-slate-400 list-disc list-inside">
                  {data.reference_data.caveats.map((c, i) => <li key={i}>{c}</li>)}
                </ul>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
