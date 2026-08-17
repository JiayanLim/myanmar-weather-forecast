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

/** Metric definition row for the "How to read these metrics" section. */
function MetricDef({ name, full, desc }: { name: string; full: string; desc: string }) {
  return (
    <div className="grid grid-cols-[4.5rem_1fr] gap-x-2 py-0.5">
      <span className="text-slate-200 font-mono font-semibold text-[10px]">{name}</span>
      <span className="text-slate-400 text-[10px]">
        <span className="text-slate-300">{full}</span> — {desc}
      </span>
    </div>
  );
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
              GraphCastOperational vs ERA5 reanalysis — measures historical consistency with the training dataset, not this forecast's real-world accuracy
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
        <div className="overflow-y-auto flex-1 px-5 py-4 space-y-4 text-[11px] text-slate-300">

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
              {/* ── Important limitations (FR-N46c) ───────────────────────── */}
              <div className="bg-amber-950/40 border border-amber-700/60 rounded p-3 space-y-1.5">
                <p className="text-amber-300 font-semibold text-[10px] uppercase tracking-wide">
                  Important limitations
                </p>
                <ul className="space-y-1 text-[10px] text-amber-200/75 list-disc list-inside leading-relaxed">
                  <li>
                    <span className="text-amber-200/90 font-medium">ERA5 is reanalysis, not observations.</span>{' '}
                    Verification is against ERA5 reanalysis data, not independent weather-station measurements.
                    ERA5 itself contains model errors and spatial smoothing.
                  </li>
                  <li>
                    <span className="text-amber-200/90 font-medium">Training-data advantage.</span>{' '}
                    GraphCastOperational was trained on ERA5-derived data. Agreement with ERA5 may be
                    systematically better than agreement with independent observations would be.
                  </li>
                  <li>
                    <span className="text-amber-200/90 font-medium">N = 1 forecast cycle.</span>{' '}
                    All metrics below come from a single initialization: {mm.init_time}, {mm.forecast_horizon_hours / 24}-day horizon.
                    One cycle is insufficient to characterise general model performance across seasons,
                    weather regimes, or different years.
                  </li>
                  <li>
                    <span className="text-amber-200/90 font-medium">Dry-season cycle.</span>{' '}
                    January is Myanmar's dry season. Precipitation categorical scores (POD, FAR, CSI)
                    are strongly season-dependent: with few rain events, even small errors can produce
                    large FAR and suppressed CSI. These scores would likely differ significantly in
                    the wet season.
                  </li>
                </ul>
              </div>

              {/* ── How to read these metrics (FR-N46a) ───────────────────── */}
              <div className="bg-slate-800/50 border border-slate-700 rounded p-3">
                <p className="text-slate-200 font-semibold text-[10px] uppercase tracking-wide mb-2">
                  How to read these metrics
                </p>
                <div className="space-y-0.5">
                  <MetricDef name="MAE"
                    full="Mean Absolute Error"
                    desc="average magnitude of the forecast error, in the variable's unit. Lower is better." />
                  <MetricDef name="RMSE"
                    full="Root Mean Square Error"
                    desc="like MAE but penalises large individual errors more strongly. Lower is better." />
                  <MetricDef name="Bias"
                    full="Mean signed error"
                    desc="systematic over- or under-prediction. Negative = model colder/weaker/drier than ERA5; positive = warmer/stronger/wetter. 0 is ideal." />
                  <MetricDef name="Circ. MAE"
                    full="Circular Mean Absolute Error"
                    desc="average angular error for wind direction, using the shortest path around the compass (avoids the 0°/360° wrap). Lower is better." />
                  <MetricDef name="POD"
                    full="Probability of Detection"
                    desc="fraction of rain events that ERA5 observed and the model also forecast. Higher is better; 1.0 = all events detected." />
                  <MetricDef name="FAR"
                    full="False Alarm Ratio"
                    desc="fraction of forecast rain events that ERA5 did not observe. Lower is better; 0 = no false alarms." />
                  <MetricDef name="CSI"
                    full="Critical Success Index"
                    desc="combined score: hits ÷ (hits + misses + false alarms). Accounts for both missed events and false alarms. Higher is better; 1.0 = perfect." />
                </div>
                <p className="text-slate-500 text-[10px] mt-2 italic">
                  Rain-event threshold: {data.precipitation_threshold_mm_hr} mm/hr.
                  Wind-direction calm threshold: &lt;{data.wind_direction_calm_threshold_kt} kt excluded.
                </p>
              </div>

              {/* ── Temporal evaluation convention (FR-N46b) ──────────────── */}
              <div className="bg-slate-800/30 border border-slate-700/50 rounded p-3 text-[10px] text-slate-400 leading-relaxed space-y-1">
                <p className="text-slate-300 font-semibold">Temporal evaluation convention</p>
                <p>
                  Temperature and wind <span className="text-slate-300">summary rows</span> include all{' '}
                  {mm.n_times} frames (t+0h through t+{mm.forecast_horizon_hours}h).
                  The t+0h frame is the ERA5 initialization state — the model's error at that point
                  is <span className="text-slate-300">zero by construction</span>, making the summary
                  slightly optimistic for forecast-only skill.
                </p>
                <p>
                  The <span className="text-slate-300">per-lead-time tables</span> below exclude t+0h
                  and show genuine forecast leads +6h–+{mm.forecast_horizon_hours}h only.
                  These are a more representative view of model performance.
                </p>
                <p>
                  Precipitation evaluation covers <span className="text-slate-300">{vars.precipitation.summary.n_frames} frames</span>{' '}
                  (t+6h–t+{mm.forecast_horizon_hours}h). t+0h precipitation is excluded because
                  GraphCastOperational outputs zero precipitation at the initialization step by design.
                </p>
              </div>

              {/* ── Forecast provenance ───────────────────────────────────── */}
              <div className="text-[10px] text-slate-500 grid grid-cols-2 gap-x-4 gap-y-0.5">
                <span>Model: {mm.model}</span>
                <span>Resolution: {mm.spatial_resolution_deg}°</span>
                <span>Init: {mm.init_time}</span>
                <span>Step: {mm.native_timestep_hours}h · Frames: {mm.n_times} · Horizon: {mm.forecast_horizon_hours}h</span>
                <span className="col-span-2">Generated: {data.generated_at}</span>
                <span className="col-span-2">Reference: {data.reference_data.source}</span>
              </div>

              <hr className="border-slate-700/60" />

              {/* ── Temperature (FR-N46d) ─────────────────────────────────── */}
              <div>
                <h3 className="text-xs font-semibold text-white mb-1">2m Temperature vs ERA5</h3>

                {/* Summary */}
                <p className="text-[10px] text-slate-400 mb-1">
                  MAE: {vars.temperature.summary.mae.toFixed(4)}°C &nbsp;
                  RMSE: {vars.temperature.summary.rmse.toFixed(4)}°C &nbsp;
                  Bias: {sign(vars.temperature.summary.bias)}{vars.temperature.summary.bias.toFixed(4)}°C &nbsp;
                  <span className="text-slate-500">({vars.temperature.summary.n_frames} frames incl. t+0h)</span>
                </p>

                {/* Cold-bias context (FR-N46d) */}
                <div className="bg-sky-950/40 border border-sky-800/40 rounded p-2.5 mb-2 text-[10px] text-sky-200/75 leading-relaxed">
                  <p className="text-sky-300/90 font-semibold mb-1">Observed cold bias — Jan 2021 cycle</p>
                  <p>
                    The R5 validation measured a mean temperature bias of{' '}
                    <span className="text-sky-200 font-medium">−0.7595°C</span> vs ERA5 across all
                    {vars.temperature.summary.n_frames} frames and 3,321 grid points.
                    A subsequent spot-check (4 representative sites × 4 frames) found approximately{' '}
                    <span className="text-sky-200 font-medium">−0.89°C</span> mean bias, with the
                    largest deviation at local midday (06Z / 12:30 MMT, avg −1.23°C).
                  </p>
                  <p className="mt-1">
                    In plain terms: the model was consistently colder than ERA5 during this evaluation,
                    particularly at midday. The negative bias value above reflects this.
                  </p>
                  <p className="mt-1 text-sky-200/55 italic">
                    This is cycle-specific evidence from a single January 2021 run, not a universal
                    GCOp correction. The displayed forecast values are raw model output — no temperature
                    offset has been applied.
                  </p>
                </div>

                {/* Per-lead table */}
                <p className="text-[10px] text-slate-500 mb-1 italic">
                  Table shows genuine forecast leads +6h–+{mm.forecast_horizon_hours}h (t+0h excluded).
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
                          <td className={`py-1 text-right ${r.bias < -0.05 ? 'text-sky-400' : r.bias > 0.05 ? 'text-orange-400' : ''}`}>
                            {sign(r.bias)}{r.bias.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>

              {/* ── Precipitation ─────────────────────────────────────────── */}
              <div>
                <h3 className="text-xs font-semibold text-white mb-1">
                  Precipitation vs ERA5 (threshold: {data.precipitation_threshold_mm_hr} mm/hr)
                </h3>

                {/* Summary */}
                <p className="text-[10px] text-slate-400 mb-1">
                  MAE: {vars.precipitation.summary.mae.toFixed(4)} mm/hr &nbsp;
                  POD: {vars.precipitation.summary.pod.toFixed(4)} &nbsp;
                  FAR: {vars.precipitation.summary.far.toFixed(4)} &nbsp;
                  CSI: {vars.precipitation.summary.csi.toFixed(4)} &nbsp;
                  <span className="text-slate-500">({vars.precipitation.summary.n_frames} frames, t+6h–t+{mm.forecast_horizon_hours}h)</span>
                </p>

                {/* Dry-season note */}
                <div className="bg-amber-950/30 border border-amber-800/40 rounded p-2 mb-2 text-[10px] text-amber-200/70 leading-relaxed">
                  January 2021 (Myanmar dry season): rain events are infrequent, which inflates FAR
                  and suppresses CSI relative to wet-season conditions. Categorical scores from a
                  single dry-season cycle require multi-cycle evaluation before generalisation.
                </div>

                {/* Per-lead table */}
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
                    {vars.precipitation.by_lead_time
                      .filter((_, i) => i % 4 === 0 || i === vars.precipitation.by_lead_time.length - 1)
                      .map((r) => (
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

              {/* ── 10m Wind (FR-N46e, FR-N46f — speed + direction grouped) ─ */}
              <div>
                <h3 className="text-xs font-semibold text-white mb-0.5">10m Wind vs ERA5</h3>
                <p className="text-[10px] text-slate-500 mb-2 italic">
                  Wind speed and direction are presented in the combined Wind view. The metrics below cover each component separately.
                </p>

                {/* Wind Speed */}
                <p className="text-[10px] text-slate-300 font-medium mb-0.5">Wind Speed</p>
                <p className="text-[10px] text-slate-400 mb-1">
                  MAE: {vars.wind_speed.summary.mae.toFixed(4)} kt &nbsp;
                  RMSE: {vars.wind_speed.summary.rmse.toFixed(4)} kt &nbsp;
                  Bias: {sign(vars.wind_speed.summary.bias)}{vars.wind_speed.summary.bias.toFixed(4)} kt &nbsp;
                  <span className="text-slate-500">({vars.wind_speed.summary.n_frames} frames incl. t+0h)</span>
                </p>
                <p className="text-[10px] text-slate-500 mb-1 italic">
                  Table shows genuine forecast leads +6h–+{mm.forecast_horizon_hours}h (t+0h excluded).
                </p>
                <table className="w-full text-[10px] border-collapse mb-3">
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

                {/* Wind Direction */}
                <p className="text-[10px] text-slate-300 font-medium mb-0.5">
                  Wind Direction <span className="text-slate-500 font-normal">(calm excluded: &lt;{data.wind_direction_calm_threshold_kt} kt)</span>
                </p>
                <p className="text-[10px] text-slate-400 mb-1">
                  Circular MAE: {vars.wind_direction.summary.circular_mae.toFixed(4)}° &nbsp;
                  <span className="text-slate-500">({vars.wind_direction.summary.n_frames} frames incl. t+0h)</span>
                </p>
                <p className="text-[10px] text-slate-500 mb-1 italic">
                  Circular MAE uses the shortest angular path around the compass, avoiding errors from the 0°/360° boundary.
                  Table shows +6h–+{mm.forecast_horizon_hours}h (t+0h excluded).
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

            </>
          )}
        </div>
      </div>
    </div>
  );
}
