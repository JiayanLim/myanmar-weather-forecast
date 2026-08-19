import { useForecastStore } from '../data/ForecastStore';
import {
  PRECIP_LUT_ALPHA, PRECIP_MIN, PRECIP_MAX, PRECIP_TICKS,
  TEMP_LUT,         TEMP_MIN,   TEMP_MAX,   TEMP_TICKS,
  WIND_LUT_ALPHA,   WIND_MIN,   WIND_MAX,   WIND_TICKS,
} from '../map/colorscales';

function buildGradientStyle(lut: Uint8ClampedArray): string {
  const stops: string[] = [];
  const n = 20;
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const idx = Math.round(t * 255);
    const r = lut[idx * 4 + 0];
    const g = lut[idx * 4 + 1];
    const b = lut[idx * 4 + 2];
    const a = lut[idx * 4 + 3] / 255;
    stops.push(`rgba(${r},${g},${b},${a}) ${(t * 100).toFixed(0)}%`);
  }
  return `linear-gradient(to right, ${stops.join(', ')})`;
}

function tickPosition(val: number, min: number, max: number, sqrtScale = false): number {
  const linear = (val - min) / (max - min);
  const norm = sqrtScale ? Math.sqrt(Math.max(0, linear)) : linear;
  return norm * 100;
}

/**
 * Gradient bar + tick marks for a scalar variable.
 * FR-W01c: when isWind, also renders the compass-arrow direction widget below a divider.
 */
export function Legend() {
  const activeVariable = useForecastStore((s) => s.activeVariable);

  const isPrecip = activeVariable === 'precipitation';
  const isWind   = activeVariable === 'wind_speed';

  const lut   = isPrecip ? PRECIP_LUT_ALPHA : isWind ? WIND_LUT_ALPHA : TEMP_LUT;
  const vmin  = isPrecip ? PRECIP_MIN       : isWind ? WIND_MIN       : TEMP_MIN;
  const vmax  = isPrecip ? PRECIP_MAX       : isWind ? WIND_MAX       : TEMP_MAX;
  const ticks = isPrecip ? PRECIP_TICKS     : isWind ? WIND_TICKS     : TEMP_TICKS;
  const gradient  = buildGradientStyle(lut);
  const sqrtScale = isPrecip;

  return (
    <div className="flex flex-col gap-1">
      {/* ── Label row ───────────────────────────────────── */}
      <div className="flex items-center justify-between">
        {isPrecip ? (
          <>
            <span className="text-[10px] text-slate-400">Precipitation (mm/hr avg)</span>
            <div className="relative group">
              <span className="text-[10px] text-slate-500 cursor-help border-b border-dashed border-slate-500">
                ⓘ rate
              </span>
              <div className="absolute bottom-5 right-0 hidden group-hover:block bg-wx-panel border border-wx-border rounded p-2 text-[10px] text-slate-300 w-56 z-50 shadow-lg">
                Estimated average rainfall rate for each 6-hour period.
                Derived from GCOp tp06 output (metres/6h × 1000 / 6 → mm/hr, clamped ≥ 0).
                Color scale is sqrt-transformed — spreads light rain across the palette.
                Calibrated for January 2021 dry season (max: 1.62 mm/hr); values above 2 mm/hr saturate.
              </div>
            </div>
          </>
        ) : isWind ? (
          <span className="text-[10px] text-slate-400">Wind Speed (10m, kt)</span>
        ) : (
          <span className="text-[10px] text-slate-400">Temperature (2m, °C)</span>
        )}
      </div>

      {/* ── Gradient bar ────────────────────────────────── */}
      <div className="relative h-3 rounded" style={{ background: gradient }}>
        {ticks.map((t) => (
          <div
            key={t}
            className="absolute top-full mt-0.5 text-[9px] text-slate-400 -translate-x-1/2"
            style={{ left: `${tickPosition(t, vmin, vmax, sqrtScale)}%` }}
          >
            {t}
          </div>
        ))}
      </div>

      {/* ── Min/max labels ──────────────────────────────── */}
      <div className="flex justify-between text-[9px] text-slate-500 mt-3">
        {isPrecip ? (
          <>
            <span>{vmin} mm/hr</span>
            <span>{vmax}+ mm/hr</span>
          </>
        ) : isWind ? (
          <>
            <span>{vmin} kt</span>
            <span>{vmax}+ kt</span>
          </>
        ) : (
          <>
            <span>{vmin}°C</span>
            <span>{vmax}°C</span>
          </>
        )}
      </div>

    </div>
  );
}
