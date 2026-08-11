import { PRECIP_LUT_ALPHA, PRECIP_MIN, PRECIP_MAX, PRECIP_TICKS } from '../map/colorscales';

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

function tickPosition(val: number, min: number, max: number): number {
  return ((val - min) / (max - min)) * 100;
}

export function Legend() {
  const gradient = buildGradientStyle(PRECIP_LUT_ALPHA);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-slate-400">Precipitation (6h accumulation)</span>
        <div className="relative group">
          <span className="text-[10px] text-slate-500 cursor-help border-b border-dashed border-slate-500">
            ⓘ accumulation
          </span>
          <div className="absolute bottom-5 right-0 hidden group-hover:block bg-wx-panel border border-wx-border rounded p-2 text-[10px] text-slate-300 w-52 z-50 shadow-lg">
            Precipitation values represent total rainfall accumulated during each 6-hour forecast period.
            These are <strong>not</strong> instantaneous rainfall rates.
          </div>
        </div>
      </div>
      <div className="relative h-3 rounded" style={{ background: gradient }}>
        {PRECIP_TICKS.map((t) => (
          <div
            key={t}
            className="absolute top-full mt-0.5 text-[9px] text-slate-400 -translate-x-1/2"
            style={{ left: `${tickPosition(t, PRECIP_MIN, PRECIP_MAX)}%` }}
          >
            {t}
          </div>
        ))}
      </div>
      <div className="flex justify-between text-[9px] text-slate-500 mt-3">
        <span>{PRECIP_MIN} mm / 6h</span>
        <span>{PRECIP_MAX}+ mm / 6h</span>
      </div>
    </div>
  );
}
