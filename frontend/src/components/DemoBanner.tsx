import { useForecastStore } from '../data/ForecastStore';

export function DemoBanner() {
  const isDemo = useForecastStore((s) => s.metadata?.is_demo ?? false);
  if (!isDemo) return null;

  return (
    <div className="bg-amber-500 text-black text-center text-xs font-semibold py-1 px-4 shrink-0 z-10">
      DEMO DATA — Synthetic forecast for development use only. Not for operational decision-making.
    </div>
  );
}
