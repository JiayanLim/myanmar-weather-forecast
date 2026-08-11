import { useForecastStore } from '../data/ForecastStore';
import type { ActiveVariable } from '../data/types';

const LABELS: Record<ActiveVariable, string> = {
  precipitation: 'Precip',
  temperature: 'Temp',
};

export function VariableSwitcher() {
  const { activeVariable, setVariable, isLoaded } = useForecastStore((s) => ({
    activeVariable: s.activeVariable,
    setVariable: s.setVariable,
    isLoaded: s.isLoaded,
  }));

  if (!isLoaded) return null;

  return (
    <div className="flex items-center gap-1 mb-2">
      {(Object.keys(LABELS) as ActiveVariable[]).map((v) => (
        <button
          key={v}
          onClick={() => setVariable(v)}
          className={`px-3 py-1 text-xs rounded font-medium transition-colors ${
            activeVariable === v
              ? 'bg-wx-accent text-white'
              : 'text-slate-400 hover:text-white border border-wx-border hover:border-slate-500'
          }`}
        >
          {LABELS[v]}
        </button>
      ))}
    </div>
  );
}
