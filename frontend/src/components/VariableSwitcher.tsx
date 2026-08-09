import { useForecastStore } from '../data/ForecastStore';
import type { VariableKey } from '../data/types';

const VARIABLES: { key: VariableKey; label: string; icon: string }[] = [
  { key: 'temperature_2m', label: 'Temperature', icon: '🌡' },
  { key: 'precipitation', label: 'Precipitation', icon: '🌧' },
];

export function VariableSwitcher() {
  const { activeVariable, setVariable } = useForecastStore((s) => ({
    activeVariable: s.activeVariable,
    setVariable: s.setVariable,
  }));

  return (
    <div className="flex rounded overflow-hidden border border-wx-border">
      {VARIABLES.map(({ key, label, icon }) => (
        <button
          key={key}
          onClick={() => setVariable(key)}
          className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium transition-colors ${
            activeVariable === key
              ? 'bg-wx-accent text-white'
              : 'bg-wx-panel text-slate-400 hover:text-white hover:bg-wx-border'
          }`}
        >
          <span>{icon}</span>
          <span>{label}</span>
        </button>
      ))}
    </div>
  );
}
