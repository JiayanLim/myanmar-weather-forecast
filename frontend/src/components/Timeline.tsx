import { useEffect, useRef, useCallback } from 'react';
import { useForecastStore } from '../data/ForecastStore';
import type { PlaybackSpeed } from '../data/types';

const SPEEDS: PlaybackSpeed[] = [0.5, 1, 2, 4];


export function Timeline() {
  const {
    metadata, currentHour, isPlaying, playbackSpeed,
    setHour, stepForward, stepBackward, togglePlay, setSpeed, setPlaying,
  } = useForecastStore();

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const maxIndex = (metadata?.n_times ?? 5) - 1;

  const tick = useCallback(() => {
    const store = useForecastStore.getState();
    if (store.currentHour >= (store.metadata?.n_times ?? 5) - 1) {
      useForecastStore.getState().setHour(0);
    } else {
      store.stepForward();
    }
  }, []);

  // Animation loop
  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (isPlaying) {
      const ms = 1000 / playbackSpeed;
      intervalRef.current = setInterval(tick, ms);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, playbackSpeed, tick]);

  const validTime = metadata?.times_utc?.[currentHour];
  const stepHours = metadata?.native_timestep_hours ?? 6;
  const leadHours = currentHour * stepHours;

  let dateStr = '—';
  let timeStr = '—';
  if (validTime) {
    const d = new Date(validTime);
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    dateStr = `${d.getUTCDate()} ${months[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
    timeStr = `${d.getUTCHours().toString().padStart(2, '0')}:00 UTC`;
  }

  return (
    <div className="flex flex-col gap-2 px-4 py-3 bg-wx-panel border-t border-wx-border shrink-0">
      {/* Time display */}
      <div className="flex items-baseline justify-between">
        <div>
          <span className="text-white text-sm font-semibold">{dateStr}</span>
          <span className="text-slate-400 text-sm ml-2">· {timeStr}</span>
        </div>
        <span className="text-slate-500 text-xs">Forecast +{leadHours}h</span>
      </div>

      {/* Slider */}
      <div className="flex items-center gap-2">
        <button
          onClick={stepBackward}
          className="text-slate-400 hover:text-white transition-colors px-1 shrink-0"
          title="Previous hour"
        >
          ◀
        </button>
        <input
          type="range"
          min={0}
          max={maxIndex}
          step={1}
          value={currentHour}
          onChange={(e) => setHour(Number(e.target.value))}
          className="flex-1"
        />
        <button
          onClick={stepForward}
          className="text-slate-400 hover:text-white transition-colors px-1 shrink-0"
          title="Next hour"
        >
          ▶
        </button>
      </div>

      {/* Controls row */}
      <div className="flex items-center justify-between">
        {/* Play/Pause */}
        <button
          onClick={togglePlay}
          className="flex items-center gap-2 px-3 py-1.5 rounded bg-wx-border hover:bg-wx-accent transition-colors text-xs font-medium text-white"
          title={isPlaying ? 'Pause' : 'Play animation'}
        >
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>

        {/* Speed selector */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-slate-500 mr-1">Speed:</span>
          {SPEEDS.map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`px-2 py-1 text-[10px] rounded transition-colors ${
                playbackSpeed === s
                  ? 'bg-wx-accent text-white'
                  : 'text-slate-400 hover:text-white border border-wx-border'
              }`}
            >
              {s}×
            </button>
          ))}
        </div>

        {/* Hour markers */}
        <div className="flex items-center gap-1 text-[9px] text-slate-600">
          <span>0h</span>
          <span className="mx-0.5">·</span>
          <span>6h</span>
          <span className="mx-0.5">·</span>
          <span>12h</span>
          <span className="mx-0.5">·</span>
          <span>18h</span>
          <span className="mx-0.5">·</span>
          <span>24h</span>
        </div>
      </div>
    </div>
  );
}
