import { create } from 'zustand';
import type { ForecastMetadata, VariableKey, PlaybackSpeed } from './types';

interface ForecastState {
  // Data
  metadata: ForecastMetadata | null;
  temperature: Float32Array | null;
  precipitation: Float32Array | null;
  isLoaded: boolean;
  isLoading: boolean;
  error: string | null;

  // UI state
  activeVariable: VariableKey;
  currentHour: number; // 0–168
  isPlaying: boolean;
  playbackSpeed: PlaybackSpeed;

  // Selected point inspector
  inspectorPoint: { lat: number; lon: number } | null;

  // Panel visibility
  showInfoPanel: boolean;

  // Actions
  setData: (
    metadata: ForecastMetadata,
    temperature: Float32Array,
    precipitation: Float32Array,
  ) => void;
  setError: (error: string) => void;
  setLoading: (loading: boolean) => void;
  setHour: (hour: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
  setVariable: (v: VariableKey) => void;
  togglePlay: () => void;
  setPlaying: (playing: boolean) => void;
  setSpeed: (s: PlaybackSpeed) => void;
  setInspectorPoint: (point: { lat: number; lon: number } | null) => void;
  toggleInfoPanel: () => void;
}

export const useForecastStore = create<ForecastState>((set, get) => ({
  metadata: null,
  temperature: null,
  precipitation: null,
  isLoaded: false,
  isLoading: false,
  error: null,

  activeVariable: 'temperature_2m',
  currentHour: 0,
  isPlaying: false,
  playbackSpeed: 1,

  inspectorPoint: null,
  showInfoPanel: false,

  setData: (metadata, temperature, precipitation) =>
    set({ metadata, temperature, precipitation, isLoaded: true, isLoading: false, error: null }),

  setError: (error) => set({ error, isLoading: false }),
  setLoading: (loading) => set({ isLoading: loading }),

  setHour: (hour) => set({ currentHour: Math.max(0, Math.min(168, hour)) }),

  stepForward: () => {
    const { currentHour, metadata } = get();
    const maxHour = (metadata?.forecast_horizon_hours ?? 168);
    set({ currentHour: Math.min(currentHour + 1, maxHour) });
  },

  stepBackward: () => {
    const { currentHour } = get();
    set({ currentHour: Math.max(currentHour - 1, 0) });
  },

  setVariable: (v) => set({ activeVariable: v }),

  togglePlay: () => set((s) => ({ isPlaying: !s.isPlaying })),
  setPlaying: (playing) => set({ isPlaying: playing }),

  setSpeed: (s) => set({ playbackSpeed: s }),

  setInspectorPoint: (point) => set({ inspectorPoint: point }),
  toggleInfoPanel: () => set((s) => ({ showInfoPanel: !s.showInfoPanel })),
}));
