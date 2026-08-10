import { create } from 'zustand';
import type { ForecastMetadata, PlaybackSpeed } from './types';

interface ForecastState {
  // Data
  metadata: ForecastMetadata | null;
  precipitation: Float32Array | null;
  isLoaded: boolean;
  isLoading: boolean;
  error: string | null;

  // UI state
  currentHour: number; // 0–48
  isPlaying: boolean;
  playbackSpeed: PlaybackSpeed;

  // Display mask (Myanmar boundary at 0.05° resolution)
  mask: Uint8Array | null;
  maskError: string | null;

  // Selected point inspector
  inspectorPoint: { lat: number; lon: number } | null;

  // Panel visibility
  showInfoPanel: boolean;

  // Actions
  setData: (metadata: ForecastMetadata, precipitation: Float32Array) => void;
  setError: (error: string) => void;
  setLoading: (loading: boolean) => void;
  setHour: (hour: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
  togglePlay: () => void;
  setPlaying: (playing: boolean) => void;
  setSpeed: (s: PlaybackSpeed) => void;
  setMask: (mask: Uint8Array) => void;
  setMaskError: (err: string) => void;
  setInspectorPoint: (point: { lat: number; lon: number } | null) => void;
  toggleInfoPanel: () => void;
}

export const useForecastStore = create<ForecastState>((set, get) => ({
  metadata: null,
  precipitation: null,
  isLoaded: false,
  isLoading: false,
  error: null,

  currentHour: 0,
  isPlaying: false,
  playbackSpeed: 1,

  mask: null,
  maskError: null,
  inspectorPoint: null,
  showInfoPanel: false,

  setData: (metadata, precipitation) =>
    set({ metadata, precipitation, isLoaded: true, isLoading: false, error: null }),

  setError: (error) => set({ error, isLoading: false }),
  setLoading: (loading) => set({ isLoading: loading }),

  setHour: (hour) => {
    const max = useForecastStore.getState().metadata?.forecast_horizon_hours ?? 48;
    set({ currentHour: Math.max(0, Math.min(max, hour)) });
  },

  stepForward: () => {
    const { currentHour, metadata } = get();
    const maxHour = metadata?.forecast_horizon_hours ?? 48;
    set({ currentHour: Math.min(currentHour + 1, maxHour) });
  },

  stepBackward: () => {
    const { currentHour } = get();
    set({ currentHour: Math.max(currentHour - 1, 0) });
  },

  togglePlay: () => set((s) => ({ isPlaying: !s.isPlaying })),
  setPlaying: (playing) => set({ isPlaying: playing }),

  setSpeed: (s) => set({ playbackSpeed: s }),

  setMask: (mask) => set({ mask, maskError: null }),
  setMaskError: (err) => set({ maskError: err }),
  setInspectorPoint: (point) => set({ inspectorPoint: point }),
  toggleInfoPanel: () => set((s) => ({ showInfoPanel: !s.showInfoPanel })),
}));
