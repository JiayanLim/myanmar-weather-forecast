import { create } from 'zustand';
import type { ActiveVariable, ForecastMetadata, PlaybackSpeed } from './types';

interface ForecastState {
  // Data
  metadata: ForecastMetadata | null;
  precipitation:  Float32Array | null;
  temperature:    Float32Array | null;
  windSpeed:      Float32Array | null;
  windDirection:  Float32Array | null;
  isLoaded: boolean;
  isLoading: boolean;
  error: string | null;

  // UI state
  currentHour: number;         // frame index 0–(n_frames-1)
  activeVariable: ActiveVariable;
  isPlaying: boolean;
  playbackSpeed: PlaybackSpeed;

  // Display mask (Myanmar boundary at 0.05° resolution)
  mask: Uint8Array | null;
  maskError: string | null;

  // Selected point inspector
  inspectorPoint: { lat: number; lon: number } | null;

  // Panel visibility
  showInfoPanel: boolean;
  showModelEvaluation: boolean;

  // Actions
  setData: (
    metadata: ForecastMetadata,
    precipitation: Float32Array,
    temperature: Float32Array,
    windSpeed: Float32Array,
    windDirection: Float32Array,
  ) => void;
  setError: (error: string) => void;
  setLoading: (loading: boolean) => void;
  setHour: (hour: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
  setVariable: (v: ActiveVariable) => void;
  togglePlay: () => void;
  setPlaying: (playing: boolean) => void;
  setSpeed: (s: PlaybackSpeed) => void;
  setMask: (mask: Uint8Array) => void;
  setMaskError: (err: string) => void;
  setInspectorPoint: (point: { lat: number; lon: number } | null) => void;
  toggleInfoPanel: () => void;
  toggleModelEvaluation: () => void;
}

export const useForecastStore = create<ForecastState>((set, get) => ({
  metadata: null,
  precipitation:  null,
  temperature:    null,
  windSpeed:      null,
  windDirection:  null,
  isLoaded: false,
  isLoading: false,
  error: null,

  currentHour: 0,
  activeVariable: 'precipitation',
  isPlaying: false,
  playbackSpeed: 1,

  mask: null,
  maskError: null,
  inspectorPoint: null,
  showInfoPanel: false,
  showModelEvaluation: false,

  setData: (metadata, precipitation, temperature, windSpeed, windDirection) =>
    set({ metadata, precipitation, temperature, windSpeed, windDirection, isLoaded: true, isLoading: false, error: null }),

  setError: (error) => set({ error, isLoading: false }),
  setLoading: (loading) => set({ isLoading: loading }),

  setHour: (hour) => {
    const max = (useForecastStore.getState().metadata?.n_frames ?? 29) - 1;
    set({ currentHour: Math.max(0, Math.min(max, hour)) });
  },

  stepForward: () => {
    const { currentHour, metadata } = get();
    const maxHour = (metadata?.n_frames ?? 29) - 1;
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

  setMask: (mask) => set({ mask, maskError: null }),
  setMaskError: (err) => set({ maskError: err }),
  setInspectorPoint: (point) => set({ inspectorPoint: point }),
  toggleInfoPanel: () => set((s) => ({ showInfoPanel: !s.showInfoPanel })),
  toggleModelEvaluation: () => set((s) => ({ showModelEvaluation: !s.showModelEvaluation })),
}));
