export interface BBox {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

export interface GridInfo {
  n_lat: number;
  n_lon: number;
  resolution_deg?: number;
}

// Schema v4.0 variable metadata — fields vary per variable; file and display_unit are guaranteed.
export interface VariableMeta {
  file: string;
  display_unit: string;
  native_variable?: string;
  native_variables?: string[];
  native_unit?: string;
  conversion?: string;
  accumulation_period_hours?: number;
  stats?: Record<string, unknown>;
  [key: string]: unknown;
}

// Schema v4.0 ForecastMetadata.
// Key renames from v3: n_times→n_frames, spatial_resolution_deg→native_resolution_deg.
// Variables expanded to all four GCOp outputs: precipitation, temperature, wind_speed, wind_direction.
export interface ForecastMetadata {
  schema_version: string;
  model: string;
  model_version?: string;
  model_checkpoint?: string;
  model_source?: string;
  model_attribution: string;
  initialization_source: string;
  initialization_time: string;
  forecast_generated_at: string;
  forecast_horizon_hours: number;
  native_timestep_hours: number;
  n_frames: number;
  native_resolution_deg: number;
  display_resolution_deg?: number | null;
  region: string;
  bbox: BBox;
  grid: GridInfo;
  lat: number[];
  lon: number[];
  times_utc: string[];
  variables: {
    precipitation: VariableMeta;
    temperature: VariableMeta;
    wind_speed: VariableMeta;
    wind_direction: VariableMeta;
  };
  data_source_attribution?: string;
  earth2studio_version?: string;
  inference_config?: {
    hardware?: string;
    device?: string;
    jax_backend?: string | null;
    rss_peak_gb?: number | null;
    [key: string]: unknown;
  };
  is_demo: boolean;
}

// wind_direction removed as selectable tab (R16). wind_direction data binary remains loaded
// and used for rendering arrows and popup display, but is no longer a selectable ActiveVariable.
export type ActiveVariable = 'precipitation' | 'temperature' | 'wind_speed';
export type PlaybackSpeed = 0.5 | 1 | 2 | 4;
