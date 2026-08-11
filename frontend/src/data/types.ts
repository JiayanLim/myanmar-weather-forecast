export interface BBox {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

export interface GridInfo {
  n_lat: number;
  n_lon: number;
}

export interface TransformationProvenance {
  source_variable: string;
  source_unit: string;
  conversion: string;
  output_unit: string;
  accumulation_period_hours?: number;  // only present for precipitation
  log_transform_applied: boolean;
  exp_transform_applied: boolean;
  pipeline: string;
}

export interface VariableMeta {
  display_name: string;
  units: string;
  source_variable: string;
  temporal_resolution: string;
  temporal_semantics: string;
  temporal_disclosure?: string;
  transformation_provenance?: TransformationProvenance;
  t0_note?: string;
  native_output?: boolean;
  file: string;
  fill_value: number | null;
}

export interface ForecastMetadata {
  schema_version: string;
  model: string;
  model_version: string;
  model_checkpoint?: string;
  model_source?: string;
  initialization_source: string;
  initialization_time: string;
  forecast_generated_at: string;
  forecast_horizon_hours: number;
  native_timestep_hours: number;
  n_times: number;
  spatial_resolution_deg: number;
  display_resolution_deg: number | null;
  region: string;
  bbox: BBox;
  grid: GridInfo;
  lat: number[];
  lon: number[];
  times_utc: string[];
  variables: {
    precipitation: VariableMeta;
    temperature: VariableMeta;
  };
  data_source_attribution: string;
  model_attribution: string;
  earth2studio_version: string;
  inference_config?: {
    device: string;
    jax_backend?: string | null;
    rss_peak_gb?: number | null;
    [key: string]: unknown;
  };
  is_demo: boolean;
}

export type ActiveVariable = 'precipitation' | 'temperature';
export type PlaybackSpeed = 0.5 | 1 | 2 | 4;
