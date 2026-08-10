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
  variable: string;
  unit: string;
  accumulation_period: string;
  source_unit: string;
  conversion: string;
  temporal_semantics: string;
  pipeline: string;
}

export interface VariableMeta {
  display_name: string;
  units: string;
  source_variable: string;
  temporal_resolution: string;
  temporal_semantics: string;
  temporal_disclosure?: string;
  transformation?: string;
  transformation_provenance?: TransformationProvenance;
  native_output?: boolean;
  file: string;
  fill_value: number;
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
  sic_handling: string;
  forecast_horizon_hours: number;
  n_times: number;
  spatial_resolution_deg: number;
  display_resolution_deg?: number;
  spatial_interpolation?: string;
  boundary_mask?: string;
  region: string;
  bbox: BBox;
  grid: GridInfo;
  lat: number[];
  lon: number[];
  times_utc: string[];
  variables: {
    temperature_2m: VariableMeta;
    precipitation: VariableMeta;
  };
  data_source_attribution: string;
  model_attribution: string;
  earth2studio_version: string;
  inference_config?: {
    device: string;
    ifs_source: string;
    patched_vars: string[];
    inference_time_seconds: number;
    total_pipeline_time_seconds: number;
  };
  is_demo: boolean;
}

export type VariableKey = 'temperature_2m' | 'precipitation';
export type PlaybackSpeed = 0.5 | 1 | 2 | 4;
