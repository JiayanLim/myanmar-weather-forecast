# Implementation Plan: Myanmar Weather Forecast Web Application

**Branch**: `001-myanmar-weather-app` | **Date**: 2026-08-11 (v3 — 48h + temperature + M4 CPU)
**Spec**: `specs/001-myanmar-weather-app/spec.md`
**Research**: `specs/001-myanmar-weather-app/research.md`
**Constitution**: `.specify/memory/constitution.md` v2.0.0

---

## Summary

Build a static GitHub Pages web application visualizing 48-hour temperature and precipitation
forecasts over Myanmar. The Earth2Studio pipeline (GraphCastSmall initialized from ARCO or IFS)
runs locally on an Apple M4 CPU (~78s end-to-end) and produces lightweight binary artifacts
(~16.6 KB). GraphCastSmall provides:
- Precipitation via native `tp06` (6-hour accumulated, metres → mm/6h)
- Temperature via native `t2m` (Kelvin → °C)

No temporal interpolation, no log transform, no sic patching, no diagnostic model, no GPU
required. The React/MapLibre frontend consumes these artifacts to render an interactive map
with a 9-step timeline and a variable switcher.

---

## Architecture Decision Summary

### Why GraphCastSmall?
- 1.0° resolution — memory footprint small enough for JAX CPU on M4
- Native `tp06` and `t2m` outputs — no diagnostic models needed
- JAX/XLA ARM64 runs efficiently on Apple M4 CPU (~54s for 8 AR steps)
- No sic gap to handle — GraphCastSmall inputs do not require sea ice concentration
- Model weights freely available via Google Cloud Storage

### Why ARCO for development, IFS for production?
- Both provide all 83 GraphCastSmall input variables including tp06 and t2m
- ARCO: historical ERA5, free GCS zarr, no credentials — ideal for local M4 inference
- IFS: near-real-time operational, free ECMWF open data — required for current forecasts
- NCAR_ERA5: NOT compatible (missing tp06 in lexicon)
- GFS: NOT verified compatible — do not use

### Why is precipitation mm / 6h (not mm/h)?
- GraphCastSmall natively outputs `tp06` = total precipitation for the 6-hour forecast period
- This is NOT an instantaneous rate; dividing by 6 would misrepresent the quantity
- The constitution (§VI) explicitly prohibits dividing tp06 by 6

### Why 48h horizon (8 steps)?
- 48h is the validated production configuration (GC_N_STEPS=8, GC_N_FRAMES=9)
- 24h was the initial target; extended to 48h after confirming M4 CPU handles 8 steps in ~54s
- 48h provides 4 extra meaningful forecast steps with minimal runtime penalty

### What data is generated offline (M4 CPU)?
- GraphCastSmall forecast: 9 frames (t+0h…t+48h) of tp06 and t2m over Myanmar
- ARCO/IFS initialization data (fetched at pipeline runtime)
- `precipitation.bin`, `temperature.bin`, `forecast.json`

### What runs on GitHub Pages?
- React + TypeScript + Vite SPA (pure JavaScript/WebGL)
- Reads pre-generated binary artifact files via `fetch()`
- No Python, no JAX, no GPU at runtime

---

## Technical Context

**Language/Version**:
- Python 3.11+ (pipeline) — uv-managed
- TypeScript 5.4+ (frontend)
- Node.js 22.x

**Primary Dependencies**:
- Pipeline: `earth2studio[aurora,data]>=0.17.0`, `xarray>=2024.1.0,<2026`, `numpy>=1.26.0`,
  `zarr>=2.17.0`, `psutil>=5.9.0`
- Pipeline JAX deps: `graphcast` (git@08cf736), `dm-haiku>=0.0.14`, `dm-tree>=0.1.9`,
  `flax>=0.10.6`
- Frontend: `react@18`, `typescript@5`, `vite@5`, `maplibre-gl@4`, `tailwindcss@3`, `zustand`

**xarray pin**: `xarray<2026` required — `xr.Dataset(existing_dataset)` was removed in
xarray 2026+. Earth2Studio 0.17.0's `_chunked_prediction_generator` uses this pattern.

**Storage**: Static files only. Binary artifacts in `data/forecast/` (production, tracked)
or `data/demo/` (development, committed).

**Performance Goals**:
- Initial load < 5 seconds on ≥25 Mbps
- Frame transition < 200ms after data loaded
- Total payload ~16.6 KB (no lazy loading required at this scale)

**Constraints**:
- No runtime server
- No interpolation of forecast variables (6h native steps only)
- No log/exp transform on tp06 or t2m
- ARCO or IFS only (not NCAR_ERA5, not GFS)
- JAX CPU backend on M4 (not MPS — MPS incompatible with JAX float64 ops)

---

## Constitution Check

| §I Static-First | No runtime compute; all data pre-generated | ✓ |
| §II Earth2Studio-Mandatory | GraphCastSmall + ARCO/IFS | ✓ |
| §III Pipeline | `scripts/` standalone; 6h steps; 48h horizon; t-6h+t+0h init | ✓ |
| §IV Myanmar-Focused | numpy bbox subset; GeoJSON boundary layer | ✓ |
| §V Map-First UX | MapLibre full-viewport; controls subordinate | ✓ |
| §VI Native-Step Navigation | 6h native steps; no interpolation; tp06 not divided | ✓ |
| §VII Model-Agnostic | All config in forecast.json; no GraphCast strings in TS | ✓ |
| §VIII Performance | Float32 binary ~16.6KB total; all frames loaded at once | ✓ |
| §IX Climate-Honest | Header + InfoPanel + tp06 tooltip mandatory | ✓ |
| §X Minimal Scope | pipeline → static → GitHub Pages; no extras | ✓ |
| §XI Hardware Transparency | M4 CPU validated; RSS recorded in forecast.json | ✓ |
| §XII Resolution Honesty | 1.0° native disclosed; interpolation policy stated | ✓ |

---

## Project Structure

```text
myanmar-weather/
│
├── .github/
│   └── workflows/
│       └── deploy-pages.yml         # 3-file check: real forecast preferred, demo fallback
│
├── .specify/                        # Spec Kit (constitution.md v2.0.0)
│
├── specs/
│   └── 001-myanmar-weather-app/
│       ├── research.md
│       ├── spec.md
│       ├── plan.md
│       └── tasks.md
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx           # App title, model info, init time, staleness
│   │   │   ├── Legend.tsx           # Variable-aware: mm/6h or °C color scale
│   │   │   ├── Timeline.tsx         # 9-step slider (0h–48h); prev/next; play/pause; speed
│   │   │   ├── InfoPanel.tsx        # About/attribution + both variable semantics
│   │   │   ├── DemoBanner.tsx       # "DEMO DATA" banner when is_demo=true
│   │   │   └── VariableSwitcher.tsx # Precip / Temp toggle buttons
│   │   ├── map/
│   │   │   ├── WeatherMap.tsx       # MapLibre GL JS container; variable-aware rendering
│   │   │   └── colorscales.ts       # PRECIP_LUT_ALPHA + TEMP_LUT; LUT-based rendering
│   │   ├── data/
│   │   │   ├── ForecastLoader.ts    # Fetch + parse both binary artifacts (Promise.all)
│   │   │   ├── ForecastStore.ts     # Zustand store; temperature + precipitation + activeVariable
│   │   │   └── types.ts             # TypeScript types; ActiveVariable = 'precipitation' | 'temperature'
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── index.css
│   ├── public/
│   │   ├── data/                    # Served data (copied at build time)
│   │   │   ├── forecast.json
│   │   │   ├── precipitation.bin
│   │   │   └── temperature.bin
│   │   └── geo/
│   │       └── myanmar-boundary.geojson
│   ├── index.html
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   └── package.json
│
├── scripts/
│   ├── generate_forecast.py         # Main pipeline (M4 CPU, ~78s)
│   ├── generate_demo_data.py        # Demo pipeline (CPU only, 9 frames, synthetic)
│   └── validate_forecast.py        # Artifact validation (schema v3.0, 25 checks)
│
├── notebooks/
│   └── graphcast_myanmar_forecast.ipynb  # Local M4 pipeline documentation notebook
│
├── data/
│   ├── demo/                        # Committed demo artifacts (is_demo=true)
│   │   ├── forecast.json
│   │   ├── precipitation.bin        # [9 × 21 × 11] float32 synthetic
│   │   └── temperature.bin          # [9 × 21 × 11] float32 synthetic
│   └── forecast/                    # Production artifacts (tracked, is_demo=false)
│       ├── forecast.json            # schema v3.0; init 2022-07-01T00:00:00Z
│       ├── precipitation.bin        # [9 × 21 × 11] float32; range [0, 17.38] mm/6h
│       └── temperature.bin          # [9 × 21 × 11] float32; range [1.94, 35.35] °C
│
├── README.md
├── pyproject.toml
└── .gitignore
```

---

## Forecast Artifact Format (Schema v3.0)

### forecast.json (authoritative schema)

```json
{
  "schema_version": "3.0",
  "model": "GraphCastSmall",
  "model_version": "1.0",
  "model_checkpoint": "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",
  "initialization_source": "ARCO",
  "initialization_time": "2022-07-01T00:00:00Z",
  "forecast_generated_at": "2026-08-11T13:09:42Z",
  "forecast_horizon_hours": 48,
  "native_timestep_hours": 6,
  "n_times": 9,
  "spatial_resolution_deg": 1.0,
  "display_resolution_deg": null,
  "region": "Myanmar",
  "bbox": { "lat_min": 9.0, "lat_max": 29.0, "lon_min": 92.0, "lon_max": 102.0 },
  "grid": { "n_lat": 21, "n_lon": 11 },
  "lat": [9.0, 10.0, "...", 29.0],
  "lon": [92.0, 93.0, "...", 102.0],
  "times_utc": ["2022-07-01T00:00:00Z", "...", "2022-07-03T00:00:00Z"],
  "variables": {
    "precipitation": {
      "file": "precipitation.bin",
      "units": "mm / 6h",
      "source_variable": "tp06",
      "transformation_provenance": { "log_transform_applied": false, "exp_transform_applied": false, ... }
    },
    "temperature": {
      "file": "temperature.bin",
      "units": "°C",
      "source_variable": "t2m",
      "transformation_provenance": { "conversion": "K - 273.15", "log_transform_applied": false, ... }
    }
  },
  "is_demo": false,
  "inference_config": {
    "device": "Apple M4 CPU",
    "jax_backend": "cpu",
    "rss_peak_gb": 2.34,
    "inference_time_seconds": 54,
    "total_pipeline_time_seconds": 78
  }
}
```

### Binary Format (per variable)

```
Layout:   [n_times × n_lat × n_lon] = [9 × 21 × 11]
Dtype:    float32, little-endian, C-order (row-major)
Size:     9 × 21 × 11 × 4 bytes = 8,316 bytes ≈ 8.1 KB each
Encoding: No NaN (fill_value = null for both variables)
Reading:  new Float32Array(await response.arrayBuffer())
Indexing: value[t][lat_i][lon_i] = array[t * 21 * 11 + lat_i * 11 + lon_i]
```

---

## Pipeline Architecture

### generate_forecast.py (key logic)

```python
# Force JAX CPU — MUST be before any JAX import
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Constants
GC_STEP_HOURS   = 6
GC_HORIZON_HOURS = 48
GC_N_STEPS      = 8   # 8 AR steps
GC_N_FRAMES     = 9   # 9 frames (t+0h through t+48h)
GC_N_LAT, GC_N_LON = 21, 11

# [1/4] Data source
data = ARCO()  # or IFS()

# [2/4] Load model (~5 GB weights from GCS)
package = GraphCastSmall.load_default_package()
model = GraphCastSmall.load_model(package)

# [3/4] Inference (~54s on M4 CPU for 8 steps)
with torch.inference_mode():
    io = e2run.deterministic(
        time=[init_time], nsteps=GC_N_STEPS,
        prognostic=model, data=data, io=ZarrBackend(),
        device=torch.device("cpu"), verbose=True,
    )

# [4/4] Post-process
tp06_raw = io.root["tp06"][0]  # (9, 181, 360)
t2m_raw  = io.root["t2m"][0]   # (9, 181, 360)

# lat ascending sort; Myanmar subset [9, 21, 11]
lat_idx_asc = ...  # np.argsort on matched lat indices
tp06_myanmar = tp06_raw[:, lat_idx_asc, :][:, :, lon_idx]
t2m_myanmar  = t2m_raw[:,  lat_idx_asc, :][:, :, lon_idx]

# Unit conversions
tp06_mm = np.maximum(tp06_myanmar * 1000.0, 0.0).astype(np.float32)
tp06_mm[0] = 0.0   # t+0h: no forecast accumulation
t2m_c   = (t2m_myanmar - 273.15).astype(np.float32)

# Write artifacts
temperature.bin   ← t2m_c (float32 C-order)
precipitation.bin ← tp06_mm (float32 C-order)
forecast.json     ← schema v3.0
```

### Unit Conversion Summary

| Variable | Raw source | Operation | Output |
|----------|-----------|-----------|--------|
| tp06 | metres (physical) | × 1000, clamp ≥ 0, t+0h=0 | mm / 6h |
| t2m | Kelvin | − 273.15 | °C |

**No log/exp transform is applied to either variable.**

---

## Frontend Architecture

### State (Zustand ForecastStore)

```typescript
interface ForecastState {
  metadata: ForecastMetadata | null;
  precipitation: Float32Array | null;  // [9 × 21 × 11]
  temperature: Float32Array | null;    // [9 × 21 × 11]
  activeVariable: 'precipitation' | 'temperature';
  isLoaded: boolean;
  error: string | null;
  maskError: string | null;

  currentHour: number;      // index 0–8 into times_utc array
  isPlaying: boolean;
  playbackSpeed: 0.5 | 1 | 2 | 4;

  setData: (metadata, precipitation, temperature) => void;
  setVariable: (v: ActiveVariable) => void;
  setHour: (h: number) => void;
  // ... stepForward, stepBackward, togglePlay, setSpeed
}
```

### Map Rendering

Frame update cycle (target < 200ms):
1. User changes step or variable → store update
2. `WeatherMap` detects change (Zustand selector)
3. `getFrame(h)` → `Float32Array` slice [21 × 11] from active variable
4. Apply LUT: `Float32Array` → `Uint8ClampedArray` [21 × 11 × 4] (RGBA)
5. Bilinear interpolation (1.0° → 0.05°) for visual smoothness
6. Update MapLibre canvas source

### Color Scales

```typescript
// Precipitation: 0–100+ mm/6h with alpha fade near zero
PRECIP_MIN = 0, PRECIP_MAX = 100
PRECIP_LUT_ALPHA: Uint8ClampedArray  // 256 RGBA entries

// Temperature: 15–40°C (Myanmar tropics), viridis-like
TEMP_MIN = 15, TEMP_MAX = 40
TEMP_LUT: Uint8ClampedArray  // 256 RGBA entries, full opacity
```

### Timeline Component

The slider operates on step index (0–8). Lead time is derived from step × native_timestep_hours.

```typescript
// Dynamic hour markers: 0h · 12h · 24h · 36h · 48h (every other frame at 6h steps)
const markers = Array.from({ length: n_times }, (_, i) => i).filter(i => i % 2 === 0);
const leadHours = currentHour * (metadata?.native_timestep_hours ?? 6);
```

---

## GitHub Actions Deployment

```yaml
# .github/workflows/deploy-pages.yml — current production state
- name: Copy forecast data into dist
  run: |
    mkdir -p frontend/dist/data
    if [ -f data/forecast/forecast.json ] && \
       [ -f data/forecast/temperature.bin ] && \
       [ -f data/forecast/precipitation.bin ]; then
      echo "Using real forecast data from data/forecast/"
      cp data/forecast/forecast.json frontend/dist/data/
      cp data/forecast/temperature.bin frontend/dist/data/
      cp data/forecast/precipitation.bin frontend/dist/data/
    else
      echo "No real forecast found — using demo data from data/demo/"
      cp data/demo/forecast.json frontend/dist/data/
      cp data/demo/temperature.bin frontend/dist/data/
      cp data/demo/precipitation.bin frontend/dist/data/
    fi
```

Logic: All 3 artifacts must exist for real forecast to be used. Missing any one → demo fallback.

---

## Technology Decisions

| Decision | Choice | Rejected | Reason |
|----------|--------|---------|--------|
| Precipitation source | GraphCastSmall native tp06 | PrecipitationAFNO | tp06 is first-class; no diagnostic model needed |
| Temperature source | GraphCastSmall native t2m | External reanalysis | t2m is first-class output; same inference pass |
| Temporal resolution | Native 6h from GraphCastSmall | Linear interpolation | Constitution §VI prohibits interpolation |
| Initialization source | ARCO (dev) / IFS (prod) | GFS, NCAR_ERA5 | NCAR_ERA5 missing tp06; GFS unverified |
| Precipitation transform | × 1000 only | exp() × 1000 | tp06 already in physical metres; no log space |
| Temperature transform | K − 273.15 only | No scaling | t2m already in Kelvin; straightforward conversion |
| Precipitation units | mm / 6h | mm/h | tp06 is genuinely 6-hourly; dividing misrepresents |
| Variables displayed | Precipitation + Temperature | Precipitation only | Both are native GraphCastSmall outputs; marginal cost |
| Frontend state | Zustand | Redux, React Context | Minimal API; sufficient scope |
| Data format | Float32 binary (one file per variable) | JSON, NetCDF | Smallest payload; native ArrayBuffer |
| Inference hardware | JAX CPU / M4 (~78s) | CUDA GPU (T4 16 GB) | T4 failed for Aurora1p5; GraphCastSmall runs on M4 CPU |
| Schema | v3.0 | v2.0 (precipitation only) | Temperature requires second binary; new schema version |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| ARCO data unavailable for recent dates | Low | Low | Fall back to IFS for recent dates |
| tp06 negative values | Low | Low | Clamp to 0 in pipeline (physical constraint) |
| GraphCastSmall weights download fails | Low | Medium | Retry; weights cached after first download |
| MapLibre canvas source API change | Low | Low | Pin MapLibre version; isolated in WeatherMap.tsx |
| xarray 2026+ incompatibility | Medium | High | Pinned xarray<2026 in pyproject.toml |
| IFS open data latency | Low | Low | Auto-detect latest available run with 6h lag |

---

## Development Phases — Status

**Phase 1**: Spec Kit update — COMPLETE (2026-08-11)

**Phase 2**: Pipeline rewrite — COMPLETE (2026-08-11)
- generate_forecast.py: GraphCastSmall + ARCO/IFS, 48h, 9 frames, temp+precip, M4 CPU

**Phase 3**: Demo data update — COMPLETE (2026-08-11)
- generate_demo_data.py: 9 frames, both variables, schema v3.0, is_demo=true

**Phase 4**: Frontend migration (24h → 48h + temperature) — COMPLETE (2026-08-11)
- VariableSwitcher, variable-aware map, legend, popup, timeline, header

**Phase 5**: Validation script update — COMPLETE (2026-08-11)
- validate_forecast.py: schema v3.0, 9 frames, both variables, 25 checks

**Phase 6**: Local M4 pipeline notebook — COMPLETE (2026-08-11)
- graphcast_myanmar_forecast.ipynb: documents local M4 inference end-to-end

**Phase 7**: Integration testing — IN PROGRESS
- T040, T041: PASS; T042 (frame profiling): pending; T043 (GitHub Pages): pending

**Phase 8**: README update — COMPLETE (2026-08-11)
