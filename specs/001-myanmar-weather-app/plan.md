# Implementation Plan: Myanmar Weather Forecast Web Application

**Branch**: `001-myanmar-weather-app` | **Date**: 2026-08-11 (revised — GraphCastSmall migration)
**Spec**: `specs/001-myanmar-weather-app/spec.md`
**Research**: `specs/001-myanmar-weather-app/research.md`
**Constitution**: `.specify/memory/constitution.md` v2.0.0

---

## Summary

Build a static GitHub Pages web application visualizing 24-hour precipitation forecasts over
Myanmar. The Earth2Studio pipeline (GraphCastSmall initialized from ARCO or IFS) runs offline
on a GPU machine and produces lightweight binary artifacts (~4.6 KB). GraphCastSmall provides
precipitation via its native `tp06` output (6-hour accumulated, metres). No temporal
interpolation, no log transform, no sic patching, no diagnostic model. The React/MapLibre
frontend consumes these artifacts to render an interactive map with a 5-step timeline.

---

## Architecture Decision Summary

### Why GraphCastSmall?
- 1.0° resolution reduces activation memory vs. Aurora1p5's 0.25°
- Native `tp06` output: 6-hour accumulated precipitation is a first-class model output
- JAX backend with bfloat16 internally — potentially fits T4 (16 GB), must be verified
- No sic gap to handle — GraphCastSmall inputs do not require sea ice concentration
- Model weights freely available via Google Cloud Storage

### Why ARCO for development, IFS for production?
- Both provide all 83 GraphCastSmall input variables including tp06
- ARCO: historical ERA5, free GCS zarr, no credentials — ideal for Colab/development
- IFS: near-real-time operational, free ECMWF open data — required for current forecasts
- NCAR_ERA5: NOT compatible (missing tp06 in lexicon)
- GFS: NOT verified compatible — do not use

### Why is precipitation mm / 6h (not mm/h)?
- GraphCastSmall natively outputs `tp06` = total precipitation for the 6-hour forecast period
- This is NOT an instantaneous rate; dividing by 6 would misrepresent the quantity
- The constitution (§VI) explicitly prohibits dividing tp06 by 6

### How does the pipeline initialize GraphCastSmall?
- GraphCastSmall requires TWO consecutive time steps: t−6h and t+0h
- Both must be fetched from ARCO or IFS before inference begins
- Earth2Studio handles this automatically via the `lead_time` coordinate in `input_coords`

### What data is generated offline?
- GraphCastSmall forecast: 5 frames (t+0h, t+6h, t+12h, t+18h, t+24h) of tp06 over Myanmar
- ARCO/IFS initialization data (fetched at pipeline runtime)
- `precipitation.bin`, `forecast.json`

### What runs on GitHub Pages?
- React + TypeScript + Vite SPA (pure JavaScript/WebGL)
- Reads pre-generated binary artifact files via `fetch()`
- No Python, no JAX, no GPU at runtime

### What requires Python/GPU?
- `scripts/generate_forecast.py`: requires GPU (T4 16 GB compatibility to be verified)
- `scripts/generate_demo_data.py`: pure NumPy, CPU-only

---

## Technical Context

**Language/Version**:
- Python 3.11+ (pipeline) — uv-managed
- TypeScript 5.4+ (frontend)
- Node.js 22.x

**Primary Dependencies**:
- Pipeline: `earth2studio>=0.17.0`, `xarray`, `numpy`, `zarr`
- Pipeline JAX deps: installed automatically by Earth2Studio GraphCastSmall
- Frontend: `react@18`, `typescript@5`, `vite@5`, `maplibre-gl@4`, `tailwindcss@3`, `zustand`

**Storage**: Static files only. Binary artifacts in `data/forecast/` (production) or
`data/demo/` (development).

**Performance Goals**:
- Initial load < 5 seconds on ≥25 Mbps
- Frame transition < 200ms after data loaded
- Total payload ~4.6 KB (no lazy loading required at this scale)

**Constraints**:
- No runtime server
- No interpolation of forecast variables (6h native steps only)
- No log/exp transform on tp06
- ARCO or IFS only (not NCAR_ERA5, not GFS)
- VRAM must be tested experimentally on target hardware

---

## Constitution Check

| §I Static-First | No runtime compute; all data pre-generated | ✓ |
| §II Earth2Studio-Mandatory | GraphCastSmall + ARCO/IFS | ✓ |
| §III Pipeline | `scripts/` standalone; 6h steps; 24h horizon; t-6h+t+0h init | ✓ |
| §IV Myanmar-Focused | xarray bbox subset; GeoJSON boundary layer | ✓ |
| §V Map-First UX | MapLibre full-viewport; controls subordinate | ✓ |
| §VI Native-Step Navigation | 6h native steps; no interpolation; tp06 not divided | ✓ |
| §VII Model-Agnostic | All config in forecast.json; no GraphCast strings in TS | ✓ |
| §VIII Performance | Float32 binary ~4.6KB; all frames loaded at once | ✓ |
| §IX Climate-Honest | Header + InfoPanel + tp06 tooltip mandatory | ✓ |
| §X Minimal Scope | pipeline → static → GitHub Pages; no extras | ✓ |
| §XI Hardware Transparency | Staged VRAM test; GPU/VRAM recorded in forecast.json | ✓ |
| §XII Resolution Honesty | 1.0° native disclosed; interpolation policy stated | ✓ |

---

## Project Structure

```text
myanmar-weather/
│
├── .github/
│   └── workflows/
│       └── deploy-pages.yml
│
├── .specify/                          # Spec Kit (existing)
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
│   │   │   ├── Header.tsx             # App title, model info, init time
│   │   │   ├── Legend.tsx             # mm/6h precipitation color scale
│   │   │   ├── Timeline.tsx           # Slider + prev/next + play/pause + speed
│   │   │   └── InfoPanel.tsx          # About/attribution + tp06 temporal semantics
│   │   ├── map/
│   │   │   ├── WeatherMap.tsx         # MapLibre GL JS container
│   │   │   └── colorscales.ts         # tp06 precipitation LUT
│   │   ├── data/
│   │   │   ├── ForecastLoader.ts      # fetch + parse binary artifacts
│   │   │   ├── ForecastStore.ts       # Zustand store
│   │   │   └── types.ts               # TypeScript types
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── index.css
│   ├── public/
│   │   └── geo/
│   │       └── myanmar-boundary.geojson
│   ├── index.html
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   └── package.json
│
├── scripts/
│   ├── generate_forecast.py           # Main pipeline (GPU required)
│   ├── generate_demo_data.py          # Demo pipeline (CPU only, 5 frames)
│   └── validate_forecast.py           # Artifact validation
│
├── notebooks/
│   └── graphcast_myanmar_forecast.ipynb  # Colab notebook (GPU inference path)
│
├── data/
│   ├── demo/                          # Committed demo artifacts
│   │   ├── forecast.json
│   │   └── precipitation.bin
│   └── forecast/                      # Production artifacts (tracked, from Colab)
│       ├── forecast.json
│       └── precipitation.bin
│
├── README.md
├── pyproject.toml
└── .gitignore
```

---

## Forecast Artifact Format

### forecast.json (complete schema)

```json
{
  "schema_version": "2.0",
  "model": "GraphCastSmall",
  "model_version": "1.0",
  "model_checkpoint": "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",
  "model_source": "gs://dm_graphcast/graphcast",
  "initialization_source": "ARCO",
  "initialization_time": "2026-08-11T00:00:00Z",
  "forecast_generated_at": "2026-08-11T04:00:00Z",
  "forecast_horizon_hours": 24,
  "native_timestep_hours": 6,
  "n_times": 5,
  "spatial_resolution_deg": 1.0,
  "display_resolution_deg": null,
  "region": "Myanmar",
  "bbox": {
    "lat_min": 9.0, "lat_max": 29.0,
    "lon_min": 92.0, "lon_max": 102.0
  },
  "grid": { "n_lat": 21, "n_lon": 11 },
  "lat": [9.0, 10.0, 11.0, "...", 29.0],
  "lon": [92.0, 93.0, 94.0, "...", 102.0],
  "times_utc": [
    "2026-08-11T00:00:00Z",
    "2026-08-11T06:00:00Z",
    "2026-08-11T12:00:00Z",
    "2026-08-11T18:00:00Z",
    "2026-08-12T00:00:00Z"
  ],
  "variables": {
    "precipitation": {
      "display_name": "Precipitation",
      "units": "mm / 6h",
      "source_variable": "tp06",
      "temporal_resolution": "6-hourly",
      "temporal_semantics": "Total precipitation accumulated over 6-hour forecast period ending at valid time",
      "temporal_disclosure": "Precipitation values represent total rainfall accumulated during the 6-hour forecast period ending at the displayed time. These are not instantaneous rainfall rates.",
      "transformation": "metres × 1000 (no log transform)",
      "transformation_provenance": {
        "source_unit": "metres",
        "conversion": "×1000",
        "accumulation_period_hours": 6,
        "log_transform_applied": false
      },
      "native_output": true,
      "file": "precipitation.bin",
      "fill_value": "NaN"
    }
  },
  "data_source_attribution": "ERA5 via ARCO (Google Cloud) or IFS HRES (ECMWF open data)",
  "model_attribution": "GraphCast (DeepMind/Google), via NVIDIA Earth2Studio",
  "earth2studio_version": ">=0.17.0",
  "is_demo": false,
  "inference_config": {
    "device": "NVIDIA T4 (16 GB)",
    "peak_vram_gb": null
  }
}
```

### Binary Format (precipitation.bin)

```
Layout:   [n_times × n_lat × n_lon] = [5 × 21 × 11]
Dtype:    float32, little-endian, C-order (row-major)
Size:     5 × 21 × 11 × 4 bytes = 4,620 bytes ≈ 4.6 KB
Encoding: NaN for missing/masked values
Reading:  new Float32Array(await response.arrayBuffer())
Indexing: value[t][lat_i][lon_i] = array[t * 21 * 11 + lat_i * 11 + lon_i]
```

---

## Pipeline Architecture

### generate_forecast.py (key logic)

```python
# scripts/generate_forecast.py
import os
import numpy as np
from datetime import datetime, timedelta, timezone

# Earth2Studio (JAX backend — no torch memory env needed)
from earth2studio.models.px import GraphCastSmall
from earth2studio.data import ARCO, IFS
from earth2studio.io import ZarrBackend
import earth2studio.run as e2run

def run_pipeline(init_time: datetime, source: str = "arco", output_dir: str = "data/forecast"):
    # --- Load model ---
    package = GraphCastSmall.load_default_package()
    model = GraphCastSmall.load_model(package)

    # --- Fetch initialization data ---
    # GraphCastSmall requires t-6h AND t+0h
    if source == "arco":
        data = ARCO()
    elif source == "ifs":
        data = IFS()
    else:
        raise ValueError(f"Unsupported source: {source}")

    # --- Run inference ---
    io = ZarrBackend()
    # e2run.deterministic handles the two-timestep init requirement internally
    io = e2run.deterministic(
        [init_time.strftime("%Y-%m-%dT%H:%M:%S")],
        n_steps=4,   # 4 steps × 6h = 24h
        prognostic=model,
        data=data,
        io=io,
    )

    # --- Post-process: extract tp06 ---
    import xarray as xr
    ds = xr.open_zarr(io.store)
    tp06 = ds["tp06"]  # shape: [n_init, n_lead, lat, lon]

    # Subset to Myanmar bbox
    tp06_mm = tp06.sel(lat=slice(9.0, 29.0), lon=slice(92.0, 102.0))

    # Convert metres → mm, clamp to ≥ 0 (NO exp transform)
    tp06_mm_values = np.maximum(tp06_mm.values * 1000.0, 0.0).astype(np.float32)

    # --- Write artifacts ---
    # Include t+0h frame (initialization state) prepended
    # Shape: [5 × 21 × 11] (t+0, t+6, t+12, t+18, t+24)
    write_artifacts(tp06_mm_values, init_time, output_dir, source)
```

### Unit Conversion (tp06 only)

```python
def tp06_metres_to_mm(tp06_m: np.ndarray) -> np.ndarray:
    """
    Convert GraphCastSmall tp06 from metres to mm / 6h.

    GraphCastSmall outputs tp06 in physical metres (NOT log space).
    Steps: metres × 1000 = mm
    Result: mm of precipitation accumulated in the 6-hour forecast period.
    Physical constraint: clamp to ≥ 0.

    NOTE: Do NOT apply exp() — there is no log transform in GraphCastSmall tp06.
    """
    mm = tp06_m * 1000.0
    return np.maximum(mm, 0.0).astype(np.float32)
```

---

## Frontend Architecture

### State (Zustand ForecastStore)

```typescript
// data/ForecastStore.ts
interface ForecastState {
  metadata: ForecastMetadata | null;
  precipitation: Float32Array | null;  // [5 × 21 × 11]
  isLoaded: boolean;
  isDemo: boolean;
  error: string | null;

  currentHour: number;      // index 0–4 into times_utc array
  isPlaying: boolean;
  playbackSpeed: 0.5 | 1 | 2 | 4;

  setHour: (h: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
  togglePlay: () => void;
  setSpeed: (s: 0.5 | 1 | 2 | 4) => void;
}
```

### Map Rendering

Frame update cycle (target < 200ms):
1. User changes step → `store.setHour(h)` (h is step index 0–4)
2. `WeatherMap` detects `currentHour` change (Zustand selector)
3. `getFrame(h)` → `Float32Array` slice [21 × 11]
4. Apply color scale: `Float32Array` → `Uint8ClampedArray` [21 × 11 × 4] (RGBA)
5. Update MapLibre canvas/image source
6. MapLibre re-renders raster layer

### Color Scale (Precipitation)

```typescript
// map/colorscales.ts

// Precipitation: sequential, 0 to 100+ mm/6h
const PRECIP_MIN = 0, PRECIP_MAX = 100;
// LUT: 256-entry RGBA lookup table
const PRECIP_LUT_ALPHA: Uint8Array = buildLUT([
  [0,   [0,   0,   0,   0]],    // 0 mm/6h: transparent
  [1,   [173, 216, 230, 200]],  // trace: light blue
  [10,  [0,   100, 255, 220]],  // 10 mm/6h: blue
  [30,  [0,   200, 0,   230]],  // 30 mm/6h: green
  [60,  [255, 255, 0,   240]],  // 60 mm/6h: yellow
  [100, [255, 0,   0,   255]],  // 100+ mm/6h: red
]);
```

### Timeline Component

The slider operates on step index (0–4), not hours directly. Lead time is derived from
`metadata.times_utc[currentHour]` relative to `metadata.initialization_time`.

```typescript
// Step mapping: index → lead hours
// index 0 → t+0h (init)
// index 1 → t+6h
// index 2 → t+12h
// index 3 → t+18h
// index 4 → t+24h
const leadHours = currentHour * (metadata?.native_timestep_hours ?? 6);
```

---

## GitHub Actions Deployment

```yaml
# .github/workflows/deploy-pages.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install frontend dependencies
        run: npm ci
        working-directory: frontend

      - name: Select forecast data (production if available, else demo)
        run: |
          mkdir -p frontend/public/data
          if [ -f data/forecast/forecast.json ] && \
             [ -f data/forecast/precipitation.bin ]; then
            echo "Using production forecast data"
            cp data/forecast/forecast.json frontend/public/data/
            cp data/forecast/precipitation.bin frontend/public/data/
          else
            echo "Using demo data"
            cp data/demo/forecast.json frontend/public/data/
            cp data/demo/precipitation.bin frontend/public/data/
          fi

      - name: Build frontend
        run: npm run build
        working-directory: frontend
        env:
          VITE_BASE_PATH: /${{ github.event.repository.name }}/

      - name: Setup Pages
        uses: actions/configure-pages@v4

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: frontend/dist

      - id: deployment
        name: Deploy to GitHub Pages
        uses: actions/deploy-pages@v4
```

---

## Technology Decisions

| Decision | Choice | Rejected | Reason |
|----------|--------|---------|--------|
| Precipitation source | GraphCastSmall native tp06 | PrecipitationAFNO | tp06 is first-class; no diagnostic model needed |
| Temporal resolution | Native 6h from GraphCastSmall | Linear interpolation | Constitution §VI prohibits interpolation |
| Initialization source | ARCO (dev) / IFS (prod) | GFS, NCAR_ERA5 | NCAR_ERA5 missing tp06; GFS unverified |
| Precipitation transform | × 1000 only | exp() × 1000 | tp06 already in physical metres; no log space |
| Precipitation units | mm / 6h | mm/h | tp06 is genuinely 6-hourly; dividing misrepresents |
| Variables displayed | Precipitation only | Temp + Precip | MVP scope; Aurora OOM necessitated simplification |
| Frontend state | Zustand | Redux, React Context | Minimal API; sufficient scope |
| Data format | Float32 binary | JSON, NetCDF | Smallest payload; native ArrayBuffer |
| Backend | JAX (GraphCastSmall) | PyTorch (Aurora1p5) | GraphCastSmall requires JAX |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| GraphCastSmall OOM on T4 (16 GB) | Medium | High | Staged VRAM test before full run; report immediately if OOM |
| JAX GPU setup issues on Colab | Low | Medium | Colab T4 has JAX pre-installed; test early in notebook |
| ARCO data unavailable for recent dates | Low | Low | Fall back to IFS for recent dates |
| tp06 negative values | Low | Low | Clamp to 0 in pipeline (physical constraint) |
| GraphCastSmall weights download fails | Low | Medium | Cache via GCS direct download in notebook |
| MapLibre canvas source API change | Low | Low | Pin MapLibre version; isolated in WeatherMap.tsx |

---

## Development Phases

**Phase 1**: Spec Kit update (constitution, research, spec, plan, tasks) — COMPLETE

**Phase 2**: Pipeline rewrite (generate_forecast.py for GraphCastSmall + ARCO/IFS)

**Phase 3**: Demo data update (generate_demo_data.py for 5 frames, 21×11 grid)

**Phase 4**: Frontend updates (Timeline 6h steps, Header title, InfoPanel metadata)

**Phase 5**: Validate pipeline (validate_forecast.py dimensions + tp06 checks)

**Phase 6**: Colab notebook rewrite (graphcast_myanmar_forecast.ipynb with staged VRAM test)

**Phase 7**: Integration testing + deployment verification

**Phase 8**: README update for GraphCastSmall architecture
