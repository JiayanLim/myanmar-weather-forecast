# Implementation Plan: Myanmar Weather Forecast Web Application

**Branch**: `001-myanmar-weather-app` | **Date**: 2026-08-09 (revised)
**Spec**: `specs/001-myanmar-weather-app/spec.md`
**Research**: `specs/001-myanmar-weather-app/research.md`
**Constitution**: `.specify/memory/constitution.md` v1.0.1

---

## Summary

Build a static GitHub Pages web application visualizing 7-day weather forecasts over Myanmar. The Earth2Studio pipeline (Aurora1p5 initialized from IFS analysis) runs offline on a GPU machine and produces lightweight binary artifacts. Aurora1p5 provides genuinely hourly temperature and precipitation via its native hourly rollout mechanism. No temporal interpolation is required or permitted. The React/MapLibre frontend consumes these artifacts to render interactive maps with a 168-hour hourly timeline.

---

## Architecture Decision Summary

### Why Aurora1p5?
- Native hourly rollout: produces t+1h through t+168h with no interpolation
- Native `tp1h` output: 1-hour accumulated precipitation is a first-class output variable
- 0.25° resolution: appropriate for Myanmar synoptic-scale visualization
- Fine-tuned on IFS analyses: highest skill with IFS initialization
- Model weights: freely available via HuggingFace `hf://microsoft/aurora`

### Why IFS initialization is required?
Aurora1p5 docstring (verified): "Aurora v1.5 was pretrained on ERA5 and fine-tuned on IFS operational analyses and as such recommended to be initialized with IFS analyses. GFS is not supported due to missing surface variables."

Aurora1p5 requires 18 surface input variables. GFS lacks 11 of them: `d2m`, `u100m`, `v100m`, `lcc`, `mcc`, `hcc`, `skt`, `stl1`, `swvl1`, `sic`, `sd`.

### Why precipitation is mm/h (1-hour accumulation)?
Aurora1p5 natively outputs `tp1h` at every hourly lead time via its internal hourly rollout. This is a genuine model prediction, not a derived or interpolated quantity. After log untransform (`exp()`) and unit conversion (m → mm), the result is mm of precipitation accumulated over the 1-hour forecast period. This is correctly displayed as "mm/h" with the disclosure that it is a 1-hour accumulation total, not an instantaneous rate.

### How hourly temperature visualization works?
Aurora1p5's `_forward_sub_steps` mechanism produces t2m at t+1h, t+2h, ..., t+168h through intermediate evaluations of its neural network within each 6-hour AR step. Each value is a distinct model prediction at that lead time.

### How does the frontend handle temporal semantics?
- Temperature: `frame[hour]` → t2m in °C → map overlay + point value
- Precipitation: `frame[hour]` → tp1h in mm/h → map overlay + point value
- Both variables use the same [169 × 81 × 41] binary layout
- `forecast.json` records variable-specific metadata: temporal resolution, unit semantics, disclosure text
- Frontend reads disclosure text from metadata → renders in tooltip and info panel

### What data is generated offline?
- Aurora1p5 forecast: all 169 hourly frames for t2m and tp1h over Myanmar bbox
- IFS/NCAR_ERA5 initialization data (fetched at pipeline runtime)
- `temperature.bin`, `precipitation.bin`, `forecast.json`

### What runs on GitHub Pages?
- React + TypeScript + Vite SPA (pure JavaScript/WebGL)
- Reads pre-generated binary artifact files via `fetch()`
- No Python, no PyTorch, no GPU at runtime

### What requires Python/GPU?
- `scripts/generate_forecast.py`: requires GPU with ≥48 GB VRAM
- `scripts/generate_demo_data.py`: pure NumPy, CPU-only, no GPU needed

### Current precipitation diagnostic recommendation?
PrecipitationAFNOv2 is the current recommended diagnostic in Earth2Studio (labeled "Improved Precipitation AFNO diagnostic model"). However, it is NOT needed for Aurora1p5, which natively outputs `tp1h`. The `PrecipitationModel` abstract interface is defined in the pipeline to support future model substitution.

---

## Technical Context

**Language/Version**:
- Python 3.11+ (pipeline) — uv-managed
- TypeScript 5.4+ (frontend)
- Node.js 22.x

**Primary Dependencies**:
- Pipeline: `earth2studio>=0.17.0`, `xarray`, `numpy`, `scipy`, `zarr`, `cfgrib`
- Frontend: `react@18`, `typescript@5`, `vite@5`, `maplibre-gl@4`, `tailwindcss@3`, `zustand`

**Storage**: Static files only. Binary artifacts in `data/forecast/` (production) or `data/demo/` (development).

**Testing**: `pytest` (pipeline), `vitest` (frontend unit), manual browser testing

**Target Platform**: GitHub Pages (static CDN). Pipeline on local/cloud GPU.

**Performance Goals**:
- Initial load < 5 seconds on ≥25 Mbps
- Frame transition < 200ms after data loaded
- Total payload < 20 MB (estimated: ~4.5 MB)

**Constraints**:
- No runtime server
- No interpolation of forecast variables
- GFS is prohibited as Aurora1p5 init source
- tp1h must pass through log untransform before storage

---

## Constitution Check

| §I Static-First | No runtime compute; all data pre-generated | ✓ |
| §II Earth2Studio-Mandatory | Aurora1p5 + IFS (with sic patch) | ✓ |
| §III Pipeline | `scripts/` standalone; sic gap documented in code | ✓ |
| §IV Myanmar-Focused | xarray bbox subset; GeoJSON boundary layer | ✓ |
| §V Map-First UX | MapLibre full-viewport; controls subordinate | ✓ |
| §VI Hourly Navigation | Aurora1p5 native hourly rollout; no interp | ✓ |
| §VII Model-Agnostic | Abstract ForecastModel; all config in forecast.json | ✓ |
| §VIII Performance | Float32 binary ~4.5MB; lazy day-chunk loading | ✓ |
| §IX Climate-Honest | Header + InfoPanel + precip tooltip mandatory | ✓ |
| §X Minimal Scope | pipeline → static → GitHub Pages; no extras | ✓ |

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
│   │   │   ├── VariableSwitcher.tsx   # Precip / Temp toggle
│   │   │   ├── Legend.tsx             # Dynamic color scale legend
│   │   │   ├── Timeline.tsx           # Slider + prev/next + play/pause + speed
│   │   │   ├── InfoPanel.tsx          # About/attribution + temporal semantics
│   │   │   ├── PointInspector.tsx     # Click popup (t2m °C + tp1h mm/h)
│   │   │   └── DemoBanner.tsx         # "DEMO DATA" banner
│   │   ├── map/
│   │   │   ├── WeatherMap.tsx         # MapLibre GL JS container
│   │   │   ├── WeatherLayer.tsx       # Float32Array → ImageData → MapLibre raster
│   │   │   └── colorscales.ts         # t2m (blue→red) + tp1h (white→blue→green→red)
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
├── forecast/
│   ├── models/
│   │   ├── base.py                    # ForecastModel abstract base
│   │   └── aurora1p5_forecast.py      # Aurora1p5Forecast concrete implementation
│   ├── sources/
│   │   ├── base.py                    # InitializationSource abstract base
│   │   ├── ifs_source.py              # IFSSource (primary, with sic handling)
│   │   └── ncar_era5_source.py        # NCAR_ERA5Source (fallback/development)
│   └── postprocessing/
│       ├── myanmar_subset.py          # xarray spatial clip
│       ├── unit_conversion.py         # K→°C, log(m)→mm/h for tp1h
│       └── artifact_writer.py         # Float32 binary + forecast.json writer
│
├── scripts/
│   ├── generate_forecast.py           # Main pipeline (GPU required)
│   ├── generate_demo_data.py          # Demo pipeline (CPU only)
│   ├── validate_forecast.py           # Artifact validation
│   └── prepare_frontend_data.py       # Optional: copy artifacts to frontend/public/
│
├── data/
│   ├── demo/                          # Committed demo artifacts
│   │   ├── forecast.json
│   │   ├── temperature.bin
│   │   └── precipitation.bin
│   └── forecast/                      # Production artifacts (gitignored)
│       ├── forecast.json
│       ├── temperature.bin
│       └── precipitation.bin
│
├── docs/
│   ├── architecture.md
│   ├── forecasting.md
│   ├── deployment.md
│   └── data-format.md
│
├── tests/
│   ├── test_unit_conversion.py
│   ├── test_myanmar_subset.py
│   ├── test_pipeline_interface.py
│   └── test_validation.py
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
  "schema_version": "1.0",
  "model": "Aurora1p5",
  "model_version": "1.5",
  "model_checkpoint": "aurora-0.25-v1.5.ckpt",
  "model_source": "hf://microsoft/aurora",
  "initialization_source": "IFS",
  "initialization_time": "2026-08-09T00:00:00Z",
  "forecast_generated_at": "2026-08-09T03:14:00Z",
  "sic_handling": "patched from ARCO ERA5 at init_time",
  "forecast_horizon_hours": 168,
  "n_times": 169,
  "spatial_resolution_deg": 0.25,
  "region": "Myanmar",
  "bbox": {
    "lat_min": 9.0, "lat_max": 29.0,
    "lon_min": 92.0, "lon_max": 102.0
  },
  "grid": { "n_lat": 81, "n_lon": 41 },
  "lat": [9.0, 9.25, "...", 29.0],
  "lon": [92.0, 92.25, "...", 102.0],
  "times_utc": ["2026-08-09T00:00:00Z", "...", "2026-08-16T00:00:00Z"],
  "variables": {
    "temperature_2m": {
      "display_name": "2m Temperature",
      "units": "°C",
      "source_variable": "t2m",
      "temporal_resolution": "hourly",
      "temporal_semantics": "Point forecast at each hour",
      "transformation": "K - 273.15",
      "file": "temperature.bin",
      "fill_value": -9999.0
    },
    "precipitation": {
      "display_name": "Precipitation",
      "units": "mm/h",
      "source_variable": "tp1h",
      "temporal_resolution": "hourly",
      "temporal_semantics": "Total precipitation accumulated over 1-hour forecast period",
      "temporal_disclosure": "Precipitation values represent total rainfall accumulated during each 1-hour forecast period. These are not instantaneous rainfall rates.",
      "transformation": "exp(raw_model_output) * 1000 (log-untransform + m to mm)",
      "native_output": true,
      "file": "precipitation.bin",
      "fill_value": -9999.0
    }
  },
  "data_source_attribution": "IFS HRES analysis (ECMWF open data)",
  "model_attribution": "Microsoft Research Aurora v1.5",
  "earth2studio_version": ">=0.17.0",
  "is_demo": false
}
```

### Binary Format (temperature.bin / precipitation.bin)

```
Layout:   [n_times × n_lat × n_lon] = [169 × 81 × 41]
Dtype:    float32, little-endian, C-order (row-major)
Size:     169 × 81 × 41 × 4 bytes = 2,251,476 bytes ≈ 2.1 MB per variable
Encoding: NaN for missing/masked values
Reading:  new Float32Array(await response.arrayBuffer())
Indexing: value[t][lat_i][lon_i] = array[t * 81 * 41 + lat_i * 41 + lon_i]
```

Total payload: ~4.3 MB for both variables.

---

## Pipeline Architecture

### Class Interfaces

```python
# forecast/models/base.py
from abc import ABC, abstractmethod
import xarray as xr
from datetime import datetime

class ForecastModel(ABC):
    """Abstract interface for Earth2Studio prognostic models."""

    @abstractmethod
    def run(self, init_data: xr.Dataset, n_hours: int) -> xr.Dataset:
        """Run forecast. Returns dataset with 'time' and 'lead_time' coords."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def version(self) -> str: ...

    @property
    @abstractmethod
    def spatial_resolution_deg(self) -> float: ...

    @property
    @abstractmethod
    def output_variables(self) -> list[str]: ...


# forecast/sources/base.py
class InitializationSource(ABC):
    """Abstract interface for forecast initialization data sources."""

    @abstractmethod
    def fetch(self, time: datetime) -> xr.Dataset:
        """Fetch analysis data at given time. Returns dataset with all required vars."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...
```

### Aurora1p5Forecast Implementation

```python
# forecast/models/aurora1p5_forecast.py
import earth2studio.run as run
from earth2studio.models.px import Aurora1p5
from earth2studio.io import ZarrBackend

class Aurora1p5Forecast(ForecastModel):
    name = "Aurora1p5"
    version = "1.5"
    spatial_resolution_deg = 0.25
    output_variables = ["t2m", "tp1h"]

    def run(self, init_source: InitializationSource, init_time: datetime, n_hours: int = 168) -> xr.Dataset:
        model = Aurora1p5.load_model(Aurora1p5.load_default_package())
        data = init_source.as_earth2studio_source()
        io = ZarrBackend()
        io = run.deterministic([init_time.strftime("%Y-%m-%d")], n_hours, model, data, io)
        return xr.open_zarr(io.store)
```

### IFS Source with sic Handling

```python
# forecast/sources/ifs_source.py
from earth2studio.data import IFS
import xarray as xr
import numpy as np

SIC_HANDLING_METHODS = {
    "zero": "sic set to 0.0 globally (appropriate for tropical Myanmar init)",
    "arco": "sic patched from ARCO ERA5 at init_time",
    "ncar": "sic patched from NCAR_ERA5 at init_time",
}

class IFSSource(InitializationSource):
    name = "IFS"

    def __init__(self, sic_method: str = "zero"):
        self.sic_method = sic_method
        self._ifs = IFS()

    def fetch(self, time: datetime) -> tuple[xr.Dataset, str]:
        """Returns (dataset, sic_handling_description)."""
        ds = self._ifs.fetch(...)
        if self.sic_method == "zero":
            ds["sic"] = xr.zeros_like(ds["t2m"])
            return ds, SIC_HANDLING_METHODS["zero"]
        elif self.sic_method == "arco":
            sic = self._fetch_arco_sic(time)
            ds["sic"] = sic
            return ds, SIC_HANDLING_METHODS["arco"]
        ...
```

### Unit Conversion

```python
# forecast/postprocessing/unit_conversion.py
import numpy as np

def kelvin_to_celsius(t2m_k: np.ndarray) -> np.ndarray:
    """Convert temperature from Kelvin to Celsius."""
    return (t2m_k - 273.15).astype(np.float32)

def tp1h_log_to_mm(tp1h_log: np.ndarray) -> np.ndarray:
    """
    Convert Aurora1p5 tp1h from log space to mm/h.

    Aurora1p5 outputs tp1h in log space (natural log of meters).
    Steps: exp(raw) → meters → multiply by 1000 → mm
    Result interpretation: mm of precipitation accumulated in the 1-hour forecast period.
    Negative values after exp() are clamped to 0 (numerical noise).
    """
    mm = np.exp(tp1h_log) * 1000.0
    mm = np.maximum(mm, 0.0)  # physical constraint: precipitation ≥ 0
    return mm.astype(np.float32)
```

---

## Frontend Architecture

### State (Zustand ForecastStore)

```typescript
// data/ForecastStore.ts
interface ForecastState {
  metadata: ForecastMetadata | null;
  temperatureData: Float32Array | null;    // [169 × 81 × 41]
  precipitationData: Float32Array | null;  // [169 × 81 × 41]
  isLoaded: boolean;
  isDemo: boolean;
  error: string | null;

  activeVariable: 'temperature_2m' | 'precipitation';
  currentHour: number;      // 0–168
  isPlaying: boolean;
  playbackSpeed: 0.5 | 1 | 2 | 4;

  setHour: (h: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
  togglePlay: () => void;
  setVariable: (v: 'temperature_2m' | 'precipitation') => void;
  setSpeed: (s: 0.5 | 1 | 2 | 4) => void;
}
```

### Map Rendering

Frame update cycle (target < 200ms):
1. User changes hour → `store.setHour(h)`
2. `WeatherLayer` detects `currentHour` change (Zustand selector)
3. `getFrame(variable, h)` → `Float32Array` slice [81 × 41]
4. Apply color scale: `Float32Array` → `Uint8ClampedArray` [81 × 41 × 4] (RGBA)
5. `map.updateImage(layerId, new ImageData(rgba, 41, 81))`
6. MapLibre re-renders raster layer

This avoids re-adding sources/layers on every frame change, keeping transitions sub-200ms.

### Color Scales

```typescript
// map/colorscales.ts

// Temperature: blue-white-red diverging, 15°C to 40°C
const TEMP_MIN = 15, TEMP_MAX = 40;
const TEMP_COLORMAP: Uint8Array = buildColormap([
  [0, [49, 54, 149]],     // 15°C: dark blue
  [50, [255, 255, 255]],  // 27.5°C: white
  [100, [165, 0, 38]],    // 40°C: dark red
]);

// Precipitation: sequential, 0 to 50+ mm/h
const PRECIP_MIN = 0, PRECIP_MAX = 50;
const PRECIP_COLORMAP: Uint8Array = buildColormap([
  [0, [255, 255, 255]],   // 0 mm/h: white
  [5, [173, 216, 230]],   // 2.5 mm/h: light blue
  [20, [0, 100, 255]],    // 10 mm/h: blue
  [60, [0, 200, 0]],      // 30 mm/h: green
  [85, [255, 255, 0]],    // 42.5 mm/h: yellow
  [100, [255, 0, 0]],     // 50+ mm/h: red
]);
```

### Lazy Loading Strategy

Load data in day-sized chunks to meet §VIII performance:
1. On app load: fetch `forecast.json` + first 24h of both variables (frames 0–23)
2. On demand: fetch frames 24–47, 48–71, etc. as user scrolls near boundary
3. All frames cached in memory after first load; no re-fetch on slider movement

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

      - name: Copy demo forecast data
        run: |
          mkdir -p frontend/public/data
          cp data/demo/forecast.json frontend/public/data/
          cp data/demo/temperature.bin frontend/public/data/
          cp data/demo/precipitation.bin frontend/public/data/

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
| Precipitation source | Aurora1p5 native tp1h | PrecipitationAFNOv2 | Aurora1p5 natively outputs tp1h; PrecipAFNO would add VRAM + variable bridging complexity for no benefit |
| Temporal resolution | Native 1h from Aurora1p5 | Linear interpolation | Constitution §VI prohibits interpolation; Aurora1p5's native rollout satisfies the requirement |
| Initialization source | IFS (open data) + sic patch | GFS | GFS explicitly unsupported per Aurora1p5 source; IFS is free/open |
| Precipitation units | mm/h (1-hour accumulation) | mm/6h (PrecipAFNO convention) | tp1h is genuinely 1-hourly |
| Map library | MapLibre GL JS | Leaflet | WebGL raster performance for frame-by-frame updates |
| State management | Zustand | Redux, React Context | Minimal API; sufficient scope |
| Data format | Float32 binary | JSON, NetCDF, Zarr | Smallest payload; native ArrayBuffer; no library needed in browser |
| Routing | Hash routing | BrowserRouter | GitHub Pages has no server-side routing support |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| IFS open data API changes | Low | Medium | Monitor ECMWF open data releases; pin Earth2Studio version |
| sic patch from ARCO fails for given init_time | Low | Medium | Fall back to climatological zero with documentation |
| Aurora1p5 48GB VRAM not available | High | Medium | Demo data path requires no GPU; document cloud GPU options (AWS p3, GCP A2) |
| tp1h log untransform produces unexpected values | Low | High | `validate_forecast.py` checks tp1h ≥ 0 and < 500 mm/h; pipeline halts on failure |
| HuggingFace model download blocked/rate-limited | Low | Medium | Cache weights locally after first download |
| MapLibre `updateImage()` API change | Low | Low | Pin MapLibre version; isolated in WeatherLayer.tsx |

---

## Development Phases

**Phase 1**: Project scaffold (dirs, pyproject.toml, package.json, configs, .gitignore)

**Phase 2**: Demo data + validation (CPU-only; generates commitment-ready demo artifacts)

**Phase 3**: Frontend map + variable switcher (MapLibre, overlay, legend, variable toggle)

**Phase 4**: Timeline + controls (slider, prev/next, play/pause, speed)

**Phase 5**: Point inspector + info panel + demo banner

**Phase 6**: Earth2Studio pipeline (Aurora1p5 + IFS + sic handling + tp1h untransform)

**Phase 7**: Integration testing (connect real artifacts → frontend)

**Phase 8**: GitHub Actions deployment + README + docs
