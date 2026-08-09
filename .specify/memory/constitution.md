<!--
Sync Impact Report
==================
Version change: 1.0.0 → 1.0.1 (MINOR — additions and clarifications)
Added sections:
  - §II: Clarified Earth2Studio-Mandatory to specify Aurora1p5 with IFS initialization
  - §III: Added sic gap handling requirement and IFS-first data source requirement
  - §VI: Clarified hourly navigation uses Aurora1p5 native hourly rollout (NOT interpolation)
  - §IX: Clarified precipitation temporal semantics (tp1h = 1-hour accumulated)
  - Architecture Constraints: Added model interface requirements
  - Data & Model Integrity: Added IFS sic gap requirement and log untransform requirement
Removed: None
Follow-up TODOs: None
Approved: User decision on 2026-08-09
-->

# Myanmar Weather Forecast Constitution

## Core Principles

### I. Static-First (NON-NEGOTIABLE)

The frontend MUST be deployable as a fully static GitHub Pages site.
No server-side inference, no backend API, no runtime Python or CUDA
is permitted at serve time. All forecast data MUST be pre-generated
and committed or uploaded as build artifacts before deployment.

**Rationale**: GitHub Pages provides zero-cost, zero-maintenance hosting.
Any runtime compute dependency would require a separate server, defeating
the purpose of a self-contained deployable application.

### II. Earth2Studio-Mandatory (NON-NEGOTIABLE)

All production forecast data MUST be generated using the NVIDIA Earth2Studio
framework. Synthetic, random, hard-coded, or third-party-API-sourced weather
values are forbidden in the production data path.

The current production model stack is:
- **Prognostic model**: `earth2studio.models.px.Aurora1p5`
- **Initialization source**: IFS HRES analysis (`earth2studio.data.IFS`) as primary;
  NCAR_ERA5 or ARCO as development/historical fallbacks
- **Precipitation**: Aurora1p5 natively outputs `tp1h` (1-hour accumulated total
  precipitation). PrecipitationAFNO/v2 MUST NOT be used as a substitute for
  Aurora1p5's native precipitation output unless Aurora1p5 is replaced by a model
  that does not natively produce precipitation.

**GFS is explicitly prohibited as an Aurora1p5 initialization source.** The model
was pretrained on ERA5 and fine-tuned on IFS operational analyses. GFS lacks the
required surface variables (d2m, u100m, v100m, lcc, mcc, hcc, skt, stl1, swvl1,
sic, sd) that Aurora1p5 requires as inputs.

**Rationale**: The application's value proposition is AI-based NWP via Earth2Studio.
Substituting fake data, incompatible data sources, or incorrect initialization
violates the project's scientific integrity.

### III. Forecast-Artifact Pipeline

The inference pipeline (Earth2Studio → Python post-processing → frontend-ready
artifacts) MUST be a standalone, independently executable set of scripts.
The frontend MUST only consume the output artifacts; it MUST NOT execute any
inference or call any live weather API at runtime.

The pipeline MUST produce artifacts that are:
- Spatially subset to the Myanmar bounding box (92°E–102°E, 9°N–29°N)
- Temporally structured as hourly frames for a 7-day horizon (168 frames for
  temperature; 28 frames for precipitation aligned to 6-hour windows, mapped
  to hourly display per §VI)
- Encoded in a browser-efficient format (evaluated and documented per release)

**IFS sic gap**: The open-data IFS (`earth2studio.data.IFS`) does not publish sea
ice concentration (`sic`). The pipeline MUST handle this gap by one of:
1. Supplementing `sic` from a compatible source (ARCO or NCAR_ERA5)
2. Padding `sic` with climatologically appropriate values (documented in pipeline)
This gap MUST be documented in pipeline code and deployment guide. The chosen
approach MUST NOT silently corrupt the forecast.

**Log untransform**: Aurora1p5's `tp1h` and `sf1h` outputs are in log space and
MUST be untransformed (exp) before unit conversion and storage.

### IV. Myanmar-Focused

All forecasts, overlays, legends, and metadata MUST target Myanmar.
The default map view MUST be centered on Myanmar. Geographic reference layers
(national boundary, states/regions) MUST be present and clearly visible.
Global forecast grids MUST be clipped to the Myanmar bbox before delivery
to the frontend.

### V. Map-First UX

The interactive weather map MUST dominate the viewport. Supporting UI elements
(timeline, legend, metadata panel, variable switcher) MUST be visually
subordinate to the map. The map MUST support pan, zoom, weather overlay
rendering, and click-to-inspect interactions.

### VI. Hourly Navigation (NON-NEGOTIABLE)

Users MUST be able to navigate the forecast hour-by-hour across the full
7-day horizon. The timeline control MUST provide:
- Slider scrubber across all 168 forecast hours (0 to 168, step 1)
- Previous / Next hour buttons
- Play / Pause animation with configurable speed (0.5×, 1×, 2×, 4×)
- Displayed forecast date, UTC time, and lead time offset

The map, legend, timestamp, and statistics MUST update synchronously
without a full page reload when the selected hour changes.

**Temporal semantics of each variable MUST be preserved and disclosed:**

- **Temperature (t2m)**: Aurora1p5's native hourly rollout produces a genuinely
  hourly temperature forecast. Each hour from t+1 to t+168 has a distinct model
  prediction. Display as-is with no interpolation.

- **Precipitation (tp1h)**: Aurora1p5's `tp1h` is 1-hour accumulated total
  precipitation at each hourly lead time. This is genuine model output, not
  interpolated. Units: mm / 1h. **Do NOT divide by time to create a rate.**
  The UI MUST display precipitation as 1-hour accumulation (mm/h interpreted as
  mm accumulated in that 1-hour window, not as an instantaneous rate).

Linear interpolation of weather variables to synthesize missing hourly frames
is PROHIBITED in the production data path. If a model with a coarser native
timestep is used in a future release, the interpolation must be clearly flagged
in the artifact metadata.

### VII. Model-Agnostic Frontend

The frontend MUST NOT be tightly coupled to any specific Earth2Studio model.
All model-specific details (name, resolution, timestep, variables) MUST be
read from a metadata JSON file at runtime. Swapping the forecasting model
MUST require only: re-running the pipeline and updating the metadata file.

The Python pipeline MUST use abstract interfaces for the forecast model and
data source, enabling model substitution without frontend changes:
```
ForecastModel (abstract)
    └── Aurora1p5Forecast (concrete)

InitializationSource (abstract)
    ├── IFSSource (primary)
    └── NCAR_ERA5Source / ARCOSource (fallback)
```

### VIII. Performance

- Initial application load MUST complete in < 5 seconds on a standard
  broadband connection (≥25 Mbps), measured from cold cache.
- Hour-to-hour frame transitions MUST feel near-instantaneous (<200 ms)
  after forecast data for the current day is loaded.
- Forecast data MUST NOT require downloading the entire 168-frame dataset
  before the first frame renders. Lazy-loading or chunked loading is required.

### IX. Climate-Honest

Every forecast display MUST clearly show:
- Model name and version
- Spatial resolution
- Forecast initialization time (UTC)
- Forecast valid time (UTC)
- Lead time offset (e.g., "+42 h")
- A disclaimer that forecast skill degrades with lead time

**Precipitation temporal semantics MUST be explicitly disclosed:**
- `tp1h` from Aurora1p5 represents **1-hour accumulated total precipitation**
  over the forecast hour. It is NOT an instantaneous rainfall rate.
- The UI MUST display "mm / 1h" (or "mm/h" clearly explained as 1-hour accumulation)
  and MUST NOT imply the value is an instantaneous point measurement.
- A tooltip or info panel MUST explain: "Precipitation values represent the total
  rainfall accumulated during each 1-hour forecast period."

Mislabelling forecast fields is a critical defect.

### X. Minimal Scope

The MVP MUST NOT include: user accounts, authentication, databases,
Redis, Kafka, Kubernetes, Docker (optional for convenience only),
microservices, cloud provider lock-in, or paid API dependencies beyond
the GPU compute needed for forecast generation.
The architecture is: pipeline scripts → static files → GitHub Pages.

## Architecture Constraints

- **Frontend stack**: React, TypeScript, Vite, MapLibre GL JS, Tailwind CSS
- **Map library**: MapLibre GL JS (evaluated over Leaflet for WebGL raster
  tile performance)
- **Basemap**: Open-source tile provider (no proprietary API keys required)
- **Forecast format**: Float32 binary arrays per variable; metadata in JSON;
  selection documented in plan.md
- **Python pipeline**: Earth2Studio ≥ 0.17.0, uv-managed virtual environment
- **GitHub Actions**: Used for frontend deployment ONLY; NOT for GPU inference
- **Model interface**: Abstract base class for ForecastModel; no Aurora-specific
  strings in frontend TypeScript code

## Data & Model Integrity

- The pipeline MUST validate output before generating frontend artifacts:
  - No NaN values in expected forecast fields over the Myanmar bbox
  - Timestamps must be monotonically increasing
  - Units must be verified against Earth2Studio source variable metadata
  - tp1h MUST be non-negative after log untransform
  - t2m MUST be within physically plausible range (−20 to 60°C over Myanmar)
- Demo data (for development/CI) MUST be clearly separated from production
  forecast data. The frontend MUST display a visible "DEMO DATA" banner
  when consuming demo artifacts.
- No secrets, API keys, or model credentials MUST be committed to the
  repository.
- The `sic` handling method (patch source or padding value) MUST be recorded
  in `forecast.json` artifact metadata so downstream validation can verify it.

## Governance

Constitution supersedes all other specifications and implementation decisions.
Any amendment requires:
1. Explicit user approval
2. A version bump per semantic versioning (MAJOR for removals/redefinitions,
   MINOR for additions, PATCH for clarifications)
3. Update to this file before implementation proceeds

All feature specifications (spec.md) and implementation plans (plan.md) MUST
include a Constitution Check section verifying compliance with each principle.

**Version**: 1.0.1 | **Ratified**: 2026-08-09 | **Last Amended**: 2026-08-09
