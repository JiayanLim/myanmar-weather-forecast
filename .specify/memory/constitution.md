<!--
Sync Impact Report
==================
Version change: 1.0.1 → 2.0.0 (MAJOR — model architecture replacement)
Changed sections:
  - §II: Aurora1p5 replaced by GraphCast Small. ARCO/IFS as init sources.
    sic-patch constraint removed (GraphCastSmall does not require sic).
    tp1h/log-untransform constraint replaced by tp06/no-transform.
  - §III: Removed sic gap requirement and log untransform.
    Updated artifact dimensions to GraphCastSmall (1° grid, 6h steps, 24h).
  - §VI: Updated to native 6h timestep navigation across 24h horizon.
    Hourly navigation requirement relaxed to match model's native step.
  - §IX: Updated precipitation disclosure to tp06 (6h accumulation).
  - Architecture Constraints: GraphCastSmall, JAX backend noted.
  - Data & Model Integrity: Updated validation rules for tp06.
  - New §XI: Hardware Transparency — VRAM must be experimentally verified.
  - New §XII: Resolution Honesty — interpolation cannot add model information.
Removed: Aurora1p5, IFS sic gap, log untransform, 168h horizon, hourly slider
Reason: Aurora1p5 exceeded T4 VRAM (16 GB). GraphCast Small selected as replacement.
Approved: User decision on 2026-08-11
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
- **Prognostic model**: `earth2studio.models.px.GraphCastSmall`
- **Initialization sources**: ARCO (`earth2studio.data.ARCO`) or
  IFS (`earth2studio.data.IFS`) — both verified compatible with all
  83 GraphCastSmall input variables. NCAR_ERA5 is NOT compatible
  (missing `tp06`).
- **Precipitation**: GraphCastSmall natively outputs `tp06` (6-hour
  accumulated total precipitation, in metres). No log transform
  or diagnostic model is required.

**GFS is NOT a verified initialization source for GraphCastSmall.**
Use ARCO (historical/reanalysis, no credentials) or IFS (operational).

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
- Temporally structured as **6-hour steps** across a **24-hour horizon**
  (4 forecast steps: t+6h, t+12h, t+18h, t+24h, plus t+0h init = 5 frames)
- Encoded in a browser-efficient format (float32 binary, documented per release)

**Initialization requirement**: GraphCastSmall requires two consecutive time
steps as input: t−6h and t+0h. Both must be fetched from the initialization
source before inference begins.

**No log transform**: GraphCastSmall's `tp06` output is already in physical
metres. The only required conversion is ×1000 to obtain mm per 6-hour period.
Applying an exponential transform would be incorrect.

### IV. Myanmar-Focused

All forecasts, overlays, legends, and metadata MUST target Myanmar.
The default map view MUST be centered on Myanmar. Geographic reference layers
(national boundary) MUST be present and clearly visible.
Global forecast grids MUST be clipped to the Myanmar bbox before delivery
to the frontend.

### V. Map-First UX

The interactive weather map MUST dominate the viewport. Supporting UI elements
(timeline, legend, metadata panel) MUST be visually subordinate to the map.
The map MUST support pan, zoom, weather overlay rendering, and
click-to-inspect interactions.

### VI. Native-Step Navigation

Users MUST be able to navigate the forecast step-by-step across the full
24-hour horizon. The timeline control MUST provide:
- Slider scrubber across all available forecast steps (0, 6, 12, 18, 24h)
- Previous / Next step buttons
- Play / Pause animation
- Displayed forecast date, UTC time, and lead time offset

The map, legend, timestamp, and point values MUST update synchronously
without a full page reload when the selected step changes.

**Temporal semantics MUST be preserved and disclosed:**

- **Precipitation (tp06)**: GraphCastSmall's `tp06` is **6-hour accumulated
  total precipitation** for the forecast period ending at the valid time.
  Units: mm / 6h. **Do NOT divide by 6 to create an hourly rate.**
  The UI MUST display precipitation as 6-hour accumulation and explicitly
  disclose: "Precipitation values represent total rainfall accumulated during
  the 6-hour forecast period ending at the displayed time."

Synthesizing intermediate hourly frames by interpolation is PROHIBITED.
Only native model output steps (every 6h) are displayed.

### VII. Model-Agnostic Frontend

The frontend MUST NOT be tightly coupled to any specific Earth2Studio model.
All model-specific details (name, resolution, timestep, variables) MUST be
read from `forecast.json` at runtime. Swapping the model MUST require only
re-running the pipeline and updating the metadata file.

No GraphCast-specific or Aurora-specific strings MUST appear in TypeScript
source code.

### VIII. Performance

- Initial application load MUST complete in < 5 seconds on a standard
  broadband connection (≥25 Mbps), measured from cold cache.
- Step-to-step frame transitions MUST feel near-instantaneous (<200 ms)
  after forecast data is loaded.
- The total forecast payload (5 frames × 21 × 11 × 4 bytes) is ~4.6 KB —
  no lazy loading is required at this scale.

### IX. Climate-Honest

Every forecast display MUST clearly show:
- Model name and version
- Spatial resolution (native and display)
- Forecast initialization time (UTC)
- Forecast valid time (UTC)
- Lead time offset (e.g., "+12 h")
- Native model timestep
- A disclaimer that forecast skill degrades with lead time

**Precipitation temporal semantics MUST be explicitly disclosed:**
- `tp06` from GraphCastSmall represents **6-hour accumulated total
  precipitation** over the forecast period ending at the valid time.
  It is NOT an instantaneous rainfall rate.
- The UI MUST display a tooltip or info panel explaining the accumulation
  period and that the value is not an instantaneous measurement.

Mislabelling forecast fields is a critical defect.

### X. Minimal Scope

The MVP MUST NOT include: user accounts, authentication, databases,
Redis, Kafka, Kubernetes, Docker (optional for convenience only),
microservices, cloud provider lock-in, or paid API dependencies beyond
the GPU compute needed for forecast generation.
The architecture is: pipeline scripts → static files → GitHub Pages.

### XI. Hardware Transparency (NEW)

The production pipeline MUST explicitly record the GPU hardware used
for inference and the measured peak VRAM consumption. The Colab notebook
MUST report GPU type, total VRAM, and peak allocated VRAM after each
inference run.

**VRAM requirements MUST be established experimentally**, not assumed from
documentation badges. The GraphCastSmall badge states "Rec VRAM: 40 GB".
Whether it runs on a 16 GB T4 MUST be verified through a staged test:
1. GPU memory baseline
2. Single-step inference memory test
3. Full 24h forecast

If a single inference step exceeds available VRAM, this MUST be reported
immediately. No further VRAM-reduction hacks MUST be attempted without
explicit user approval.

### XII. Resolution Honesty (NEW)

The application MUST clearly communicate the distinction between native
model resolution and display resolution.

GraphCastSmall operates at **1.0° resolution** (~111 km grid spacing).
If the display uses bilinear interpolation to a finer grid for visual
clarity, the UI MUST state:

> "Display resolution is interpolated from the 1.0° native model grid
>  and does not represent additional meteorological information."

The application MUST NOT imply that interpolation adds forecast information
beyond the native 1.0° model grid.

## Architecture Constraints

- **Frontend stack**: React, TypeScript, Vite, MapLibre GL JS, Tailwind CSS
- **Map library**: MapLibre GL JS (WebGL raster performance)
- **Basemap**: Open-source tile provider (no proprietary API keys required)
- **Forecast format**: Float32 binary arrays per variable; metadata in JSON
- **Python pipeline**: Earth2Studio ≥ 0.17.0, uv-managed virtual environment
- **GraphCastSmall backend**: JAX + Haiku (DeepMind) wrapped by Earth2Studio;
  runs on CUDA GPU via JAX's GPU backend; uses bfloat16 internally
- **GitHub Actions**: Used for frontend deployment ONLY; NOT for GPU inference
- **Model interface**: No model-specific strings in frontend TypeScript

## Data & Model Integrity

- The pipeline MUST validate output before generating frontend artifacts:
  - No NaN values in expected forecast fields over the Myanmar bbox
  - Timestamps must be monotonically increasing
  - tp06 MUST be non-negative (physical constraint: no negative precipitation)
  - tp06 MUST be within a physically plausible range (max < configurable threshold)
- Demo data (for development/CI) MUST be clearly separated from production
  forecast data. The frontend MUST display a visible "DEMO DATA" banner
  when consuming demo artifacts.
- No secrets, API keys, or model credentials MUST be committed to the
  repository.
- `forecast.json` MUST record the transformation provenance for tp06,
  including source unit (metres), conversion (×1000), and accumulation period.

## Architecture Decision Log

### ADR-010: Aurora1p5 → GraphCastSmall (2026-08-11)

**Decision**: Replace Aurora1p5 with GraphCastSmall.

**Reason**: Aurora1p5 attempted inference on a free Colab NVIDIA T4 (16 GB VRAM)
and failed with `OutOfMemoryError` at the first inference step. The model attempted
to allocate 824 MB when only 571 MB remained after loading 13.38 GB of weights and
activations. Memory optimizations (bfloat16, inference_mode, expandable_segments)
were applied but insufficient for Aurora1p5's attention mechanism.

**Why GraphCastSmall**:
- Native 1.0° resolution reduces activation memory vs. 0.25°
- Already uses bfloat16 internally (Bfloat16Cast in DeepMind's GraphCast)
- JAX's memory management differs from PyTorch — may have different footprint
- Still a scientifically valid global medium-range forecast model

**What remains unknown**:
- Whether GraphCastSmall actually fits on T4 (16 GB) — MUST be tested

**Constraints accepted**:
- 1.0° resolution (coarser than Aurora1p5's 0.25°)
- 6h native timestep (not hourly)
- 24h forecast horizon (vs. 48h)
- tp06 (6h accumulated) rather than tp1h (1h accumulated)

## Governance

Constitution supersedes all other specifications and implementation decisions.
Any amendment requires:
1. Explicit user approval
2. A version bump per semantic versioning (MAJOR for removals/redefinitions,
   MINOR for additions, PATCH for clarifications)
3. Update to this file before implementation proceeds

All feature specifications (spec.md) and implementation plans (plan.md) MUST
include a Constitution Check section verifying compliance with each principle.

**Version**: 2.0.0 | **Ratified**: 2026-08-09 | **Last Amended**: 2026-08-11
