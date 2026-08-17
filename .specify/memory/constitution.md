<!--
Sync Impact Report
==================
Version change: 1.0.1 → 2.0.0 (MAJOR — model architecture replacement)
  - §II: Aurora1p5 replaced by GraphCast Small. ARCO/IFS as init sources.
    sic-patch constraint removed. tp1h/log-untransform → tp06/no-transform.
  - §III: 1° grid, 6h steps, 24h horizon (5 frames).
  - §VI: Native 6h timestep navigation across 24h horizon.
  - §IX: Precipitation disclosure for tp06 (6h accumulation).
  - New §XI: Hardware Transparency — VRAM experimentally verified.
  - New §XII: Resolution Honesty — interpolation cannot add model information.
Removed: Aurora1p5, IFS sic gap, log untransform, 168h horizon, hourly slider
Reason: Aurora1p5 exceeded T4 VRAM (16 GB). GraphCast Small selected as replacement.
Approved: User decision on 2026-08-11

Version change: 2.0.0 → 2.1.0 (MINOR — 48h + temperature + M4 CPU)
  - §III: 24h → 48h horizon; 5 → 9 frames; added t2m/temperature variable.
  - §VI: 24h → 48h horizon; slider covers 0–48h.
  - §VIII: Payload ~16.6 KB (two binaries, 9 frames each).
  - §XI: M4 CPU validated (not T4 GPU); hardware metric is RSS not VRAM.
  - Architecture Constraints: JAX CPU (XLA ARM64) noted as validated path.
  - ADR-011: 24h → 48h horizon extension with temperature addition.
Reason: Full 48h inference fits on M4 CPU in ~78s. Temperature (t2m) is a
  native GraphCastSmall output with no additional inference cost.
Approved: User decision on 2026-08-11

Version change: 2.1.0 → 3.0.0 (MAJOR — new target architecture; legacy baseline preserved)
  - §II: Distinguishes Legacy Validated Architecture (GraphCastSmall/M4/48h) from
    New Target Architecture (model TBD, 7-day, 4 variables, ERA5 1990–2020 init,
    2021-01-01 initialization date). Model selection explicitly pending.
  - §III: New target pipeline adds wind components (u10m/v10m or pressure-level winds),
    four display variables (precipitation mm/hr, wind direction °, wind speed kt, temp °C).
    Legacy pipeline (48h, GraphCastSmall, 2 variables) preserved as historical validated.
  - §VI: Target horizon updated to 168h / 7-day at model-native timestep (TBD).
    Precipitation display target updated to mm/hr with model-output-to-display-rate
    conversion clearly distinguished from native model semantics.
  - §VIII: Payload estimate deferred until model and variable count confirmed.
  - §IX: Four-variable disclosure requirements added (wind direction/speed, precipitation
    as mm/hr, temperature). Existing tp06 6h accumulation disclosure preserved for legacy.
  - §XI: Hardware requirements for new target are unknown until model is selected.
  - New ADR-012: Legacy vs New Target architecture distinction.
  - New ADR-013: 7-day forecast horizon requirement.
  - New ADR-014: Four-variable meteorological product.
  - New ADR-015: ERA5 1990–2020 as initialization source; scientific role distinction.
Reason: Project requirements changed. Investigation of 7-day Myanmar forecast using
  ERA5 historical data and 2021-01-01 initialization. Model selection pending research.
Approved: User decision on 2026-08-12

Version change: 3.0.0 → 3.1.0 (MINOR — GCOp model confirmed, R1–R3 complete, hardware validated)
  - §II: GraphCastOperational selected and hardware-validated (R3 PASS, ADR-018).
    tp06 metres/6h, t2m Kelvin, u10m/v10m m/s confirmed in output. "TBD" language removed.
  - §III: New target pipeline fully specified: GCOp 0.25°, 168h/28 AR steps/29 frames,
    4 variables, schema v4.0. Myanmar 81×41 at 0.25°. Two-timestep ARCO init.
  - §VI: New target precipitation semantics confirmed: tp06 metres×1000/6→mm/hr (ADR-021).
    Wind direction FROM convention, vector-component interpolation required (ADR-020).
  - §VIII: Payload confirmed 1.54 MB (4×29×81×41×4 bytes). Load-all-at-startup policy (ADR-022).
  - §XI: M4 hardware validated for GCOp: peak ~5–6 GB RSS during XLA compile, 1.99 GB post.
    Per-step: ~25 min measured; full 7-day: ~12–20 h. Research/demo pipeline, not daily production.
  - §XII: GCOp native 0.25° (~27.8 km) documented. Bilinear interpolation prohibited for wind
    direction (circular quantity); vector-component bilinear required (ADR-020).
  - ADR-012: Closed with GraphCastOperational as final approved selection.
  - ADR-019: Schema v4.0 artifact design (4 variables, 0.25°, 168h, 29 frames).
  - ADR-020: Wind direction visualization — vector-component interpolation (sin/cos).
  - ADR-021: GCOp precipitation conversion — tp06 metres×1000/6 → mm/hr.
  - ADR-022: Payload loading policy — all 4 variable binaries loaded at startup.
Reason: R1–R3 phases complete. GraphCastOperational validated on M4 (ADR-018, R3 PASS).
  Phase R4 7-day pipeline implementation approved and in progress (2026-08-16).
Approved: User decision on 2026-08-16
-->

# Myanmar Weather Forecast Constitution

---

## ARCHITECTURE STATUS (as of 2026-08-12)

### Legacy / Current Validated Architecture

The following architecture has been implemented and validated:

| Property | Value |
|----------|-------|
| Model | GraphCastSmall (Earth2Studio 0.17.0) |
| Resolution | 1.0° global |
| Timestep | 6h native |
| Horizon | 48h (8 AR steps + init = 9 frames) |
| Variables | tp06 (precipitation, mm/6h) + t2m (temperature, °C) |
| Initialization | ARCO (historical ERA5) or IFS (operational) |
| Hardware | Apple M4 CPU, JAX XLA ARM64, ~78s, 2.34 GB RSS |
| Schema | v3.0 |
| Deployed | GitHub Pages — live, working |

This implementation MUST NOT be broken by Spec Kit updates.

### New Target Architecture

The new target architecture has been selected and hardware-validated (R1–R3 COMPLETE):

| Property | Value |
|----------|-------|
| Model | GraphCastOperational (Earth2Studio 0.17.0) |
| Resolution | 0.25° global (721 × 1440); Myanmar subset 81 × 41 |
| Timestep | 6h native |
| Horizon | 168h / 7 days (28 AR steps + t+0h init = 29 frames) |
| Variables | tp06→mm/hr, t2m→°C, u10m+v10m→speed(kt)+dir(°FROM) |
| Initialization | ERA5 via ARCO; two timesteps: 2020-12-31T18Z + 2021-01-01T00Z |
| Hardware | Apple M4 CPU, JAX XLA ARM64, peak ~6 GB RSS (JIT), ~25 min/step |
| Schema | v4.0 |
| Status | R4 IN PROGRESS — 7-day pipeline running; R1/R2/R3 COMPLETE |

**Model selection: CLOSED.** GraphCastOperational approved (ADR-012, ADR-018).
Hardware validated on M4 24 GB (R3 PASS). See research.md for full ADR record.

---

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

**Legacy validated model stack (current deployment)**:
- Prognostic model: `earth2studio.models.px.GraphCastSmall`
- Initialization sources: ARCO or IFS — both verified compatible
- Precipitation: `tp06` (6h accumulated, metres → mm/6h, no log transform)

**New target model stack (SELECTED AND VALIDATED — R3 PASS)**:
- Prognostic model: `earth2studio.models.px.GraphCastOperational`
- Resolution: 0.25° global (721 × 1440 lat-lon); Myanmar subset 81 × 41 at 0.25°
- Initialization source: ERA5 via ARCO; two timesteps (2020-12-31T18Z + 2021-01-01T00Z)
- Variables confirmed: tp06 (metres/6h, output-only), t2m (Kelvin), u10m (m/s), v10m (m/s)
- Backend: JAX + Haiku (same as GraphCastSmall); XLA ARM64 validated on M4 CPU
- Model selection documented in research.md ADR-012 (CLOSED) and ADR-018 (R3 PASS)
- GFS: NOT a verified initialization source; ARCO ERA5 is the confirmed init path

**ERA5 data role distinction** (NON-NEGOTIABLE):
- ERA5 used to PRE-TRAIN a foundation model ≠ ERA5 used as INITIALIZATION INPUT
- Providing ERA5 analysis as the model's initial state is INITIALIZATION, not training
- The project does NOT fine-tune any model — ERA5 1990–2020 is used as the
  atmospheric state from which the forecast begins, not as training data
- Documentation MUST NOT conflate initialization with training or fine-tuning

**Rationale**: The application's value proposition is AI-based NWP via Earth2Studio.
Substituting fake data, incompatible data sources, or incorrect initialization
violates the project's scientific integrity.

### III. Forecast-Artifact Pipeline

The inference pipeline (Earth2Studio → Python post-processing → frontend-ready
artifacts) MUST be a standalone, independently executable set of scripts.
The frontend MUST only consume the output artifacts; it MUST NOT execute any
inference or call any live weather API at runtime.

**Legacy pipeline (current validated)**:
- Model: GraphCastSmall
- Horizon: 48h, 9 frames (t+0h through t+48h), 6h step
- Variables: tp06 (precipitation, mm/6h) and t2m (temperature, °C)
- Grid: Myanmar 21 × 11 at 1.0°
- Format: float32 binary per variable + forecast.json schema v3.0

**New target pipeline (CONFIRMED — R4 IN PROGRESS)**:
- Model: GraphCastOperational (0.25° global, 721 × 1440)
- Horizon: 168h / 7 days; 28 AR steps; 29 frames (t+0h through t+168h)
- Variables and conversions (all confirmed from R3 smoke test, ADR-018):
  - precipitation: tp06 (metres/6h) → metres × 1000 / 6 → mm/hr, clamp ≥ 0
  - temperature: t2m (Kelvin) → K − 273.15 → °C
  - wind_speed: sqrt(u10m² + v10m²) × 1.94384 → knots
  - wind_direction: (270 − atan2d(v10m, u10m)) mod 360 → °FROM (meteorological convention)
- Grid: Myanmar 81 × 41 at 0.25° (lat 9.0–29.0°N, lon 92.0–102.0°E, ascending)
- Format: float32 binary per variable (4 files) + forecast.json schema v4.0
- Initialization: ERA5 via ARCO; two timesteps (2020-12-31T18Z + 2021-01-01T00Z)
- Output directory: data/forecast_v4/ (does NOT overwrite data/forecast/ schema v3.0 artifacts)
- ARCO compatibility: CONFIRMED — all 82 GCOp input variables available (ADR-017)

**Conversion and semantics (CONFIRMED — ADR-018, ADR-021)**:
- Precipitation: GCOp tp06 is 6-hour accumulated total in physical metres (no log transform).
  Conversion: metres × 1000 / 6 = mm/hr (average rate over 6-hour step). Clamp ≥ 0.
  Display label: "Estimated average rainfall rate (mm/hr)" — NOT an instantaneous rate.
- Wind speed: u10m, v10m in m/s. speed_kt = sqrt(u² + v²) × 1.94384.
- Wind direction: meteorological FROM convention. formula: (270 − atan2d(v, u)) mod 360.
  Wind direction is a CIRCULAR quantity. Bilinear interpolation of degree values is
  mathematically incorrect. Use vector-component interpolation (interpolate sin/cos of
  direction, reconstruct angle) when upsampling to display resolution. See ADR-020.
- Temperature: t2m in Kelvin. K − 273.15 → °C.

**No interpolation between native timesteps is permitted.**
Native model output steps ONLY are displayed.

### IV. Myanmar-Focused

All forecasts, overlays, legends, and metadata MUST target Myanmar.
The default map view MUST be centered on Myanmar. Geographic reference layers
(national boundary) MUST be present and clearly visible.
Global forecast grids MUST be clipped to the Myanmar bbox before delivery
to the frontend.

**Myanmar bounding box**: 92°E–102°E, 9°N–29°N

### V. Map-First UX

The interactive weather map MUST dominate the viewport. Supporting UI elements
(timeline, legend, metadata panel) MUST be visually subordinate to the map.
The map MUST support pan, zoom, weather overlay rendering, and
click-to-inspect interactions.

**New target**: The variable selector MUST support four variables:
Precipitation / Wind Direction / Wind Speed / Temperature.
The current legacy selector (Precip / Temp) is a subset of the new target.

### VI. Native-Step Navigation

Users MUST be able to navigate the forecast step-by-step across the full
forecast horizon. The timeline control MUST provide:
- Slider scrubber across all available forecast steps
- Previous / Next step buttons
- Play / Pause animation
- Displayed forecast date, UTC time, and lead time offset

The map, legend, timestamp, and point values MUST update synchronously
without a full page reload when the selected step changes.

**Temporal semantics MUST be preserved and disclosed for each variable:**

- **Precipitation (legacy, tp06, mm/6h)**: GraphCastSmall's `tp06` is 6-hour
  accumulated total precipitation. Units: mm / 6h. Do NOT divide by 6.
  Disclosure: "Precipitation values represent total rainfall accumulated during
  the 6-hour forecast period ending at the displayed time."

- **Precipitation (new target, display unit mm/hr, CONFIRMED)**: GCOp tp06 is 6-hour
  accumulated total precipitation in physical metres (no log transform). Confirmed from
  ADR-018 R3 smoke test. Conversion: `mm/hr = tp06_metres × 1000 / 6`, clamp ≥ 0.
  Display label: "Estimated average rainfall rate (mm/hr) — 6-hour period average".
  The UI MUST disclose that this is an average rate, not an instantaneous measurement.
  See ADR-021 for full provenance.

- **Wind Direction (new target, degrees, CONFIRMED)**: Meteorological FROM convention
  (direction FROM which wind blows, clockwise from North). Variables: u10m, v10m (m/s).
  Formula: `direction = (270 − atan2d(v10m, u10m)) mod 360`. Range: [0, 360).
  **CRITICAL**: Wind direction is a circular quantity. Direct bilinear interpolation
  of degree values produces nonsensical results near the 0°/360° boundary (e.g.,
  averaging 10° and 350° linearly gives 180° instead of 0°). Rendering MUST use
  vector-component bilinear interpolation: interpolate sin(dir) and cos(dir) across
  the model grid, then reconstruct direction = atan2d(interp_sin, interp_cos). See ADR-020.

- **Wind Speed (new target, knots, CONFIRMED)**: u10m, v10m in m/s (confirmed ADR-018).
  `speed_kt = sqrt(u10m² + v10m²) × 1.94384`. Standard bilinear interpolation applies.

- **Temperature**: K − 273.15 → °C. Native units MUST be confirmed (Kelvin assumed).

**Synthesizing intermediate frames by interpolation is PROHIBITED.**
Only native model output steps are displayed.

**Legacy horizon**: 48h (9 frames, 6h step) — currently deployed.
**New target horizon**: 168h / 7 days at model-native timestep (TBD).

### VII. Model-Agnostic Frontend

The frontend MUST NOT be tightly coupled to any specific Earth2Studio model.
All model-specific details (name, resolution, timestep, variables, units) MUST be
read from `forecast.json` at runtime. Swapping the model MUST require only
re-running the pipeline and updating the metadata file.

No model-specific strings (GraphCast, Aurora, GenCast, FourCastNet, etc.) MUST
appear in TypeScript source code.

**This principle is especially important during the new model selection phase.**
Any frontend code written for the legacy GraphCastSmall pipeline MUST use
`forecast.json` metadata for all model-specific values, so the frontend can
accommodate the new model without source changes.

### VIII. Performance

- Initial application load MUST complete in < 5 seconds on a standard
  broadband connection (≥25 Mbps), measured from cold cache.
- Step-to-step frame transitions MUST feel near-instantaneous (<200 ms)
  after forecast data is loaded.
- **Legacy payload**: ~16.6 KB (2 variables × 9 frames × 21 × 11 × 4 bytes).
- **New target payload**: **1,540,944 bytes ≈ 1.54 MB** (4 variables × 29 frames × 81 × 41 × 4 bytes).
  This exceeds the 1 MB threshold. Loading assessment (ADR-022): at ≥25 Mbps (3.1 MB/s),
  1.54 MB loads in ~0.5 seconds — within the 5-second total budget. Variable-by-variable
  lazy loading would force re-fetch on every variable switch (unacceptable UX latency).
  **Decision (ADR-022): Load all 4 variable binaries in parallel at startup.** All frames
  are kept in memory for instant step-to-step transitions (< 200ms constraint).

### IX. Climate-Honest

Every forecast display MUST clearly show:
- Model name and version
- Spatial resolution (native and display, distinguished)
- Forecast initialization time (UTC)
- Forecast valid time (UTC)
- Lead time offset (e.g., "+12 h")
- Native model timestep
- A disclaimer that forecast skill degrades with lead time

**Per-variable temporal semantics MUST be explicitly disclosed:**

- Precipitation: Whether values are accumulations or rates; the period
  of accumulation; the formula used to convert model output to display unit.
  "Estimated hourly average rainfall rate" is acceptable if derived from
  a model accumulation by dividing by the accumulation period.
- Wind speed: Unit (knots); derivation from u/v components if applicable.
- Wind direction: Convention (meteorological FROM direction); derivation formula.
- Temperature: Unit (°C); conversion from Kelvin if applied.

**Mislabelling forecast fields is a critical defect.**
Displaying accumulation as instantaneous rate, or rate as accumulation,
is a critical scientific error and MUST be caught in validation.

**Verification output requirement (R5 COMPLETE — 2026-08-17)**:
- `data/verification/verification.json` schema v2.0 produced by `verify_forecast.py`
- Reference dataset: ERA5 via ARCO; documented as reanalysis, not observations
- Caveats MUST be displayed wherever verification metrics appear in the UI:
  - "ERA5 is a reanalysis product, not direct observations"
  - "Based on a single 7-day forecast cycle — not statistically robust"
  - "GCOp was trained on ERA5; comparison may be optimistic"
- Verification methodology: ADR-023 in research.md

### X. Minimal Scope

The MVP MUST NOT include: user accounts, authentication, databases,
Redis, Kafka, Kubernetes, Docker (optional for convenience only),
microservices, cloud provider lock-in, or paid API dependencies beyond
the compute needed for forecast generation.
The architecture is: pipeline scripts → static files → GitHub Pages.

### XI. Hardware Transparency (NEW in v2.0.0)

The production pipeline MUST explicitly record the inference hardware used
and the measured peak memory consumption. These MUST appear in `forecast.json`
under `inference_config`.

**Hardware requirements MUST be established experimentally**, not assumed from
documentation badges.

**Legacy validated configuration (2026-08-11)**:
- Hardware: Apple M4 CPU, 24 GB unified memory
- Backend: JAX CPU / XLA ARM64 (`JAX_PLATFORM_NAME=cpu`)
- Peak RSS: ~2.34 GB for the full 48h pipeline
- Runtime: ~78s end-to-end (~54s inference + model load)
- MPS (Metal) is NOT compatible — JAX requires float64 ops not available on MPS

**New target hardware (VALIDATED — ADR-018, R3 PASS)**:
- Hardware: Apple M4 CPU, 24 GB unified memory
- Backend: JAX CPU / XLA ARM64 (`JAX_PLATFORM_NAME=cpu`) — same as legacy
- Peak RSS during XLA compilation: ~5.0–6.0 GB
- Peak RSS after inference (post-JIT): ~1.99–2.18 GB
- JIT cold-start (per Python process): ~27–34 min (XLA kernel compilation)
- Post-JIT per-step forward pass: ~25 min measured on M4 (current R4 run)
- Full 7-day pipeline (28 steps): ~12–20 hours total (research/demo only; not daily production)
- JAX persistent compilation cache: configured; cache directory empty after prior runs
  (cache write behaviour requires investigation in Phase R4b)

**This pipeline is a research and demonstration system.** The ~25 min/step runtime
makes daily automated production use impractical on M4 CPU without hardware acceleration.
This constraint MUST be disclosed in the frontend.

If any hardware test fails (OOM, crash, assertion error), this MUST be reported
immediately. No workarounds may be attempted without explicit user approval.

### XII. Resolution Honesty (NEW in v2.0.0)

The application MUST clearly communicate the distinction between native
model resolution and display resolution.

If the display uses bilinear interpolation to a finer grid for visual
clarity, the UI MUST state that interpolation does not add meteorological
information beyond the native model grid.

**Legacy**: GraphCastSmall operates at 1.0° resolution (~111 km grid spacing).

**New target (CONFIRMED)**: GraphCastOperational operates at 0.25° resolution (~27.8 km
grid spacing). The UI MUST display "0.25° (~28 km)" and disclose that bilinear interpolation
to 0.05° display resolution adds visual smoothness only, not additional forecast information.

**Wind direction exception**: Wind direction values MUST NOT be bilinearly interpolated
as scalar degree values. The correct method is vector-component interpolation: interpolate
sin(direction) and cos(direction) over the 0.25° model grid, then reconstruct direction via
atan2d. This preserves circular continuity across the 0°/360° boundary. See ADR-020.

---

## Architecture Constraints

- **Frontend stack**: React, TypeScript, Vite, MapLibre GL JS, Tailwind CSS
- **Map library**: MapLibre GL JS (WebGL raster performance)
- **Basemap**: Open-source tile provider (no proprietary API keys required)
- **Forecast format**: Float32 binary arrays per variable; metadata in JSON
- **Python pipeline**: Earth2Studio ≥ 0.17.0, uv-managed virtual environment
- **Legacy backend**: GraphCastSmall — JAX + Haiku (DeepMind), Apple M4 CPU validated
- **New target backend**: JAX + Haiku (same as GraphCastSmall — confirmed compatible, ADR-018)
- **GitHub Actions**: Used for frontend deployment ONLY; NOT for inference
- **Model interface**: No model-specific strings in frontend TypeScript

---

## Data & Model Integrity

- The pipeline MUST validate output before generating frontend artifacts:
  - No NaN values in expected forecast fields over the Myanmar bbox
  - Timestamps must be monotonically increasing
  - Precipitation MUST be non-negative (physical constraint)
  - Precipitation MUST be within a physically plausible range
  - Temperature MUST be in a plausible range (−90°C to +70°C)
  - Wind components: physically plausible range to be determined per model
- Demo data (for development/CI) MUST be clearly separated from production
  forecast data. The frontend MUST display a visible "DEMO DATA" banner
  when consuming demo artifacts.
- No secrets, API keys, or model credentials MUST be committed to the repository.
- `forecast.json` MUST record the transformation provenance for all variables,
  including source variable names, native units, and conversion formulas.

---

## Architecture Decision Log

### ADR-010: Aurora1p5 → GraphCastSmall (2026-08-11)

**Decision**: Replace Aurora1p5 with GraphCastSmall.

**Reason**: Aurora1p5 attempted inference on a free Colab NVIDIA T4 (16 GB VRAM)
and failed with `OutOfMemoryError` at the first inference step.

**Constraints accepted**: 1.0° resolution (coarser than Aurora1p5's 0.25°); 6h native
timestep; tp06 (6h accumulated) rather than tp1h (1h accumulated).

### ADR-011: 24h → 48h Horizon + Temperature (2026-08-11)

**Decision**: Extend forecast horizon from 24h to 48h and add temperature display.

**Reason**: GraphCastSmall on M4 CPU runs 8 AR steps in ~54s. t2m is a native output
with zero additional inference cost.

**Changes**: GC_N_STEPS 4→8, GC_N_FRAMES 5→9, GC_HORIZON_HOURS 24→48, schema v2→v3.

### ADR-012: GraphCastOperational Selected for New Target Architecture (2026-08-12, CLOSED)

**Decision**: GraphCastOperational is the selected model for the 7-day / 4-variable
new target architecture. Selection gate R010 CLOSED. Hardware validation R3 PASS.

**Selected architecture**:
- Model: `earth2studio.models.px.GraphCastOperational` (Earth2Studio 0.17.0)
- Resolution: 0.25° global (721 × 1440); Myanmar 81 × 41
- Timestep: 6h; Horizon: 168h / 7 days; Frames: 29
- Initialization: ERA5 via ARCO, two timesteps: 2020-12-31T18Z + 2021-01-01T00Z
- Variables: tp06 (metres/6h) → mm/hr; t2m (K) → °C; u10m+v10m → speed(kt)+dir(°FROM)
- Schema: v4.0
- Hardware: M4 CPU, JAX XLA ARM64, peak ~6 GB RSS (JIT), ~25 min/step

**Rejected candidates**:
- FuXi: R2 FAIL — onnxruntime-gpu has no macOS ARM64 wheel; ARCO lacks r* variables
- AIFS/AIFS2: R2 FAIL — flash-attn dependency is CUDA-only, no macOS ARM64 support
- Aurora1p5: OOM on T4 GPU; log-space tp1h adds pipeline complexity
- Pangu3/6/24, FCN, FengWu, GenCastMini: No precipitation output variable

**Legacy preservation**: The GraphCastSmall/48h/2-variable/schema-v3.0 deployment MUST NOT
be broken. data/forecast/ (schema v3.0) remains untouched. New artifacts go to data/forecast_v4/.

**Status**: CLOSED — GraphCastOperational selected, R1/R2/R3 COMPLETE, R4 IN PROGRESS.

### ADR-013: Target Forecast Horizon — 168h / 7 Days (2026-08-12, CONFIRMED)

**Decision**: The new target is a 168-hour (7-day) forecast.

**Rationale**: Provides meteorologically useful medium-range guidance for Myanmar
monsoon season planning. GraphCastOperational produces meaningful skill through 168h.

**Frame count**: 29 frames — `(168h / 6h) + 1 = 29`. Confirmed from GCOp output coords.

### ADR-014: Four-Variable Meteorological Product (2026-08-12, CONFIRMED)

**Decision**: The new target exposes four display variables. All confirmed from GCOp output.

| Variable | Display unit | Source variable | Native unit | Conversion |
|----------|-------------|----------------|------------|-----------|
| Precipitation | mm/hr | tp06 | metres/6h | ×1000/6, clamp ≥ 0 |
| Wind Direction | ° (FROM) | u10m, v10m | m/s | (270−atan2d(v,u)) mod 360 |
| Wind Speed | knots | u10m, v10m | m/s | √(u²+v²)×1.94384 |
| Temperature | °C | t2m | Kelvin | K−273.15 |

**Precipitation (CONFIRMED — ADR-021)**: GCOp tp06 is 6-hour accumulated total in
physical metres. No log transform. Conversion: metres×1000/6=mm/hr. Clamp ≥ 0.

**Wind (CONFIRMED — ADR-018)**: u10m and v10m at 10m level, in m/s. Confirmed present
in GCOp output_coords. Near-surface wind used as required.

### ADR-015: ERA5 1990–2020 as Initialization Source (2026-08-12)

**Decision**: Use ERA5 historical data (via ARCO) to provide the initialization
state for the 2021-01-01 forecast.

**Scientific role clarification**:
- ERA5 (1990–2020) is used to initialize the forecast, not to train or fine-tune the model
- Providing ERA5 analysis as the model's initial atmospheric state is initialization
- The foundation model was PRE-TRAINED on ERA5 by the model's original authors
  (e.g., GraphCastSmall trained on ERA5 1979–2015)
- The project does NOT perform any additional training or fine-tuning
- The "1990–2020 ERA5 data" in this project refers to the available historical reanalysis
  that covers dates prior to 2021-01-01, from which the initialization state is drawn

**ARCO coverage**: ARCO ERA5 covers approximately 1959–2023.
The 2021-01-01 date is within ARCO coverage. IFS is NOT needed for this date.

**Required**: Confirm that ARCO provides all input variables required by the selected
model for 2021-01-01T00:00:00Z (and T−6h, T−12h if the model requires multi-step init).

---

## Governance

Constitution supersedes all other specifications and implementation decisions.
Any amendment requires:
1. Explicit user approval
2. A version bump per semantic versioning (MAJOR for removals/redefinitions,
   MINOR for additions, PATCH for clarifications)
3. Update to this file before implementation proceeds

All feature specifications (spec.md) and implementation plans (plan.md) MUST
include a Constitution Check section verifying compliance with each principle.

**Version**: 3.2.0 | **Ratified**: 2026-08-09 | **Last Amended**: 2026-08-17
