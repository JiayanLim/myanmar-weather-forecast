# Feature Specification: Myanmar Weather Forecast Web Application

**Feature Branch**: `001-myanmar-weather-app`
**Created**: 2026-08-09
**Revised**: 2026-08-11 v2 — Aurora1p5 → GraphCastSmall; 168h/0.25° → 24h/1.0°; temp removed
**Revised**: 2026-08-11 v3 — MAJOR: 24h → 48h; temperature + precipitation; schema v3.0; M4 CPU validated
**Revised**: 2026-08-12 v4 (draft) — MAJOR: new target 7-day / 4-variable / ERA5 init / model TBD
**Revised**: 2026-08-16 v5 — Section B updated: GCOp confirmed, all TBDs resolved, FRs updated
**Revised**: 2026-08-17 v6 — FR-N40–N45 COMPLETE; ADR-023 closed; R4/R5 COMPLETE
**Status**: LEGACY (v3) deployed. NEW TARGET (v4) R4/R5 COMPLETE — R6 NOT STARTED.
**Constitution**: `.specify/memory/constitution.md` v3.2.0

---

## DOCUMENT SCOPE

This spec contains two sections:

**Section A — Legacy Baseline (v3.0, DEPLOYED)**
The current production architecture: GraphCastSmall, 48h, 2 variables, M4 CPU.
These requirements are implemented and live. Do NOT modify this section to reflect
the new target unless implementing the new target.

**Section B — New Target (v4.0, RESEARCH PHASE)**
The new project direction: 7-day forecast, 4 variables, ERA5 2021-01-01 initialization,
model TBD. These requirements are DRAFTS — they will be revised once the model is selected.

---

## SECTION A — LEGACY BASELINE (v3.0, DEPLOYED)

### Constitution Check (Legacy)

| Principle | Requirement | Design Decision | Status |
|-----------|-------------|-----------------|--------|
| I. Static-First | No runtime server | All data pre-generated; GitHub Pages CDN only | ✓ DONE |
| II. Earth2Studio-Mandatory | GraphCastSmall + ARCO/IFS init | `GraphCastSmall` + `earth2studio.data.ARCO` or `IFS` | ✓ DONE |
| III. Forecast-Artifact Pipeline | Standalone scripts; 6h steps; 48h horizon | `scripts/generate_forecast.py`; 9 frames | ✓ DONE |
| IV. Myanmar-Focused | bbox 92°E–102°E, 9°N–29°N | xarray/numpy spatial subset in pipeline | ✓ DONE |
| V. Map-First UX | Map dominates viewport | MapLibre GL JS full-viewport | ✓ DONE |
| VI. Native-Step Navigation | 6h steps, no interpolation | 9 native frames; 6h slider steps | ✓ DONE |
| VI. tp06 semantics | 6h accumulation, not instantaneous | UI label: mm/6h with mandatory disclosure | ✓ DONE |
| VII. Model-Agnostic Frontend | metadata.json drives all model display | No GraphCast strings in TypeScript | ✓ DONE |
| VIII. Performance | Load < 5s; transition < 200ms | Float32 binary ≈ 16.6 KB total | ✓ DONE |
| IX. Climate-Honest | All metadata shown; tp06 semantics disclosed | Header + InfoPanel + tp06 tooltip | ✓ DONE |
| X. Minimal Scope | No databases, accounts, paid APIs | pipeline → static files → GitHub Pages | ✓ DONE |
| XI. Hardware Transparency | Hardware validated; recorded in forecast.json | M4 CPU: ~78s, 2.34 GB RSS | ✓ DONE |
| XII. Resolution Honesty | Disclose native 1.0° vs. display resolution | InfoPanel states interpolation policy | ✓ DONE |

### Legacy Validated Architecture Summary

- **Model**: GraphCastSmall (Earth2Studio 0.17.0)
- **Initialization**: ARCO (historical ERA5, 1959–2023) or IFS (operational)
- **Horizon**: 48h (8 AR steps + t+0h init = 9 frames)
- **Variables**: tp06 → mm/6h (precipitation); t2m → °C (temperature)
- **Grid**: Myanmar 21 × 11 at 1.0° (lat 9–29°N, lon 92–102°E)
- **Hardware**: Apple M4 CPU, JAX XLA ARM64, ~78s, 2.34 GB RSS
- **Schema**: v3.0
- **Deployment**: GitHub Pages (live)

### Legacy Functional Requirements (Python Pipeline) — IMPLEMENTED

- **FR-001**: Uses `earth2studio.models.px.GraphCastSmall`
- **FR-002**: ARCO or IFS only; NCAR_ERA5 and GFS prohibited
- **FR-003**: Fetches two consecutive timesteps (t−6h, t+0h) before inference
- **FR-004**: No log or exponential transform on tp06
- **FR-005**: tp06 extracted, metres × 1000 → mm/6h, clamped ≥ 0; t+0h = 0.0
- **FR-006**: 9 frames (t+0h through t+48h, 6h step)
- **FR-007**: No interpolation between frames
- **FR-008**: Myanmar bbox subset: lat 9–29°N, lon 92–102°E, 21 × 11 at 1.0°
- **FR-009**: forecast.json schema v3.0 with full transformation provenance
- **FR-010**: Validation: no NaN, monotonic timestamps, tp06 ≥ 0, tp06 < threshold, t2m in range
- **FR-011**: Inference hardware and peak RSS recorded in forecast.json
- **FR-012**: Demo data: no GPU; is_demo=true; 9 frames synthetic; 21 × 11 grid
- **FR-013**: t2m extracted, K − 273.15 → °C

### Legacy Functional Requirements (Frontend) — IMPLEMENTED

- **FR-020**: Interactive MapLibre GL JS map centered on Myanmar (~96°E, 19°N)
- **FR-021**: Precipitation OR temperature colored raster overlay (activeVariable)
- **FR-022**: Myanmar national boundary GeoJSON layer
- **FR-023**: Timeline slider across 9 steps (t+0h to t+48h)
- **FR-024**: Previous / Next step buttons (6h increment, boundary clamping)
- **FR-025**: Play/Pause animation with speed selection (0.5×, 1×, 2×, 4×)
- **FR-026**: Forecast valid date, UTC time, lead time offset displayed
- **FR-027**: Variable-aware legend (mm/6h for precipitation, °C for temperature)
- **FR-028**: Point-inspect popup on map click with both variable values
- **FR-029**: "DEMO DATA" banner when forecast.json has is_demo=true
- **FR-030**: Model name, resolution, init time in header (from forecast.json)
- **FR-031**: Info/About panel with full metadata and attribution
- **FR-032**: Precipitation disclosure: 6-hour accumulation period
- **FR-033**: All model details read from forecast.json; no hardcoded strings in TypeScript
- **FR-034**: Interpolation disclosure if display resolution differs from native 1.0°
- **FR-035**: Variable switcher (Precip / Temp) updating map, legend, popup simultaneously

### Legacy Non-Functional Requirements — VALIDATED

- **NFR-001**: Initial load < 5s on ≥25 Mbps, cold cache
- **NFR-002**: Step-to-step transition < 200ms after data loaded
- **NFR-003**: Total forecast data payload < 1 MB (validated: ~16.6 KB)
- **NFR-004**: No secrets/credentials committed to repository
- **NFR-005**: No proprietary API keys in deployed static frontend
- **NFR-006**: TypeScript: no errors
- **NFR-007**: Python pipeline: ruff linting
- **NFR-008**: Precipitation correctly labeled as 6-hour accumulation
- **NFR-009**: Native 1.0° resolution disclosed; interpolation policy stated

---

## SECTION B — NEW TARGET (v4.0, RESEARCH PHASE)

**Status**: Architecture confirmed. R1/R2/R3 COMPLETE. R4 IN PROGRESS.
Requirements marked [CONFIRMED] are locked. Phase-gated implementation per tasks.md.

### Constitution Check (New Target — UPDATED 2026-08-16)

| Principle | New Target Design | Status |
|-----------|------------------|--------|
| I. Static-First | All data pre-generated; static GitHub Pages | ✓ PRESERVED |
| II. Earth2Studio-Mandatory | GraphCastOperational; ERA5/ARCO init | ✓ CONFIRMED (ADR-012, R3 PASS) |
| III. Pipeline | GCOp 0.25°/168h/29 frames/4 vars/schema v4.0 | ✓ CONFIRMED (R4 in progress) |
| IV. Myanmar-Focused | Myanmar 81×41 at 0.25°; lat 9–29°N, lon 92–102°E | ✓ CONFIRMED (ADR-017, R3) |
| V. Map-First UX | 4-variable selector; map dominates | Pending Phase R7 |
| VI. Native-Step Navigation | 6h native timestep; 29 frames; no interpolation | ✓ CONFIRMED (GCOp 6h) |
| VI. Precipitation semantics | tp06 metres×1000/6=mm/hr; 6h period average | ✓ CONFIRMED (ADR-021) |
| VI. Wind direction semantics | Meteorological FROM; vector-component interpolation | ✓ CONFIRMED (ADR-020) |
| VII. Model-Agnostic | All model details from forecast.json; no model strings in TS | Pending Phase R6/R7 |
| VIII. Performance | 1.54 MB payload; load-all-at-startup; < 5s | ✓ CONFIRMED (ADR-022) |
| IX. Climate-Honest | 4-variable disclosures; ERA5 init role; limitation text | Pending Phase R7 |
| XI. Hardware Transparency | M4 24 GB, peak 6 GB, ~25 min/step recorded | ✓ CONFIRMED (ADR-018) |
| XII. Resolution Honesty | 0.25° (~28 km); display 0.05° interpolated; wind dir special | ✓ CONFIRMED (ADR-020) |

### New Target Overview

**Objective**: Produce a scientifically meaningful 7-day forecast over Myanmar using
GraphCastOperational with ERA5 historical initialization (2021-01-01), and expose four
meteorological variables (precipitation, wind direction, wind speed, temperature) in
an interactive map UI.

**Confirmed architecture** (R1/R2/R3 COMPLETE):
- Model: `earth2studio.models.px.GraphCastOperational` (0.25°, 6h, JAX/Haiku)
- Initialization: ERA5 via ARCO; two timesteps: 2020-12-31T18Z + 2021-01-01T00Z
- Init role: atmospheric state for forecast start date (NOT training or fine-tuning)
- Forecast period: 2021-01-01T00:00:00Z → 2021-01-08T00:00:00Z (168h / 7 days)
- Output: 29 frames at 6h steps, 4 variables, Myanmar 81×41 at 0.25°, schema v4.0

### New Target User Stories (DRAFT)

#### User Story 1 — View Myanmar 7-Day Weather Map [P1, DRAFT]

A user opens the app and immediately sees a weather map centered on Myanmar, showing
one of four meteorological variables at the initialization hour.

**Acceptance scenarios**:
1. Map renders Myanmar with colored overlay for the selected variable; legend shows correct units
2. Header shows model name, resolution, init time (2021-01-01T00:00:00Z), data source (ERA5)
3. Variable selector shows four options: Precipitation / Wind Direction / Wind Speed / Temperature

#### User Story 2 — Navigate 7-Day Forecast [P1, CONFIRMED]

A user steps through all forecast frames from t+0h to t+168h at 6h native steps.

**Acceptance scenarios**:
1. Timeline shows 29 frames from t+0h to t+168h at 6h intervals
2. Lead time markers at 0·24·48·72·96·120·144·168h (every 4th frame)
3. All four variables update synchronously when step changes

#### User Story 3 — Switch Between Four Variables [P1, DRAFT]

A user switches between precipitation, wind direction, wind speed, and temperature.
Each switch updates the map overlay, legend, and popup.

**Acceptance scenarios**:
1. Precipitation: map shows mm/hr raster; legend shows mm/hr scale
2. Wind Direction: map shows direction field (color wheel or compass); legend shows degrees
3. Wind Speed: map shows speed raster; legend shows knot scale
4. Temperature: map shows °C raster; legend shows °C scale

#### User Story 4 — View Point Values [P2, DRAFT]

A user clicks Myanmar map; popup shows all four variable values at that grid point.

#### User Story 5 — View Model Evaluation Popup [P3, FUTURE]

A user opens a "Historical Model Evaluation" popup showing skill metrics vs ERA5
for the 2021-01-01 to 2021-01-08 period. This is a FUTURE requirement.
Do NOT implement until the forecast pipeline is working and verified.

**Evaluation popup content (FUTURE)**:
- Section header: "Historical Model Evaluation" (NOT "accuracy of current forecast")
- Reference: ERA5 analysis (not station observations) — clearly disclosed
- Temperature: MAE, RMSE, Bias by lead time
- Precipitation: MAE, RMSE, Bias; POD/FAR/CSI at key thresholds
- Wind speed: MAE, RMSE, Bias
- Wind direction: circular MAE
- Caveats: reanalysis reference; single forecast cycle; scale mismatch

### New Target Functional Requirements (CONFIRMED — locked 2026-08-16)

#### Pipeline Requirements [CONFIRMED]

- **FR-N01**: Pipeline uses `earth2studio.models.px.GraphCastOperational`
- **FR-N02**: Initialization source: ARCO ERA5; two timesteps: 2020-12-31T18Z + 2021-01-01T00Z
- **FR-N03**: Pipeline fetches two consecutive timesteps (t−6h, t+0h) via Earth2Studio ARCO source
- **FR-N04**: Pipeline produces 168h / 7-day forecast; 28 AR steps; 29 frames (t+0h through t+168h)
- **FR-N05**: Pipeline extracts and converts all four variables with documented provenance [CONFIRMED]:
  - Precipitation: tp06 (metres/6h) → metres×1000/6 → mm/hr, clamp ≥ 0 (ADR-021)
  - Wind direction: u10m, v10m (m/s) → (270−atan2d(v,u)) mod 360 → °FROM
  - Wind speed: u10m, v10m (m/s) → sqrt(u²+v²)×1.94384 → knots
  - Temperature: t2m (Kelvin) → K−273.15 → °C
- **FR-N06**: No interpolation between native timesteps (6h steps only)
- **FR-N07**: Myanmar bbox subset after global inference: lat 9.0–29.0°N, lon 92.0–102.0°E,
  81×41 at 0.25° (lat ascending, verified from actual model output_coords)
- **FR-N08**: forecast.json schema v4.0 with full transformation provenance for all 4 variables (ADR-019)
- **FR-N09**: Validation: all four variables checked for NaN, Inf, physical plausibility, no negatives
  for precipitation after clamping. Pre-clamp negative count recorded for provenance.
- **FR-N10**: forecast.json records: init_time=2021-01-01T00:00:00Z, init_source=ARCO/ERA5,
  model=GraphCastOperational, model_version, native_timestep_hours=6, forecast_horizon_hours=168,
  n_times=29, spatial_resolution_deg=0.25, hardware config, per-variable provenance,
  per-step timing, peak RSS, JAX cache hit/miss status

#### Display Unit Requirements [CONFIRMED — LOCKED]

| Variable | Display unit | Native variable | Conversion | ADR |
|----------|-------------|----------------|-----------|-----|
| Precipitation | mm/hr | tp06 (metres/6h) | ×1000/6, clamp ≥ 0 | ADR-021 |
| Wind Direction | °FROM [0, 360) | u10m, v10m (m/s) | (270−atan2d(v,u)) mod 360 | ADR-020 |
| Wind Speed | knots | u10m, v10m (m/s) | √(u²+v²)×1.94384 | ADR-018 |
| Temperature | °C | t2m (Kelvin) | K−273.15 | ADR-018 |

**Units are LOCKED. Conversions are verified from R3 smoke test output (ADR-018).**

#### Frontend Requirements [CONFIRMED — implementation pending Phase R6/R7]

- **FR-N20**: Four-variable selector (Precipitation / Wind Direction / Wind Speed / Temperature)
- **FR-N21**: Timeline derived dynamically from forecast.json (n_times=29, native_timestep_hours=6)
- **FR-N22**: Lead time markers at 24h intervals — every 4th frame at 6h step (frames 0,4,8,12,16,20,24,28)
- **FR-N23**: Variable-aware legend for all four variables with correct units and color scales.
  Scales calibrated to R4 January 2021 validation dataset (see FR-N23a):
  Precipitation 0–2 mm/hr; Wind Speed 0–30 kt; Wind Direction HSL hue wheel with compass labels;
  Temperature 15–40 °C.

- **FR-N23a**: Precipitation color scale calibration (R10, 2026-08-17).
  PRECIP_MAX = 2 mm/hr; fully-transparent cutoff at 0.02 mm/hr (norm < 0.002).
  Rationale: R4 January 2021 data maximum = 1.62 mm/hr (no saturation at PRECIP_MAX=2);
  P95 = 0.084 mm/hr (visible above cutoff); median ≈ 0.0003 mm/hr (correctly suppressed as noise).
  Ticks: [0, 0.1, 0.25, 0.5, 1.0, 2.0] mm/hr.
  This is a display calibration, not a physical upper bound on Myanmar precipitation.
  Monsoon-season data routinely exceeds 2 mm/hr; recalibration will be required before
  serving wet-season forecasts. The color scale MUST be documented as dry-season calibrated
  in the legend or info panel.
- **FR-N24**: Popup displays all four variable values at clicked grid point for current step
- **FR-N25**: Header: "Myanmar 7-Day AI Weather Forecast" (derived from forecast_horizon_hours=168)
- **FR-N26**: Info panel: model, 0.25° resolution (~28 km), timestep, init time (2021-01-01T00Z),
  source (ERA5), per-variable unit disclosures, initialization vs training distinction
- **FR-N27**: Precipitation disclosure: "Estimated average rainfall rate (mm/hr) during the
  6-hour period ending at the displayed time. Derived from 6-hour accumulated total (tp06)."
- **FR-N28**: Wind direction disclosure: "Meteorological convention — direction FROM which wind
  blows, measured clockwise from North."
- **FR-N29**: All forecast metadata consumed from forecast.json; no model-specific constants in TypeScript
- **FR-N11**: `colorscales.ts` MODEL_STEP MUST be derived from `metadata.spatial_resolution_deg`
  at runtime. It MUST NOT be hardcoded. (Critical bug: current code has MODEL_STEP=1.0 which
  produces incorrect bilinear interpolation at 0.25° grid spacing.)
- **FR-N12**: Wind direction rendering MUST use vector-component bilinear interpolation (ADR-020).
  The generic `renderWithInterpolation` MUST NOT be called with wind direction degree values.
  A dedicated `renderWindDirection` function (or equivalent) MUST implement sin/cos interpolation.
- **FR-N13**: `ForecastLoader.ts` MUST fetch all 4 binary files in parallel using `Promise.all`.
  `ForecastStore.ts` MUST hold all 4 Float32Arrays from startup (ADR-022).
  The loading spinner MUST remain until all 4 arrays are available.

#### Wind Vector Overlay Requirements [Phase R11 — 2026-08-17]

- **FR-W01**: When wind_speed or wind_direction is the active variable, render a sparse
  vector field of directional arrows overlaid on the existing color raster using the same
  overlay canvas. No separate canvas or MapLibre source is required.

- **FR-W02**: Sampling constant: `WIND_ARROW_GRID_STEP = 3` (initial value).
  Sample every 3rd model grid point in both latitude and longitude (~27×14 = ~378 candidate points).
  Confirmed by visual testing at default zoom (5.2) and maxZoom (10).
  If visually overcrowded at default zoom, increase to 4; if too sparse, decrease to 2.
  Do not implement adaptive density — use a fixed constant.

- **FR-W03**: Calm threshold: grid points where `wind_speed < 2.0 kt` must not render an arrow.
  This matches `wind_direction_calm_threshold_kt = 2.0 kt` from verification.json.
  No marker or dot is rendered at calm points.

- **FR-W04**: Arrow direction is the TO direction (where wind blows toward), opposite of the stored
  meteorological FROM convention. Arrow head rotated by `(stored_direction + 180) mod 360°`.
  Mandatory direction sanity: 90°FROM east → arrow points west; 180°FROM south → arrow points north.
  The stored binary values and popup display remain FROM convention (unchanged).

- **FR-W05**: Arrow length proportional to wind speed. Arrow style: solid filled arrowhead with
  a short tail. Maximum arrow length: ≤ half the spacing between adjacent arrows to prevent
  overlap at max speed. Arrow color: white or near-white at ≥80% opacity for contrast against
  both the color raster and the basemap. Performance: arrow rendering must not cause a noticeable
  regression against the existing step-transition requirement (< 200ms total). No premature
  optimization — measure first if needed.

#### Evaluation Requirements [FR-N40–FR-N45 COMPLETE; FR-N46 FUTURE]

- **FR-N40**: verify_forecast.py produces verification.json for the 2021-01-01 forecast
  — COMPLETE: exit 0; data/verification/verification.json schema v2.0 written (2026-08-17)

- **FR-N41**: Temperature: MAE, RMSE, Bias at all native lead times (29 frames)
  — COMPLETE: MAE=1.3137°C, RMSE=1.6975°C, bias=−0.7595°C (see ADR-023)

- **FR-N42**: Precipitation: MAE, RMSE, Bias; POD/FAR/CSI at 0.1 mm/hr threshold
  — COMPLETE: MAE=0.0172 mm/hr; POD=0.6343, FAR=0.7279, CSI=0.2352 (28 frames; ADR-023)
  — ERA5 tp: 1-hour accumulation per timestamp (empirically confirmed); no seam handling
  — 6h aggregation: sum of 6 consecutive hourly values, clamped ≥0, ×1000/6 → mm/hr
  — Summary POD/FAR/CSI from total contingency counts (not averaged per-frame ratios)
  — t+0h excluded from precipitation metrics (GCOp convention: tp06=0 at init)

- **FR-N43**: Wind speed: MAE, RMSE, Bias
  — COMPLETE: MAE=1.0548 kt, RMSE=1.4027 kt, bias=−0.3911 kt (see ADR-023)

- **FR-N44**: Wind direction: circular MAE; calm wind exclusion < 2 kt
  — COMPLETE: circular MAE=16.9113°; n_points_active and n_points_calm_excluded recorded
  — Formula: diff = ((fcst_dir − era5_dir + 180) % 360) − 180; MAE = mean(|diff|)

- **FR-N45**: ERA5 reference: ARCO `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
  — COMPLETE: 0.25° exact grid match; lat_err=0°, lon_err=0° (no interpolation)
  — variable `total_precipitation`: 1-hour accumulation ending at each timestamp (confirmed)

- **FR-N46**: ModelEvaluation popup in frontend (FUTURE; do NOT implement yet)

### New Target Non-Functional Requirements [CONFIRMED]

- **NFR-N01**: Initial load < 5s on ≥25 Mbps, cold cache. Payload 1.54 MB loads in ~0.5s. ✓ WITHIN BUDGET
- **NFR-N02**: Step-to-step transition < 200ms after data loaded (all 4 arrays in memory). ✓ ACHIEVABLE
- **NFR-N03**: Forecast payload: **1,540,944 bytes ≈ 1.54 MB** (4×29×81×41×4 bytes). CONFIRMED (ADR-022)
- **NFR-N04**: No secrets committed to repository. GCOp checkpoint is public GCS; ARCO needs no auth.
- **NFR-N05**: Hardware validated: M4 24 GB, peak ~6 GB RSS during JIT, 1.99 GB post (ADR-018, R3 PASS)

### Key Entities (New Target — CONFIRMED)

- **ForecastRun**: init_time=2021-01-01T00:00:00Z, model=GraphCastOperational,
  init_source=ARCO/ERA5, horizon=168h, n_times=29, native_timestep_h=6
- **ForecastFrame**: one model step of one variable at one lead time
- **ForecastArtifact**: forecast.json (schema v4.0) + 4 float32 binary files
- **Variable**: precipitation (mm/hr), wind_direction (°FROM), wind_speed (kt), temperature (°C)
- **ActiveVariable**: currently displayed variable (one of four)
- **GridPoint**: one lat/lon point in Myanmar 81×41 subset at 0.25°

### Resolved Architecture Questions (all confirmed R1–R3, 2026-08-12–16)

1. **Model**: `earth2studio.models.px.GraphCastOperational` (ADR-012, R3 PASS)
2. **Precipitation semantics**: tp06, metres/6h accumulation, no log transform, →mm/hr (ADR-021)
3. **Wind variables**: u10m, v10m at 10m level in m/s (ADR-018)
4. **Native timestep**: 6h (confirmed from GCOp output_coords)
5. **Native resolution**: 0.25° (confirmed 721×1440 global grid)
6. **Hardware**: M4 24 GB; peak ~6 GB RSS during JIT; 1.99 GB post; ~25 min/step (ADR-018)
7. **Init variables**: 82 variables; q* (specific humidity, not r*); all in ARCO (ADR-017)
8. **ARCO availability**: CONFIRMED — all 82 vars at 2020-12-31T18Z + 2021-01-01T00Z (ADR-017)

### Out of Scope (New Target MVP)

- Ensemble / probabilistic forecasting
- Forecast horizons beyond 7 days
- Wind at pressure levels above the surface (display only near-surface)
- Real-time automated data ingestion
- User accounts, authentication, or personalization
- Mobile-native applications
- Paid API integrations
- Fine-tuning or retraining of any model
