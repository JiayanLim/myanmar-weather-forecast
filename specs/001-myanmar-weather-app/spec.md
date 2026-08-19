# Feature Specification: Myanmar Weather Forecast Web Application

**Feature Branch**: `001-myanmar-weather-app`
**Created**: 2026-08-09
**Revised**: 2026-08-11 v2 — Aurora1p5 → GraphCastSmall; 168h/0.25° → 24h/1.0°; temp removed
**Revised**: 2026-08-11 v3 — MAJOR: 24h → 48h; temperature + precipitation; schema v3.0; M4 CPU validated
**Revised**: 2026-08-12 v4 (draft) — MAJOR: new target 7-day / 4-variable / ERA5 init / model TBD
**Revised**: 2026-08-16 v5 — Section B updated: GCOp confirmed, all TBDs resolved, FRs updated
**Revised**: 2026-08-17 v6 — FR-N40–N45 COMPLETE; ADR-023 closed; R4/R5 COMPLETE
**Revised**: 2026-08-17 v7 — FR-W01–FR-W07 rewritten (SVG overlay, R12); ADR-025; FR-N23 updated
**Revised**: 2026-08-17 v8 — FR-N23b (precip sqrt scale), FR-N30 (MMT local time); R13/R14 diagnostic findings
**Revised**: 2026-08-17 v9 — FR-N20 updated (Wind tab consolidation, R16); FR-N46 expanded (Model Eval context, R15); FR-W01c added
**Revised**: 2026-08-19 v10 — FR-W01c revised (compass widget removed, R17); FR-N25 refined (Init/Source removed from header, R17)
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
one of three meteorological views (Precipitation / Wind / Temperature) at the initialization hour.

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

- **FR-N20**: Three-tab variable selector (Precipitation / Wind / Temperature). (R16, 2026-08-17)
  The separate "Wind Speed" and "Wind Direction" tabs are consolidated into a single "Wind" tab.
  When "Wind" is active:
  - The wind-speed raster is rendered (speed colorscale, 0–30 kt).
  - SVG wind-direction arrows are overlaid (same as current wind_speed + wind_direction views).
  - The legend panel shows both the wind-speed gradient bar and the compass-arrow direction widget.
  - Popup continues to display wind speed (kt) and wind direction (°FROM / compass label) separately.
  The `wind_speed` and `wind_direction` binary data variables are NOT merged; they remain separate
  in ForecastStore, ForecastLoader, and verification.json. `ActiveVariable` type is updated:
  `wind_direction` is retired as a selectable value; `wind_speed` becomes the canonical internal
  state for the Wind tab. `WindArrowOverlay` triggers on `activeVariable === 'wind_speed'` only.
  No change to WeatherMap canvas logic for wind_speed rendering.
- **FR-N21**: Timeline derived dynamically from forecast.json (n_times=29, native_timestep_hours=6)
- **FR-N22**: Lead time markers at 24h intervals — every 4th frame at 6h step (frames 0,4,8,12,16,20,24,28)
- **FR-N23**: Variable-aware legend for all four variables with correct units and color scales.
  Scales calibrated to R4 January 2021 validation dataset (see FR-N23a):
  Precipitation 0–2 mm/hr; Wind Speed 0–30 kt; Wind Direction: compass description (no hue raster;
  arrows are the sole direction encoding); Temperature 15–40 °C.

- **FR-N23a**: Precipitation color scale calibration (R10, 2026-08-17) — superseded by FR-N23b
  for the sqrt-scale redesign. PRECIP_MAX = 2 mm/hr remains locked.

- **FR-N23b**: Precipitation sqrt color scale (R13, 2026-08-17).
  Supersedes the linear alpha ramp from FR-N23a. Rationale from diagnostic investigation:
  R4 January 2021 distribution is heavily right-skewed (P50=0.000288, P75=0.005, P90=0.028,
  P95=0.084, P99=0.406, max=1.620 mm/hr). The linear scale compressed 99% of non-zero data
  into 22% of the color ramp; the current 0.020 mm/hr cutoff suppresses 87.8% of all values.
  Fix:
  - Color norm: `norm_display = sqrt(v / PRECIP_MAX)` — a sqrt transform so that P90 (0.028
    mm/hr) maps to norm=0.12, P95 (0.084) to 0.20, P99 (0.406) to 0.45, max (1.62) to 0.90.
  - Visibility cutoff: v < 0.003 mm/hr → fully transparent (noise suppression).
  - Alpha ramp: v 0.003–0.010 mm/hr → partial opacity (in sqrt-norm space: 0.039–0.071).
  - Full opacity: v ≥ 0.010 mm/hr (norm_display ≥ 0.071).
  - PRECIP_MAX = 2.0 mm/hr unchanged (no saturation; R4 max = 1.62 mm/hr).
  - Ticks: [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0] mm/hr (spaced for sqrt visual scale).
  - Implementation: add `sqrtScale?: boolean` param to `renderWithInterpolation`; pass
    `true` for precipitation only. Legend note updated to "sqrt scale calibrated for Jan 2021".
  - Underlying precipitation.bin values and verification.json metrics: UNCHANGED.

- **FR-N30**: Temperature display: Myanmar local time (MMT) in popup and timeline.
  Diagnostic confirmed: temperature binary values are correct (K−273.15 applied, no
  double conversion, correct grid indexing, diurnal cycle present). The perceived cold bias
  is explained by:
  (1) The default display frame (t+0h = 2021-01-01T00:00:00Z = 06:30 MMT local) is pre-dawn,
      the coolest part of the diurnal cycle. Yangon at t+0h: 21.1°C; at t+6h (12:30 MMT): 28.9°C.
  (2) GCOp documented cold bias: −0.76°C average vs ERA5; strongly diurnal (−1.45°C at 12Z,
      near zero at 18Z). This is a known model characteristic, not a data defect.
  (3) External weather sites show current/recent temperatures — a different year and/or time
      of day, not January 2021 at the displayed UTC hour.
  Frontend fix (no data change): display Myanmar time alongside UTC so users understand which
  part of the diurnal cycle is displayed.
  - Popup time line: show both UTC and MMT (UTC+6:30), e.g. "Mon 01 Jan 2021 00:00 UTC · 06:30 MMT"
  - Timeline component: existing UTC label unchanged; add "MMT" suffix or parenthetical at key markers
  - Info panel: add note that times are UTC; Myanmar local time = UTC+6:30 (MMT, no DST)
  - The cold bias (−0.76°C systematic) MUST NOT be "corrected" in the data. Document only.
  - No modification to temperature.bin, forecast.json, or verification.json.

- **FR-N24**: Popup displays all four variable values at clicked grid point for current step
- **FR-N25**: Header title: "Myanmar {N}-Day Weather Forecast" (N derived from forecast_horizon_hours).
  Header sub-row: model name, native resolution (e.g. "0.25° model"), and staleness warning if
  the forecast was generated more than forecast_horizon_hours ago.
  Init time and initialization source are NOT shown in the header sub-row (R17); they are
  available in the Info panel (FR-N26) where there is sufficient space to display them clearly.
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

#### Wind Vector Overlay Requirements [Phase R12 — 2026-08-17, supersedes R11]

**Note**: Phase R11 (commit 2ee8c60) implemented canvas-drawn arrows with two confirmed defects:
(1) distribution bug — minimum-length filter suppressed ~P75% of arrows (only southern/coastal
high-wind points visible); (2) HSL rasterization not user-readable. Phase R12 and ADR-025 replace
the R11 implementation entirely.

- **FR-W01**: When `wind_speed` is the active variable (the Wind tab), render a sparse SVG
  vector field of directional arrows positioned via `map.project([lon, lat])` screen coordinates.
  The SVG overlay is a sibling `<div>` of the map container div, styled
  `position: absolute; inset: 0; pointer-events: none`. No canvas drawing for arrows.
  Recompute on every `map.move` and `map.resize` event.

- **FR-W01a**: SUPERSEDED by R16. The `wind_direction` tab is retired as a selectable ActiveVariable
  (FR-N20, R16). The transparent-canvas branch is removed from WeatherMap.tsx. Arrow display is
  consolidated into the Wind (wind_speed) tab. The hue-wheel HSL raster remains retired (R12).

- **FR-W01b**: For `wind_speed`: retain the existing speed color raster. SVG arrows are overlaid,
  encoding direction (angle) and relative speed (length).

- **FR-W01c**: Wind tab legend (R16, revised R17). When `activeVariable === 'wind_speed'` (the Wind
  tab), the Legend component renders the wind-speed gradient bar only:
  (a) Wind-speed gradient bar (0–30 kt, existing WIND_LUT_ALPHA scale, existing tick marks).
  The compass-arrow direction widget (four-arrow N/E/S/W SVG) is NOT rendered in the legend.
  Rationale (R17): arrows on the map are self-explanatory; the compass widget added visual clutter
  that obstructed map area on mobile without adding actionable information.
  The arrow overlay (WindArrowOverlay.tsx) and all wind-direction data/calculations are unchanged.
  The `wind_direction` branch of the Legend component remains absent (retired in R16).

- **FR-W02**: Sampling: `WIND_ARROW_GRID_STEP = 3`. Sample every 3rd model grid point in both
  lat and lon. 81×41 → ~27×14 = ~378 candidates before calm filtering. Sampling is uniform in
  model grid index space. Do not implement adaptive density.

- **FR-W03**: Calm threshold: `wind_speed < 2.0 kt` → no arrow rendered. No dot or marker for
  calm points. Matches `wind_direction_calm_threshold_kt = 2.0 kt` in verification.json.

- **FR-W04**: Arrow direction: TO direction = `(stored_FROM + 180) mod 360°`.
  SVG rotation: `transform="rotate(toDeg)"` where 0° = north, clockwise.
  Mandatory sanity tests (must pass before gate):
    90°FROM east  → toDeg=270° → SVG rotate(270°) → arrow points west ✓
    180°FROM south → toDeg=0°  → SVG rotate(0°)   → arrow points north ✓
    0°FROM north  → toDeg=180° → SVG rotate(180°) → arrow points south ✓
    270°FROM west → toDeg=90°  → SVG rotate(90°)  → arrow points east ✓
  Binary values and popup display remain FROM convention (unchanged).

- **FR-W05**: Arrow length scales with wind speed. Any wind ≥ 2 kt MUST produce a visible arrow;
  no suppress-if-too-short filter. Linear interpolation: 8 CSS px at 2 kt → 22 CSS px at 30 kt.
  Arrow style: `<line>` shaft + `<polygon>` filled arrowhead. Color: white, 85–90% opacity,
  stroke-width 1.5–2 px. Arrowhead: filled equilateral or isoceles triangle, 5–8 CSS px.
  Arrows must be readable over both the speed raster and the basemap at zoom 5.2 through 10.

- **FR-W06**: Geographic distribution acceptance criterion: arrows must be visible across all
  regions of Myanmar — northern (Kachin/Shan), central (Mandalay), and southern (Yangon/Tenasserim).
  Visual QA at three zoom levels: default (5.2), mid (7), maximum (10).

- **FR-W07**: Bug record (R11). Distribution failure mechanism documented in ADR-025:
  `len = (speed/30) × 6.77px`; at median 3.10 kt `len = 0.70px < 2px` → filtered.
  Effective drawing threshold was ≈8.87 kt (≈P75). Fix: `map.project()` + guaranteed
  minimum arrow length for any non-calm point.

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

- **FR-N46**: ModelEvaluation popup in frontend — partially implemented (R8, commit untracked);
  contextualization requirements added R15 (2026-08-17). See FR-N46a–FR-N46g below.

- **FR-N46a**: Metric definitions panel (R15). The Model Evaluation panel MUST include a clearly
  labelled "How to interpret these metrics" section above the data tables, defining:
  - **MAE**: Mean Absolute Error — average magnitude of the forecast error.
  - **RMSE**: Root Mean Square Error — penalises larger errors more strongly than MAE.
  - **Bias**: Signed mean error; negative = systematic under-forecasting; positive = over-forecasting.
  - **Wind-direction circular MAE**: Shortest angular difference between forecast and ERA5 direction,
    avoiding the 0°/360° discontinuity. Formula: `diff = ((fcst − era5 + 180) % 360) − 180`.
  - **POD**: Probability of Detection — proportion of observed precipitation events correctly detected.
  - **FAR**: False Alarm Ratio — proportion of forecast precipitation events that were false alarms.
  - **CSI**: Critical Success Index — combined detection score accounting for hits, misses, and
    false alarms. Formula: `CSI = hits / (hits + misses + false_alarms)`.
  The definitions must be rendered as a collapsible or always-visible block with heading.
  No qualitative labels ("good", "bad", "accurate") unless a stated baseline is provided.

- **FR-N46b**: Temporal evaluation convention note (R15). The panel MUST display:
  - Temperature and wind summary metrics span 29 frames including t+0h.
  - t+0h is the ERA5 analysis/initialization state; forecast error = 0 by construction.
    This makes the 29-frame summary slightly optimistic for forecast-only skill assessment.
  - The per-lead-time tables show genuine forecast leads +6h through +168h only (t+0h excluded
    from the table rows), and are the more representative view of model performance.
  - Precipitation evaluation covers 28 forecast frames only because t+0h precipitation is
    excluded by the GCOp convention (tp06 = 0 at initialization).
  This note must appear before or alongside the summary metric rows.

- **FR-N46c**: Important limitations block (R15). The panel MUST display a prominently styled
  limitations/caveats block containing all four of:
  (1) Verification is against ERA5 reanalysis, not independent station observations.
  (2) GCOp was trained on ERA5; agreement may be optimistic relative to independent observations
      due to an inherent reanalysis/training-data advantage.
  (3) Results represent one forecast cycle only: 2021-01-01T00Z, 168h horizon. N=1 is insufficient
      to characterise general model performance across seasons or weather regimes.
  (4) January 2021 is a dry-season cycle; precipitation categorical metrics (POD/FAR/CSI) are
      particularly sensitive to the small number of rain events in this period.
  Style: amber/warning-tone border, visible at-a-glance.

- **FR-N46d**: Temperature cold-bias context (R15). The panel MUST include, in the temperature
  section, a factual note stating:
  - The R5 Jan 2021 validation measured a mean temperature bias of −0.7595°C against ERA5.
  - A subsequent live QA spot-check (4 sites × 4 frames) found approximately −0.89°C mean bias,
    with the largest deviation at local midday (06Z / 12:30 MMT, −1.23°C avg).
  - This is described as an "observed cold bias in the Jan 2021 validation cycle" only.
    It MUST NOT be described as a universal GCOp correction or characteristic.
  - The displayed temperature values are the raw GCOp forecast output; no offset is applied.
  - One cycle is insufficient to determine whether this bias generalises to other seasons or
    initialization dates.

- **FR-N46e**: Wind section consolidation in Model Evaluation (R16). Under a single "10m Wind"
  section heading, present wind speed and wind direction metrics separately:
  - Wind speed: MAE, RMSE, Bias (kt) — summary row + per-lead-time table.
  - Wind direction: circular MAE (°) — summary + per-lead-time table.
  These remain distinct subsections with their own column headers and units.
  Do not combine into a single score.

- **FR-N46f**: Model Evaluation must note the wind tab consolidation (R16):
  "Wind speed and direction are presented in a combined Wind view. The metrics below cover each
  component separately."

- **FR-N46g**: No qualitative ratings. The panel MUST NOT include language such as "good accuracy",
  "reliable", "accurate forecast", "performs well", or equivalent unless paired with a stated
  quantitative baseline or skill score relative to a reference (e.g. climatology or persistence).

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
- **ActiveVariable**: currently displayed variable (one of three after R16): `precipitation | wind_speed | temperature`.
  `wind_direction` remains a data variable (binary + verification) but is no longer a selectable tab value.
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
