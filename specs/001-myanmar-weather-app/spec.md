# Feature Specification: Myanmar Weather Forecast Web Application

**Feature Branch**: `001-myanmar-weather-app`
**Created**: 2026-08-09
**Revised**: 2026-08-11 v2 — Aurora1p5 → GraphCastSmall; 168h/0.25° → 24h/1.0°; temp removed
**Revised**: 2026-08-11 v3 — MAJOR: 24h → 48h; temperature + precipitation; schema v3.0; M4 CPU validated
**Status**: Approved — Architecture validated on Apple M4 CPU
**Constitution**: `.specify/memory/constitution.md` v2.0.0

---

## Constitution Check

| Principle | Requirement | Design Decision | Status |
|-----------|-------------|-----------------|--------|
| I. Static-First | No runtime server | All data pre-generated; GitHub Pages CDN only | ✓ |
| II. Earth2Studio-Mandatory | GraphCastSmall + ARCO/IFS init | `GraphCastSmall` + `earth2studio.data.ARCO` or `IFS` | ✓ |
| III. Forecast-Artifact Pipeline | Standalone scripts; 6h steps; 48h horizon | `scripts/generate_forecast.py`; 9 frames (t+0…t+48h) | ✓ |
| III. Two-timestep init | t-6h and t+0h required | Both fetched before inference | ✓ |
| III. No log transform | tp06 already in physical metres | Convert ×1000 only; no exp() | ✓ |
| IV. Myanmar-Focused | bbox 92°E–102°E, 9°N–29°N | xarray/numpy spatial subset in pipeline | ✓ |
| V. Map-First UX | Map dominates viewport | MapLibre GL JS full-viewport | ✓ |
| VI. Native-Step Navigation | 6h steps, no interpolation | 9 native frames; 6h slider steps | ✓ |
| VI. tp06 semantics | 6h accumulation, not instantaneous | UI label: mm/6h with mandatory disclosure | ✓ |
| VII. Model-Agnostic Frontend | metadata.json drives all model display | No GraphCast strings in TypeScript | ✓ |
| VIII. Performance | Load < 5s; transition < 200ms | Float32 binary ≈ 16.6 KB total; all loaded at once | ✓ |
| IX. Climate-Honest | All metadata shown; tp06 semantics disclosed | Header + InfoPanel + tp06 tooltip | ✓ |
| X. Minimal Scope | No databases, accounts, paid APIs | pipeline → static files → GitHub Pages | ✓ |
| XI. Hardware Transparency | Hardware validated; recorded in forecast.json | M4 CPU validated: ~78s, 2.34 GB RSS | ✓ |
| XII. Resolution Honesty | Disclose native 1.0° vs. display resolution | InfoPanel states interpolation cannot add model info | ✓ |

---

## User Scenarios & Testing

### User Story 1 — View Myanmar Weather Map (Priority: P1)

A meteorologist or general user opens the application and immediately sees an interactive weather
map centered on Myanmar, showing temperature or precipitation at the forecast initialization hour.

**Why this priority**: The map is the application's entire value proposition.

**Independent Test**: Open the app with demo data — the map renders Myanmar with a colored
overlay, a legend, the forecast timestamp, and the model name in the header.

**Acceptance Scenarios**:

1. **Given** a user opens the app, **When** the page loads, **Then** the map is centered on
   Myanmar (~96°E, 19°N), the precipitation overlay is visible by default, a mm/6h legend is
   shown, and the header displays the model name and initialization time.
2. **Given** the app has loaded, **When** the user pans the map, **Then** the weather overlay
   remains synchronized.
3. **Given** the app has loaded, **When** the user zooms, **Then** the overlay scales correctly.
4. **Given** the app is running with demo data, **When** the page loads, **Then** a clearly
   visible "DEMO DATA" banner appears.

---

### User Story 2 — Navigate Forecast Step by Step (Priority: P1)

A user steps through 8 forecast steps (t+6h, …, t+48h), watching the map, timestamp, and lead
time update at each 6-hour step.

**Why this priority**: The constitution (§VI) makes native-step navigation non-negotiable.

**Independent Test**: With demo data, moving the slider through all 9 frames produces 9 distinct
map states with correct timestamps (init_time + N hours).

**Acceptance Scenarios**:

1. **Given** the app has loaded, **When** the user drags the slider to step t+12h, **Then**
   the map shows the t+12h forecast, the timestamp shows init_time + 12h UTC, and the lead time
   displays "+12 h".
2. **Given** the user presses Next Step, **Then** the step advances by 6h and the map updates.
3. **Given** the current step is t+48h, **When** the user presses Next, **Then** nothing changes
   (boundary clamping).
4. **Given** the current step is t+0h, **When** the user presses Previous, **Then** nothing changes.

---

### User Story 3 — Animate the Forecast (Priority: P2)

A user presses Play and watches the forecast animate through all 9 steps. Speed controls change
the tick interval.

**Acceptance Scenarios**:

1. **Given** the user presses Play, **Then** the forecast advances 1 step per tick at the
   selected playback speed.
2. **Given** animation is playing, **When** Pause is pressed, **Then** animation stops.
3. **Given** animation reaches t+48h, **Then** it loops back to t+0h.
4. **Given** 4× speed is selected, **Then** the tick interval is 1000/4 = 250ms.

---

### User Story 4 — Switch Between Temperature and Precipitation (Priority: P2)

A user clicks "Temp" or "Precip" in the VariableSwitcher and the map, legend, and popup all
update to show the selected variable.

**Acceptance Scenarios**:

1. **Given** the app has loaded, **When** the user clicks "Temp", **Then** the map shows the
   temperature overlay (°C color scale) and the legend updates.
2. **Given** temperature is active, **When** the user clicks "Precip", **Then** the map reverts
   to precipitation (mm/6h).
3. **Given** a popup is open showing temperature, **When** the user switches to precipitation,
   **Then** the popup immediately shows precipitation values.

---

### User Story 5 — Click Map for Point Values (Priority: P2)

A user clicks anywhere on the Myanmar map and sees a popup with the values for both variables
at that grid point for the currently selected forecast step.

**Acceptance Scenarios**:

1. **Given** the user clicks within Myanmar bbox, **Then** a popup shows: nearest 1° grid
   point lat/lon, active variable value prominently, other variable dimmed.
2. **Given** a popup is open and the user moves the slider, **Then** popup values update.
3. **Given** the user clicks outside the Myanmar bbox, **Then** the popup shows "Outside
   forecast domain."

---

### User Story 6 — View Forecast Metadata & Attribution (Priority: P3)

A user opens the About panel and reads forecast model details, data sources, variable
semantics, and limitations.

**Acceptance Scenarios**:

1. **Given** the user clicks the Info button, **Then** a panel opens showing: model (from
   forecast.json), resolution (1.0°, native), init source, init time, both variable descriptions
   (precipitation 6-hour accumulation, mm/6h; temperature 2m, °C), limitations, and attribution.
2. **Given** the forecast is demo data, **Then** the panel explicitly states
   "DEMO DATA — not for operational use."

---

### Edge Cases

- **forecast.json fails to load**: Show error message "Forecast data unavailable. Try refreshing."
- **NaN values in forecast array**: Render as transparent (no color), not zero.
- **Mobile viewport (≥320px)**: Map remains usable; controls stack vertically.
- **tp06 negative**: Clamp to 0 (negative precipitation is physically impossible).
- **t+0h frame**: Temperature is analysis state; precipitation is 0.0 (no accumulation).
- **Very high precipitation values**: Cap legend at 100 mm/6h; higher values at maximum color.

---

## Requirements

### Functional Requirements — Forecast Pipeline (Python)

- **FR-001**: Pipeline MUST use `earth2studio.models.px.GraphCastSmall` as the prognostic model
- **FR-002**: Pipeline MUST use `earth2studio.data.ARCO` or `earth2studio.data.IFS` as the
  initialization source; NCAR_ERA5 and GFS MUST NOT be used
- **FR-003**: Pipeline MUST fetch TWO consecutive time steps (t−6h and t+0h) from the
  initialization source before inference begins
- **FR-004**: Pipeline MUST NOT apply a log or exponential transform to `tp06`; the only
  required conversion is metres × 1000 = mm / 6h
- **FR-005**: Pipeline MUST extract `tp06` from GraphCastSmall output and convert from metres
  to mm (× 1000), clamping to ≥ 0; set t+0h frame to 0.0
- **FR-006**: Pipeline MUST produce 8 genuine 6-hourly forecast frames (t+6h through t+48h)
  plus a t+0h initialization frame = 9 total frames
- **FR-007**: Pipeline MUST NOT interpolate between 6-hourly frames
- **FR-008**: Pipeline MUST subset output to Myanmar bbox: lat 9°N–29°N, lon 92°E–102°E
  (21 × 11 points at 1.0°)
- **FR-009**: Pipeline MUST write `forecast.json` with full schema v3.0 metadata including
  provenance for both tp06 and t2m transformations
- **FR-010**: Pipeline MUST validate output (no NaN in expected fields, monotonic timestamps,
  tp06 ≥ 0, tp06 < configurable max threshold, t2m in [-90°C, 70°C])
- **FR-011**: Pipeline MUST record the inference hardware and peak RSS in `inference_config`
  field of forecast.json
- **FR-012**: Demo data generation MUST NOT require a GPU; must set `is_demo: true`; must
  produce 9 frames of synthetic precipitation and temperature data in the 21 × 11 grid
- **FR-013**: Pipeline MUST extract `t2m` from GraphCastSmall output and convert from Kelvin
  to Celsius (K − 273.15)

### Functional Requirements — Frontend (React + TypeScript)

- **FR-020**: App MUST render an interactive MapLibre GL JS map centered on Myanmar (~96°E, 19°N)
- **FR-021**: App MUST render precipitation (tp06) OR temperature (t2m) as a colored raster
  overlay depending on `activeVariable`
- **FR-022**: App MUST display Myanmar national boundary as a GeoJSON line layer
- **FR-023**: App MUST provide a timeline slider across all 9 available forecast steps (t+0h to t+48h)
- **FR-024**: App MUST provide Previous / Next step buttons stepping in 6h increments, with
  boundary clamping
- **FR-025**: App MUST provide Play/Pause animation with speed selection (0.5×, 1×, 2×, 4×)
- **FR-026**: App MUST display: forecast valid date, UTC time, and lead time offset for the
  current step (Forecast +0h through +48h)
- **FR-027**: App MUST display a variable-aware legend — mm/6h for precipitation, °C for
  temperature
- **FR-028**: App MUST show a point-inspect popup on map click with both variable values at
  the clicked grid point
- **FR-029**: App MUST display a "DEMO DATA" banner when `forecast.json` has `is_demo: true`
- **FR-030**: App MUST display model name, resolution, and init time in the header (from
  `forecast.json`)
- **FR-031**: App MUST include an Info/About panel with full metadata and attribution for both
  variables
- **FR-032**: App MUST include a precipitation disclosure: "Precipitation values represent
  total rainfall accumulated during the 6-hour forecast period ending at the displayed time."
- **FR-033**: App MUST read ALL model details from `forecast.json`; no model names, resolutions,
  or variable names MUST be hard-coded in TypeScript source
- **FR-034**: If display resolution differs from native 1.0° model resolution, the UI MUST
  state that interpolation does not add forecast information
- **FR-035**: App MUST include a variable switcher (Precip / Temp buttons) that updates the
  map overlay, legend, and popup simultaneously

### Functional Requirements — Deployment

- **FR-040**: App MUST deploy to GitHub Pages as a fully static site
- **FR-041**: GitHub Actions `deploy-pages.yml` MUST build frontend and copy either real or
  demo data on push to main; real data preferred when all 3 artifacts are present
- **FR-042**: All asset paths MUST work under a GitHub Pages repository subpath
- **FR-043**: Page refresh MUST NOT break the application

---

## Non-Functional Requirements

- **NFR-001**: Initial load < 5 seconds on ≥25 Mbps broadband, cold cache
- **NFR-002**: Step-to-step transition < 200ms after data loaded
- **NFR-003**: Total forecast data payload < 1 MB (validated: ~16.6 KB — no lazy loading required)
- **NFR-004**: No secrets, credentials, or API keys committed to repository
- **NFR-005**: No proprietary API keys required for the deployed static frontend
- **NFR-006**: TypeScript compilation MUST have no avoidable `any` types or errors
- **NFR-007**: Python pipeline code MUST pass `ruff` linting and formatting
- **NFR-008**: Precipitation MUST be correctly labeled as 6-hour accumulation, not instantaneous
- **NFR-009**: The InfoPanel MUST disclose native 1.0° model resolution; if display is
  interpolated, the disclosure MUST state interpolation does not add model information

---

## Key Entities

- **ForecastRun**: A pipeline execution. Attributes: `init_time`, `model`, `init_source`,
  `resolution`, `generated_at`, `is_demo`, `inference_config` (hardware + RSS record).
- **ForecastFrame**: One 6h timestep of one variable. Attributes: `valid_time`, `lead_hours`,
  `variable`, `data [n_lat × n_lon]`.
- **ForecastArtifact**: Binary files (`temperature.bin`, `precipitation.bin`) + `forecast.json`
  consumed by the frontend.
- **Variable**: `precipitation` (tp06, mm/6h, 6h accumulation) and `temperature` (t2m, °C).
- **ActiveVariable**: The currently displayed variable — toggled by VariableSwitcher.
- **GridPoint**: One 1.0° lat/lon point in the Myanmar subset (21 × 11 grid).

---

## Success Criteria

- **SC-001**: User opens GitHub Pages URL and sees Myanmar weather map within 5 seconds
- **SC-002**: User navigates slider through all 9 steps (t+0 to t+48h); map updates with
  correct timestamp at each step
- **SC-003**: User plays animation through all 9 steps without freeze
- **SC-004**: User clicks Myanmar map; popup shows both tp06 (mm/6h) and t2m (°C)
- **SC-005**: Earth2Studio GraphCastSmall pipeline generates a Myanmar forecast with ARCO/IFS
  initialization
- **SC-006**: Precipitation is labeled "mm / 6h" with disclosure explaining 6-hour accumulation
- **SC-007**: GitHub Actions workflow deploys frontend to GitHub Pages on push to main
- **SC-008**: `forecast.json` records both tp06 and t2m transformation provenance
- **SC-009**: InfoPanel discloses 1.0° native model resolution and interpolation policy

---

## Assumptions

- Earth2Studio ≥ 0.17.0 with xarray < 2026 (pinned; xr.Dataset constructor changed in 2026)
- Apple M4 CPU is the validated production inference hardware (~78s, 2.34 GB RSS)
- ARCO accessible via Google Cloud (no credentials; historical data only to ~2023)
- IFS open data accessible at forecast generation time (operational, no credentials)
- GraphCastSmall weights downloaded automatically via `GraphCastSmall.load_default_package()`
- Myanmar boundary GeoJSON sourced from Natural Earth (public domain)
- Basemap: OSM-based open tiles (no API key required)
- Python pipeline: uv virtual environment, pyproject.toml
- Frontend: React 18, TypeScript 5, Vite 5, MapLibre GL JS 4, Tailwind CSS 3, Zustand

---

## Out of Scope (MVP)

- Forecast horizon beyond 48h
- Ensemble/probabilistic forecasting
- Wind speed/direction visualization
- Real-time automated data ingestion
- User accounts, authentication, or personalization
- Mobile-native applications
- Paid API integrations
- ERA5 verification against forecast (separate script; see verify_forecast.py)
