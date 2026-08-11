# Feature Specification: Myanmar Weather Forecast Web Application

**Feature Branch**: `001-myanmar-weather-app`
**Created**: 2026-08-09
**Revised**: 2026-08-11 (MAJOR — Aurora1p5 → GraphCastSmall; 168h/0.25° → 24h/1.0°; temp removed)
**Status**: Approved
**Constitution**: `.specify/memory/constitution.md` v2.0.0

---

## Constitution Check

| Principle | Requirement | Design Decision | Status |
|-----------|-------------|-----------------|--------|
| I. Static-First | No runtime server | All data pre-generated; GitHub Pages CDN only | ✓ |
| II. Earth2Studio-Mandatory | GraphCastSmall + ARCO/IFS init | `GraphCastSmall` + `earth2studio.data.ARCO` or `IFS` | ✓ |
| III. Forecast-Artifact Pipeline | Standalone scripts; 6h steps; 24h horizon | `scripts/generate_forecast.py`; 5 frames (t+0,6,12,18,24h) | ✓ |
| III. Two-timestep init | t-6h and t+0h required | Both fetched before inference | ✓ |
| III. No log transform | tp06 already in physical metres | Convert ×1000 only; no exp() | ✓ |
| IV. Myanmar-Focused | bbox 92°E–102°E, 9°N–29°N | xarray spatial subset in pipeline | ✓ |
| V. Map-First UX | Map dominates viewport | MapLibre GL JS full-viewport | ✓ |
| VI. Native-Step Navigation | 6h steps, no interpolation | 5 native frames; 6h slider steps | ✓ |
| VI. tp06 semantics | 6h accumulation, not instantaneous | UI label: mm/6h with mandatory disclosure | ✓ |
| VII. Model-Agnostic Frontend | metadata.json drives all model display | No GraphCast strings in TypeScript | ✓ |
| VIII. Performance | Load < 5s; transition < 200ms | Float32 binary ≈ 4.6 KB total; no lazy loading needed | ✓ |
| IX. Climate-Honest | All metadata shown; tp06 semantics disclosed | Header + InfoPanel + tp06 tooltip | ✓ |
| X. Minimal Scope | No databases, accounts, paid APIs | pipeline → static files → GitHub Pages | ✓ |
| XI. Hardware Transparency | VRAM verified experimentally; staged test | Staged VRAM test in notebook before full run | ✓ |
| XII. Resolution Honesty | Disclose native 1.0° vs. display resolution | InfoPanel states interpolation cannot add model info | ✓ |

---

## User Scenarios & Testing

### User Story 1 — View Myanmar Precipitation Map (Priority: P1)

A meteorologist or general user opens the application and immediately sees an interactive weather
map centered on Myanmar, showing precipitation at the forecast initialization hour.

**Why this priority**: The map is the application's entire value proposition.

**Independent Test**: Open the app with demo data — the map renders Myanmar with a colored
precipitation overlay, a legend showing mm/6h values, the forecast timestamp, and the model
name in the header.

**Acceptance Scenarios**:

1. **Given** a user opens the app, **When** the page loads, **Then** the map is centered on
   Myanmar (~96°E, 19°N), the precipitation overlay is visible, a mm/6h legend is shown, and
   the header displays the model name and initialization time.
2. **Given** the app has loaded, **When** the user pans the map, **Then** the weather overlay
   remains synchronized.
3. **Given** the app has loaded, **When** the user zooms, **Then** the overlay scales correctly.
4. **Given** the app is running with demo data, **When** the page loads, **Then** a clearly
   visible "DEMO DATA" banner appears.

---

### User Story 2 — Navigate Forecast Step by Step (Priority: P1)

A user steps through 4 forecast steps (t+6h, t+12h, t+18h, t+24h), watching the map,
timestamp, and lead time update at each 6-hour step.

**Why this priority**: The constitution (§VI) makes native-step navigation non-negotiable.

**Independent Test**: With demo data, moving the slider through all 5 frames produces 5 distinct
map states with correct timestamps (init_time + N hours).

**Acceptance Scenarios**:

1. **Given** the app has loaded, **When** the user drags the slider to step t+12h, **Then**
   the map shows the t+12h forecast, the timestamp shows init_time + 12h UTC, and the lead time
   displays "+12 h".
2. **Given** the user presses Next Step, **Then** the step advances by 6h and the map updates.
3. **Given** the current step is t+24h, **When** the user presses Next, **Then** nothing changes
   (boundary clamping).
4. **Given** the current step is t+0h, **When** the user presses Previous, **Then** nothing changes.

---

### User Story 3 — Animate the Forecast (Priority: P2)

A user presses Play and watches the forecast animate through all 5 steps. Speed controls change
the tick interval.

**Acceptance Scenarios**:

1. **Given** the user presses Play, **Then** the forecast advances 1 step per tick at the
   selected playback speed.
2. **Given** animation is playing, **When** Pause is pressed, **Then** animation stops.
3. **Given** animation reaches t+24h, **Then** it loops back to t+0h.
4. **Given** 4× speed is selected, **Then** the tick interval is 1000/4 = 250ms.

---

### User Story 4 — Click Map for Point Values (Priority: P2)

A user clicks anywhere on the Myanmar map and sees a popup with the precipitation at that grid
point for the currently selected forecast step.

**Acceptance Scenarios**:

1. **Given** the user clicks within Myanmar bbox, **Then** a popup shows: nearest 1° grid
   point lat/lon, tp06 in mm/6h (2 decimals), current valid time, accumulation period note.
2. **Given** a popup is open and the user moves the slider, **Then** popup values update.
3. **Given** the user clicks outside the Myanmar bbox, **Then** the popup shows "Outside
   forecast domain."

---

### User Story 5 — View Forecast Metadata & Attribution (Priority: P3)

A user opens the About panel and reads forecast model details, data sources, variable
semantics, and limitations.

**Acceptance Scenarios**:

1. **Given** the user clicks the Info button, **Then** a panel opens showing: model (from
   metadata.json), resolution (1.0°, native), display resolution (if interpolated), init
   source, init time, precipitation description (6-hour accumulation, mm/6h), limitations
   (resolution, uncertainty, interpolation disclosure), and attribution.
2. **Given** the forecast is demo data, **Then** the panel explicitly states
   "DEMO DATA — not for operational use."

---

### Edge Cases

- **forecast.json fails to load**: Show error message "Forecast data unavailable. Try refreshing."
- **NaN values in forecast array**: Render as transparent (no color), not zero.
- **Mobile viewport (≥320px)**: Map remains usable; controls stack vertically.
- **tp06 negative**: Clamp to 0 (negative precipitation is physically impossible).
- **t+0h frame**: Derived from initialization data; displayed identically to forecast frames.
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
  to mm (× 1000), clamping to ≥ 0
- **FR-006**: Pipeline MUST produce 4 genuine 6-hourly forecast frames (t+6h through t+24h)
  plus a t+0h initialization frame = 5 total frames
- **FR-007**: Pipeline MUST NOT interpolate between 6-hourly frames
- **FR-008**: Pipeline MUST subset output to Myanmar bbox: lat 9°N–29°N, lon 92°E–102°E
  (21 × 11 points at 1.0°)
- **FR-009**: Pipeline MUST write `forecast.json` with full metadata including tp06 provenance
  (source unit: metres, conversion: ×1000, accumulation period: 6h)
- **FR-010**: Pipeline MUST validate output (no NaN in expected fields, monotonic timestamps,
  tp06 ≥ 0, tp06 < configurable max threshold)
- **FR-011**: Pipeline MUST record the GPU hardware used and peak VRAM consumed
- **FR-012**: Demo data generation MUST NOT require a GPU; must set `is_demo: true`; must
  produce 5 frames of synthetic precipitation data in the 21 × 11 grid
- **FR-013**: Pipeline MUST perform a staged VRAM test before full inference: (1) baseline,
  (2) single-step inference; abort and report if any stage exceeds available VRAM

### Functional Requirements — Frontend (React + TypeScript)

- **FR-020**: App MUST render an interactive MapLibre GL JS map centered on Myanmar (~96°E, 19°N)
- **FR-021**: App MUST render precipitation (tp06) as a colored raster overlay
- **FR-022**: App MUST display Myanmar national boundary as a GeoJSON line layer
- **FR-023**: App MUST provide a timeline slider across all available forecast steps (t+0h to t+24h)
- **FR-024**: App MUST provide Previous / Next step buttons stepping in 6h increments, with
  boundary clamping
- **FR-025**: App MUST provide Play/Pause animation with speed selection (0.5×, 1×, 2×, 4×)
- **FR-026**: App MUST display: forecast valid date, UTC time, and lead time offset for the
  current step
- **FR-027**: App MUST display a synchronized mm/6h legend for precipitation
- **FR-028**: App MUST show a point-inspect popup on map click with tp06 (mm/6h) value
- **FR-029**: App MUST display a "DEMO DATA" banner when `forecast.json` has `is_demo: true`
- **FR-030**: App MUST display model name, resolution, and init time in the header (from
  `forecast.json`)
- **FR-031**: App MUST include an Info/About panel with full metadata and attribution
- **FR-032**: App MUST include a precipitation disclosure: "Precipitation values represent
  total rainfall accumulated during the 6-hour forecast period ending at the displayed time."
- **FR-033**: App MUST read ALL model details from `forecast.json`; no model names, resolutions,
  or variable names MUST be hard-coded in TypeScript source
- **FR-034**: If display resolution differs from native 1.0° model resolution, the UI MUST
  state that interpolation does not add forecast information

### Functional Requirements — Deployment

- **FR-040**: App MUST deploy to GitHub Pages as a fully static site
- **FR-041**: GitHub Actions `deploy-pages.yml` MUST build frontend and copy demo or production
  data on push to main
- **FR-042**: All asset paths MUST work under a GitHub Pages repository subpath
- **FR-043**: Page refresh MUST NOT break the application

---

## Non-Functional Requirements

- **NFR-001**: Initial load < 5 seconds on ≥25 Mbps broadband, cold cache
- **NFR-002**: Step-to-step transition < 200ms after data loaded
- **NFR-003**: Total forecast data payload < 1 MB (estimated: ~4.6 KB — no lazy loading required)
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
  `resolution`, `generated_at`, `is_demo`, `inference_config` (GPU/VRAM record).
- **ForecastFrame**: One 6h timestep of one variable. Attributes: `valid_time`, `lead_hours`,
  `variable`, `data [n_lat × n_lon]`.
- **ForecastArtifact**: Binary files + `forecast.json` consumed by the frontend.
- **Variable**: `precipitation` only (tp06, mm/6h, 6h accumulation).
- **GridPoint**: One 1.0° lat/lon point in the Myanmar subset (21 × 11 grid).

---

## Success Criteria

- **SC-001**: User opens GitHub Pages URL and sees Myanmar precipitation map within 5 seconds
- **SC-002**: User navigates slider through all 5 steps (t+0 to t+24h); map updates with
  correct timestamp at each step
- **SC-003**: User plays animation through all 5 steps without freeze
- **SC-004**: User clicks Myanmar map; popup shows tp06 (mm/6h) for current step
- **SC-005**: Earth2Studio GraphCastSmall pipeline generates a Myanmar forecast with ARCO/IFS
  initialization
- **SC-006**: Precipitation is labeled "mm / 6h" with disclosure explaining 6-hour accumulation
- **SC-007**: GitHub Actions workflow deploys frontend to GitHub Pages on push to main
- **SC-008**: `forecast.json` records tp06 transformation provenance (source unit, conversion,
  accumulation period)
- **SC-009**: InfoPanel discloses 1.0° native model resolution and interpolation policy

---

## Assumptions

- Earth2Studio ≥ 0.17.0
- GPU with sufficient VRAM available for production runs (T4 16 GB compatibility unverified;
  must be established experimentally before full inference)
- ARCO accessible via Google Cloud (no credentials; historical data only)
- IFS open data accessible at forecast generation time (operational, no credentials)
- GraphCastSmall weights downloaded automatically via `GraphCastSmall.load_default_package()`
- Myanmar boundary GeoJSON sourced from Natural Earth (public domain)
- Basemap: OSM-based open tiles (no API key required)
- Python pipeline: uv virtual environment, pyproject.toml
- Frontend: React 18, TypeScript 5, Vite 5, MapLibre GL JS 4, Tailwind CSS 3, Zustand

---

## Out of Scope (MVP)

- Temperature visualization (precipitation only in this MVP)
- Variable switcher (precipitation is the only displayed variable)
- Forecast horizon beyond 24h
- Ensemble/probabilistic forecasting
- Wind speed/direction visualization
- Real-time automated data ingestion
- User accounts, authentication, or personalization
- Mobile-native applications
- Paid API integrations
