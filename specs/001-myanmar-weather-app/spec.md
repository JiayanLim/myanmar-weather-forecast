# Feature Specification: Myanmar Weather Forecast Web Application

**Feature Branch**: `001-myanmar-weather-app`
**Created**: 2026-08-09
**Revised**: 2026-08-09 (ADR updates: IFS init, native tp1h, native hourly rollout)
**Status**: Approved
**Constitution**: `.specify/memory/constitution.md` v1.0.1

---

## Constitution Check

| Principle | Requirement | Design Decision | Status |
|-----------|-------------|-----------------|--------|
| I. Static-First | No runtime server | All data pre-generated; GitHub Pages CDN only | ✓ |
| II. Earth2Studio-Mandatory | Aurora1p5 + IFS init | `Aurora1p5` + `earth2studio.data.IFS` + sic patch | ✓ |
| III. Forecast-Artifact Pipeline | Standalone scripts | `scripts/generate_forecast.py` + `validate_forecast.py` | ✓ |
| III. sic gap handling | Must be documented and handled | Sic patch from ARCO or climatological zero; recorded in forecast.json | ✓ |
| IV. Myanmar-Focused | bbox 92°E–102°E, 9°N–29°N | xarray spatial subset in pipeline | ✓ |
| V. Map-First UX | Map dominates viewport | MapLibre GL JS full-viewport | ✓ |
| VI. Hourly Navigation | Native hourly from Aurora1p5 | 168 genuine model predictions; no interpolation | ✓ |
| VI. Temporal semantics | tp1h = 1h accumulated, not interpolated | Native tp1h output; UI label: mm/h with disclosure | ✓ |
| VII. Model-Agnostic Frontend | metadata.json drives all model display | No Aurora strings in TypeScript; abstract pipeline interfaces | ✓ |
| VIII. Performance | Load < 5s; transition < 200ms | Float32 binary ≈ 4.5 MB total; lazy day-chunk loading | ✓ |
| IX. Climate-Honest | All metadata shown; precip semantics disclosed | Header + InfoPanel + tp1h tooltip | ✓ |
| X. Minimal Scope | No databases, accounts, paid APIs | pipeline → static files → GitHub Pages | ✓ |

---

## User Scenarios & Testing

### User Story 1 — View Myanmar Weather Map (Priority: P1)

A meteorologist or general user opens the application and immediately sees an interactive weather map centered on Myanmar, showing the default variable (temperature) at the forecast initialization hour.

**Why this priority**: The map is the application's entire value proposition. Without a working overlay, nothing else matters.

**Independent Test**: Open the app with demo data — the map renders Myanmar with a colored temperature overlay, a legend showing °C values, the forecast timestamp, and the model name in the header.

**Acceptance Scenarios**:

1. **Given** a user opens the app, **When** the page loads, **Then** the map is centered on Myanmar (~96°E, 19°N), the temperature overlay is visible, a °C legend is shown, and the header displays "Aurora 1.5" and the initialization time.
2. **Given** the app has loaded, **When** the user pans the map, **Then** the weather overlay remains synchronized.
3. **Given** the app has loaded, **When** the user zooms, **Then** the overlay scales correctly.
4. **Given** the app is running with demo data, **When** the page loads, **Then** a clearly visible "DEMO DATA" banner appears.

---

### User Story 2 — Navigate Forecast Hour by Hour (Priority: P1)

A user steps through 168 hours of forecast, watching the map, timestamp, lead time, and values update at each hour.

**Why this priority**: The constitution (§VI) makes hourly navigation non-negotiable. Each Aurora1p5 hourly step is a distinct model prediction.

**Independent Test**: With demo data, moving the slider from hour 0 to hour 24 produces 25 distinct map states with correct timestamps (init_time + N hours).

**Acceptance Scenarios**:

1. **Given** the app has loaded, **When** the user drags the slider to hour 42, **Then** the map shows the t+42h forecast, the timestamp shows init_time + 42h UTC, and the lead time displays "+42 h".
2. **Given** the user presses Next Hour, **Then** the hour advances by 1 and the map updates.
3. **Given** the current hour is 168, **When** the user presses Next, **Then** nothing changes (boundary clamping).
4. **Given** the current hour is 0, **When** the user presses Previous, **Then** nothing changes.

---

### User Story 3 — Animate the Forecast (Priority: P2)

A user presses Play and watches the forecast animate. The animation advances 1 hour per tick. Speed controls change the tick interval.

**Acceptance Scenarios**:

1. **Given** the user presses Play, **Then** the forecast advances 1 frame per tick at the selected playback speed.
2. **Given** animation is playing, **When** Pause is pressed, **Then** animation stops at the current hour.
3. **Given** animation reaches hour 168, **Then** it loops back to hour 0.
4. **Given** 4× speed is selected, **Then** the tick interval is 1000/4 = 250ms.

---

### User Story 4 — Switch Between Variables (Priority: P1)

A user switches between Temperature and Precipitation. The map overlay, legend, units, and color scale change immediately. The timeline position is preserved.

**Acceptance Scenarios**:

1. **Given** temperature is active, **When** the user clicks Precipitation, **Then** the overlay switches to precipitation coloring, the legend shows mm/h, and the active button changes.
2. **Given** precipitation is active, **When** the user selects Temperature, **Then** the overlay reverts to temperature coloring with °C legend.
3. **Given** precipitation is displayed, **When** the user hovers over the "?" icon, **Then** a tooltip appears: "Precipitation values represent total rainfall accumulated during each 1-hour forecast period."

---

### User Story 5 — Click Map for Point Values (Priority: P2)

A user clicks anywhere on the Myanmar map and sees a popup with the temperature and precipitation at that grid point for the currently selected forecast hour.

**Acceptance Scenarios**:

1. **Given** the user clicks within Myanmar bbox, **Then** a popup shows: nearest grid point lat/lon, t2m in °C (1 decimal), tp1h in mm/h (2 decimals), current valid time.
2. **Given** a popup is open and the user moves the slider, **Then** popup values update for the new hour.
3. **Given** the user clicks outside the Myanmar bbox, **Then** the popup shows "Outside forecast domain."

---

### User Story 6 — View Forecast Metadata & Attribution (Priority: P3)

A user opens the About panel and reads forecast model details, data sources, variable semantics, and limitations.

**Acceptance Scenarios**:

1. **Given** the user clicks the Info button, **Then** a panel opens showing: model (Aurora 1.5), resolution (0.25°), init source (IFS), init time, valid range, temperature description (hourly), precipitation description (1-hour accumulation, mm/h), limitations (resolution, uncertainty, sic patch disclosure), and attribution.
2. **Given** the forecast is demo data, **Then** the panel explicitly states "DEMO DATA — not for operational use."

---

### Edge Cases

- **forecast.json fails to load**: Show error message "Forecast data unavailable. Try refreshing."
- **NaN values in forecast array**: Render as transparent (no color), not zero.
- **Mobile viewport (≥320px)**: Map remains usable; controls stack vertically.
- **tp1h negative after log untransform**: Clamp to 0 (negative precipitation is physically impossible; negative values indicate numerical noise).
- **t+0h frame**: Derived from initialization data; displayed identically to forecast frames.
- **Very high precipitation values**: Cap legend at 50 mm/h; higher values rendered at maximum color.

---

## Requirements

### Functional Requirements — Forecast Pipeline (Python)

- **FR-001**: Pipeline MUST use `earth2studio.models.px.Aurora1p5` as the prognostic model
- **FR-002**: Pipeline MUST use `earth2studio.data.IFS` as the primary initialization source
- **FR-003**: Pipeline MUST handle the `sic` (sea ice concentration) gap in IFS open data by patching from a documented secondary source or climatological value; the method MUST be recorded in `forecast.json`
- **FR-004**: Pipeline MUST NOT use GFS as an Aurora1p5 initialization source
- **FR-005**: Pipeline MUST extract `t2m` (temperature) from Aurora1p5 output and convert from K to °C
- **FR-006**: Pipeline MUST extract `tp1h` (total precipitation) from Aurora1p5 output, apply log untransform (`exp()`), convert from m to mm, and store as mm/h (1-hour accumulation)
- **FR-007**: Pipeline MUST produce 168 genuine hourly forecast frames (t+1h through t+168h) plus a t+0h initialization frame = 169 total frames
- **FR-008**: Pipeline MUST NOT interpolate between hourly frames; Aurora1p5's native hourly rollout provides all 168 steps
- **FR-009**: Pipeline MUST subset output to Myanmar bbox: lat 9°N–29°N, lon 92°E–102°E
- **FR-010**: Pipeline MUST write `forecast.json` containing all metadata in `plan.md` format including `sic_handling` field
- **FR-011**: Pipeline MUST validate output (no NaN in expected fields, monotonic timestamps, tp1h ≥ 0, t2m in [−20, 60] °C range)
- **FR-012**: Demo data generation MUST NOT require a GPU; must produce physically plausible synthetic data; must set `is_demo: true`
- **FR-013**: Pipeline MUST use abstract `ForecastModel` interface enabling model substitution
- **FR-014**: Pipeline MUST use abstract `InitializationSource` interface with IFS and NCAR_ERA5/ARCO implementations

### Functional Requirements — Frontend (React + TypeScript)

- **FR-020**: App MUST render an interactive MapLibre GL JS map centered on Myanmar (~96°E, 19°N)
- **FR-021**: App MUST render weather overlays (t2m or tp1h) as colored raster images
- **FR-022**: App MUST display Myanmar national boundary as a GeoJSON line layer
- **FR-023**: App MUST provide a timeline slider: range 0–168 (hours), step 1
- **FR-024**: App MUST provide Previous / Next hour buttons with boundary clamping
- **FR-025**: App MUST provide Play/Pause animation with speed selection (0.5×, 1×, 2×, 4×)
- **FR-026**: App MUST display: forecast valid date, UTC time, and lead time offset for the current hour
- **FR-027**: App MUST provide a variable switcher: Temperature | Precipitation
- **FR-028**: App MUST display a synchronized legend for the active variable (°C or mm/h)
- **FR-029**: App MUST show a point-inspect popup on map click with t2m (°C) and tp1h (mm/h) values
- **FR-030**: App MUST display a "DEMO DATA" banner when `forecast.json` has `is_demo: true`
- **FR-031**: App MUST display model name, resolution, and init time in the header (from `forecast.json`)
- **FR-032**: App MUST include an Info/About panel with full metadata and attribution
- **FR-033**: App MUST include a precipitation tooltip: "Precipitation values represent total rainfall accumulated during each 1-hour forecast period. These are not instantaneous rainfall rates."
- **FR-034**: App MUST read ALL model details from `forecast.json`; no model names, resolutions, or variable names MUST be hard-coded in TypeScript source

### Functional Requirements — Deployment

- **FR-040**: App MUST deploy to GitHub Pages as a fully static site
- **FR-041**: GitHub Actions `deploy-pages.yml` MUST build frontend and copy demo data on push to main
- **FR-042**: All asset paths MUST work under a GitHub Pages repository subpath
- **FR-043**: Page refresh MUST NOT break the application (hash routing or no-router single-page design)

---

## Non-Functional Requirements

- **NFR-001**: Initial load < 5 seconds on ≥25 Mbps broadband, cold cache
- **NFR-002**: Hour-to-hour transition < 200ms after data loaded for current day
- **NFR-003**: Total forecast data payload < 20 MB for 169 frames × 2 variables
- **NFR-004**: No secrets, credentials, or API keys committed to repository
- **NFR-005**: No proprietary API keys required for the deployed static frontend
- **NFR-006**: TypeScript compilation MUST have no avoidable `any` types or errors
- **NFR-007**: Python pipeline code MUST pass `ruff` linting and formatting
- **NFR-008**: Precipitation MUST be correctly labeled as 1-hour accumulation, not instantaneous rate

---

## Key Entities

- **ForecastRun**: A pipeline execution. Attributes: `init_time`, `model`, `init_source`, `sic_handling`, `resolution`, `generated_at`, `is_demo`.
- **ForecastFrame**: One timestep of one variable. Attributes: `valid_time`, `lead_hours`, `variable`, `data [n_lat × n_lon]`.
- **ForecastArtifact**: Binary files + `forecast.json` consumed by the frontend.
- **Variable**: Either `temperature_2m` (°C, hourly genuine) or `precipitation` (mm/h, 1h accumulation genuine).
- **GridPoint**: One 0.25° lat/lon point in the Myanmar subset (~81×41 grid).

---

## Success Criteria

- **SC-001**: User opens GitHub Pages URL and sees Myanmar weather map within 5 seconds
- **SC-002**: User navigates slider from hour 0 to 168; map updates at each step with correct timestamp
- **SC-003**: User plays animation through all 168 hours without freeze
- **SC-004**: User switches variables; legend, color scale, and overlay change correctly
- **SC-005**: User clicks Myanmar map; popup shows t2m (°C) and tp1h (mm/h) for current hour
- **SC-006**: Earth2Studio Aurora1p5 pipeline generates a Myanmar forecast with IFS initialization
- **SC-007**: Precipitation is labeled "mm/h" with tooltip explaining 1-hour accumulation semantics
- **SC-008**: GitHub Actions workflow deploys frontend to GitHub Pages on push to main
- **SC-009**: The `forecast.json` sic_handling field is populated and non-empty

---

## Assumptions

- Earth2Studio ≥ 0.17.0
- GPU with ≥48 GB VRAM available for production runs (not required for demo data)
- IFS open data accessible at forecast generation time (ECMWF open data initiative; no credentials)
- NCAR_ERA5 accessible via AWS (no credentials; possible transfer cost)
- Aurora1p5 model weights downloaded automatically via HuggingFace (`load_default_package()`)
- Myanmar boundary GeoJSON sourced from Natural Earth (public domain)
- Basemap: OSM-based open tiles (no API key required)
- Python pipeline: uv virtual environment, pyproject.toml
- Frontend: React 18, TypeScript 5, Vite 5, MapLibre GL JS 4, Tailwind CSS 3, Zustand

---

## Out of Scope (MVP)

- Aurora1p5Ensemble (probabilistic/ensemble forecasting)
- Wind speed/direction visualization
- Additional variables beyond t2m and tp1h
- Real-time IFS data ingestion automation (MVP: manual trigger)
- Sub-daily forecast refresh automation
- User accounts, authentication, or personalization
- Mobile-native applications
- Paid API integrations
