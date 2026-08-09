# Tasks: Myanmar Weather Forecast Web Application

**Input**: `specs/001-myanmar-weather-app/` (spec.md v2, plan.md v2, research.md v2)
**Feature**: 001-myanmar-weather-app
**Revised**: 2026-08-09

**Key architectural facts for task authors**:
- Aurora1p5 natively produces hourly t2m AND tp1h — no PrecipitationAFNO, no interpolation
- IFS is the required init source; GFS is prohibited
- sic gap in IFS open data must be patched (see plan.md IFSSource)
- tp1h requires log untransform: `exp(raw) * 1000` → mm/h
- Binary format: [169 × 81 × 41] float32 per variable (~2.1 MB each)

---

## Phase 1: Setup — Project Scaffold

**Purpose**: Create all directories, configs, and dependency files before any code.

- [ ] T001 Create directory structure: `frontend/src/{components,map,data}/`, `forecast/{models,sources,postprocessing}/`, `scripts/`, `data/{demo,forecast}/`, `docs/`, `tests/`
- [ ] T002 Create `pyproject.toml` with uv config: `earth2studio>=0.17.0`, `xarray`, `numpy`, `scipy`, `zarr`, `ruff`, `pytest`, `cfgrib` dependencies
- [ ] T003 [P] Create `frontend/package.json`: react@18, typescript@5, vite@5, maplibre-gl@4, tailwindcss@3, zustand, vitest, @vitejs/plugin-react
- [ ] T004 [P] Create `frontend/vite.config.ts`: base path from `VITE_BASE_PATH` env var; react plugin
- [ ] T005 [P] Create `frontend/tailwind.config.ts` and `frontend/src/index.css` with Tailwind directives
- [ ] T006 [P] Create `frontend/tsconfig.json`: strict mode, no implicit any, ES2022 target
- [ ] T007 [P] Create `.gitignore`: `data/forecast/`, `**/__pycache__/`, `node_modules/`, `.env`, `*.zarr/`, `dist/`, `.venv/`, `*.ckpt`, `*.pickle`
- [ ] T008 Create `frontend/index.html` with correct base href and React root div
- [ ] T009 Create `frontend/src/main.tsx`: React 18 root render

**Checkpoint**: `uv sync` succeeds. `npm install` in `frontend/` succeeds. Directory tree matches plan.md.

---

## Phase 2: Foundational — Data Contract + Demo Data

**Purpose**: Lock the binary artifact format and produce committed demo data that all frontend phases depend on.

**⚠️ CRITICAL GATE**: Frontend development cannot start until `data/demo/` artifacts are committed.

- [ ] T010 Create `frontend/src/data/types.ts`: TypeScript interfaces — `ForecastMetadata`, `VariableMeta`, `GridInfo`, `BBoxInfo`; typed to match `forecast.json` schema in plan.md
- [ ] T011 Create `forecast/postprocessing/artifact_writer.py`: `write_artifacts(t2m_celsius, tp1h_mm, metadata, output_dir)` — writes `temperature.bin`, `precipitation.bin` as float32 [169×81×41] C-order, writes `forecast.json` with full schema
- [ ] T012 Create `forecast/postprocessing/unit_conversion.py`: `kelvin_to_celsius(arr)` and `tp1h_log_to_mm(arr)` functions per plan.md docstrings
- [ ] T013 Create `scripts/generate_demo_data.py`: CPU-only; generates synthetic t2m (sinusoidal, 15–40°C range across Myanmar lat/lon/time) and tp1h (Gaussian blobs, 0–30 mm/h); writes to `data/demo/` with `is_demo: true`; must complete in < 60s; no imports of earth2studio or torch
- [ ] T014 Run `scripts/generate_demo_data.py` and verify `data/demo/` contains three files
- [ ] T015 Create `scripts/validate_forecast.py`: checks — file existence, shape [169×81×41], NaN count (FAIL if > 0 in expected fields), tp1h ≥ 0, t2m in [−20, 60]°C, timestamps monotonically increasing, `sic_handling` field non-empty; prints PASS/FAIL per check
- [ ] T016 Run `scripts/validate_forecast.py --data-dir data/demo/` and verify all PASS
- [ ] T017 Commit `data/demo/` artifacts to git

**Checkpoint**: `data/demo/forecast.json`, `data/demo/temperature.bin`, `data/demo/precipitation.bin` committed. `validate_forecast.py` reports all PASS.

---

## Phase 3: User Stories 1 & 4 — Interactive Map + Variable Switcher (P1)

**Goal**: Map renders Myanmar with colored weather overlay. Variable switcher changes overlay and legend.

**Independent Test**: `npm run dev` → browser shows Myanmar temp overlay with °C legend. Clicking Precipitation changes to mm/h overlay.

### Data Loading

- [ ] T018 Create `frontend/src/data/ForecastLoader.ts`: `loadForecast(baseUrl)` async function — fetches `forecast.json`, fetches `temperature.bin` + `precipitation.bin` as ArrayBuffer, wraps in Float32Array; exports `getFrame(data, variable, hour, nLat, nLon)` returning `Float32Array[nLat × nLon]`
- [ ] T019 Create `frontend/src/data/ForecastStore.ts`: Zustand store with all state fields from plan.md; `initialize()` action calls `ForecastLoader.loadForecast()`; auto-loads demo data on startup via `VITE_DATA_URL` env (default: `./data/`)

### Map + Overlay (US1)

- [ ] T020 Download Myanmar boundary GeoJSON from Natural Earth Admin-0, clip to Myanmar, save as `frontend/public/geo/myanmar-boundary.geojson` (< 200 KB)
- [ ] T021 Create `frontend/src/map/colorscales.ts`: `buildColormap(stops)` utility; `TEMP_COLORMAP` (15°C blue → 40°C red); `PRECIP_COLORMAP` (0 white → 50+ red); both return `Uint8Array[256×4]`; exports `applyColorscale(frame, colormap, min, max): Uint8ClampedArray`
- [ ] T022 Create `frontend/src/map/WeatherMap.tsx`: MapLibre GL JS map, center `[96, 19]`, zoom 5, OSM raster basemap (no API key), Myanmar GeoJSON boundary line layer; exposes `mapRef` for child layers
- [ ] T023 Create `frontend/src/map/WeatherLayer.tsx`: reads `activeVariable` + `currentHour` from store; calls `getFrame()` → `applyColorscale()` → `new ImageData(rgba, nLon, nLat)` → `map.addImage()` on mount, `map.updateImage()` on frame change; positions image over Myanmar bbox
- [ ] T024 Wire `WeatherLayer` inside `WeatherMap`

### Variable Switcher + Legend (US4)

- [ ] T025 Create `frontend/src/components/VariableSwitcher.tsx`: two-button toggle (🌡 Temperature | 🌧 Precipitation); calls `store.setVariable()`; active button has distinct visual state
- [ ] T026 Create `frontend/src/components/Legend.tsx`: reads `activeVariable` from store; renders gradient bar with tick labels — temp: 15/20/25/30/35/40°C; precip: 0/1/5/10/25/50 mm/h; precipitation tooltip with disclosure text from `metadata.variables.precipitation.temporal_disclosure`
- [ ] T027 Create `frontend/src/App.tsx`: layout — header area, full-viewport map, bottom controls (legend + variable switcher + timeline); wire all components
- [ ] T028 Create `frontend/src/components/DemoBanner.tsx`: fixed amber banner shown when `metadata.is_demo === true`

**Checkpoint**: `npm run dev` → Myanmar map visible, temperature overlay colored, legend shows °C. Switching to Precipitation changes overlay and legend to mm/h. Demo banner visible.

---

## Phase 4: User Story 2 — Timeline + Time Display (P1)

**Goal**: Slider steps through hours 0–168. Map, timestamp, and lead time update synchronously.

**Independent Test**: Drag slider to hour 48 → map changes, header shows init_time + 48h, lead time shows "+48 h".

- [ ] T029 Create `frontend/src/components/Header.tsx`: reads `metadata` from store; shows app title ("MYANMAR WEATHER / AI FORECAST"), model name, spatial resolution, init time (UTC); shows current valid time + lead time for `currentHour`
- [ ] T030 Create `frontend/src/components/Timeline.tsx`: `<input type="range" min=0 max=168 step=1>`; reads `currentHour` from store; calls `setHour()` on change; shows forecast date (day+month) and time (HH:00 UTC) for current hour; shows "+N h" lead time
- [ ] T031 Implement `store.stepForward()` and `store.stepBackward()` with clamping to [0, 168]
- [ ] T032 Add Prev (◀) and Next (▶) buttons to `Timeline.tsx`
- [ ] T033 Wire `Timeline` and `Header` into `App.tsx`

**Checkpoint**: Slider moves → map and time display update together. Prev/Next work. Boundary clamping works.

---

## Phase 5: User Story 3 — Playback Animation (P2)

**Goal**: Play button animates forecast. Speed controls change tick rate.

- [ ] T034 Add `isPlaying`, `playbackSpeed`, `togglePlay()`, `setSpeed()` to `ForecastStore.ts`
- [ ] T035 Implement animation loop in `App.tsx` or `Timeline.tsx`: `useEffect` with `setInterval` driven by `isPlaying` + `playbackSpeed`; advance `currentHour` by 1 per tick; loop back to 0 at 168; clear interval on unmount
- [ ] T036 Add Play/Pause (▶ / ⏸) button to `Timeline.tsx`
- [ ] T037 Add speed selector (0.5× / 1× / 2× / 4×) to `Timeline.tsx`

**Checkpoint**: Play animates through all 168 hours. Pause stops. Speed buttons change rate. Map stays synchronized.

---

## Phase 6: User Story 5 — Point Inspector (P2)

**Goal**: Click on Myanmar map → popup shows t2m + tp1h at nearest grid point.

- [ ] T038 Create `frontend/src/components/PointInspector.tsx`: popup showing lat/lon, valid time, t2m (°C, 1dp), tp1h (mm/h, 2dp), note about 0.25° grid resolution
- [ ] T039 Add click handler to `WeatherMap.tsx`: on click → compute nearest grid indices `(lat_i, lon_i)` by rounding to 0.25° grid; lookup both variables from Float32Array at `currentHour`; show `PointInspector` at click lngLat
- [ ] T040 Handle out-of-bounds click: if lngLat outside Myanmar bbox, show "Outside forecast domain"
- [ ] T041 Update popup values reactively when `currentHour` changes while popup is open

**Checkpoint**: Click within Myanmar bbox → popup with correct values. Moving slider → popup updates. Outside bbox → message shown.

---

## Phase 7: User Story 6 — Info Panel + Attribution (P3)

- [ ] T042 Create `frontend/src/components/InfoPanel.tsx`: modal/panel with all metadata from `forecast.json` — model name/version, resolution, init source, init time, valid range, temperature semantics (hourly), precipitation semantics (1-hour accumulation, mm/h disclosure), limitations (resolution, uncertainty, sic patch disclosure), attribution (Aurora/Microsoft, Earth2Studio/NVIDIA, ECMWF open data, OSM basemap)
- [ ] T043 Add Info (ℹ) button to `Header.tsx` that opens/closes `InfoPanel`
- [ ] T044 Verify `DemoBanner` (T028) is wired correctly; add clear language in InfoPanel when `is_demo: true`

**Checkpoint**: Info button opens panel with correct metadata. Demo banner visible in demo mode.

---

## Phase 8: Earth2Studio Pipeline (Production GPU Forecast)

**Goal**: Full Aurora1p5 + IFS pipeline producing real forecast artifacts.

- [ ] T045 Create `forecast/models/base.py`: `ForecastModel` abstract base class per plan.md interface
- [ ] T046 Create `forecast/sources/base.py`: `InitializationSource` abstract base class
- [ ] T047 Create `forecast/sources/ncar_era5_source.py`: `NCAR_ERA5Source` — wraps `earth2studio.data.NCAR_ERA5`; provides all 83 Aurora1p5 input variables including sic; for development/historical init times
- [ ] T048 Create `forecast/sources/ifs_source.py`: `IFSSource` — wraps `earth2studio.data.IFS`; handles `sic` gap via configurable method (`"zero"` or `"arco"`); returns `sic_handling` description string for metadata
- [ ] T049 Create `forecast/models/aurora1p5_forecast.py`: `Aurora1p5Forecast(ForecastModel)` — loads model weights via `Aurora1p5.load_default_package()`; runs `earth2studio.run.deterministic()` with `ZarrBackend`; extracts `t2m` and `tp1h` from output; returns xarray Dataset
- [ ] T050 Create `forecast/postprocessing/myanmar_subset.py`: `subset_myanmar(ds)` — `ds.sel(lat=slice(9.0, 29.0), lon=slice(92.0, 102.0))`; verifies resulting grid is 81×41
- [ ] T051 Create `scripts/generate_forecast.py`: CLI (`--init-time`, `--init-source`, `--sic-method`, `--output-dir`); orchestrates T047–T050 and T011–T012; writes to `data/forecast/`
- [ ] T052 Test `generate_forecast.py` with NCAR_ERA5 source and a historical init time (e.g., 2024-06-01T00:00:00Z)
- [ ] T053 Run `validate_forecast.py --data-dir data/forecast/` and verify all checks PASS

**Checkpoint**: `generate_forecast.py` produces valid artifacts for a historical init time. All validation PASS.

---

## Phase 9: Tests

- [ ] T054 [P] `tests/test_unit_conversion.py`: test `kelvin_to_celsius` (273.15K → 0°C, negative results for sub-freezing); test `tp1h_log_to_mm` (log(0.001) → ~1mm; negative input → 0 after clamp)
- [ ] T055 [P] `tests/test_myanmar_subset.py`: test `subset_myanmar` returns shape (n_time, 81, 41); test lat/lon bounds are correct; test out-of-bounds global data is clipped
- [ ] T056 [P] `tests/test_pipeline_interface.py`: test that `Aurora1p5Forecast` implements `ForecastModel` interface; test `IFSSource` implements `InitializationSource`; mock Earth2Studio models to avoid GPU in tests
- [ ] T057 [P] `tests/test_validation.py`: test `validate_forecast.py` catches NaN values (FAIL expected); catches negative tp1h (FAIL); catches wrong shape (FAIL); passes on valid demo data (PASS)
- [ ] T058 [P] `frontend/src/data/ForecastLoader.test.ts`: test `getFrame()` returns correct [81×41] slice for given hour; test frame 0 returns initialization data; test frame 168 returns last forecast
- [ ] T059 [P] `frontend/src/map/colorscales.test.ts`: test min value → first color; test max value → last color; test NaN → transparent (alpha=0)
- [ ] T060 Run `pytest tests/` → all pass; run `npm test` in `frontend/` → all pass

---

## Phase 10: Integration Testing

- [ ] T061 Copy `data/forecast/` artifacts (from T052) to `frontend/public/data/`
- [ ] T062 Run `npm run dev` with production artifacts; verify: correct timestamps, realistic t2m values, tp1h ≥ 0
- [ ] T063 Profile frame transitions in Chrome DevTools: confirm < 200ms per transition
- [ ] T064 Profile initial load on simulated 25 Mbps (Chrome Network throttle): confirm < 5s to interactive
- [ ] T065 Test on mobile viewport (375px): map usable, controls accessible

---

## Phase 11: GitHub Actions + Deployment

- [ ] T066 Create `.github/workflows/deploy-pages.yml` per plan.md spec
- [ ] T067 Verify `frontend/vite.config.ts` reads `VITE_BASE_PATH` and sets Vite `base` correctly
- [ ] T068 Enable GitHub Pages in repository settings: Source → GitHub Actions
- [ ] T069 Push to main; verify GitHub Actions workflow completes (green checkmark)
- [ ] T070 Open deployed GitHub Pages URL; verify: map loads, demo banner visible, timeline works, variable switcher works, info panel opens with correct metadata
- [ ] T071 Refresh the deployed page; verify app loads correctly (no broken routing)

---

## Phase 12: Documentation

- [ ] T072 Fill in `README.md`: Project Overview, Architecture diagram (ASCII), Model description (Aurora1p5 + IFS, tp1h semantics), Installation commands, Local Development commands, Forecast Generation commands (with GPU note), Data Format, Testing commands, Deployment guide, Limitations
- [ ] T073 Create `docs/architecture.md`: system diagram, component roles, data flow
- [ ] T074 [P] Create `docs/forecasting.md`: Aurora1p5 details, IFS initialization, sic gap, tp1h log untransform, hourly rollout mechanism, GPU requirements, cloud GPU options, model interface
- [ ] T075 [P] Create `docs/deployment.md`: Steps 1–19 per spec with exact commands and expected results
- [ ] T076 [P] Create `docs/data-format.md`: complete `forecast.json` schema, binary format spec, JavaScript and Python reading examples

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1** (Setup): Start immediately — no dependencies
- **Phase 2** (Demo Data): Depends on Phase 1 — **BLOCKS all frontend work**
- **Phase 3** (Map + Variable Switcher): Depends on Phase 2
- **Phase 4** (Timeline): Depends on Phase 3 map foundation
- **Phase 5** (Animation): Depends on Phase 4 store
- **Phase 6** (Point Inspector): Depends on Phase 3 map + Phase 2 data
- **Phase 7** (Info Panel): Depends on Phase 2 metadata + Phase 3 layout
- **Phase 8** (Pipeline): Depends on Phase 1; independent of frontend phases
- **Phase 9** (Tests): Write alongside implementation; run after Phase 8 for pipeline tests
- **Phase 10** (Integration): Depends on Phase 8 + Phase 5
- **Phase 11** (Deployment): Depends on all frontend phases + Phase 2 demo data
- **Phase 12** (Docs): Can start any time; finalize after Phase 11

### Parallel Tracks

Phase 8 (Earth2Studio pipeline) can run entirely in parallel with Phases 3–7 (frontend), as both depend only on Phase 2's locked data contract.

### MVP Path (Phases 1–4 + 11)

Complete Phases 1 → 2 → 3 → 4 → 11 for a working deployed GitHub Pages app with:
- Myanmar weather map
- Temperature + precipitation overlay
- Hourly timeline navigation
- Demo data with DEMO banner

**STOP and VALIDATE** before continuing to Phase 5+.

---

## Acceptance Criteria Verification Map

| Criterion | Tasks |
|-----------|-------|
| Earth2Studio actually used | T049–T052 |
| Myanmar forecast generated | T050, T052 |
| 7-day forecast produced | T049 |
| Hourly frames native (no interpolation) | T049 (Aurora1p5 native rollout) |
| Temperature available | T049, T050 |
| Precipitation available (native tp1h) | T049, T050 |
| Units correct (°C, mm/h) | T012, T054 |
| tp1h log untransform applied | T012, T054 |
| Timestamps correct | T015, T057 |
| Resolution documented | T029, T042 |
| Myanmar map visible | T022 |
| Zoom/pan works | T022 |
| Weather overlay works | T023–T024 |
| Legend works | T026 |
| Location inspector | T038–T041 |
| Hourly slider | T030–T031 |
| Prev/Next buttons | T032 |
| Play/Pause | T035–T036 |
| Timestamp updates | T030 |
| Map updates with slider | T023 (reactive) |
| Lead time updates | T030 |
| Precipitation mode | T025–T026 |
| Temperature mode | T025–T026 |
| Precip tooltip disclosure | T026, T033 (FR-033) |
| TypeScript strict | T006 |
| Python ruff | T002 |
| Tests exist | T054–T059 |
| Forecast validation | T015–T016 |
| Demo data | T013–T017 |
| sic gap handled + documented | T048 |
| Production path documented | T072–T074 |
| No secrets committed | T007 |
| README complete | T072 |
| GitHub Pages works | T069–T071 |
| GitHub Actions works | T069 |
| Refresh doesn't break | T071 |
