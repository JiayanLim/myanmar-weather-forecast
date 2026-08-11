# Tasks: Myanmar Weather Forecast Web Application

**Input**: `specs/001-myanmar-weather-app/` (spec.md v3, plan.md v3, research.md v3)
**Feature**: 001-myanmar-weather-app
**Revised**: 2026-08-11 v3 — 48h + temperature + M4 CPU validated

**Key architectural facts (validated)**:
- GraphCastSmall (Earth2Studio 0.17.0) natively produces tp06 and t2m — no diagnostic model
- ARCO (dev) or IFS (prod) are the only verified init sources; NCAR_ERA5 and GFS are out
- GraphCastSmall requires t-6h AND t+0h at initialization — two timesteps fetched automatically
- tp06: metres × 1000 → mm/6h, clamp ≥ 0, t+0h=0; NO log/exp transform
- t2m: K − 273.15 → °C; NO log/exp transform
- Binary format: [9 × 21 × 11] float32 per variable (8,316 bytes each)
- Timeline: 9 steps at 0, 6, 12, 18, 24, 30, 36, 42, 48h
- Myanmar at 1.0°: lat 9–29°N (21 pts), lon 92–102°E (11 pts)
- Hardware: Apple M4 CPU, JAX XLA ARM64; ~78s end-to-end; ~2.34 GB peak RSS
- Schema: v3.0 (two variable binaries + forecast.json)

---

## Phase 1: Spec Kit Update

**Status**: COMPLETE (2026-08-11 v3 — updated for 48h+temperature)

- [x] T001 Update `.specify/memory/constitution.md` v1.0.1 → v2.0.0 (GraphCastSmall, 6h steps, tp06)
- [x] T002 Update `specs/001-myanmar-weather-app/research.md` — v3: ADR-004 48h/9 frames, ADR-006 schema v3.0, ADR-007 M4 CPU validated
- [x] T003 Update `specs/001-myanmar-weather-app/spec.md` — v3: 48h/9 frames, temperature, schema v3.0, M4 CPU
- [x] T004 Update `specs/001-myanmar-weather-app/plan.md` — v3: 48h+temp, M4 pipeline, validated status
- [x] T005 Update `specs/001-myanmar-weather-app/tasks.md` (this file)

---

## Phase 2: Pipeline Rewrite

**Status**: COMPLETE (2026-08-11) — Validated on Apple M4 CPU

- [x] T010 Rewrite `scripts/generate_forecast.py`:
  - Uses `earth2studio.models.px.GraphCastSmall`
  - Default source: ARCO (historical); flag `--source [arco|ifs]`
  - Two-timestep init handled by Earth2Studio via e2run.deterministic
  - nsteps=8 (GC_N_STEPS), GC_N_FRAMES=9 (t+0h through t+48h)
  - Extracts tp06 from zarr output; applies ×1000 (no exp); clamps ≥ 0; t+0h=0
  - Extracts t2m from zarr output; converts K → °C (K − 273.15)
  - Subsets to Myanmar bbox: lat 9–29°N, lon 92–102°E (21×11 at 1.0°)
  - Sorts lat ascending (south→north) via np.argsort on matched lat indices
  - Writes `precipitation.bin` [9×21×11] float32 C-order
  - Writes `temperature.bin` [9×21×11] float32 C-order
  - Writes `forecast.json` with schema v3.0
  - Records M4 CPU device, JAX backend, peak RSS in inference_config
  - JAX env vars set before imports: JAX_PLATFORM_NAME=cpu, XLA_PYTHON_CLIENT_PREALLOCATE=false
  - psutil.Process().memory_info().rss used for peak RSS tracking

- [x] T011 Hardware validation (M4 CPU):
  - JAX CPU (XLA ARM64) confirmed working — no GPU required
  - 8-step inference: ~54s; full pipeline: ~78s; peak RSS: 2.34 GB
  - MPS backend is incompatible (float64/SIGABRT); JAX CPU is the correct M4 backend
  - VRAM staging test (T4 GPU) is NOT applicable; CPU RSS tracking is the validated metric

- [x] T012 Update `frontend/src/data/types.ts`:
  - `ActiveVariable = 'precipitation' | 'temperature'`
  - `variables: { precipitation: VariableMeta; temperature: VariableMeta; }` (both required)
  - `inference_config?: { device: string; jax_backend?: string; rss_peak_gb?: number | null; ... }`
  - `accumulation_period_hours?: number` (optional, precipitation only)
  - Removed `peak_vram_gb` (no GPU used)

**Checkpoint**: generate_forecast.py with `--source arco --init-time 2022-07-01T00:00:00Z` produces:
- `data/forecast/precipitation.bin` (8,316 bytes) ✓
- `data/forecast/temperature.bin` (8,316 bytes) ✓
- `data/forecast/forecast.json` (4,134 bytes, schema v3.0) ✓
- Validation: 25/25 PASS ✓

---

## Phase 3: Demo Data Update

**Status**: COMPLETE (2026-08-11)

- [x] T013 Rewrite `scripts/generate_demo_data.py`:
  - N_FRAMES = 9, STEP_HOURS = 6 → 48h horizon
  - N_LAT = 21, N_LON = 11 (Myanmar at 1.0°)
  - Generate precipitation: Gaussian blobs, range 0–98 mm/6h, frame_scales for rain progression
  - Generate temperature: latitudinal gradient + diurnal cycle, range ~23–42°C
  - Write `precipitation.bin` [9 × 21 × 11] float32
  - Write `temperature.bin` [9 × 21 × 11] float32
  - Write `forecast.json` with schema v3.0, both variables, is_demo=true
  - inference_config.device = "DEMO (no inference)"

- [x] T014 Run `scripts/generate_demo_data.py` and verify `data/demo/`:
  - `data/demo/forecast.json` (3,829 bytes, schema v3.0, is_demo=true) ✓
  - `data/demo/precipitation.bin` (8,316 bytes) ✓
  - `data/demo/temperature.bin` (8,316 bytes) ✓
  - Verified ranges: precip [0, 97.7] mm/6h; temp [23.4, 41.3] °C ✓

- [x] T015 Demo artifacts committed in git (in `data/demo/`) ✓

**Checkpoint**: `data/demo/` has 3 files (8,316 bytes each for binaries). `forecast.json` has
`n_times: 9`, `schema_version: "3.0"`, `is_demo: true`. All validated by validate_forecast.py. ✓

---

## Phase 4: Frontend Migration (24h → 48h + Temperature)

**Status**: COMPLETE (2026-08-11)

### Variable infrastructure

- [x] T016 `frontend/src/data/types.ts` — Added `ActiveVariable`, temperature to variables, fixed inference_config
- [x] T017 `frontend/src/data/ForecastLoader.ts` — Loads both binaries via Promise.all using metadata.variables.*.file
- [x] T018 `frontend/src/data/ForecastStore.ts` — Added temperature, activeVariable, setVariable(); default '?? 9'

### Variable switcher and map

- [x] T019 `frontend/src/components/VariableSwitcher.tsx` — Full rewrite: Precip/Temp toggle buttons
- [x] T020 `frontend/src/map/WeatherMap.tsx` — Variable-aware rendering (TEMP_LUT vs PRECIP_LUT_ALPHA); dual-variable popup

### Timeline

- [x] T021 `frontend/src/components/Timeline.tsx`:
  - Slider max from `metadata?.n_times ?? 9`
  - `metadata?.n_times ?? 9` fallback (not 5)
  - Dynamic hour markers: `0h · 12h · 24h · 36h · 48h` (every other frame)
  - Lead time: `currentHour * (metadata?.native_timestep_hours ?? 6)`

### Legend, Header, InfoPanel

- [x] T022 `frontend/src/components/Legend.tsx` — Variable-aware (precip/temp LUT, labels, units, ticks)
- [x] T023 `frontend/src/components/Header.tsx` — "Myanmar 48h Weather Forecast"; staleness from `forecast_horizon_hours`
- [x] T024 `frontend/src/components/InfoPanel.tsx` — Both variables shown; horizon from metadata; no hardcoded 24h

### App composition

- [x] T025 `frontend/src/App.tsx` — VariableSwitcher imported; `setData(d.metadata, d.precipitation, d.temperature)`

### Demo data for dev server

- [x] T026 Copy schema v3.0 demo artifacts to `frontend/public/data/`:
  - `forecast.json` (3,829 bytes) ✓
  - `precipitation.bin` (8,316 bytes) ✓
  - `temperature.bin` (8,316 bytes) ✓

**Checkpoint (completed)**:
- `npx tsc --noEmit`: 0 errors ✓
- `npm run build`: Success (1.41s) ✓
- Dev server: schema v3.0, n_times=9, is_demo=true ✓
- Stale term grep (`24h|tp1h|mm.*1h|Aurora|schema.*2.0|5-frame`): 0 matches ✓
- Binary shapes: precip (9,21,11) range [0, 97.7]; temp (9,21,11) range [23.4, 41.3] ✓

---

## Phase 5: Validation Script Update

**Status**: COMPLETE (2026-08-11)

- [x] T027 Rewrite `scripts/validate_forecast.py`:
  - Schema v3.0 checks
  - EXPECTED_N_TIMES = 9, EXPECTED_HORIZON_H = 48, EXPECTED_TIMESTEP_H = 6
  - Checks 3 required files (forecast.json, precipitation.bin, temperature.bin)
  - temperature.bin checks: range [-90°C, 70°C], no NaN
  - precipitation.bin checks: ≥0, <500 mm/6h, no NaN
  - Variable metadata checks for both temperature and precipitation
  - log_transform_applied + exp_transform_applied checks for tp06
  - 25 total checks

- [x] T028 Run validate_forecast.py on both data directories:
  - `data/demo/`: validated ✓
  - `data/forecast/`: 25/25 PASS ✓ (real M4 forecast, init 2022-07-01T00:00:00Z)

**Checkpoint**: 25/25 PASS for both demo and real forecast data. ✓

---

## Phase 6: Local Pipeline Notebook

**Status**: COMPLETE (2026-08-11)

Note: The original `aurora_myanmar_forecast.ipynb` was renamed to `graphcast_myanmar_forecast.ipynb`
in an earlier migration. The notebook is now rewritten around the validated local M4 pipeline.

- [x] T030 `notebooks/graphcast_myanmar_forecast.ipynb` renamed (git mv from aurora notebook)

- [x] T031 Rewrite notebook for local M4 pipeline:
  - Section 0: Overview (GraphCastSmall, tp06, t2m, 48h, schema v3.0, M4 CPU)
  - Section 1: Environment setup (uv, pyproject.toml, xarray<2026 pin)
  - Section 2: Apple M4 hardware detection (JAX_PLATFORM_NAME=cpu, psutil for RSS)
  - Section 3: ARCO data source setup (or IFS for near-real-time)
  - Section 4: GraphCastSmall checkpoint load with RSS tracking
  - Section 5: 6h smoke test (nsteps=1) — OOM/crash → stop
  - Section 6: Full 48h inference (nsteps=8) — ~78s
  - Section 7: tp06 post-processing → mm/6h (×1000, clamp ≥ 0, t+0h=0)
  - Section 8: t2m post-processing → °C (K − 273.15)
  - Section 9: Myanmar subset [9×21×11]
  - Section 10: Artifact generation (forecast.json, temperature.bin, precipitation.bin)
  - Section 11: validate_forecast.py (25/25 PASS gate)
  - Section 12: Provenance report
  - Section 13: Optional git commit + push
  - No Colab/T4/A100 as primary path (documented as alternative only)
  - No cudf monkey-patch (not applicable for local Python environment)
  - xarray<2026 constraint documented explicitly

- [ ] T032 Verify notebook runs top-to-bottom on a fresh Python environment on M4 with
  `uv run jupyter nbconvert --to notebook --execute ...`
  (Optional: can be verified manually during next forecast regeneration)

**Checkpoint**: Notebook documents the complete local M4 pipeline. All sections use
`generate_forecast.py` constants (GC_N_STEPS=8, GC_N_FRAMES=9, schema v3.0). ✓

---

## Phase 7: Integration Testing

- [x] T040 Dev server test with demo data:
  - 9-frame timeline with 0h·12h·24h·36h·48h markers ✓
  - Precipitation overlay renders on Myanmar map ✓
  - Slider max = 8; steps advance correctly ✓
  - Header shows "Myanmar 48h Weather Forecast" ✓
  - Variable switcher (Precip/Temp) toggles map + legend + popup ✓
  - InfoPanel shows both variables with correct metadata ✓
  - DEMO DATA banner visible (is_demo=true) ✓

- [x] T041 `npm run build` in `frontend/`: 0 TypeScript errors, built in 1.41s ✓

- [ ] T042 Profile frame transitions in browser DevTools: confirm < 200ms
  (Pending — no blockers identified; data payload is 16.6 KB with all frames in memory)

- [ ] T043 End-to-end test with REAL forecast data:
  - Copy `data/forecast/` artifacts to `frontend/public/data/`
  - Run `npm run build`
  - Verify: schema v3.0 loads, 9 frames, is_demo=false, no DEMO banner
  - Verify: temperature range [1.94, 35.35] °C, precip range [0, 17.38] mm/6h
  - Status: IN PROGRESS (this session)

- [ ] T044 Push to main; confirm GitHub Actions green; GitHub Pages loads with real forecast
  - Verify: is_demo=false (no DEMO banner in production)
  - Verify: Header shows real init time (2022-07-01T00:00:00Z)
  - Verify: Temperature toggle shows plausible Myanmar monsoon temperatures

---

## Phase 8: README Update

- [x] T050 Update `README.md`:
  - Architecture: GraphCastSmall 1.0°, 6h, 48h, tp06+t2m, M4 CPU (~78s)
  - Remove Aurora1p5 references from setup instructions
  - Update pipeline: ARCO/IFS, two-timestep init, tp06 → mm/6h, t2m → °C
  - Data format: [9 × 21 × 11], schema v3.0, two binaries
  - Hardware: Apple M4 CPU validated; GPU optional (accelerates inference)

---

## Dependencies & Execution Order

- **Phases 1–5**: COMPLETE
- **Phase 6**: COMPLETE (T031 done; T032 optional verification)
- **Phase 7**: T040, T041 COMPLETE; T042, T043 pending (T043 in this session)
- **Phase 8**: COMPLETE

### Critical Path to GitHub Pages Deployment

T043 (local build with real data) → T044 (push to main) → GitHub Actions → Pages

---

## Acceptance Criteria Verification Map

| Criterion | Tasks | Status |
|-----------|-------|--------|
| Earth2Studio actually used (GraphCastSmall) | T010 | ✓ DONE |
| Myanmar 48h forecast generated | T010 | ✓ DONE |
| 6h native steps (no interpolation) | T010, T021 | ✓ DONE |
| tp06 precipitation available | T010 | ✓ DONE |
| t2m temperature available | T010 | ✓ DONE |
| No log/exp transform applied | T010, T027 | ✓ DONE |
| Units correct (mm/6h, °C) | T010, T013 | ✓ DONE |
| Two-timestep init (t-6h + t+0h) | T010 | ✓ DONE |
| Hardware validated (M4 CPU, ~78s, 2.34 GB) | T011 | ✓ DONE |
| Timestamps correct (6h spacing) | T013, T027 | ✓ DONE |
| Resolution documented (1.0°) | T016, T024 | ✓ DONE |
| Myanmar map visible | T020 | ✓ DONE |
| Precipitation overlay works | T020 | ✓ DONE |
| Temperature overlay works | T020 | ✓ DONE |
| Variable switcher (Precip/Temp) | T019 | ✓ DONE |
| Legend correct (mm/6h and °C) | T022 | ✓ DONE |
| 9-step slider (0h…48h) | T021 | ✓ DONE |
| Lead time display correct | T021 | ✓ DONE |
| Play/Pause animation | T021 | ✓ DONE |
| Point inspector (popup, both vars) | T020 | ✓ DONE |
| tp06 disclosure in UI | T024 | ✓ DONE |
| Header: 48h title | T023 | ✓ DONE |
| InfoPanel: 6h accumulation disclosure | T024 | ✓ DONE |
| Resolution honesty disclosure | T024 | ✓ DONE |
| TypeScript strict (no errors) | T041 | ✓ DONE |
| Demo data correct format (schema v3.0) | T013–T015 | ✓ DONE |
| Validation 25/25 PASS | T028 | ✓ DONE |
| Local notebook for M4 pipeline | T031 | ✓ DONE |
| Frame transition < 200ms | T042 | pending |
| GitHub Pages deployment | T044 | pending |
| README updated | T050 | ✓ DONE |
| No secrets committed | (existing .gitignore) | ✓ |
