# Implementation Plan: Myanmar Weather Forecast Web Application

**Branch**: `001-myanmar-weather-app`
**Date**: 2026-08-12 (v4 — new target: 7-day / 4-variable / ERA5 2021-01-01 / model TBD)
**Revised**: 2026-08-16 (v5 — Spec Kit update phase RS; R4 timings corrected; phases added)
**Revised**: 2026-08-17 (v6 — Phase R11 superseded; Phase R12 added; ADR-025)
**Revised**: 2026-08-17 (v7 — Phase R13 precip sqrt scale; Phase R14 temperature context)
**Revised**: 2026-08-17 (v8 — Phase R15 Model Eval contextualization; Phase R16 Wind UI consolidation)
**Revised**: 2026-08-19 (v9 — R15/R16 marked COMPLETE; Phase R17 Wind UI simplification + header cleanup added)
**Spec**: `specs/001-myanmar-weather-app/spec.md`
**Research**: `specs/001-myanmar-weather-app/research.md`
**Constitution**: `.specify/memory/constitution.md` v3.0.0

---

## Document Purpose

This plan describes:

1. **Legacy Baseline** (completed, deployed): The GraphCastSmall/48h/2-variable/M4 CPU
   implementation. Do NOT modify this section to mark phases as incomplete.

2. **New Target Plan** (research phase, not started): A new 9-phase plan for the
   7-day / 4-variable / ERA5 init / model-TBD architecture.

---

## PART A — LEGACY IMPLEMENTATION (v3.0, COMPLETED AND DEPLOYED)

All phases below were completed on 2026-08-11 and the application is live on GitHub Pages.

### Phase 1: Spec Kit Update — COMPLETE (2026-08-11)

Updated constitution, research, spec, plan, tasks for GraphCastSmall/48h/temperature.

### Phase 2: Pipeline Rewrite — COMPLETE (2026-08-11)

`generate_forecast.py`: GraphCastSmall + ARCO/IFS, 48h, 9 frames, tp06+t2m, M4 CPU.
Validated: 2022-07-01T00:00:00Z init, ~78s, 2.34 GB RSS, 25/25 validation checks passed.

### Phase 3: Demo Data Update — COMPLETE (2026-08-11)

`generate_demo_data.py`: 9 frames, both variables, schema v3.0, is_demo=true.

### Phase 4: Frontend Migration (24h → 48h + Temperature) — COMPLETE (2026-08-11)

Variable switcher, variable-aware map, legend, info panel, timeline, header.
VariableSwitcher.tsx fixed for Zustand 5 / React #185 (commit 5041b07, 2026-08-12).

### Phase 5: Validation Script Update — COMPLETE (2026-08-11)

`validate_forecast.py`: schema v3.0, 9 frames, both variables, 25 checks.

### Phase 6: Local M4 Pipeline Notebook — COMPLETE (2026-08-11)

`notebooks/graphcast_myanmar_forecast.ipynb`: documents local M4 inference end-to-end.

### Phase 7: Integration Testing — COMPLETE (2026-08-11/12)

T040, T041, T044 complete. React #185 fixed. Live on GitHub Pages with is_demo=false.

### Phase 8: README Update — COMPLETE (2026-08-11)

README updated for GraphCastSmall/48h/M4.

---

## PART B — NEW TARGET PLAN (v4.0 — 2026-08-17)

**Status**: Phases 1–5 COMPLETE. Phase RS COMPLETE. R4/R5/R6/R9 COMPLETE (2026-08-17). RS10 COMPLETE. RS11–RS14 DEFERRED (ADR-024). R10 COMPLETE (commit 2ee8c60). R11 COMPLETE (commit 2ee8c60) — SUPERSEDED by R12 (distribution defect + HSL visualization). R12 IN PROGRESS. Live on GitHub Pages (commit 2ee8c60).

**Model**: GraphCastOperational (0.25°, JAX/Haiku, ARCO/ERA5 init)
**R3 result**: PASS — M4 24 GB, peak RSS 1.99 GB post-compile, JIT cold ~27–34 min
**R4 performance (actual)**: 18.7h total; 25–34 min/step post-JIT; step 10 anomaly ~80 min (thermal)
**R5 result**: PASS — exit 0; verification.json v2.0 written; ADR-023 closed.
**Model selection gate**: PASSED — GraphCastOperational approved by user (ADR-012/018).

---

### Phase 1: Model and Data Research — COMPLETE (2026-08-12)

**Objective**: Select the Earth2Studio model for the 7-day / 4-variable / ERA5 2021-01-01
forecast. Produce ADR-012 (see research.md) with a fully documented model selection.

**Inputs**:
- Earth2Studio model registry: `earth2studio.models.px.*`
- Earth2Studio documentation and source code
- Research.md ADR-008 through ADR-016

**Tasks**:
- Enumerate all models in `earth2studio.models.px` and `earth2studio.models.dx`
- For each candidate: inspect `input_coords`, `output_coords`, supported variables
- Verify presence of: precipitation variable, u-wind, v-wind, temperature
- Document native timestep, native resolution, initialization requirements for each
- Verify ARCO compatibility (can ARCO provide all required init variables for 2021-01-01?)
- Document compute requirements (RAM, VRAM) for each candidate
- Evaluate 7-day forecast skill (academic literature, Earth2Studio model cards)
- Produce a model comparison table
- Select one model with documented rationale
- Obtain explicit user approval of model selection

**Gate**: Model selected, documented in research.md ADR-012 (Updated), approved.

**Deliverables**:
- Updated ADR-012 in research.md with "Selected model" filled in
- Model variable specification table (input variables, output variables, units)
- Hardware feasibility assessment

**Do NOT modify any application code during Phase 1.**

---

### Phase 2: ERA5 Data Availability and Initialization Validation — COMPLETE (2026-08-12)

**Prerequisite**: Phase 1 gate passed (model selected).

**Objective**: Verify that ERA5 via ARCO provides all required input variables for
the selected model at 2021-01-01T00:00:00Z (and any required preceding timesteps).

**Tasks**:
- Write a short script to fetch required init variables from ARCO for 2021-01-01
- Confirm all model input_coords variables are available and non-NaN
- If multi-step init required (e.g., t−6h AND t+0h), verify all required timesteps
- Document any missing or problematic variables
- If ARCO cannot provide required variables: report to user and reassess

**Gate**: Confirmed that ARCO can provide a complete, valid initialization for
the selected model at 2021-01-01T00:00:00Z.

**Deliverables**:
- Short validation script or notebook output confirming data availability
- List of confirmed available variables

**Do NOT run full inference until Phase 3.**

---

### Phase 3: Minimal Model Inference / Hardware Feasibility — COMPLETE / PASS (2026-08-12)

**Prerequisite**: Phase 2 gate passed (data availability confirmed).

**Objective**: Run a 1-step (smoke test) inference to confirm the selected model
runs on available hardware without OOM or assertion errors.

**Result**: PASS — ADR-018 in research.md.

| Measurement | Value |
|---|---|
| Model | GraphCastOperational (0.25°, 721×1440) |
| Hardware | Apple M4 MacBook Air 24 GB |
| JAX backend | cpu (XLA ARM64) |
| nsteps=1 time | 2081s = 34.7 min (JIT dominates on cold start) |
| Peak RSS (compilation) | ~5.0 GB |
| RSS post-inference | 1.99 GB |
| OOM / crash | None |
| tp06 in output | ✓ metres/6h, max 0.0872 m = 14.5 mm/hr |
| t2m in output | ✓ Kelvin, 221–316 K → -52 to +43°C |
| u10m in output | ✓ m/s |
| v10m in output | ✓ m/s |

**R4 measured per-step timing (2026-08-16, 29-step run)**:
- Step 2 (JIT + 1st forward): 27:14 total elapsed
- Steps 3–9 (post-JIT): 24–26 min each (stable; ~25 min mean)
- Estimated 7-day total: ~27 min JIT + 28 × 25 min ≈ **12–14 hours**
- Note: First aborted run (2026-08-13) showed 35–48 min/step — attributed to thermal
  throttling or background load. Current run on a quieter machine is significantly faster.

**Gate**: PASSED. R4 approved by user.

**Deliverables**:
- ADR-018 in research.md (full hardware feasibility results)
- Variable specification confirmed: tp06 (metres/6h), t2m (K), u10m/v10m (m/s)

---

### Phase RS: Spec Kit Update — IN PROGRESS (2026-08-16)

**Prerequisite**: R3 PASS; R4 pipeline running; user approves planning output.

**Objective**: Update all Spec Kit files to reflect confirmed GCOp architecture.
Lock schema v4.0, ADR-019–022. No application code modified.

**Tasks**: RS01–RS05 (see tasks.md)

**Gate**: All five Spec Kit files updated and reviewed by user.

---

### Phase 4: 7-Day Four-Variable Forecast Pipeline — COMPLETE (2026-08-17)

**Prerequisite**: Phase 3 gate passed (smoke test succeeded). Phase RS approved.

**Objective**: Produce validated 7-day GCOp forecast artifacts in data/forecast_v4/.

**Result**: PASS

- Pipeline: 12:33 SGT 2026-08-16 → 07:15 SGT 2026-08-17 (~18.7h total; caffeinate active)
- Final step timing: JIT 27 min (step 2); steps 3–29: 25–34 min/step; step 10 anomaly ~80 min
- Total peak RSS: 3.13 GB (during 28-step run); post-inference: 2.06 GB (per forecast.json)
- JAX cache: configured but remained empty (warm-start not achieved; cold JIT on every run)

**R044 validation**: PASS — 0 errors, all 4 variables at 385,236 bytes each

| Stat | Value |
|---|---|
| Precipitation (mm/hr) | min=0.000, median=0.0003, P95=0.084, max=1.620 |
| Temperature (°C) | min=−26.88, median=21.30, P95=27.92, max=33.18 |
| Wind Speed (kt) | min=0.005, median=3.10, P95=12.72, max=24.20 |
| Wind Direction (°FROM) | min=0.003, median=83.40, P95=339.53, max=359.99 |

**RS10**: generate_forecast.py corrected to emit `schema_version: "4.0"` (was "5.0").

**Gate**: PASSED — data/forecast_v4/ validated; RS10 complete.

**Deliverables**:
- `data/forecast_v4/` — 4 binaries × 385,236 bytes + forecast.json (8,063 bytes); is_demo=false
- `scripts/generate_forecast.py` — corrected to emit schema v4.0
- `scripts/validate_forecast.py` — schema v4.0/v5.0, 4 variables, 29 frames
- `data/demo/` — contains stale v3.0 artifacts; RS11–RS14 DEFERRED per ADR-024 (not on critical path; production serves data/forecast_v4/)

---

### Phase 5: ERA5 Evaluation Pipeline — COMPLETE (2026-08-17)

**Prerequisite**: Phase 4 gate passed (7-day forecast produced and validated).

**Objective**: Compare the 2021-01-01 7-day forecast against ERA5 analysis.

**Result**: PASS — exit 0; verification.json schema v2.0 written.

**ARCO investigation (pre-implementation, 2026-08-16)**:
- Confirmed ERA5 `total_precipitation` = 1-hour accumulation ending at each timestamp
- Not cumulative; no seam handling at 00Z/12Z; each hourly value independent
- 11.73% negative values in dry season (spectral artifacts, clamped to 0)
- Grid alignment: 0.25° exact match — lat_err=0°, lon_err=0° (no interpolation needed)
- Results locked in ADR-023

**R5 Verification Results — 2021-01-01 GCOp vs ERA5**:

| Variable | MAE | RMSE | Bias |
|---|---|---|---|
| Temperature (29 frames) | 1.3137°C | 1.6975°C | −0.7595°C |
| Wind speed (29 frames) | 1.0548 kt | 1.4027 kt | −0.3911 kt |
| Wind direction (29 frames) | 16.9113° circ. MAE | — | — |
| Precipitation (28 frames) | 0.0172 mm/hr | 0.0645 | +0.0110 |

Precipitation categorical (0.1 mm/hr threshold; totals from 28 frames × 3,321 points):
POD=0.6343 FAR=0.7279 CSI=0.2352 (1,148 hits / 662 misses / 3,071 false alarms)

**Gate**: PASSED — all checks pass; caveats documented in ADR-023 and verification.json.

**Deliverables**:
- `scripts/verify_forecast.py` — fully rewritten for GCOp, 4 variables, schema v2.0
- `data/verification/verification.json` — schema v2.0, 31,482 bytes

---

### Phase 6: Schema v4.0 Full Frontend Migration — COMPLETE (2026-08-17)

**Prerequisite**: Phase 4 gate passed; Phase RS complete.

**Objective**: Migrate all frontend layers (types, loading, store, rendering, UI) to schema v4.0.
Phases 6, 7, and 8 were completed in a single unified R6 authorization on 2026-08-17.

**Files changed**:
- `frontend/src/data/types.ts`: schema v4.0; 4-variable ActiveVariable; n_frames; native_resolution_deg
- `frontend/src/data/ForecastLoader.ts`: 4-binary Promise.all (ADR-022)
- `frontend/src/data/ForecastStore.ts`: windSpeed + windDirection arrays; n_frames fallbacks
- `frontend/src/App.tsx`: setData passes all 4 arrays
- `frontend/src/map/colorscales.ts`: MODEL_STEP removed (ADR-019); renderWithInterpolation(modelStep);
  WIND_LUT_ALPHA; renderWindDirectionWithInterpolation (vector-component bilinear, ADR-020)
- `frontend/src/map/WeatherMap.tsx`: all 4 variables; modelStep=metadata.native_resolution_deg; popup
- `frontend/src/components/VariableSwitcher.tsx`: 4 buttons
- `frontend/src/components/Legend.tsx`: precipitation mm/hr (0–10); wind speed kt (0–30); wind dir hue
- `frontend/src/components/InfoPanel.tsx`: display_unit/conversion field names; 4 variables
- `frontend/src/components/Header.tsx`: native_resolution_deg; optional model_version
- `frontend/src/components/Timeline.tsx`: n_frames
- `frontend/src/components/ModelEvaluation.tsx`: schema v2.0; all 4 variables; GCOp label
- `frontend/public/data/`: 5 v4 forecast files + verification.json for dev mode

**Validation results**:
- `npx tsc --noEmit` → 0 errors
- `npm run build` → ✓ 1.50s
- All 4 binaries: 385,236 bytes each ✓
- No MODEL_STEP=1.0 hardcoded; no n_times; no spatial_resolution_deg in active layer
- No raw wind-direction degree interpolation

**Gate**: PASSED — TypeScript 0 errors; build passes; all 4 variables integrated.

---

### Phase 7: Frontend Four-Variable Migration — COMPLETE (merged into Phase 6, 2026-08-17)

See Phase 6.

---

### Phase 8: Evaluation / Accuracy UI — COMPLETE (merged into Phase 6, 2026-08-17)

See Phase 6. ModelEvaluation.tsx rewritten for verification.json schema v2.0.

---

### Phase 8: Evaluation / Accuracy UI — FUTURE (NOT STARTED)

**Prerequisite**: Phase 5 (verification.json) and Phase 7 (frontend migration) complete.

**Objective**: Implement the Model Evaluation popup in the frontend.

**Tasks**:
- ModelEvaluation.tsx: fetches verification.json; displays 4-variable skill tables
- Section header: "Historical Model Evaluation" (NOT "this forecast's accuracy")
- Tables: temperature MAE/RMSE/Bias; precipitation MAE/RMSE/Bias/POD/FAR/CSI;
  wind speed MAE/RMSE; wind direction circular MAE
- Caveats prominently displayed
- Button in Header to open popup
- Graceful fallback if verification.json is absent

**Gate**: TypeScript 0 errors; popup renders correctly; caveats clearly visible.

---

### Phase 9: Deployment — NOT STARTED

**Prerequisite**: All phases 1–8 complete; user gives explicit approval to push.

**Objective**: Deploy the new 7-day / 4-variable forecast to GitHub Pages.

**Tasks**:
- Update `.github/workflows/deploy-pages.yml` for schema v4.0 (4 artifact files)
- Commit all Spec Kit updates, pipeline scripts, and demo data
- Run validate_forecast.py on all artifacts
- TypeScript check and production build
- Push to main
- Monitor GitHub Actions
- Verify live GitHub Pages: model, init time, 4 variables, 7-day timeline
- Confirm is_demo=false on production

**Gate**: Live site shows correct model, 2021-01-01 init, 7-day timeline, 4 variables, no errors.

---

### Phase R10: Precipitation Color Scale Calibration — COMPLETE (superseded by R13)

**Classification**: Implementation defect fix (display calibration).
**Prerequisite**: Binary spot-check passed (confirmed 2026-08-17 — see pre-R10 analysis).
**Scope**: `frontend/src/map/colorscales.ts` and `frontend/src/components/Legend.tsx` only.
No data files, verification.json, or backend scripts are modified.

**Objective**: Make the precipitation field visible for the January 2021 dry-season dataset.
The existing PRECIP_MAX=10 mm/hr and transparency ramp suppress 95% of the field (P95=0.084 mm/hr < old 0.2 mm/hr cutoff). After calibration, 12.2% of the field will be visible (values > 0.02 mm/hr).

**Changes** (per FR-N23a):
- `PRECIP_MAX`: 10 → 2 mm/hr
- Alpha ramp: `norm < 0.02 → 0` threshold reduces to `norm < 0.01` (= 0.02 mm/hr at new scale)
- `PRECIP_TICKS`: [0, 0.5, 1, 2, 5, 10] → [0, 0.1, 0.25, 0.5, 1.0, 2.0] mm/hr
- Legend label: add "dry-season calibration" note (per FR-N23a)

**Validation**:
- tsc 0 errors
- npm run build passes
- P95 (0.084 mm/hr) visible
- Max (1.62 mm/hr) distinguishable
- Values ≤ 0.02 mm/hr fully transparent

**Gate**: tsc 0 errors; build passes; all visual criteria met; commit to main + push.

---

### Phase R11: Wind Vector Arrow Overlay — COMPLETE / SUPERSEDED (2026-08-17)

**Status**: SUPERSEDED by Phase R12 (ADR-025).

Commit 2ee8c60 implemented `drawWindArrows()` in `colorscales.ts` and integrated it into
`WeatherMap.tsx`. TypeScript 0 errors; build passed; direction sanity tests passed. Deployed.

**Two defects confirmed post-deployment**:

1. **Distribution defect** — arrows appeared predominantly at the bottom (southern Myanmar).
   Root cause: `if (len < 2) continue` filter combined with `maxLen ≈ 6.77 intrinsic px`.
   Effective draw threshold ≈ 8.87 kt (≈P75). January median = 3.10 kt → most arrows filtered.
   Only high-wind coastal/southern points produced visible arrows.

2. **HSL wind direction** — hue-based coloring not user-readable. Direction requires
   decoding a hue wheel; arrows alone are sufficient and clearer.

Phase R12 replaces the canvas implementation with an SVG/`map.project()` approach (ADR-025).

---

### Phase R12: SVG Wind Arrow Overlay — Replacement for R11 — COMPLETE (2026-08-17, commit fb9a7cf)

**Classification**: Defect fix + UX improvement. Replaces Phase R11 canvas arrows.
**Prerequisite**: Phase R10 complete (✓). ADR-025 accepted (✓). R11 defects documented (✓).
**Scope**: New `WindArrowOverlay.tsx`; update `WeatherMap.tsx`, `colorscales.ts`, `Legend.tsx`.

**Objective**: Fix the distribution defect and replace HSL wind-direction coloring with
SVG arrows positioned via `map.project()` for correct Mercator-projected placement.

**Architecture** (ADR-025):
- New `WindArrowOverlay.tsx` component: receives `map: maplibregl.Map | null` prop
- Reads `windSpeed`, `windDirection`, `currentHour`, `metadata` from Zustand store
- Samples every WIND_ARROW_GRID_STEP=3 grid points (lat and lon)
- `map.project([lon, lat])` → CSS pixel coordinates; recomputes on `map.move`/`map.resize`
- SVG overlay: `<svg style="position:absolute;inset:0;width:100%;height:100%">`
- Arrow: `<g transform="translate(x,y) rotate(toDeg)">` with `<line>` shaft + `<polygon>` head
- `toDeg = (fromDir + 180) % 360`; SVG rotation: 0°=north, clockwise
- Arrow length: 8 CSS px (2 kt) → 22 CSS px (30 kt), guaranteed minimum — no suppression

**Wind direction view**: No raster; canvas transparent; arrows over basemap only.
**Wind speed view**: Speed raster retained; SVG arrows overlaid.

**Validation**:
- tsc 0 errors; npm run build passes
- Arrows visible in northern (Kachin), central (Mandalay), and southern (Yangon) Myanmar
- Direction sanity: all four cardinal directions verified
- Popup ground truth: Yangon 95°FROM → arrow points west; Mandalay 79°FROM → points SW
- All 29 frames step without lag (playback 4×)
- No regression against 200ms step-transition requirement

**Gate**: tsc 0 errors; build passes; full geographic distribution confirmed; deployed to main.

---

### Phase R13: Precipitation Sqrt Color Scale — COMPLETE (2026-08-17)

**Classification**: Visualization calibration fix (FR-N23b). No data modification.
**Prerequisite**: Phase R12 complete ✓. Diagnostic investigation complete ✓.
**Scope**: `frontend/src/map/colorscales.ts`, `frontend/src/map/WeatherMap.tsx`,
`frontend/src/components/Legend.tsx`.

**Problem** (from diagnostic): R4 January 2021 precipitation is heavily right-skewed
(P50=0.000288, P75=0.005, P90=0.028, P95=0.084, P99=0.406, max=1.620 mm/hr). Current linear
scale with 0.020 mm/hr cutoff suppresses 87.8% of all values. Light rain (0.003–0.020 mm/hr)
is invisible despite representing meaningful forecast signal in frames 12–28.

**Solution** (FR-N23b):
- `renderWithInterpolation`: add `sqrtScale?: boolean` param; apply `Math.sqrt(norm)` when true
- `PRECIP_LUT_ALPHA`: recalibrated cutoffs in sqrt-norm space; v < 0.003 mm/hr → transparent
- `PRECIP_TICKS`: [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0] mm/hr
- WeatherMap.tsx precipitation call: `sqrtScale: true`
- Legend tooltip: describe sqrt scale and dry-season calibration

**Key metrics after fix** (sqrt at PRECIP_MAX=2):
- P75 (0.005 mm/hr): norm=0.050 → faint visible trace
- P90 (0.028 mm/hr): norm=0.118 → clearly visible
- P95 (0.084 mm/hr): norm=0.205 → strong color
- P99 (0.406 mm/hr): norm=0.450 → near-max color
- Max (1.620 mm/hr): norm=0.900 → no saturation

**Gate**: tsc 0 errors; build passes; light rain visible in frames 12–28 (>15% above 0.01);
color differentiation preserved across the 0.01–1.62 mm/hr range.

---

### Phase R14: Temperature Display Context (Myanmar Local Time) — COMPLETE (2026-08-17)

**Classification**: Display context improvement (FR-N30). No data modification.
**Prerequisite**: Phase R13 complete.
**Scope**: `frontend/src/map/WeatherMap.tsx` (popup), `frontend/src/components/InfoPanel.tsx`.

**Problem** (from diagnostic): Default frame (t+0h = 00:00Z = 06:30 MMT pre-dawn) is the
coolest part of the diurnal cycle. Yangon: 21.1°C at 06:30 MMT vs 28.9°C at 12:30 MMT.
GCOp has a documented −0.76°C cold bias vs ERA5 (diurnal: −1.45°C at 12Z, ≈0 at 18Z).
External weather sites show current temperatures, not January 2021 UTC timestamps.
Temperature values are correct — the fix is display context, not data.

**Solution** (FR-N30):
- Popup: "Mon 01 Jan 2021 00:00:00 UTC · 06:30 MMT" format (UTC+6:30 fixed offset)
- InfoPanel: "All times UTC. Myanmar Standard Time = UTC+6:30 (MMT, no daylight saving)."
- InfoPanel temperature note: "GCOp cold bias −0.76°C vs ERA5 (documented); stronger at midday."
- No data changes; no bias correction.

**Gate**: tsc 0 errors; build passes; popup shows both UTC and MMT time.

---

### Phase R15: Model Evaluation Contextualization — COMPLETE (2026-08-19, commit c9c7c89)

**Classification**: UI content improvement (FR-N46a–FR-N46g). No data modification.
**Prerequisite**: Phase R14 COMPLETE ✓.
**Spec refs**: FR-N46a (metric definitions), FR-N46b (temporal convention), FR-N46c (limitations),
  FR-N46d (temperature cold-bias context), FR-N46g (no qualitative ratings).

**Scope**: `frontend/src/components/ModelEvaluation.tsx` only.
No changes to: verification.json, verification scripts, forecast binaries, or any other file.

**Deliverables**:
1. "How to interpret these metrics" section: 7 metric definitions (MAE, RMSE, Bias, circ. MAE, POD, FAR, CSI).
2. Temporal convention note: t+0h = ERA5 init, error=0 by construction; tables show +6h–+168h only.
3. 4-point amber limitations block: ERA5 not obs; training advantage; N=1; dry season.
4. Temperature sky-blue cold-bias box: −0.7595°C R5 mean; −0.89°C QA spot-check; Jan 2021 only; no offset.
5. "10m Wind vs ERA5" grouped section: Wind Speed + Wind Direction subsections (FR-N46e/f).
6. No qualitative labels anywhere (FR-N46g).

**Gate**: PASSED — tsc 0 errors; build passes; all FR-N46a–FR-N46d,g,e,f visible.

---

### Phase R16: Wind UI Consolidation — COMPLETE (2026-08-19, commit c9c7c89)

**Classification**: UX simplification (FR-N20, FR-W01c, FR-N46e, FR-N46f). No data modification.
**Prerequisite**: Phase R15 COMPLETE ✓.
**Spec refs**: FR-N20 (three-tab selector), FR-W01c (consolidated wind legend), FR-N46e/f (eval).

**Files changed**:
- `frontend/src/data/types.ts` — `ActiveVariable` reduced to 3 values (wind_direction removed)
- `frontend/src/components/VariableSwitcher.tsx` — three-tab UI (Precip / Wind / Temp)
- `frontend/src/components/Legend.tsx` — Wind tab shows speed bar + compass widget
- `frontend/src/map/WindArrowOverlay.tsx` — trigger condition: `activeVariable === 'wind_speed'` only
- `frontend/src/map/WeatherMap.tsx` — wind_direction canvas branch removed (unreachable)
- `frontend/src/components/ModelEvaluation.tsx` — "10m Wind vs ERA5" grouped section

No changes to: wind_speed.bin, wind_direction.bin, ForecastLoader.ts,
ForecastStore.ts wind arrays, WeatherMap.tsx canvas rendering logic (wind_speed path), colorscales.ts.

**TypeScript sweep**: 7 stale `wind_direction` references eliminated across 4 files. tsc: 0 errors.

**Note (R17)**: Legend.tsx compass widget (added here) will be removed in Phase R17.
FR-W01c revised in spec.md v10: Wind tab shows speed bar ONLY.

**Gate**: PASSED — tsc 0 errors; build passes; 3 tabs; wind_direction removed from type system; deployed.

---

### Phase R17: Wind UI Simplification + Header Metadata Cleanup — NOT STARTED

**Classification**: UX refinement (FR-W01c revised, FR-N25 refined). No data modification.
**Prerequisite**: Phase R16 COMPLETE ✓.
**Spec refs**: FR-W01c (revised spec.md v10 — speed bar only), FR-N25 (refined spec.md v10 — init/source not in header).

**Rationale**:
- Wind compass widget added in R16 adds visual clutter on mobile — SVG arrows on the map are self-explanatory.
- Init time and Source in Header sub-row duplicate InfoPanel content and consume limited header space.

**Scope**:
- `frontend/src/components/Legend.tsx` — remove compass widget block from Wind tab
- `frontend/src/components/Header.tsx` — remove Init and Source spans from sub-row
- `frontend/src/components/InfoPanel.tsx` — read-only verification that init time + source are already present (FR-N26)

**No changes to**: WindArrowOverlay.tsx, WeatherMap.tsx, colorscales.ts, ForecastStore.ts,
wind_speed.bin, wind_direction.bin, verification.json, any pipeline script.

**Work items**:

1. `Legend.tsx`: Remove the `{isWind && (<>...</>)}` JSX block that renders the compass
   direction widget — the `border-t` divider + "Wind Direction (FROM)" label + N/E/S/W SVG arrows + text.
   Wind tab then renders speed gradient bar + tick marks + min/max labels ONLY.
   All WIND_LUT_ALPHA, WIND_TICKS, vmin/vmax retained.

2. `Header.tsx`: Remove `<span>Init: {formatInitTime(metadata.initialization_time)}</span>`,
   `<span>Source: {metadata.initialization_source}</span>`, and their `<span>·</span>` separators.
   Header sub-row retains: model name, native resolution, and staleness warning only.

3. `InfoPanel.tsx`: Read and confirm init time + source are rendered (FR-N26 — present since R6).
   No code addition required.

**Gate**: tsc 0 errors; build passes; Wind tab legend shows speed bar only (no compass widget);
Header sub-row shows model name + resolution only (Init and Source absent);
InfoPanel still shows Init time and Source; no data files modified.

---

## Architecture Decision Summary (New Target)

### Why ERA5 via ARCO for 2021-01-01?

- ARCO covers 1959–2023; 2021-01-01 is within range (no IFS needed)
- No credentials required
- Provides historical atmospheric state for initialization
- 2021 ERA5 is also available for evaluation against the forecast

### Why 2021-01-01 as initialization date?

- Falls within ARCO coverage
- Post-2020 makes it a quasi out-of-sample test for models trained on ERA5 ≤2019/2020
- ERA5 for 2021 is fully processed and available for evaluation

### What is the role of "ERA5 1990–2020 historical data"?

ERA5 1990–2020 is the historical reanalysis available prior to 2021-01-01.
The project uses ARCO to fetch the atmospheric state at 2021-01-01T00:00:00Z
(and preceding timesteps if required by the model) as the forecast initialization.
This is initialization, NOT training.
The foundation model's weights were pre-trained on ERA5 by the model's original authors.
This project does NOT perform training or fine-tuning.

### Constitution Check (New Target Plan)

| §I Static-First | No runtime compute; static GitHub Pages | ✓ Preserved |
| §II Earth2Studio-Mandatory | Earth2Studio model (TBD); ARCO/ERA5 init | Pending model selection |
| §III Pipeline | 7-day, 4 variables, model-native steps | Pending Phase 4 |
| §IV Myanmar-Focused | Myanmar bbox subset | Preserved |
| §V Map-First UX | 4-variable map | Pending Phase 7 |
| §VI Native-Step Navigation | Model-native timestep; no interpolation | Pending model selection |
| §VII Model-Agnostic Frontend | All config from forecast.json | Preserved |
| §VIII Performance | Payload TBD | Pending model resolution |
| §IX Climate-Honest | 4-variable disclosures; ERA5 init disclosure | Pending Phase 7 |
| §XI Hardware Transparency | Smoke test first; hardware recorded | Pending Phase 3 |
| §XII Resolution Honesty | Native resolution documented and disclosed | Pending model selection |

---

## Risk Register (New Target)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Selected model OOM on available hardware | Medium | High | Phase 3 smoke test; fallback to lower-resolution model |
| ARCO missing required init variables | Low | High | Phase 2 data availability check before any inference |
| Precipitation semantics differ from GraphCastSmall | Medium | Medium | Inspect output_coords in Phase 1; document per model |
| Wind variable not at 10m level | Medium | Medium | Inspect output_coords; use nearest available level |
| 7-day skill insufficient for display | Low | Medium | Document skill in Phase 5; add disclaimer in UI |
| Schema v4.0 breaks legacy frontend | Low | High | Demo data and legacy data kept separate; schema checked at load |
| Model selection takes longer than expected | Low | Medium | Research phase has no time constraint; gate enforced |
