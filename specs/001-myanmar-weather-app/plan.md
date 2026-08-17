# Implementation Plan: Myanmar Weather Forecast Web Application

**Branch**: `001-myanmar-weather-app`
**Date**: 2026-08-12 (v4 — new target: 7-day / 4-variable / ERA5 2021-01-01 / model TBD)
**Revised**: 2026-08-16 (v5 — Spec Kit update phase RS; R4 timings corrected; phases added)
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

**Status**: Phases 1–5 COMPLETE. Phase RS COMPLETE. R4/R5/R6/R9 COMPLETE (2026-08-17). RS10 COMPLETE. RS11–RS14 DEFERRED (ADR-024 — data/forecast_v4/ is authoritative production dataset). Live on GitHub Pages (commit 30ff08c).

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
