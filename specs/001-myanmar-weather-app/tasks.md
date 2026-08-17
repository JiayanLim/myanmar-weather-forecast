# Tasks: Myanmar Weather Forecast Web Application

**Input**: `specs/001-myanmar-weather-app/` (spec.md v5, plan.md v5, research.md v4+ADRs)
**Feature**: 001-myanmar-weather-app
**Revised**: 2026-08-12 v4 — New target: 7-day / 4-variable / ERA5 init / model TBD
**Revised**: 2026-08-16 v5 — Schema v4.0 locked; GCOp confirmed; RS tasks added; timings corrected
**Revised**: 2026-08-17 v6 — R4 COMPLETE; R044 PASS; RS10 COMPLETE; R5 COMPLETE; ADR-023 closed
**Revised**: 2026-08-17 v7 — R6 COMPLETE (types+loader+store+all components+build); R7 COMPLETE; R8 COMPLETE
**Revised**: 2026-08-17 v8 — R10 COMPLETE; R11 COMPLETE (superseded by R12 — defect); R12 tasks added
**Revised**: 2026-08-17 v10 — R13 COMPLETE (precip sqrt scale); R14 COMPLETE (MMT local time)

---

## PART A — LEGACY TASKS (v3.0, ALL COMPLETE)

The following tasks were completed on 2026-08-11/12. The resulting application is live
on GitHub Pages. No modification required unless addressing a bug in the legacy deployment.

- [x] T001 Update constitution.md v1.0.1 → v2.0.0 (GraphCastSmall, 6h, tp06)
- [x] T002 Update research.md v3 (ADR-001 through ADR-007)
- [x] T003 Update spec.md v3 (48h, temperature, schema v3.0, M4 CPU)
- [x] T004 Update plan.md v3 (48h+temp, M4 pipeline)
- [x] T005 Update tasks.md (this file, legacy portion)
- [x] T010 Rewrite scripts/generate_forecast.py (GraphCastSmall, ARCO/IFS, 48h, 9 frames, tp06+t2m, M4 CPU)
- [x] T011 Hardware validation — M4 CPU: ~78s, 2.34 GB RSS, JAX XLA ARM64
- [x] T012 Update frontend/src/data/types.ts (ActiveVariable, temperature, inference_config)
- [x] T013 Rewrite scripts/generate_demo_data.py (9 frames, schema v3.0, is_demo=true)
- [x] T014 Run generate_demo_data.py; verify data/demo/ (8,316 bytes each, validated)
- [x] T015 Commit demo artifacts (data/demo/)
- [x] T016 Update types.ts (legacy)
- [x] T017 Update ForecastLoader.ts (loads both binaries)
- [x] T018 Update ForecastStore.ts (temperature, activeVariable, setVariable)
- [x] T019 VariableSwitcher.tsx (Precip/Temp toggle with useShallow — React #185 fix)
- [x] T020 WeatherMap.tsx (variable-aware rendering, dual-variable popup)
- [x] T021 Timeline.tsx (dynamic n_times, 6h markers, fallback)
- [x] T022 Legend.tsx (variable-aware: mm/6h or °C)
- [x] T023 Header.tsx (Myanmar 48h Weather Forecast; staleness threshold)
- [x] T024 InfoPanel.tsx (both variables, horizon from metadata)
- [x] T025 App.tsx (VariableSwitcher, setData with both variables)
- [x] T026 Copy schema v3.0 demo artifacts to frontend/public/data/
- [x] T027 Rewrite scripts/validate_forecast.py (schema v3.0, 25 checks)
- [x] T028 Run validate_forecast.py on data/demo/ and data/forecast/ (25/25 PASS)
- [x] T030 Rename notebook to graphcast_myanmar_forecast.ipynb
- [x] T031 Rewrite notebook for local M4 pipeline (12 sections)
- [x] T040 Dev server test with demo data (9-frame timeline, both variables, DEMO banner)
- [x] T041 npm run build: 0 TypeScript errors, built in 1.41s
- [x] T042 React #185 production fix (useShallow in VariableSwitcher.tsx, commit 5041b07)
- [x] T043 Push to main; GitHub Actions green; GitHub Pages live; is_demo=false confirmed
- [x] T050 README updated for GraphCastSmall/48h/M4

**Known legacy status**: validate_forecast.py and generate_demo_data.py were partially
updated toward 29 frames / schema v4.0 during an interrupted 7-day extension session
(2026-08-12). These changes should be reviewed for consistency with the new target plan
before being used. The live deployment and data/forecast/ artifacts remain schema v3.0.

---

## PART B — NEW TARGET TASKS (v4.0, IN PROGRESS — R4 running 2026-08-12)

**EXPLICIT MODEL SELECTION GATE**

> ⛔ No task in Phase 2 through Phase 9 may begin until Task R010 is complete
> and the model selection has been explicitly approved by the user.
>
> The model is NOT assumed to be GraphCastSmall, Aurora, or any other specific model.
> Every implementation detail (variables, timestep, resolution, hardware) is
> determined by the selected model.

---

## Phase R1: Model and Data Research — COMPLETE (2026-08-12)

**Objective**: Select the Earth2Studio model for the new target. Produce a documented,
approved model selection in research.md ADR-012.

- [x] R001 Inspect Earth2Studio source: enumerate all models in `earth2studio.models.px.*`
      and `earth2studio.models.dx.*`; record model names, availability, import paths

- [x] R002 For each candidate model, inspect `input_coords`:
      - What variables are required for initialization?
      - What is the initialization time structure (1 step, 2 steps, N steps)?
      - Are all required variables available in ARCO for 2021-01-01T00:00:00Z?

- [x] R003 For each candidate model, inspect `output_coords`:
      - Is there a precipitation variable? What is its name? Is it accumulated or rate?
        Over what period? What is its native unit?
      - Are there near-surface u/v wind components (u10m, v10m)?
        If not, what wind variables are available?
      - Is there a 2m temperature variable (t2m)? In what unit?
      - What is the native temporal resolution (timestep in hours)?
      - What is the native spatial resolution?
      - How many output variables does the model produce?

- [x] R004 Document precipitation variable semantics for each candidate:
      - Name of the precipitation variable
      - Whether it is a rate (mm/hr) or an accumulation (mm over N hours, or metres)
      - If accumulated: over what period? Is the period equal to the model timestep?
      - What conversion is needed to produce mm/hr for display?
      - Are there any model-specific transforms (e.g., log-space like Aurora tp1h)?
      NOTE: Do NOT assume GraphCastSmall tp06 semantics apply to other models.

- [x] R005 Document wind variable availability for each candidate:
      - Confirm presence of u10m and v10m (or equivalent near-surface wind components)
      - If only pressure-level winds are available, document which levels
      - Confirm native units (m/s expected)
      - Derive: speed_kt = sqrt(u²+v²) × 1.94384; direction = (270 − atan2d(v,u)) mod 360

- [x] R006 Research 7-day forecast skill for each candidate (documented in ADR-012)

- [x] R007 Research compute requirements for each candidate on available hardware (M4 CPU)

- [x] R008 Produce a model comparison table (see ADR-012 in research.md)

- [x] R009 Identify preferred candidate: GraphCastOperational (ADR-012 rationale)

- [x] R010 **MODEL SELECTION GATE — COMPLETE**
      ADR-012 in research.md records:
      - FuXi initially selected → R2 FAIL (onnxruntime-gpu no macOS ARM64; ARCO lacks r* vars)
      - AIFS rejected → R2 FAIL (flash-attn CUDA-only)
      - **Final selection: GraphCastOperational** (user approved)
      - Variables: tp06 (metres/6h), t2m (K), u10m (m/s), v10m (m/s)
      - Timestep: 6h; Resolution: 0.25°; Backend: JAX/Haiku (CPU)
      - Init: ARCO ERA5, two timesteps (t-6h, t+0h); all 82 input vars in ARCO

**Gate output**: ADR-012 in research.md — GraphCastOperational approved by user.

---

## Phase R2: ERA5 Data Availability and Initialization Validation — COMPLETE (2026-08-12)

**Prerequisite**: R010 gate passed (model selected and approved).

- [x] R020 Identify all variables required by GCOp input_coords: 82 variables
      (z, q, t, u, v, w at 13 pressure levels + msl). Uses q* (specific humidity),
      NOT r* (relative humidity) — confirmed compatible with ARCO.

- [x] R021 ARCO availability verified: all 82 GCOp input variables available at
      2021-01-01T00:00:00Z and 2020-12-31T18:00:00Z (t-6h init). NaN count: 0.
      Shape: (2, 82, 721, 1440). Fetch time: ~7s (warm cache).

- [x] R022 All required variables available — no gaps. ADR-017 in research.md records
      full ARCO compatibility table.

- [x] R023 N/A — no missing variables. PASS.

**Gate**: PASS — ARCO confirmed for all 82 GCOp init variables at 2021-01-01T00:00:00Z.
See ADR-017 in research.md.

---

## Phase R3: Minimal Model Inference / Hardware Feasibility — COMPLETE / PASS (2026-08-12)

**Prerequisite**: R022 gate passed (data availability confirmed).

- [x] R030 nsteps=1 smoke test with GraphCastOperational on M4 (ARCO, 2021-01-01T00Z):
      - SUCCESS. No OOM. No crash. exit code 0.
      - nsteps=1 total time: 2081s = 34.7 min (JIT compilation dominates on cold start)
      - Peak RSS during XLA compilation: ~5.0 GB; post-inference: 1.99 GB

- [x] R031 N/A — smoke test SUCCEEDED.

- [x] R032 All four variable groups confirmed in output:
      - tp06: metres/6h accumulation, min=-0.0002 (float noise), max=0.0872 m/6h = 14.5 mm/hr
      - t2m: Kelvin, 221.1–315.8 K → -52.1 to +42.6°C after conversion
      - u10m: m/s, ±21 m/s range
      - v10m: m/s, ±21 m/s range
      - Output shape: (1, 2, 721, 1440) float32 for nsteps=1

- [x] R033 Hardware feasibility documented in research.md ADR-018:
      - Hardware: Apple M4 MacBook Air 24 GB, JAX CPU (XLA ARM64)
      - Peak RSS: 1.99 GB post-inference (~5–6 GB during XLA compilation)
      - JIT cold-start: ~27–34 min; post-JIT step time: **~25 min measured** (R4 pipeline)
      - 7-day run feasible: YES; estimated ~12–14 hours total (~25 min/step × 28 steps)
      - NOTE: First aborted run (2026-08-13) showed 35–48 min/step; current run is ~25 min/step
      - This pipeline is research/demo quality — NOT practical for daily production use

- [x] R034 Variable specifications confirmed and recorded in ADR-018:
      - tp06: metres/6h → mm/hr via ×1000/6, clamp ≥ 0
      - t2m: K → °C via −273.15
      - wind speed: sqrt(u²+v²) × 1.94384 = knots
      - wind direction: (270−atan2(v,u)×180/π) mod 360 = °FROM

**Gate**: PASS — ADR-018 in research.md. Hardware verified on M4 24 GB. R4 approved by user.

---

## Phase RS: Spec Kit Update — COMPLETE (2026-08-16)

**Prerequisite**: R3 PASS; user approves planning document.
**Scope**: Spec Kit files ONLY. No application code, frontend, or pipeline scripts modified.

- [x] RS01 Update `.specify/memory/constitution.md` v3.0.0 → v3.1.0
      - Filled in all "TBD" fields with confirmed GCOp values
      - Added v3.1.0 sync impact record
      - Updated new target architecture table, §II, §III, §VI, §VIII, §XI, §XII, ADR-012

- [x] RS02 Update `specs/001-myanmar-weather-app/research.md`
      - Corrected ADR-018: replaced "1.5–2 hours" estimate with actual measured timings (~25 min/step)
      - Corrected ADR-018: JAX cache status documented (empty after run; warm-start unconfirmed)
      - Added ADR-019: schema v4.0 artifact design (locked canonical schema version)
      - Added ADR-020: wind direction visualization — vector-component bilinear interpolation
      - Added ADR-021: GCOp precipitation conversion — tp06 metres×1000/6 → mm/hr
      - Added ADR-022: payload loading policy — all 4 variables at startup

- [x] RS03 Update `specs/001-myanmar-weather-app/spec.md` v4 → v5
      - Section B constitution check: updated all "Pending model selection" → confirmed GCOp values
      - New target overview: replaced "model TBD" with confirmed GCOp architecture
      - User stories: updated frame count from TBD to 29; timestep from TBD to 6h
      - FR-N01 through FR-N10: replaced all TBD with confirmed values and ADR references
      - FR-N11: added MODEL_STEP must be derived from metadata (colorscales.ts bug)
      - FR-N12: added wind direction vector-component interpolation requirement (ADR-020)
      - FR-N13: added 4-binary parallel load requirement (ADR-022)
      - Display unit table: confirmed with ADR references; removed "TBD by model" language
      - NFRs: confirmed payload size and load budget
      - Unresolved questions block: replaced with resolved answers

- [x] RS04 Update `specs/001-myanmar-weather-app/plan.md` v4 → v5
      - Added Phase RS (Spec Kit update) to Part B sequence
      - Phase 4: corrected runtime estimates; added current run status and measured timings
      - Phase 6: renamed to "Data Layer Migration"; updated scope description
      - PART B status: updated to reflect RS in progress

- [x] RS05 Update `specs/001-myanmar-weather-app/tasks.md` v4 → v5
      - R033: corrected "estimated 1.5–2 hours" → actual ~25 min/step, ~12–14h total
      - R041: resolved "schema v5.0" → "schema v4.0"; updated run status with measured timings
      - Added RS01–RS05 tasks (this block)
      - Added RS10–RS14 tasks (schema v4.0 finalization, Phase R4b)

**Gate**: PASSED — All five Spec Kit files updated and approved by user (2026-08-16).

---

## Phase R4b: Schema v4.0 Finalization — PARTIALLY COMPLETE — RS10 DONE; RS11–RS14 DEFERRED (OPTIONAL)

**Prerequisite**: R041 complete (pipeline finishes); R044 passes; RS gate approved.
**Scope**: Finalize schema version string in pipeline, regenerate demo data.
**No frontend changes in this phase.**
**Note (2026-08-17)**: RS10 complete. RS11–RS14 deferred per ADR-024 — `data/forecast_v4/` (real R4 data, `is_demo=false`) is the authoritative production dataset; synthetic demo regeneration is not on the critical path.

- [x] RS10 Update `scripts/generate_forecast.py`: change `schema_version` emit from "5.0" → "4.0"
      COMPLETE (2026-08-17) — single string change at line 524; verified via grep.

- [DEFERRED] RS11 Update `scripts/generate_demo_data.py` for schema v4.0:
      Deferred per ADR-024. Not required — production serves data/forecast_v4/ (real data).
      Resume only if: offline dev needs synthetic fixtures independent of committed real data,
      or CI requires a lightweight fixture. Requires rewrite for 4 vars, 81×41, 29 frames.
      - 4 variables (precipitation, temperature, wind_speed, wind_direction)
      - 81 × 41 grid (matching GCOp Myanmar subset)
      - 29 frames at 6h steps
      - schema_version: "4.0", model: "GraphCastOperational"
      - is_demo: true; is_demo banner text preserved
      - Synthetic data: physically plausible ranges for all 4 variables

- [DEFERRED] RS12 Run `uv run python scripts/generate_demo_data.py` and verify output:
      Deferred per ADR-024.
      - 4 binary files created in data/demo/
      - File sizes: each ~376 KB (29 × 81 × 41 × 4 bytes)
      - forecast.json: schema_version "4.0", 29 times, 4 variables, is_demo=true

- [DEFERRED] RS13 Run `uv run python scripts/validate_forecast.py --data-dir data/demo/`:
      Deferred per ADR-024.
      - 0 validation errors
      - All 4 variables present and within physical bounds

- [DEFERRED] RS14 Copy demo artifacts to `frontend/public/data/`:
      Deferred per ADR-024. `frontend/public/data/` already populated with data/forecast_v4/
      artifacts (real data) in R9 — npm run dev works without synthetic demo data.
      - Copy data/demo/{forecast.json, precipitation.bin, temperature.bin, wind_speed.bin, wind_direction.bin}

**Gate (RS10)**: PASSED — schema_version "4.0" in generate_forecast.py.
**Gate (RS11–RS14)**: DEFERRED — reauthorize explicitly before starting.

---

## Phase R4: 7-Day Four-Variable Forecast Pipeline — COMPLETE (2026-08-17)

**Prerequisite**: R033 gate passed (smoke test succeeded, hardware feasible).

- [x] R040 Implemented `scripts/generate_forecast.py` for GraphCastOperational:
      - nsteps = 168 / native_timestep_h (e.g., 28 for 6h, 56 for 3h, 168 for 1h)
      - Extract and convert all four variables with documented provenance
      - Precipitation: native_output → mm/hr (model-specific conversion, documented)
      - Wind direction: (u, v) → degrees, meteorological FROM convention
      - Wind speed: (u, v) → knots (via × 1.94384)
      - Temperature: Kelvin → °C (if required)
      - Myanmar bbox subset at model-native resolution
      - Write forecast.json schema v4.0 (corrected from "5.0" in RS10)
      - Record hardware and transformation provenance

- [x] R041 Full pipeline run: COMPLETE (2026-08-17 07:15 SGT)
      - Runtime: 12:33 SGT 2026-08-16 → 07:15 SGT 2026-08-17 (~18.7h; caffeinate active)
      - Timings: JIT 27 min (step 2); steps 3–9: ~25 min; step 10: ~80 min (thermal); steps 11–29: ~27–34 min
      - JAX persistent cache: configured but remained empty — cold JIT (~27 min) on every run
      - Output: data/forecast_v4/ (did NOT touch data/forecast/ schema v3.0)
      - 4 binaries: 385,236 bytes each; forecast.json: 8,063 bytes; is_demo=false
      - Pipeline built-in sanity check: PASS; rss_peak_gb=3.13

- [ ] R042 Write/update `scripts/generate_demo_data.py` for schema v4.0 (NOT STARTED — Phase R4b)

- [x] R043 Wrote `scripts/validate_forecast.py` for schema v4.0/v5.0:
      - All four variables checked (precipitation, temperature, wind_speed, wind_direction)
      - Physical plausibility checks per variable; 385,236 byte check

- [x] R044 Run validate_forecast.py on data/forecast_v4/: PASS — 0 errors (2026-08-17)
      - All 4 binaries: 385,236 bytes ✓ | is_demo=false ✓ | model=GraphCastOperational ✓
      - All physical plausibility checks pass; no NaN/Inf

**Gate**: PASSED — 7-day forecast validated; all four variables; is_demo=false; RS10 complete.

---

## Phase R5: ERA5 Evaluation Pipeline — COMPLETE (2026-08-17)

**Prerequisite**: R044 gate passed (7-day forecast validated).

- [x] R050 Rewrote `scripts/verify_forecast.py` for GCOp and 4 variables:
      - 4-variable verification: temperature (°C), wind speed (kt), wind direction (°FROM circ. MAE), precipitation (mm/hr)
      - ARCO tp semantics: 1-hour accumulation per timestamp; no seam handling (confirmed empirically)
      - 168 hourly tp timestamps fetched (t+1h → t+168h); summed into 28 × 6h windows
      - Per-hour clamp ≥ 0 before sum; aggregate clamp ≥ 0; convert × 1000/6 → mm/hr
      - t+0h excluded from precipitation metrics (GCOp pipeline convention)
      - Calm wind exclusion: ERA5 speed < 2 kt excluded from circular MAE
      - Summary POD/FAR/CSI from aggregated total counts, not averaged per-frame ratios
      - verification.json schema v2.0; exit 0 on PASS, exit 1 on any failure
      - All pre/post invariants checked; output written only on success

- [x] R051 Ran verify_forecast.py — PASS (2026-08-17)
      - Exit code: 0
      - All 29 frames verified for temperature, wind speed, wind direction
      - 28 frames verified for precipitation (t+6h through t+168h)
      - No NaN in any metric; all post-computation invariants pass
      - ARCO grid alignment: lat_err=0°, lon_err=0° (exact match; no interpolation)
      - ERA5 tp negatives: 11.73% (max −0.000004 mm) — clamped as expected

      Results:
      | Variable | MAE | RMSE | Bias |
      |---|---|---|---|
      | Temperature | 1.3137°C | 1.6975°C | −0.7595°C |
      | Wind speed | 1.0548 kt | 1.4027 kt | −0.3911 kt |
      | Wind direction | 16.9113° circ. MAE | — | — |
      | Precipitation | 0.0172 mm/hr | 0.0645 | +0.0110 |
      POD=0.6343 FAR=0.7279 CSI=0.2352 (1,148/662/3,071 from 4,881 events)

- [x] R052 Commit data/verification/verification.json — COMPLETE (included in R6 commit)

**Gate**: PASSED — verification.json schema v2.0 written; exit 0; ADR-023 closed.

---

## Phase R6: Schema v4.0 Full Frontend Migration — COMPLETE (2026-08-17)

**Prerequisite**: R044 gate passed.

- [x] R060 types.ts: n_times→n_frames; spatial_resolution_deg→native_resolution_deg;
      VariableMeta simplified (display_unit, conversion, file); model_version optional;
      ActiveVariable = 'precipitation' | 'temperature' | 'wind_speed' | 'wind_direction';
      ForecastMetadata.variables has all 4 GCOp outputs.
- [x] R061 ForecastLoader.ts: ForecastData has precipitation, temperature, windSpeed, windDirection;
      loadForecast() fetches all 4 binaries concurrently via Promise.all (ADR-022).
- [x] R062 ForecastStore.ts: windSpeed + windDirection arrays added; setData takes all 5 args;
      n_times→n_frames fallback 9→29; all references updated.
- [x] R063 App.tsx: setData call updated to pass all 4 arrays.
- [x] R064 TypeScript check: `npx tsc --noEmit` → 0 errors ✓
- [x] R065 colorscales.ts: MODEL_STEP=1.0 removed (ADR-019/FR-N11); renderWithInterpolation
      accepts modelStep derived from metadata.native_resolution_deg; WIND_LUT_ALPHA added;
      renderWindDirectionWithInterpolation implements vector-component bilinear (ADR-020).
- [x] R066 WeatherMap.tsx: all 4 variables; modelStep from metadata; vector-interp for wind_direction;
      popup shows all 4 values with compass labels.
- [x] R067 VariableSwitcher.tsx: 4 buttons (Precip / Temp / Wind Speed / Wind Dir).
- [x] R068 Legend.tsx: precipitation (mm/hr, 0–10); temperature (°C); wind speed (kt, 0–30);
      wind direction (circular hue gradient with compass labels N/E/S/W/N).
- [x] R069 InfoPanel.tsx: display_unit/conversion (v4.0 field names); all 4 variables listed;
      native_resolution_deg used; model_version optional; hardware/device fallback.
- [x] R070 Header.tsx: native_resolution_deg; model_version optional.
- [x] R071 Timeline.tsx: n_frames (replaces n_times) in all 3 references.
- [x] R072 ModelEvaluation.tsx: rewritten for verification.json schema v2.0; all 4 variables;
      "GraphCastOperational" (not "GraphCastSmall"); correct field names.
- [x] R073 frontend/public/data/: 5 v4 forecast files + verification.json copied for dev mode.
- [x] R074 npm run build → ✓ built in 1.50s; 0 type errors.

**Validation results (2026-08-17)**:
- `npx tsc --noEmit` → 0 errors
- `npm run build` → ✓ 1.50s; no type errors
- All 4 binaries served at 385,236 bytes each ✓
- forecast.json: n_frames=29, native_resolution_deg=0.25, 4 variables, is_demo=false ✓
- verification.json: schema v2.0, 4 variables, wind_dir circular_mae=16.9113° ✓
- No MODEL_STEP=1.0 in source; no n_times; no spatial_resolution_deg in active data layer
- No raw degree interpolation (vector-component bilinear confirmed in renderWindDirectionWithInterpolation)
- GraphCastSmall: 0 references in frontend/src/

**Gate**: PASSED — TypeScript 0 errors; build passes; all 4 variables integrated.

---

## Phase R7: Frontend Four-Variable Migration — COMPLETE (merged into R6, 2026-08-17)

All R7 tasks completed as part of the unified R6 authorization. See Phase R6 above for details.

**Gate**: PASSED — see R6 gate.

---

## Phase R8: Evaluation / Accuracy UI — COMPLETE (merged into R6, 2026-08-17)

- [x] R090 ModelEvaluation.tsx rewritten for verification.json schema v2.0: all 4 variables;
      temperature MAE/RMSE/Bias; precipitation MAE/POD/FAR/CSI; wind speed MAE/RMSE/Bias;
      wind direction circular MAE; caveats block; graceful error fallback.
- [x] R091 "ⓘ Model Eval" button in Header.tsx — already existed, preserved.
- [x] R092 toggleModelEvaluation in ForecastStore.ts — already existed, preserved.
- [x] R093 TypeScript check: 0 errors ✓
- [x] R094 Modal closes on backdrop click and × button ✓
- [x] R095 No "X% accurate" language; only MAE/RMSE/Bias/POD/FAR/CSI/circular-MAE ✓

**Gate**: PASSED — see R6 gate.

---

## Phase R9: Deployment — COMPLETE (2026-08-17)

**Prerequisite**: Phases R1–R8 complete; EXPLICIT USER APPROVAL to deploy.

- [x] R100 deploy-pages.yml updated: 5-file v4.0 check; verification.json copy;
      per-file verify step; triggers on data/forecast_v4/**, data/verification/**, data/demo/**
- [x] R101 Spec Kit committed (constitution v3.2.0, research+ADR-023, spec v6, plan, tasks v7)
- [x] R102 Pipeline scripts committed (generate_forecast.py RS10 fix, verify_forecast.py R5)
- [x] R103 data/demo/ committed (intermediate state — contains stale v3.0 artifacts; RS11–RS14 DEFERRED per ADR-024; not served to production)
- [x] R104 data/forecast_v4/ committed (5 files: 385,236 bytes each + forecast.json 8,063 bytes)
- [x] R105 data/verification/verification.json committed (schema v2.0, 31,482 bytes)
- [x] R106 tsc → 0 errors; npm run build → ✓ 1.50s
- [x] R107 Pushed to main (commit 30ff08c)
- [x] R108 GitHub Actions: build ✓ 26s | deploy ✓ 10s | both jobs success
- [x] R109 Live GitHub Pages verified (2026-08-17):
      - Model: GraphCastOperational
      - Init time: 2021-01-01T00:00:00Z
      - n_frames: 29, horizon_hours: 168, native_resolution_deg: 0.25
      - Variables: ['precipitation', 'temperature', 'wind_speed', 'wind_direction']
      - is_demo: false ✓
      - All 4 binaries: 385,236 bytes ✓
      - verification.json schema v2.0: wind_dir MAE=16.9113°, temp MAE=1.3137°C ✓
      - Old data/forecast/ path: HTTP 404 (v3.0 assets absent) ✓

**Gate**: PASSED — Live GitHub Pages confirmed working. Commit 30ff08c deployed.

---

## Phase R10: Precipitation Color Scale Calibration — COMPLETE (2026-08-17)

**Classification**: Display defect fix.
**Commit**: 2ee8c60

- [x] R10a Binary spot-check confirmed (385,236 bytes/file, max=1.62 mm/hr < 2, 12.2% > 0.02 mm/hr)
- [x] R10b colorscales.ts: PRECIP_MAX 10→2; ramp cutoff norm<0.01; PRECIP_TICKS updated
- [x] R10c Legend.tsx: dry-season calibration note in tooltip
- [x] R10d tsc 0 errors; build passes
- [x] R10e Committed and pushed (2ee8c60); GitHub Pages redeployed ✓

**Gate**: PASSED.

---

## Phase R11: Wind Vector Arrow Overlay — COMPLETE / SUPERSEDED (2026-08-17)

**Commit**: 2ee8c60
**Status**: SUPERSEDED by Phase R12. Two defects confirmed post-deployment (ADR-025):

1. **Distribution defect**: arrows appeared only at bottom (southern Myanmar). Root cause:
   `if (len < 2) continue` with maxLen≈6.77px; at January median 3.10 kt, len=0.70px → filtered.
   Effective draw threshold ≈8.87 kt (≈P75+).

2. **HSL rasterization**: hue-based wind_direction coloring not user-readable.

- [x] R11a Spec confirmed (FR-W01–FR-W05, WIND_ARROW_GRID_STEP=3, calm threshold=2 kt) ✓
- [SUPERSEDED] R11b drawWindArrows() in colorscales.ts — implemented; distribution defect identified
- [SUPERSEDED] R11c WeatherMap.tsx integration — implemented; visual defect confirmed
- [SUPERSEDED] R11d Visual test at zoom 5.2 — arrows appeared only at bottom (defect confirmed)
- [SUPERSEDED] R11e Visual test at zoom 10 — not performed (defect requires architectural fix)
- [SUPERSEDED] R11f Direction sanity — math verified (formula correct); visual layout defective
- [x] R11g tsc 0 errors; npm run build passes ✓
- [x] R11h Committed 2ee8c60; deployed ✓

**Gate**: PARTIALLY PASSED — TypeScript/build OK; visual distribution FAILED → Phase R12 required.

---

## Phase R12: SVG Wind Arrow Overlay — COMPLETE (2026-08-17, commit fb9a7cf)

**Classification**: Defect fix + UX improvement. Replaces Phase R11.
**Prerequisite**: R10 COMPLETE ✓; ADR-025 ACCEPTED ✓; R11 defects documented ✓.
**Spec**: FR-W01–FR-W07 (spec.md v7).

- [ ] R12a Spec and architecture review:
      - FR-W01–FR-W07 locked in spec.md v7 ✓; ADR-025 in research.md ✓
      - Architecture: SVG overlay div + map.project() + guaranteed min arrow length
      - WIND_ARROW_GRID_STEP = 3; calm threshold 2.0 kt; toDeg = (fromDir+180) % 360

- [ ] R12b Create `frontend/src/map/WindArrowOverlay.tsx`:
      - Props: `{ map: maplibregl.Map | null }`
      - Reads from store: windSpeed, windDirection, currentHour, metadata, activeVariable, isLoaded
      - Computes arrows array on change: map.project([lon, lat]) → {x, y, toDeg, len}
      - len = lerp(8, 22, clamp((speed - 2) / 28, 0, 1)) CSS px; any speed ≥ 2 kt → len ≥ 8 px
      - SVG: `<svg style="position:absolute;inset:0;width:100%;height:100%;overflow:visible">`
      - Per arrow: `<g key transform="translate(x,y) rotate(toDeg)">` containing shaft + head
      - Shaft: `<line x1="0" y1="2" x2="0" y2="-len" stroke="white" strokeOpacity="0.85" strokeWidth="1.5"/>`
      - Head: `<polygon points="0,-len -3,-len+6 3,-len+6" fill="white" fillOpacity="0.85"/>`
      - Only render when `activeVariable === 'wind_speed' || 'wind_direction'`
      - Recompute on map.move and map.resize events

- [ ] R12c Update `frontend/src/map/WeatherMap.tsx`:
      - Import WindArrowOverlay
      - Remove `drawWindArrows` import and call from draw effect
      - For wind_direction activeVariable: skip canvas raster (do not call renderWindDirectionWithInterpolation)
        — or render a neutral transparent canvas (either approach; blank canvas acceptable)
      - Pass `mapRef.current` to `<WindArrowOverlay map={mapRef.current} />`
      - Place WindArrowOverlay in returned JSX alongside map div and canvas

- [ ] R12d Remove `drawWindArrows` from `frontend/src/map/colorscales.ts`:
      - Remove the function body and export
      - Remove `WIND_ARROW_GRID_STEP` and `WIND_ARROW_CALM_KT` constants
      - Keep all other exports unchanged (renderWindDirectionWithInterpolation may be retained or removed)

- [ ] R12e Update `frontend/src/components/Legend.tsx`:
      - Remove WindDirectionLegend hue-wheel gradient component
      - Replace with a simple text description: "Wind direction shown by arrows (TO direction)"
        with a minimal compass icon or N/E/S/W labels

- [ ] R12f Direction sanity tests (all four cardinal directions + Yangon + Mandalay):
      - 90°FROM east  → toDeg=270° → SVG rotate(270°) → arrow points left (west) ✓
      - 180°FROM south → toDeg=0°  → SVG rotate(0°)   → arrow points up (north) ✓
      - 0°FROM north  → toDeg=180° → SVG rotate(180°) → arrow points down (south) ✓
      - 270°FROM west → toDeg=90°  → SVG rotate(90°)  → arrow points right (east) ✓
      - Yangon frame 0: popup ~95°FROM east → arrow points ~west (toDeg≈275°) ✓
      - Mandalay frame 0: popup ~79°FROM → arrow points ~SW (toDeg≈259°) ✓

- [ ] R12g Visual QA — geographic distribution (must cover full Myanmar domain):
      - zoom 5.2: arrows visible in northern Myanmar (Kachin/Shan ~25–28°N)
      - zoom 5.2: arrows visible in central Myanmar (Mandalay ~22°N)
      - zoom 5.2: arrows visible in southern Myanmar (Yangon ~17°N, Tenasserim ~10–14°N)
      - zoom 7: density appropriate; not overcrowded, not too sparse
      - zoom 10: arrows readable; individual arrows not overlapping excessively
      - If overcrowded at default zoom: increase WIND_ARROW_GRID_STEP to 4 and re-test
      - If too sparse: decrease to 2 and re-test

- [ ] R12h Performance check:
      - Step through all 29 frames at 4× playback speed
      - No visible lag or stutter between frames
      - Accept if step transition < 200ms; profile before optimizing if exceeded

- [ ] R12i Validate:
      - `npx tsc --noEmit` → 0 errors
      - `npm run build` → passes
      - All direction sanity tests pass (R12f)
      - Geographic distribution confirmed (R12g)

- [ ] R12j Commit and push to main; confirm GitHub Pages redeploy succeeds.

**Gate**: tsc 0 errors; build passes; full-domain distribution confirmed; all cardinal sanity tests pass; no step-transition regression; deployed.

---

## Phase R12: SVG Wind Arrow Overlay — COMPLETE (2026-08-17, commit fb9a7cf)

See Phase R12 in plan.md. Distribution defect fixed; HSL raster retired; deployed.

---

## Phase R13: Precipitation Sqrt Color Scale — COMPLETE (2026-08-17)

**Classification**: Visualization calibration fix (FR-N23b). No data modification.
**Prerequisite**: R12 COMPLETE ✓. Diagnostic investigation complete ✓.

**Diagnostic summary** (read-only investigation, 2026-08-17):
- P50=0.000288, P75=0.005, P90=0.028, P95=0.084, P99=0.406, max=1.620 mm/hr
- 87.8% of values invisible at current 0.020 mm/hr cutoff
- Sqrt scale: P90 maps to norm=0.118, P95 to 0.205, P99 to 0.450 — spread across ramp
- With cutoff 0.003 mm/hr + sqrt: 25.5% of values visible (vs 12.2% currently)
- precipitation.bin and verification.json: UNCHANGED

- [x] R13a Diagnostic pre-read confirmed (above). No code changes until spec approved.

- [x] R13b Edit `frontend/src/map/colorscales.ts`:
      - `renderWithInterpolation`: add `sqrtScale = false` parameter (7th positional param after mask)
        When true: `const norm = Math.sqrt(Math.max(0, Math.min(1, (v-vmin)/range)));`
        When false: linear norm as before
      - `PRECIP_LUT_ALPHA`: update alpha cutoffs for sqrt-norm space:
        `const CUT_LO = Math.sqrt(0.003 / 2.0); // ≈0.039 — fully transparent below`
        `const CUT_HI = Math.sqrt(0.010 / 2.0); // ≈0.071 — fully opaque above`
        `alpha = norm < CUT_LO ? 0 : norm < CUT_HI ? round((norm-CUT_LO)/(CUT_HI-CUT_LO)*200) : 200`
      - `PRECIP_TICKS`: [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
      - Update PRECIP comment to describe sqrt scale

- [x] R13c Edit `frontend/src/map/WeatherMap.tsx`:
      - Precipitation `renderWithInterpolation` call: add `true` for sqrtScale parameter
      - All other calls (temperature, wind_speed): unchanged (sqrtScale defaults to false)

- [x] R13d Edit `frontend/src/components/Legend.tsx`:
      - Precipitation tooltip: update to describe sqrt scale and the extended tick range

- [x] R13e Validate:
      - `npx tsc --noEmit` → 0 errors
      - `npm run build` → passes
      - Visual: light rain visible in frames 12–28 (which have 15–33% of points > 0.01 mm/hr)
      - Visual: differentiation preserved — 0.084 mm/hr (P95) clearly different from 0.406 (P99)
      - Visual: frame 0 remains fully transparent (all zeros by GCOp convention)
      - Visual: max (1.62 mm/hr) is at/near full color, not saturated

- [x] R13f Commit and push to main; confirm GitHub Pages redeploy.

**Gate**: tsc 0 errors; build passes; light rain visible in frames 12–28; data and verification unchanged.

---

## Phase R14: Temperature Display Context (Myanmar Local Time) — COMPLETE (2026-08-17)

**Classification**: Display context improvement (FR-N30). No data modification.
**Prerequisite**: R13 COMPLETE.

**Diagnostic summary** (read-only investigation, 2026-08-17):
- Temperature pipeline confirmed correct: K−273.15 applied; no double conversion; grid indexing correct
- t+0h = 00:00Z = 06:30 MMT (pre-dawn). Yangon: 21.1°C at t+0h; 28.9°C at t+6h (12:30 MMT)
- GCOp bias: −0.76°C avg vs ERA5. Diurnal: −1.45°C at 12Z daytime, ≈0 at 18Z midnight
- Root cause of perceived cold: temporal mismatch (00Z = pre-dawn MMT) + model characteristic
- temperature.bin and verification.json: UNCHANGED

- [x] R14a Confirm diagnostic findings above. No code changes until pre-read confirmed.

- [x] R14b Edit `frontend/src/map/WeatherMap.tsx` — popup time display:
      - Add MMT time alongside UTC:
        `const mmtOffsetMs = (6 * 60 + 30) * 60000;`
        `const mmtStr = new Date(dt.getTime() + mmtOffsetMs).toUTCString().replace(' GMT','').slice(5)+'MMT';`
      - Popup time row: `<div class="wx-popup-time">${timeStr} · ${mmtStr}</div>`
        OR separate rows: UTC on first line, MMT on second line

- [x] R14c Edit `frontend/src/components/InfoPanel.tsx`:
      - Add timezone note: "All times UTC. Myanmar Standard Time = UTC+6:30 (MMT, no DST)."
      - For temperature: add note about documented GCOp cold bias (−0.76°C avg vs ERA5,
        diurnal pattern; see model evaluation panel for details).

- [x] R14d Validate:
      - `npx tsc --noEmit` → 0 errors
      - `npm run build` → passes
      - Popup shows e.g. "Fri, 01 Jan 2021 00:00:00 UTC · 06:30 MMT"
      - t+6h popup shows "Fri, 01 Jan 2021 06:00:00 UTC · 12:30 MMT"

- [x] R14e Commit and push to main; confirm GitHub Pages redeploy.

**Gate**: tsc 0 errors; build passes; popup shows UTC + MMT; deployed.

---

## Acceptance Criteria (R13 + R14)

| Criterion | Specification |
|---|---|
| Temperature displayed value | Exactly matches temperature.bin at the documented K−273.15 conversion. No additional conversion. |
| Frame/location mapping | latIdx = round((lat − 9.0) / 0.25); lonIdx = round((lon − 92.0) / 0.25). Verified in diagnostic. |
| External website comparison | Diagnostic only, not ground truth. January 2021 vs current year; 00Z (06:30 MMT) vs daytime. |
| Temperature data | temperature.bin and verification.json: UNCHANGED. |
| Non-zero rainfall visible | P75 (0.005 mm/hr) and above: visible with sqrt scale. |
| Light rainfall not disappearing | P90 (0.028 mm/hr): clearly visible; P95 (0.084): strong color. |
| Color differentiation | P90 (0.028) visually distinct from P99 (0.406) and max (1.62). |
| Precipitation data | precipitation.bin and verification.json: UNCHANGED. |

---

## Dependencies and Execution Order

```
R001–R009 (research) → R010 (MODEL SELECTION GATE) → ...
                                                       ↓
                                              R020–R023 (data availability)
                                                       ↓
                                              R030–R034 (smoke test / hardware)
                                                       ↓
                                         RS01–RS05 (Spec Kit update) ← CURRENT PHASE
                                                       ↓
                                         USER APPROVES SPEC KIT
                                                       ↓
                                              R040–R044 (7-day pipeline + validate)
                                                       ↓
                                         RS10 (schema v4.0 string fix — COMPLETE)
                                         RS11–RS14 (demo data — DEFERRED, ADR-024)
                                                       ↓
                                         ┌─────────────┴─────────────┐
                                  R050–R052 (verification)    R060–R064 (data layer types)
                                         └─────────────┬─────────────┘
                                                        ↓
                                              R070–R080 (frontend components)
                                                        ↓
                                              R090–R095 (eval popup)
                                                        ↓
                                         USER APPROVAL → R100–R109 (deploy)
                                                        ↓
                                         R10a–R10e (precip calibration fix) ← COMPLETE
                                                        ↓
                         R11a–R11i (canvas arrows, commit 2ee8c60) ← SUPERSEDED
                                                        ↓
                                         R12a–R12j (SVG wind arrow overlay)
```

---

## Acceptance Criteria (New Target)

| Criterion | Task(s) | Status |
|-----------|---------|--------|
| Model selected, documented, approved | R010 | **COMPLETE** (GCOp, user approved) |
| ARCO provides all init variables for 2021-01-01 | R022 | **COMPLETE** (82 vars, NaN=0) |
| Smoke test passes on available hardware | R030 | **COMPLETE** (R3 PASS, ADR-018) |
| Output variables confirmed (names, units, semantics) | R032 | **COMPLETE** (see ADR-018) |
| Spec Kit updated (constitution/research/spec/plan/tasks) | RS01–RS05 | **COMPLETE** (2026-08-16) |
| Full 7-day forecast produced (schema v4.0) | R041 | **COMPLETE** (2026-08-17, 18.7h) |
| 0 validation errors (schema v4.0) | R044 | **COMPLETE** — 0 errors, PASS |
| ERA5 evaluation produced (2021-01-01 to 2021-01-08) | R051 | **COMPLETE** — exit 0, PASS |
| Wind direction verified with circular MAE | R050 | **COMPLETE** — 16.91° |
| 4-variable frontend renders correctly | R080 | NOT STARTED |
| Timeline dynamically derived from metadata | R073 | NOT STARTED |
| Precipitation disclosed as mm/hr with conversion documented | R075 | NOT STARTED |
| Wind direction disclosed with FROM convention | R075 | NOT STARTED |
| Evaluation popup: "Historical Model Evaluation" heading | R090 | NOT STARTED |
| Evaluation popup: no "X% accurate" language | R095 | NOT STARTED |
| Deployed to GitHub Pages, is_demo=false | R109 | NOT STARTED |
