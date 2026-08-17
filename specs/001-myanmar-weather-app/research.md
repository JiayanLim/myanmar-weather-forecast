# Research: Myanmar Weather Forecast App — Earth2Studio Discovery

**Feature**: 001-myanmar-weather-app
**Date**: 2026-08-12 (revised — ADR-012 updated with Phase R1 model research results)
**Phase**: Phase R1 Research complete — model recommendation pending user approval (ADR-012)

---

## ARCHITECTURE DECISION RECORD

### ADR-001: Model Selection — GraphCastSmall (supersedes Aurora1p5)

**Decision**: Use `earth2studio.models.px.GraphCastSmall` as the primary prognostic model.

**Context**: Aurora1p5 was the original model selection. It failed at the first inference step
on a free Colab NVIDIA T4 (16 GB VRAM) with `OutOfMemoryError` — 13.38 GB already allocated,
attempted to allocate 824 MB when only 571 MB remained. Memory optimizations (bfloat16,
inference_mode, expandable_segments) were applied but insufficient for Aurora1p5's attention
mechanism at 0.25° resolution.

**Why GraphCastSmall**:
- 1.0° global resolution — substantially smaller activation memory than Aurora1p5's 0.25°
- JAX backend (not PyTorch) — XLA ARM64 runs efficiently on Apple M4 CPU
- Native `tp06` output: 6-hour accumulated total precipitation is a first-class model output
- Native `t2m` output: 2m temperature included in the 83-variable output set
- No diagnostic model required (no PrecipitationAFNO)
- Scientifically valid global medium-range forecast model (DeepMind/Google)
- Model weights: freely available via Google Cloud Storage
- Checkpoint: `params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz`
- Package source: `gs://dm_graphcast/graphcast`

**Hardware validation (completed)**:
- Validated on Apple M4 CPU (24 GB unified memory), JAX CPU / XLA ARM64
- Full 8-step (48h) inference: ~54s; total pipeline: ~78s end-to-end
- Peak RSS: ~2.34 GB
- No GPU required for production inference on this hardware
- GPU (CUDA) path remains available for future use but is not the validated production path

**Constraints accepted vs. Aurora1p5**:
- 1.0° resolution (coarser than Aurora1p5's 0.25°)
- 6h native timestep (not hourly)
- No sea ice concentration (sic) gap to handle — GraphCastSmall inputs do not require sic

---

### ADR-002: Initialization Source — ARCO (primary) / IFS (operational)

**Decision**: ARCO as primary historical/development source; IFS for operational real-time.
NCAR_ERA5 is NOT compatible. GFS is NOT a verified compatible source.

**Why ARCO**:
- Provides all 83 GraphCastSmall input variables including tp06 and t2m
- Free via Google Cloud (zarr) — no credentials required
- Historical ERA5 reanalysis (1959–2023) — suitable for development and historical validation
- Verified compatible via WB2 lexicon cross-check against GraphCastSmall input_coords

**Why IFS**:
- Provides all 83 GraphCastSmall input variables including tp06
- Real-time operational data — no credentials required (ECMWF open data)
- Verified compatible — IFS open data includes tp06 unlike its Aurora1p5 variant
- Required for near-real-time production forecasts

**Why NOT NCAR_ERA5**:
- NCAR_ERA5 lexicon (verified) does NOT include `tp06`
- GraphCastSmall requires tp06 as an initialization variable
- NCAR_ERA5 cannot be used without custom variable bridging

**Why NOT GFS**:
- GFS compatibility with GraphCastSmall has NOT been verified
- The constitution (§II) explicitly states: "GFS is NOT a verified initialization source for GraphCastSmall"
- Do not use GFS until compatibility is confirmed via lexicon audit

**Two-timestep initialization requirement** (verified from Earth2Studio source):
- GraphCastSmall requires TWO consecutive time steps as input: t−6h and t+0h
- Both must be fetched from the initialization source before inference begins
- This means a single `earth2studio.data.ARCO` or `IFS` fetch must cover both times

---

### ADR-003: Precipitation — GraphCastSmall Native tp06 (No Transform)

**Decision**: Use GraphCastSmall's native `tp06` output directly. No log transform or diagnostic
model is required. The only conversion is ×1000 to obtain mm/6h.

**What tp06 represents**:
- 6-hour accumulated total precipitation
- Produced natively by GraphCastSmall at each 6h lead time (t+6h, t+12h, …, t+48h)
- Units in raw model output: metres (physical space, not log space)
- Conversion: metres × 1000 = mm per 6-hour accumulation period

**CRITICAL: No log/exp transform**:
Unlike Aurora1p5's `tp1h` (which required `exp()` before use), GraphCastSmall's `tp06` is
already in physical metres. Applying an exponential transform would be incorrect and produce
wildly inflated values.

**Physical constraint**:
- tp06 MUST be non-negative (no negative precipitation)
- Clamp any sub-zero values to 0.0 (numerical noise only)
- Plausible range: 0–150 mm / 6h (extreme tropical threshold ~100 mm/6h)

**t+0h frame**:
- Set to 0.0 mm/6h explicitly (no forecast accumulation has occurred at the init hour)

**PrecipitationAFNO status**:
Not required. GraphCastSmall natively outputs tp06. Do not chain a diagnostic precipitation
model.

---

### ADR-004: Temporal Resolution — Native 6h Steps, 48h Horizon (Validated)

**Decision**: GraphCastSmall produces one forecast step every 6 hours. The MVP covers a 48-hour
horizon with 8 forecast steps plus the t+0h initialization frame = 9 total frames.

**Validated configuration**:
- `GC_N_STEPS = 8` (autoregressive rollout steps)
- `GC_N_FRAMES = 9` (including t+0h init frame)
- `GC_HORIZON_HOURS = 48`
- `GC_STEP_HOURS = 6`

**Native steps**: t+0h, t+6h, t+12h, t+18h, t+24h, t+30h, t+36h, t+42h, t+48h

**Interpolation is PROHIBITED** (Constitution §VI):
- Synthesizing intermediate hourly frames is forbidden
- Only the 9 native 6h frames are displayed
- Timeline slider steps in 6h increments

**Lead time display**:
- t+0h: initialization state (from analysis)
- t+6h through t+48h: forecast frames

---

### ADR-005: Precipitation Display — 6-Hour Accumulation

**Decision**: Display precipitation as mm / 6h with mandatory disclosure that values represent
the total rainfall accumulated during the 6-hour forecast period ending at the displayed time.

**Rationale**:
- tp06 is a 6-hour accumulated variable — NOT an instantaneous rate
- Dividing by 6 to create an "mm/h" rate is PROHIBITED (Constitution §VI)
- The UI MUST display a tooltip/info note: "Precipitation values represent total rainfall
  accumulated during the 6-hour forecast period ending at the displayed time."

---

### ADR-006: Forecast Data Format — Schema v3.0 (Validated)

**Decision**: Float32 binary arrays (one per variable) + `forecast.json` metadata.

**Validated artifact layout**:
```
data/
├── demo/ or forecast/
│   ├── forecast.json          # All metadata (schema v3.0)
│   ├── precipitation.bin      # [9 × 21 × 11] float32
│   └── temperature.bin        # [9 × 21 × 11] float32
```

**Grid dimensions** (Myanmar 1.0° bbox):
- lat: 9.0°N to 29.0°N → 21 points (step 1.0°, ascending south-to-north)
- lon: 92.0°E to 102.0°E → 11 points (step 1.0°)
- times: t+0h to t+48h inclusive at 6h steps → 9 frames (indices 0–8)

**Validated sizes**: 9 × 21 × 11 × 4 bytes = 8,316 bytes ≈ 8.1 KB per variable; ~16.6 KB total

**Note on t+0h frame**:
- Precipitation: set to 0.0 (no forecast accumulation at init hour)
- Temperature: the analysis t2m from the data source

**Schema version**: 3.0 (incompatible with v2.0; `schema_version` field must be checked)

---

### ADR-007: Hardware — Apple M4 CPU (Validated Production Path)

**Decision**: Local M4 CPU with JAX CPU / XLA ARM64 is the validated production inference path.
No GPU or Colab is required for the current deployment workflow.

**Validated configuration**:
- Hardware: Apple Silicon M4, 24 GB unified memory
- JAX backend: CPU (XLA ARM64)
- `JAX_PLATFORM_NAME=cpu` must be set before any JAX import
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` prevents pre-allocation
- Full 48h pipeline (8 AR steps): ~78s end-to-end, ~54s inference
- Peak RSS: ~2.34 GB

**Aurora1p5 GPU context (historical)**:
Aurora1p5 failed on T4 (16 GB) with CUDA OOM at the first inference step. After switching
to GraphCastSmall (1.0°, JAX), the model became runnable on CPU, eliminating the GPU
requirement entirely for the current 48h horizon.

**MPS incompatibility (Apple M4 — documented, do not retry)**:
GraphCastSmall / JAX cannot run on MPS (Metal Performance Shaders):
- `x.double()` in fourier.py fails (MPS has no float64)
- `from_numpy(lat).to(device)` fails (float64→MPS)
- `MPSNDArrayDescriptor sliceDimension` SIGABRT (unresolvable)
JAX CPU (XLA ARM64) is the correct backend for M4.

**GPU path**:
A CUDA GPU (any cloud GPU with ≥16 GB VRAM) can accelerate inference further. The pipeline
automatically uses `torch.device("cpu")` via the `device` parameter in `e2run.deterministic`.
The GraphCastSmall badge lists 40 GB recommended VRAM, but JAX CPU is validated and sufficient
for the current 48h horizon.

---

## 2. GraphCastSmall Variable Specification (Verified)

### Input Variables (83 total)

Includes atmospheric variables at 13 pressure levels plus surface variables including:
- `tp06` (6-hour total precipitation — required as initialization input)
- `t2m` (2-metre temperature — required as initialization input)
- Surface meteorological fields (wind, humidity, pressure, etc.)
- Pressure-level fields at 13 levels

GraphCastSmall uses the WeatherBench2 (WB2) lexicon for variable mapping.

### Output Variables

GraphCastSmall outputs the same 83 variables it ingests as input at the next 6h lead time.
Both `tp06` and `t2m` are native outputs.

### Variables used in frontend (validated)

| Variable | GraphCast name | Unit (raw output) | Transformation | Frontend unit |
|----------|---------------|-------------------|---------------|---------------|
| 6h precipitation | `tp06` | metres (physical) | × 1000, clamp ≥ 0 | mm / 6h |
| 2m temperature | `t2m` | Kelvin | K − 273.15 | °C |

Validated ranges for init_time 2022-07-01T00:00:00Z (Myanmar monsoon, ARCO):
- Temperature: [1.94, 35.35] °C across 9 frames
- Precipitation: [0.000, 17.381] mm/6h across 9 frames

---

## 3. Data Source Analysis

### earth2studio.data.ARCO

**Access**: Free via Google Cloud (zarr) — no credentials required
**Coverage**: 1959–2023 (historical only)
**Resolution**: 0.25° (upsampled to 1.0° by GraphCastSmall's Earth2Studio wrapper)
**Variables**: Full ERA5 including tp06 and t2m
**Compatible with GraphCastSmall**: YES — all 83 required variables including tp06
**Use case**: Development runs, historical validation, local M4 inference

### earth2studio.data.IFS

**Access**: ECMWF open data — no credentials required
**Resolution**: 0.25°
**Coverage**: Near-real-time (4× daily: 00, 06, 12, 18 UTC, ~6h latency)
**Variables**: Full IFS variable set including tp06 and t2m
**Compatible with GraphCastSmall**: YES — all 83 required variables confirmed
**Note**: Unlike the Aurora1p5 use case, IFS for GraphCastSmall does NOT have a sic gap
  problem. GraphCastSmall's 83 input variables do not include sic.
**Use case**: Near-real-time production forecasts

### earth2studio.data.NCAR_ERA5

**Compatible with GraphCastSmall**: NO — missing `tp06` in lexicon
**Use case**: NOT suitable for GraphCastSmall initialization

### GFS

**Compatible with GraphCastSmall**: UNVERIFIED — do not use
**Constitution requirement**: Must be explicitly validated before use

---

## 4. Pipeline Architecture (Validated)

```
ARCO (earth2studio.data.ARCO) or IFS (earth2studio.data.IFS)
    ↓ [fetch t-6h AND t+0h — two consecutive timesteps required]
GraphCastSmall (earth2studio.models.px.GraphCastSmall)
    │ JAX CPU / XLA ARM64 (Apple M4) — ~54s for 8 steps
    │ Native 6h auto-regressive rollout: t+6h, t+12h, ..., t+48h
    │ 83 output variables per step
    ↓
ZarrBackend (earth2studio.io.ZarrBackend)
    ↓
Post-processing (root["tp06"][0] and root["t2m"][0]):
    ├── tp06: lat ascending sort → Myanmar subset → metres × 1000 → mm/6h → clamp ≥ 0 → t+0h=0
    └── t2m:  lat ascending sort → Myanmar subset → K − 273.15 → °C
    ↓
myanmar_subset: lat_arr[9–29°N] × lon_arr[92–102°E] → [9 × 21 × 11]
    ↓
artifact_writer:
    ├── temperature.bin   [9 × 21 × 11] float32, C-order, little-endian
    ├── precipitation.bin [9 × 21 × 11] float32, C-order, little-endian
    └── forecast.json     schema v3.0 with full provenance
    ↓
data/forecast/ (or data/demo/)
```

**No log transform. No diagnostic precipitation model. No sic patching. No interpolation.**

---

## 5. Model Weights and Licensing

**GraphCastSmall weights**:
- Package source: `gs://dm_graphcast/graphcast`
- Checkpoint: `params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz`
- Download: Handled automatically by `GraphCastSmall.load_default_package()`
- License: DeepMind/Google license — check GCS terms for current conditions

**ARCO / ERA5 data**:
- ERA5: Copernicus Climate Data Store license (see ECMWF terms)
- ARCO mirror: Free via Google Cloud

**IFS open data**:
- ECMWF open data license (Creative Commons-compatible; check ECMWF terms)

**No secrets, tokens, or API keys are required for the production data path.**

---

## 6. Relevant Examples

- Deterministic workflow: https://nvidia.github.io/earth2studio/examples/01_getting_started/01_deterministic_workflow.html
- GraphCast example (if available): check https://nvidia.github.io/earth2studio/examples/

---

## 7. Summary: What Changed (Aurora1p5 → GraphCastSmall → 48h migration)

| Item | Aurora1p5 | GraphCastSmall (v2.0, 24h) | GraphCastSmall (v3.0, 48h) — CURRENT |
|------|-----------|----------------------------|--------------------------------------|
| Resolution | 0.25° | 1.0° | 1.0° |
| Native timestep | 1h | 6h | 6h |
| Forecast horizon | 168h (7 days) | 24h | **48h** |
| Total frames | 169 | 5 | **9** |
| Myanmar grid | 81 × 41 | 21 × 11 | 21 × 11 |
| Precipitation variable | tp1h | tp06 | tp06 |
| Precip transform | exp() × 1000 | × 1000 only | × 1000 only |
| Precipitation unit | mm / 1h | mm / 6h | mm / 6h |
| Temperature variable | t2m (displayed) | NOT displayed | **t2m displayed (°C)** |
| Variable switcher | YES | NO | **YES (Precip / Temp)** |
| Sea ice gap (sic) | YES — IFS missing sic | NO | NO |
| Init timesteps required | 1 (t+0h only) | 2 (t-6h AND t+0h) | 2 (t-6h AND t+0h) |
| Compatible init sources | IFS (sic patch), NCAR_ERA5, ARCO | ARCO, IFS | ARCO, IFS |
| Backend | PyTorch | JAX + Haiku | JAX + Haiku |
| Inference hardware | T4 GPU (OOM — failed) | CPU (unverified) | **M4 CPU (validated, ~78s)** |
| Peak RSS / VRAM | N/A (OOM) | N/A | **2.34 GB RAM** |
| Pipeline complexity | High (sic patch, log untransform) | Low | Low |
| Schema version | — | v2.0 | **v3.0** |
| Payload per variable | ~4.3 MB | ~4.6 KB | **~8.1 KB** |
| Total payload | ~4.3 MB | ~4.6 KB | **~16.6 KB** |

---

---

# NEW TARGET ARCHITECTURE RESEARCH (v4.0 target — 2026-08-12)

**Status**: Research phase. No model selected. No implementation begun.
**Objective**: Investigate whether an Earth2Studio weather foundation model can generate
a useful 7-day Myanmar forecast using ERA5 historical data with 2021-01-01 initialization.

---

### ADR-008: ERA5 1990–2020 Dataset Role Clarification

**Context**: The new requirements specify "ERA5 1990–2020 historical data" as an input.
This phrasing could be misread as training or fine-tuning.

**Decision and clarification**:

There are four distinct roles ERA5 can play in an ML weather forecasting system:

| Role | Description | Applies to this project? |
|------|-------------|--------------------------|
| A. Pre-training data | ERA5 used to train the foundation model's weights from scratch | NO — this is done by model authors |
| B. Initialization / input state | ERA5 analysis for the forecast start date provided as model input | YES — this is what is meant |
| C. Fine-tuning | ERA5 used to further adapt pre-trained weights to a region or period | NO — no fine-tuning is performed |
| D. Evaluation / ground truth | ERA5 analysis used as reference for measuring forecast skill | YES — for the verification pipeline |

**For this project**:
- "ERA5 1990–2020" = the historical ERA5 reanalysis record available prior to 2021-01-01
- The initialization date is 2021-01-01T00:00:00Z
- ARCO provides ERA5 analysis for this date (confirmed within ARCO coverage ~1959–2023)
- No training or fine-tuning is performed
- **DO NOT** claim that using ERA5 as the initialization state constitutes training

**Documentation requirement**: All specs, artifacts, and UI text MUST use "initialization"
or "initial atmospheric state" — never "training data" to describe this role.

---

### ADR-009: 2021-01-01 Forecast Initialization Date

**Decision**: The forecast initialization date is 2021-01-01T00:00:00Z.

**Rationale**:
- Falls within ARCO ERA5 coverage (~1959–2023) — data availability confirmed
- Allows comparison with ERA5 ground truth (2021-01-01 through 2021-01-08)
- 2021 is after the GraphCastSmall training cutoff (ERA5 1979–2015) and after
  many other foundation models' training cutoffs — this is a quasi out-of-sample test
- Provides a meaningful verification period (ERA5 available with ~5-day latency;
  2021 data is fully processed and freely available via ARCO)

**Multi-timestep initialization**: Many foundation models require multiple consecutive
time steps at initialization. For example, GraphCastSmall requires t−6h and t+0h.
The selected model's initialization requirements MUST be documented. ARCO can provide
any historical timestamp, so multi-step init is feasible.

**IFS note**: IFS provides operational real-time data and is NOT needed for 2021-01-01.
ARCO is the appropriate source.

---

### ADR-010 (new target): 7-Day Forecast Horizon

**Decision**: The new target forecast horizon is 168 hours / 7 days.

**Rationale**:
- Provides medium-range guidance useful for agricultural, humanitarian, and general
  planning in Myanmar's monsoon-prone geography
- 7 days is a well-established medium-range forecast window for operational NWP
- The selected model must be scientifically capable of producing useful forecasts
  to 7 days (168h)

**Temporal resolution**:
- The model's native timestep determines the frame count
- 6h timestep: 168h / 6h = 28 AR steps + init = 29 frames
- 1h timestep: 169 frames
- 3h timestep: 57 frames
- The actual timestep MUST be documented once the model is selected

**DO NOT** interpolate between native model timesteps to create sub-native resolution
frames. Only native model outputs are displayed (Constitution §VI).

---

### ADR-011 (new target): Four-Variable Meteorological Product

**Decision**: The new target exposes four display variables.

**Variables and derivation requirements**:

| Variable | Display unit | Derivation | Key questions to resolve |
|----------|-------------|------------|--------------------------|
| Precipitation | mm/hr | Model native output (TBD) → mm/hr | What is the model's native precipitation variable? Is it accumulated or rate? Over what period? |
| Wind Direction | degrees (meteorological) | `(270 − atan2d(v, u)) mod 360` | Does the model output u10m/v10m or pressure-level winds? What level? |
| Wind Speed | knots | `sqrt(u² + v²) × 1.94384` | Same as above; confirm m/s native units |
| Temperature | °C | `K − 273.15` (if Kelvin) | Does the model output t2m? What height/level? |

**Precipitation semantics — must be resolved per model**:
- If model outputs accumulated total precipitation over each timestep (like GraphCastSmall's tp06):
  `mm/hr = accumulated_mm_per_step / step_hours`
  Example: tp06 = 12mm over 6h → 2 mm/hr
- If model outputs instantaneous precipitation rate in mm/hr: no conversion needed
- If model outputs in metres/step: first convert to mm, then divide by step_hours
- The conversion formula MUST be documented in the pipeline and disclosed in the UI
- Aurora's tp1h used `exp(x) × 0.001 → metres → mm` — this is model-specific and
  MUST NOT be assumed for other models

**Wind convention**:
- Meteorological wind direction = direction FROM which wind blows, clockwise from North
- Derived from u (eastward) and v (northward) components:
  `wind_dir_deg = (270 − atan2(v, u) × 180/π) mod 360`
- A model providing 10-metre winds (u10m, v10m) is preferred for near-surface display
- If only pressure-level winds are available, the level must be documented

**Open questions to resolve during research**:
1. Which Earth2Studio model provides all four required variable groups?
2. Does the model output near-surface wind components (u10m, v10m)?
3. What is the model's native precipitation variable and its temporal semantics?
4. What is the model's native timestep?
5. Is 7-day skill meaningful for the selected model at Myanmar's latitude?

---

### ADR-012 (new target): Candidate Earth2Studio Model Comparison

**Status**: REVISED AND APPROVED — GraphCastOperational selected. Gate R010 RE-CLOSED.
**Research date**: 2026-08-12
**Initial approval (FuXi)**: 2026-08-12 — superseded after R2 failure
**Revised approval (GraphCastOperational)**: 2026-08-12
**Method**: Inspection of Earth2Studio 0.17.0 installed source files; R2 ARCO/M4 compatibility verification

---

#### Step 1: Full Model Enumeration (earth2studio.models.px, v0.17.0)

All models confirmed present in installed Earth2Studio 0.17.0:

```
ACE2ERA5, AIFS, AIFS2, AIFS2ENS, AIFSENS, Atlas, Aurora, Aurora1p5,
Aurora1p5Ensemble, CBottleVideo, DLESyM, DLWP, FCN, FCN3, FengWu, FuXi,
GenCastMini, GraphCastOperational, GraphCastSmall, Pangu3, Pangu6, Pangu24,
SFNO, StormCast, UCast
```

---

#### Step 2: Variable Screening — Four-Variable Coverage

Required: **t2m**, **u10m**, **v10m**, **any precipitation variable**.

Inspected each model's VARIABLES list and/or output_coords from source code.

**DISQUALIFIED — missing precipitation (tp) variable**:

| Model | Variables | Reason |
|-------|-----------|--------|
| Pangu3 / Pangu6 / Pangu24 | z, q, t, u, v (pressure levels) + msl, u10m, v10m, t2m | **NO precipitation output** |
| GenCastMini | t2m, msl, u10m, v10m, sst (5 vars only) | **NO precipitation variable** |
| FCN | pressure-level z/q/t/u/v + surface vars; no tp* | **NO precipitation variable** |
| FCN3 | 72 pressure-level and surface vars; no tp* | **NO precipitation variable** |
| FengWu | Similar to Pangu; pressure + surface; no tp* | **NO precipitation variable** |
| DLWP, DLESyM, StormCast, UCast | Specialized models; no tp* at surface | Not applicable to this use case |
| Atlas, SFNO, ACE2ERA5 | Research/specialized; missing one or more required vars | Not applicable |
| AuroraEnsemble / AIFS2ENS / AIFSENS | Ensemble variants — same var sets as deterministic, higher compute | Out of scope (ensemble) |

---

**VIABLE — all four required variable groups confirmed**:

| Model | t2m | u10m | v10m | Precip var | Precip type |
|-------|-----|------|------|-----------|-------------|
| GraphCastSmall | ✓ | ✓ | ✓ | tp06 | 6h accum, metres |
| GraphCastOperational | ✓ | ✓ | ✓ | tp06 (output-only) | 6h accum, metres; zeros for input |
| AIFS | ✓ | ✓ | ✓ | tp06, cp06 | 6h accum, metres |
| AIFS2 | ✓ | ✓ | ✓ | tp06, u100m, v100m | 6h accum, metres |
| FuXi | ✓ | ✓ | ✓ | tp06 | 6h accum, metres; internally mm→m |
| Aurora1p5 | ✓ | ✓ | ✓ | tp1h | 1h, log-space (`aurora_log_untransform` required) |

---

#### Step 3: Detailed Specification per Viable Candidate

All specs sourced from `input_coords`, `output_coords`, and source code in Earth2Studio 0.17.0.

---

**GraphCastSmall**
- Native resolution: 1.0° global (lat 181pt, lon 360pt)
- Myanmar subset at 1.0°: 21 × 11 = 231 grid points
- Native timestep: 6h (single AR step = +6h)
- Init requirement: two timesteps — t−6h and t+0h
- ARCO compatibility: confirmed (ARCO covers ERA5 1959-2023; t−6h = 2020-12-31T18Z available)
- Checkpoint: `gs://dm_graphcast/graphcast_small` — GCS, public, no auth
- Training data: `ERA5 1979-2015 - resolution 1.0 - pressure levels 13`
- Total variables: 83 (pressure-level + surface including tp06, t2m, u10m, v10m)
- 7-day (168h): 28 AR steps — no known hard limit in E2Studio implementation
- tp06 semantics: 6h accumulated total precipitation in **metres** → ×1000 = mm/6h → ÷6 = mm/hr
- Backend: JAX + Haiku (not PyTorch); XLA ARM64 runs on M4 CPU
- **Hardware validated**: M4 CPU, 8 steps/48h in ~54s, peak RSS 2.34 GB — PROVEN
- Earth2Studio integration: battle-tested in this project; two-timestep ARCO fetch confirmed

**GraphCastOperational**
- Native resolution: 0.25° global (lat 721pt, lon 1440pt) — 16× more points than GraphCastSmall
- Myanmar subset at 0.25°: ~81 × 41 = 3,321 grid points
- Native timestep: 6h
- Init requirement: two timesteps — t−6h and t+0h
- ARCO compatibility: ARCO provides 0.25° ERA5; should satisfy all 82 input variables
- Checkpoint: `gs://dm_graphcast/graphcast` (params file: `GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz`)
- Training data: `ERA5-HRES 1979-2021 - resolution 0.25`
- Total variables: 82 input + tp06 as output-only (zeros injected for input — model does NOT require tp06 as input)
- 7-day (168h): 28 AR steps — no hard limit in E2Studio implementation
- tp06 semantics: 6h accumulated total precipitation in **metres** — same as GraphCastSmall
- Backend: JAX + Haiku — same backend as GraphCastSmall
- Hardware: 0.25° means ~16× memory over GraphCastSmall; estimated RSS significantly higher (UNVERIFIED)
- Earth2Studio integration: same API as GraphCastSmall; additional tp06 zeros injection handled internally

**AIFS (ECMWF AI Forecasting System)**
- Native resolution: 0.25° global (lat 721pt, lon 1440pt)
- Myanmar subset at 0.25°: ~81 × 41 = 3,321 grid points
- Native timestep: 6h
- Init requirement: two timesteps — t−6h and t+0h
- ARCO compatibility: ECMWF AIFS trained on ERA5; ARCO provides ERA5; all required vars expected
- Checkpoint: `hf://ecmwf/aifs-single-1.1` — HuggingFace, public, no auth
- Checkpoint file: `aifs-single-mse-1.1.ckpt` (loaded via `torch.load`)
- Total variables: 115 including tp06, cp06, u10m, v10m, t2m, u100m, v100m
- 7-day (168h): 28 AR steps — no E2Studio limit; ECMWF publishes 10-day forecasts operationally
- tp06 semantics: 6h accumulated total precipitation in **metres** — same conversion as GraphCastSmall
- Backend: PyTorch — M4 CPU via `torch.cpu`; MPS has float64 issues (avoid MPS)
- Scientific maturity: ECMWF production model; strong published skill vs. NWP
- Earth2Studio integration: mature; metadata from `ai-models.json` in checkpoint

**AIFS2 (ECMWF AI Forecasting System v2)**
- Native resolution: 0.25° global (same as AIFS)
- Native timestep: 6h; two-timestep init
- ARCO compatibility: same as AIFS; ERA5-based training
- Checkpoint: `hf://ecmwf/aifs-single-2.0@08286fc...` — HuggingFace, pinned commit
- Total variables: 134 (larger than AIFS; includes additional wave and surface fields)
- Precipitation: tp06 in metres; same semantics as AIFS
- Backend: PyTorch
- Hardware concern: 134 variables × 721 × 1440 grid points — significantly larger than AIFS
- Status: newer than AIFS but less field-proven at the time of this research

**FuXi**
- Native resolution: 0.25° global (lat 721pt, lon 1440pt) — south-pole including
- Myanmar subset at 0.25°: ~81 × 41 = 3,321 grid points
- Native timestep: 6h; two-timestep init (t−6h and t+0h)
- ARCO compatibility: FuXi trained on ERA5 0.25°; all 70 input variables from ERA5
- Checkpoint: `hf://NickGeneva/earth_ai/fuxi` — HuggingFace community repo, 3 ONNX files (short.onnx, medium.onnx, long.onnx)
- Total variables: 70 (pressure-level z/t/u/v/r + surface t2m, u10m, v10m, msl, tp06)
- 7-day (168h): 28 AR steps
  - Steps 0–19 (0–120h / 5 days): short ONNX model
  - Steps 20–39 (120–240h / 10 days): medium ONNX model (auto-loaded at step 20)
  - For 7-day forecast: short (steps 0–19) + first 8 steps of medium (steps 20–27)
- tp06 semantics: 6h accumulated precipitation in metres; Earth2Studio internally converts to mm for computation then back to metres for output
- Backend: ONNX Runtime (`onnxruntime`) — portable; M4 CPU expected to work well; no MPS/CUDA dependency
- Published capability: 15-day forecasts (3 model cascade); 7-day is within short+medium range
- Earth2Studio integration: functional; three-ONNX cascade is automatic in `_default_generator`

**Aurora1p5 (Microsoft)**
- Native resolution: 0.25° global (lat 720pt, lon 1440pt — south-pole NOT included)
- Native timestep: 1h (sub-stepped; 6h AR steps internally divided to produce 1h output)
- Init requirement: two timesteps — t−6h and t+0h
- ARCO compatibility: requires ERA5 surface+pressure vars; sic (sea ice concentration) gap known
- Checkpoint: `hf://microsoft/aurora` — HuggingFace, public
- Total variables: 18 surface (including tp1h, t2m, u10m, v10m) + pressure-level z/q/t/u/v
- tp1h semantics: **log-space** accumulated 1h precipitation; physical units require `aurora_log_untransform(v)` call before use
  - `from aurora.normalisation import log_untransform as aurora_log_untransform`
  - tp1h → mm/hr: `log_untransform(tp1h) × 1000` (convert m to mm; already 1h rate)
- 7-day (168h): 168 AR steps at 1h — significantly more steps than 6h models
- Hardware concern: **OOM on T4 16 GB VRAM** at first inference step in earlier testing (see ADR-001)
- Backend: PyTorch; MPS issues (float64 SIGABRT on M4)
- Earth2Studio integration: functional but complex precipitation pipeline

---

#### Step 4: Decision Matrix

Weights defined in Phase R1 specification (sum = 100%):

| Criterion | Weight | GraphCastSmall | GraphCastOp | AIFS | AIFS2 | FuXi | Aurora1p5 |
|-----------|--------|----------------|-------------|------|-------|------|-----------|
| Four-variable coverage | 20% | 9 | 9 | 10 | 10 | 10 | 9 |
| 7-day forecast capability | 15% | 8 | 8 | 9 | 9 | 10 | 8 |
| ERA5 init compatibility | 15% | 10 | 10 | 9 | 9 | 9 | 8 |
| Spatial resolution | 10% | 4 | 10 | 10 | 10 | 10 | 10 |
| Temporal resolution | 10% | 6 | 6 | 6 | 6 | 6 | 10 |
| Forecast quality / scientific maturity | 15% | 7 | 8 | 10 | 9 | 8 | 7 |
| M4 / available hardware feasibility | 10% | 10 | 5 | 5 | 4 | 8 | 2 |
| Earth2Studio integration maturity | 5% | 10 | 9 | 8 | 7 | 7 | 8 |
| **Weighted total** | **100%** | **7.90** | **8.05** | **8.55** | **8.30** | **8.90** | **7.45** |

Score rationale (0–10 per criterion):
- **GraphCastSmall**: 1.0° resolution (4/10) heavily penalises it for the new target; proven hardware (10/10)
- **GraphCastOperational**: 0.25° JAX backend scales from proven GCSmall path; tp06 output-only is acceptable; GCS checkpoint (not HuggingFace) adds minor friction; hardware unverified at 0.25°
- **AIFS**: ECMWF production model, highest scientific maturity (10/10); all vars; PyTorch on M4 CPU unverified (5/10); HuggingFace checkpoint easy to download
- **AIFS2**: Slightly larger and newer than AIFS; more variables but higher memory risk (4/10 hardware)
- **FuXi**: Highest overall (8.90); ONNX Runtime is most portable for M4 CPU (8/10 hardware); automatic short+medium cascade covers 7-day natively; all required vars; community HuggingFace checkpoint; 7-day capability proven in published results (10/10)
- **Aurora1p5**: 1h resolution excellent for temporal detail (10/10) but hardware risk is severe — OOM on T4 (2/10); log-space tp1h adds pipeline complexity; 168 steps at 1h is significantly more compute than 28 steps at 6h

---

#### Step 5: Recommendation

**Primary candidate: FuXi** (weighted score 8.90)

Rationale:
1. All four required variables (tp06, t2m, u10m, v10m) confirmed in source
2. tp06 semantics: metres, 6h accumulation — same as GraphCastSmall (familiar, tested pipeline)
3. ONNX Runtime backend: portable across hardware; M4 CPU expected to work without MPS/CUDA issues
4. Automatic short+medium cascade in Earth2Studio covers 7-day forecast natively
5. 0.25° resolution matches the new target (vs. legacy 1.0° of GraphCastSmall)
6. HuggingFace community checkpoint (`hf://NickGeneva/earth_ai/fuxi`) — public, no auth
7. Precipitation conversion: `metres × 1000 ÷ 6 = mm/hr` — no log transform needed

Risks:
- Community HuggingFace checkpoint (not first-party) — ONNX files must be verified on first load
- Short+medium model cascade increases checkpoint download size (3 ONNX files)
- Hardware feasibility at 0.25° is UNVERIFIED — smoke test (Phase R3) required before committing

**Secondary candidate: AIFS** (weighted score 8.55)

Rationale:
- ECMWF production model — highest scientific credibility
- All required variables; tp06 semantics identical to GraphCastSmall
- If FuXi fails Phase R3 smoke test (OOM), AIFS is the recommended fallback

**NOT recommended (with reasons)**:
- GraphCastSmall: Correct for legacy (v3.0), but 1.0° resolution is insufficient for the new target
- Aurora1p5: OOM failure on T4 GPU documented; log-space tp1h adds complexity; 2/10 hardware risk
- GraphCastOperational: Viable backup but GCS checkpoint (not HuggingFace); hardware unverified
- AIFS2: More variables and higher memory than AIFS with no clear benefit for this project
- Pangu3/6/24: **No precipitation output** — disqualified
- GenCastMini, FCN, FCN3, FengWu: **No precipitation output** — disqualified

---

#### MODEL SELECTION GATE (R010)

**Status: CLOSED — APPROVED 2026-08-12**

**Initial approved selection (superseded)**: `earth2studio.models.px.FuXi`
**Rejection reason**: R2 FAIL — onnxruntime-gpu has no macOS ARM64 wheel; ARCO lacks r* (relative humidity) variables required by FuXi.

**Secondary rejected candidate**: `earth2studio.models.px.AIFS`
**Rejection reason**: R2 FAIL — flash-attn dependency is CUDA-only; cannot be built or installed on macOS ARM64.

**Secondary rejected candidate**: `earth2studio.models.px.AIFS2`
**Rejection reason**: Same flash-attn requirement as AIFS.

---

**Approved selection (final)**: `earth2studio.models.px.GraphCastOperational`
**Approval date**: 2026-08-12
**ADR-012 weighted score**: 8.05 / 10.00 (see decision matrix above)

**Why GraphCastOperational over all other candidates**:
1. Only 0.25° model that is both M4-compatible AND ARCO-compatible
2. Same JAX/Haiku backend as proven GraphCastSmall — highest confidence for M4 CPU execution
3. All 82 input variables available in ARCO (uses q* not r*; no custom preprocessing)
4. tp06 as output-only (Earth2Studio injects zeros; ARCO tp06 not needed at init)
5. All required Earth2Studio[graphcast] dependencies already installed
6. Same trusted DeepMind GCS checkpoint source
7. No cascade complexity — single JAX model

**Approved model specification**:

| Property | Value |
|----------|-------|
| Earth2Studio class | `earth2studio.models.px.GraphCastOperational` |
| Resolution | 0.25° global (721 × 1440) |
| Native timestep | 6h |
| Forecast horizon | 168h / 7 days |
| Forecast steps | 28 AR steps + t+0h = 29 frames |
| Backend | JAX + Haiku (same as GraphCastSmall) |
| Init variables | 82 vars: z/q/t/u/v/w at 13 levels + msl; tp06 zeros-only input |
| Output variables | 82 + tp06 = all four required groups |
| Precipitation native units | physical metres, 6-hour accumulation (output-only) |
| Precipitation log transform | **NONE** (straight metres) |
| Precipitation display conversion | metres × 1000 ÷ 6 = mm/hr (verify in Phase R3) |
| Wind speed derivation | sqrt(u10m² + v10m²) × 1.94384 → knots |
| Wind direction convention | meteorological FROM; (270 − atan2d(v,u)) mod 360 |
| Temperature conversion | K − 273.15 → °C |
| Checkpoint | gs://dm_graphcast/graphcast |
| Checkpoint auth | public GCS, no authentication required |
| Myanmar grid | 81 × 41 = 3,321 grid points at 0.25° |
| Payload estimate | ≈ 1.47 MB total (4 variables, 29 frames) |
| Hardware | M4 CPU — UNVERIFIED at 0.25°; Phase R3 smoke test mandatory |

---

### ADR-013 (new target): Wind Speed and Direction Derivation

**Decision**: Wind speed and direction will be derived from u/v wind components.

**Meteorological wind direction formula**:
```
wind_dir_deg = (270 - atan2(v, u) × 180/π) mod 360
```
This gives the direction FROM which the wind blows, measured clockwise from North.
Example: u=0, v=−10 m/s → wind blows FROM North (0°/360°).

**Wind speed formula (m/s → knots)**:
```
wind_speed_kt = sqrt(u² + v²) × 1.94384
```

**Variable selection**:
- Near-surface (10m): use u10m, v10m if available
- If the model only outputs 10m winds via pressure-level interpolation, document the level
- The model's exact variable names MUST be confirmed from Earth2Studio output_coords

**Unresolved**: Which wind level is available in the selected model's native outputs.

---

### ADR-014 (new target): Precipitation Temporal Semantics and mm/hr Display

**Decision**: The display unit for precipitation is mm/hr.

**Conversion requirements by model type**:

| Native model output | Native unit | Conversion to mm/hr |
|--------------------|-------------|---------------------|
| Accumulated total per step (e.g., GraphCastSmall tp06) | metres / step_h hours | `metres × 1000 / step_h` |
| 1-hour accumulated (e.g., Aurora tp1h after untransform) | mm / 1h | Already mm/hr if 1h step |
| Instantaneous rate | mm/hr | No conversion |
| Accumulated from forecast start | metres | Requires differencing consecutive steps |

**Documentation requirements**:
- The pipeline MUST record: native_variable_name, native_unit, accumulation_period_h,
  conversion_formula, output_unit in the precipitation transformation provenance
- The UI MUST display a disclosure: "Precipitation shown as estimated hourly average
  [mm/hr] calculated from [model native output description]."
- The accumulation period and conversion formula MUST appear in forecast.json

**Prohibition**: Do NOT apply the Aurora-specific log-untransform (0.001 × (eˣ − 1))
to any other model's precipitation output. This transform is Aurora-specific and will
produce incorrect results if applied to GraphCastSmall, GenCast, or other models.

---

### ADR-015 (new target): ERA5-Based Evaluation Methodology

**Decision**: The verification pipeline will compare the 2021-01-01 forecast against
ERA5 analysis for the corresponding period (2021-01-01 through 2021-01-08).

**Data availability**:
- ERA5 for 2021 is fully processed and available via ARCO
- No latency issues (unlike operational forecasts compared to near-real-time ERA5)
- ERA5 spatial resolution: 0.25° — interpolation to model-native grid needed

**Evaluation metrics by variable**:

Temperature (t2m):
- MAE (°C), RMSE (°C), Bias (°C) at each lead time
- Domain-average and spatial error maps over Myanmar bbox

Precipitation:
- MAE (mm/hr), RMSE (mm/hr), Bias (mm/hr) at each lead time
- Categorical scores at appropriate thresholds: POD, FAR, CSI
- Thresholds in mm/hr: 0.1, 0.5, 1.0, 5.0 (to be revised once model output range is known)

Wind speed:
- MAE (kt), RMSE (kt), Bias (kt)
- Vector error if ERA5 u/v components available

Wind direction:
- Circular/angular MAE (degrees) — MUST use circular difference, not ordinary subtraction
  `angular_error = atan2d(sin(fcst−obs), cos(fcst−obs))` or equivalent
- Ordinary subtraction of angles is incorrect (e.g., 359° vs 1° should give 2°, not 358°)

**ERA5 precipitation reference**:
- ERA5 tp is cumulative from analysis start; 6h accumulation = tp(t) − tp(t−6h)
- Converting to mm/hr: divide 6h accumulation by 6
- ERA5 native temporal resolution is 1h; verify how ARCO provides tp at analysis times

**Caveats to document in verification output**:
1. ERA5 is a reanalysis, not direct observations
2. If the model was trained on ERA5 (same data family), skill scores may be optimistic
3. Double-penalty problem applies to precipitation categorical scores
4. 1.0° model resolution vs ERA5's 0.25° creates inherent scale mismatch
5. This verification covers one forecast cycle; single-cycle statistics are not robust

**Implementation timing**: The evaluation pipeline is a FUTURE requirement.
It should NOT be implemented until the forecast pipeline is working and validated.

---

### ADR-016 (new target): Compute and Hardware Strategy

**Status**: OPEN — pending model selection.

**Known constraints**:
- Available hardware: Apple M4 CPU, 24 GB unified memory (validated for GraphCastSmall)
- No dedicated CUDA GPU available locally
- GraphCastSmall 48h: ~78s, 2.34 GB RSS on M4 CPU (validated)
- Aurora1p5 on T4 16 GB VRAM: FAILED (OOM at first inference step)

**Hardware strategy by scenario**:

| Scenario | Hardware | Risk | Action |
|----------|---------|------|--------|
| Selected model fits on M4 CPU | M4 CPU | Low | Proceed as with GraphCastSmall |
| Selected model needs GPU ≤16 GB VRAM | Cloud GPU (T4 or equivalent) | Medium | Smoke test before committing |
| Selected model needs GPU >16 GB VRAM | A100/H100 or equivalent | High | Budget/access required; verify first |
| Selected model is 0.25° resolution | Higher memory requirement | Medium | RSS/VRAM test with nsteps=1 |

**Recommendation**: Run a 1-step smoke test on available hardware before implementing
the full 7-day pipeline. If the smoke test fails, report the failure and reassess.

**Preliminary assessment (FuXi, pending approval)**:
- Backend: ONNX Runtime — no JAX/CUDA/MPS dependency; M4 CPU should work
- Resolution: 0.25° → ~16× more grid points than GraphCastSmall at 1.0°
- Variable count: 70 (vs. 83 for GraphCastSmall); global grid 721×1440
- Estimated memory: significantly higher than GraphCastSmall's 2.34 GB RSS; UNVERIFIED
- Action required: Phase R3 smoke test (nsteps=1) before committing to full 7-day pipeline
- Cascade: short ONNX (0–19 steps) + medium ONNX (20–39 steps); both files loaded for 7-day run

**Unresolved**: Actual RAM usage for FuXi at 0.25° on M4 CPU is unknown until Phase R3 smoke test.

---

## 9. Phase R2 — ARCO / Initialization Validation (2026-08-12)

---

### ADR-017: Phase R2 — FuXi Initialization Architecture Verification

**Date**: 2026-08-12
**Status**: COMPLETE — R2 FAIL for FuXi. Model recommendation revised. User approval required.

---

#### R2.1 ERA5 Role Clarification

**The project requirement states**: "Use 1990–2020 ERA5 data to do a 1-week prediction for 1 Jan 2021."

**Verified role**: **Type B only** — only the ERA5 atmospheric state at two timesteps is required for inference:
- t−6h: 2020-12-31T18:00Z
- t+0h: 2021-01-01T00:00Z

**ERA5 does NOT play these roles in this project**:
- Type A (pre-training): The model authors trained FuXi/GCOp on ERA5 1979–2021. This project does NOT modify model weights. Pre-training data is irrelevant to inference.
- Type C (fine-tuning): This project performs NO fine-tuning.
- Type D (evaluation reference): Phase R5, future work.

**"1990–2020 ERA5"** in the requirement = the initialization data source is ERA5, which covers 1990 through (at least) 2021. Only two timestamps are fetched for initialization.

**ARCO provides ERA5 1959–2023** at 0.25° resolution, hourly. The two required timestamps (2020-12-31T18Z and 2021-01-01T00Z) are within ARCO coverage.

---

#### R2.2 FuXi Initialization Requirements (from source inspection)

**FuXi input_coords** (verified from `fuxi.py` in Earth2Studio 0.17.0):
```
batch: (empty)
time: (empty)
lead_time: [timedelta64(-6, 'h'), timedelta64(0, 'h')]
variable: VARIABLES (70 vars — see below)
lat: linspace(90, -90, 721, endpoint=True)   [0.25°, south-pole inclusive]
lon: linspace(0, 360, 1440, endpoint=False)  [0.25°]
```

**FuXi VARIABLES (70 total, in order)**:

| Index | Group | Variables |
|-------|-------|-----------|
| 0–12 | Geopotential (z) | z50, z100, z150, z200, z250, z300, z400, z500, z600, z700, z850, z925, z1000 |
| 13–25 | Temperature (t) | t50, t100, t150, t200, t250, t300, t400, t500, t600, t700, t850, t925, t1000 |
| 26–38 | U-wind (u) | u50, u100, u150, u200, u250, u300, u400, u500, u600, u700, u850, u925, u1000 |
| 39–51 | V-wind (v) | v50, v100, v150, v200, v250, v300, v400, v500, v600, v700, v850, v925, v1000 |
| 52–64 | **Relative humidity (r)** | r50, r100, r150, r200, r250, r300, r400, r500, r600, r700, r850, r925, r1000 |
| 65 | 2m temperature | t2m |
| 66 | 10m u-wind | u10m |
| 67 | 10m v-wind | v10m |
| 68 | Mean sea-level pressure | msl |
| 69 | **Total precipitation 6h** | tp06 |

**All 13 pressure levels**: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa

---

#### R2.3 ARCO Compatibility — FuXi

Tested via `ARCOLexicon[var]` for all 70 FuXi variables:

| Variable group | ARCO available? | Notes |
|---------------|----------------|-------|
| z50–z1000 (13 vars) | ✓ YES | `geopotential::hPa` |
| t50–t1000 (13 vars) | ✓ YES | `temperature::hPa` |
| u50–u1000 (13 vars) | ✓ YES | `u_component_of_wind::hPa` |
| v50–v1000 (13 vars) | ✓ YES | `v_component_of_wind::hPa` |
| **r50–r1000 (13 vars)** | **✗ NO** | `r*` (relative humidity) **NOT in ARCO lexicon** |
| t2m | ✓ YES | `2m_temperature::` |
| u10m | ✓ YES | `10m_u_component_of_wind::` |
| v10m | ✓ YES | `10m_v_component_of_wind::` |
| msl | ✓ YES | `mean_sea_level_pressure::` |
| tp06 | ✓ YES | `total_precipitation::` (ARCO de-accumulates hourly → 6h) |

**Result**: 57/70 available. 13 relative humidity variables (`r*`) are NOT in ARCO.

**Why**: ARCO stores ERA5 `specific_humidity` (`q*`), not `relative_humidity` (`r*`).
The CDS ERA5 API (`earth2studio.data.CDS`) provides `r*`, but CDS requires credentials.

**Possible workaround** (NOT implemented): Derive relative humidity from ARCO:
```
e   = q × P / (0.622 + 0.378 × q)            # partial vapor pressure [Pa]
e_s = 611.2 × exp(17.67 × (T−273.15) / (T−29.65))  # saturation pressure [Pa]
RH  = (e / e_s) × 100                         # relative humidity [%]
```
where q = ARCO `q*`, T = ARCO `t*` [K], P = pressure level [Pa].
This is a custom preprocessing step not performed by Earth2Studio automatically.

---

#### R2.4 FuXi M4 Hardware Feasibility — BLOCKED

**Blocker #1: onnxruntime-gpu required, no macOS ARM64 wheel**

FuXi's Earth2Studio extra (`earth2studio[fuxi]`) requires `onnxruntime-gpu>=1.21.0`.

Verified via PyPI:
- `onnxruntime-gpu` available platforms: `win_amd64`, `manylinux_x86_64` (Linux CUDA)
- **macOS ARM64 wheel: DOES NOT EXIST**
- onnxruntime (CPU-only) IS available for macOS ARM64 but is NOT what earth2studio[fuxi] installs

Earth2Studio's `check_optional_dependencies` will raise `OptionalDependencyError` if `onnxruntime` cannot be imported. Without installation of `onnxruntime`, FuXi cannot be instantiated.

**Blocker #2 (secondary)**: Even if onnxruntime (CPU-only) were installed manually:
- `r*` relative humidity still not available from ARCO
- Would require custom preprocessing before any inference

**FuXi ONNX file sizes** (from HuggingFace HEAD requests):
- `fuxi/short.onnx`: 40.1 MB (ONNX graph, references external data)
- `fuxi/short` (external weights): 3,129.9 MB = **3.13 GB**
- `fuxi/medium` (external weights): 3,129.3 MB = **3.13 GB**
- `fuxi/long` (external weights): 3,129.3 MB = **3.13 GB**

For 7-day (steps 0–27): short model (steps 0–19) + medium model (steps 20–27).
Peak memory includes model weights (~3.13 GB) + activations (70 vars × 721 × 1440 × 2 timesteps × float32 ≈ 572 MB). M4 with 24 GB: uncertain; `gpu:40gb` badge noted.

**FuXi R2 verdict: FAIL — onnxruntime-gpu not available on macOS ARM64**

---

#### R2.5 AIFS / AIFS2 M4 Hardware Feasibility — BLOCKED

Evaluated as potential alternative. AIFS uses `q*` (specific_humidity) — fully ARCO-compatible (0/115 variables missing). However:

`earth2studio[aifs]` requires **`flash-attn`** (Flash Attention library).
- `flash-attn` builds from source only (no pre-built wheels)
- Building requires CUDA-capable GPU
- **No macOS ARM64 support**

AIFS checkpoint: ~0.99 GB (`aifs-single-mse-1.1.ckpt`, also `gpu:40gb` badge).

**AIFS R2 verdict: FAIL — flash-attn build requires CUDA; unavailable on macOS ARM64**

**AIFS2 R2 verdict: FAIL — same flash-attn requirement**

---

#### R2.6 GraphCastOperational — ARCO and M4 Compatibility Assessment

GraphCastOperational uses the **same JAX/Haiku backend** as GraphCastSmall (proven on M4).
It is implemented in the same `graphcast` Python package, under the same Earth2Studio extra.

**GraphCastOperational VARIABLES (82 total)**:

| Group | Variables | ARCO? |
|-------|-----------|-------|
| z50–z1000 (13) | Geopotential | ✓ YES |
| **q50–q1000 (13)** | **Specific humidity** (NOT r*) | ✓ **YES** |
| t50–t1000 (13) | Temperature | ✓ YES |
| u50–u1000 (13) | U-wind | ✓ YES |
| v50–v1000 (13) | V-wind | ✓ YES |
| w50–w1000 (13) | Vertical velocity | ✓ YES |
| msl | Mean sea-level pressure | ✓ YES |
| tp06 | Precipitation (output-only) | N/A — input set to zeros by E2Studio |

**Result: 82/82 available from ARCO. ZERO missing variables.**

**tp06 initialization**: Earth2Studio's GCOp wrapper adds a zeros slice for tp06 at input (`np.zeros(...)`). ARCO does NOT need to provide tp06 for GCOp initialization.

**Checkpoint**: `gs://dm_graphcast/graphcast` — same GCS bucket as GraphCastSmall.
Checkpoint file: `params/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz`

**Optional dependencies** (earth2studio[graphcast] extra):
```
dm-haiku>=0.0.14    ✓ installed
dm-tree>=0.1.9      ✓ installed
flax>=0.10.6        ✓ installed
graphcast (git)     ✓ installed
jax (CPU)           ✓ installed (XLA ARM64)
xarray<2026.4.0     ✓ installed
```
No CUDA, no onnxruntime-gpu, no flash-attn. **M4 CPU compatible.**

**Hardware badge**: NOT present in GCOp docstring (no explicit GPU requirement stated).
**GraphCastSmall at 1.0°**: 83 vars × 181 × 360 × 2 timesteps × float32 = ~43 MB input → 2.34 GB RSS peak
**GraphCastOperational at 0.25°**: 82 vars × 721 × 1440 × 2 timesteps × float32 = ~681 MB input → UNVERIFIED RSS (needs Phase R3 smoke test)

**GCOp R2 verdict: CONDITIONALLY PASS** — ARCO and M4 deps satisfied. Memory unverified. Phase R3 smoke test required.

---

#### R2.7 Output Variable Semantics — GraphCastOperational

From `graphcast_operational.py` source, confirmed for all four required variables:

**tp06 (precipitation)**:
- Output-only variable: GCOp does not require tp06 as input (zeros are injected)
- Output semantics: 6-hour accumulated total precipitation, **metres**
- Conversion to mm/hr: `metres × 1000 ÷ 6`
- No log/exponential transform — straight metres
- t+0h frame: all zeros (same as GraphCastSmall convention)
- Negative values: clamp to ≥ 0 (same as legacy pipeline)

**t2m (temperature)**:
- Output: 2m temperature in **Kelvin**
- Conversion: `K − 273.15 = °C`
- Range verification: roughly −50°C to +60°C for physically plausible values

**u10m, v10m (wind)**:
- Output: 10m u/v wind components in **m/s**
- Wind speed: `sqrt(u10m² + v10m²) × 1.94384 → knots`
- Wind direction: `(270 − atan2(v10m, u10m) × 180/π) mod 360 → degrees FROM`
- Meteorological convention: direction FROM which wind blows, clockwise from North

---

#### R2.8 Myanmar Grid at 0.25°

```
Bounding box: lat 9–29°N, lon 92–102°E
Global grid: lat=linspace(90, -90, 721), lon=linspace(0, 360, 1440)
Myanmar lat subset: 81 points (29.00, 28.75, ..., 9.25, 9.00)
Myanmar lon subset: 41 points (92.00, 92.25, ..., 101.75, 102.00)
Myanmar grid: 81 × 41 = 3,321 grid points
```

**Comparison to legacy**:
- GraphCastSmall (1.0°): 21 × 11 = 231 grid points
- GraphCastOperational (0.25°): 81 × 41 = 3,321 grid points (14.4× more)

**Binary file size estimate (7-day, 29 frames)**:
- Per variable: 29 × 81 × 41 × 4 bytes = 386,676 bytes ≈ 0.37 MB
- 4 variables total: ≈ 1.47 MB (well within 5 MB payload budget)

---

#### R2.9 Seven-Day Cascade Architecture — GraphCastOperational

GraphCastOperational is a **single model** — no cascade. Unlike FuXi's short/medium/long ONNX cascade, GCOp is one unified JAX model run auto-regressively:

```python
e2run.deterministic(
    time=["2021-01-01T00:00:00"],
    nsteps=28,            # 28 × 6h = 168h = 7 days
    prognostic=model,     # GraphCastOperational instance
    data=arco,            # ARCO data source
    io=zarr_backend,
)
```

Earth2Studio's `run.deterministic` auto-fetches t−6h and t+0h, then runs 28 AR steps, yielding frames t+6h through t+168h (plus the t+0h init frame = 29 total frames).

No manual step-routing or model switching needed.

---

#### R2.10 Revised Model Recommendation

**FuXi selected at gate R010 → R2 FAIL (onnxruntime-gpu not available on macOS ARM64)**

| Model | R2 Status | Reason |
|-------|----------|--------|
| FuXi | **FAIL** | onnxruntime-gpu requires CUDA; no macOS ARM64 wheel |
| AIFS | **FAIL** | flash-attn requires CUDA to build; no macOS ARM64 support |
| AIFS2 | **FAIL** | Same flash-attn requirement |
| GraphCastOperational | **CONDITIONAL PASS** | ARCO OK, M4 deps OK; memory unverified |
| Aurora1p5 | Deferred | PyTorch OK on macOS; OOM risk at 0.25°; log-space tp1h |
| GraphCastSmall | Viable | Proven on M4; 1.0° resolution below new target |

**Recommended revision**: `earth2studio.models.px.GraphCastOperational`

Rationale:
1. Only 0.25° model with ALL required variables (tp06, t2m, u10m, v10m) fully in ARCO
2. Same JAX/Haiku backend as proven GraphCastSmall → highest M4 confidence
3. No CUDA-only dependencies; all Earth2Studio[graphcast] deps already installed
4. tp06 as output-only with zero input initialization — no ARCO tp06 needed at init
5. Trained on ERA5-HRES 1979–2021 at 0.25° — includes 2021 data range
6. Single model, no cascade complexity — simpler pipeline than FuXi
7. Checkpoint source: `gs://dm_graphcast` — same trusted DeepMind GCS bucket

**Risk**: Memory at 0.25° is unverified on M4. Phase R3 smoke test (nsteps=1) is mandatory.

**Gate R010 revision required**: User must approve model change from FuXi to GraphCastOperational before Phase R3 may begin.

---

#### R2 Answers to Required Questions

1. **FuXi initialization requirements**: Two-timestep init (t−6h, t+0h); 70 variables (z/t/u/v/r at 13 levels + 5 surface); 0.25° global grid.
2. **ARCO compatibility**: FuXi BLOCKED (r* missing). GCOp CLEAR (q* used, all 82 vars in ARCO).
3. **Required ERA5 variables for GCOp**: z, q, t, u, v, w at 13 levels + msl. Two timestamps only.
4. **Required historical timesteps**: 2020-12-31T18:00Z and 2021-01-01T00:00Z.
5. **Four-variable output semantics**: tp06 (metres, 6h accum), t2m (K), u10m/v10m (m/s).
6. **Wind derivation**: speed_kt = sqrt(u²+v²)×1.94384; dir° = (270−atan2(v,u)×180/π) mod 360.
7. **Precipitation conversion**: metres × 1000 ÷ 6 = mm/hr. No log transform.
8. **Myanmar grid at 0.25°**: 81 × 41 = 3,321 grid points.
9. **Cascade architecture**: GCOp = no cascade (single JAX model, 28 AR steps).
10. **M4 feasibility**: FuXi BLOCKED (onnxruntime-gpu). GCOp: ARCO deps satisfied; memory UNVERIFIED.
11. **Files changed**: research.md only (ADR-017 added).
12. **R2 RESULT**: **FAIL for FuXi**. GCOp proposed as replacement, pending user approval.

---

## 8. Summary: Architecture Evolution

| Item | Aurora1p5 (failed) | GraphCastSmall v3.0 (CURRENT) | New Target (GCOp proposed) |
|------|-------------------|-------------------------------|---------------------------|
| Model | Aurora1p5 | GraphCastSmall | **GraphCastOperational** (pending approval) |
| Resolution | 0.25° | 1.0° | **0.25°** |
| Timestep | 1h | 6h | **6h** |
| Horizon | 168h (target) | 48h (validated) | **168h (28 steps, 7 days)** |
| Variables | tp1h, t2m | tp06, t2m | **tp06, t2m, u10m, v10m** |
| Precip semantics | log-space 1h | metres, 6h accum | **metres, 6h accum → mm/hr** |
| Init source | IFS (sic patched) | ARCO / IFS | **ARCO (ERA5, 2021-01-01)** |
| Init date | operational | 2022-07-01 | **2021-01-01T00:00:00Z** |
| Init timesteps | t−6h, t+0h | t−6h, t+0h | **t−6h, t+0h** |
| Backend | PyTorch | JAX/Haiku | **JAX/Haiku (same as GCSmall)** |
| Hardware | T4 GPU (OOM) | M4 CPU (~78s) | **M4 CPU — R3 PASS (JIT 34 min, peak RSS 1.99 GB)** |
| Myanmar grid | — | 21 × 11 @ 1.0° | **81 × 41 @ 0.25°** |
| Checkpoint | HuggingFace/microsoft | gs://dm_graphcast | **gs://dm_graphcast (same GCS)** |
| ARCO compat | — | ✓ (all vars) | **✓ all 82 vars in ARCO (q* not r*)** |
| Schema | — | v3.0 | v4.0 (design pending) |
| Note | — | — | FuXi BLOCKED: onnxruntime-gpu no macOS ARM64 |
| Status | FAILED | DEPLOYED | **R3 PASS — awaiting Phase R4 approval** |

---

## 10. Phase R3 — M4 Hardware Smoke Test

### ADR-018: GraphCastOperational nsteps=1 Smoke Test on M4

**Date**: 2026-08-12
**Init**: 2021-01-01T00:00:00Z (ARCO ERA5)
**Hardware**: Apple M4 MacBook Air, 24 GB unified memory, JAX_PLATFORM_NAME=cpu

#### Measured Results

| Measurement | Value |
|---|---|
| Package load (cached) | 23.7s |
| Model load (cached weights) | 0.7s |
| RSS after model load | 1.07 GB |
| ARCO fetch (82 vars, 2 timesteps, 721×1440) | 6.9s |
| RSS after ARCO fetch | 3.36 GB |
| nsteps=1 inference (incl. JAX JIT compilation) | **2081s = 34.7 min** |
| Peak RSS (during XLA compilation) | ~5.0 GB |
| RSS after inference complete | **1.99 GB** |
| JAX backend | cpu ✓ |
| No OOM, no crashes | ✓ |

#### JAX JIT Profile

The `e2run.deterministic` progress bar showed two items:
- **Item 1** (init feed / ARCO → model input tensor): **2.83s**
- **Item 2** (JAX forward pass + XLA JIT compilation): **~2077s = 34.6 min**

The 34.6 min is almost entirely JAX XLA kernel compilation, not actual forward pass execution.
On subsequent AR steps within the same process the kernel is cached and will run in seconds.
There is no persistent XLA cache on disk; each cold-start Python process recompiles.

**Actual measured per-step times (R4 pipeline, 2026-08-16)**:
- Step 2 (JIT + 1st forward pass): 27:14 elapsed total
- Step 3 (post-JIT): 25:24 min
- Step 4 (post-JIT): 25:17 min
- Step 5 (post-JIT): 25:32 min
- Step 6 (post-JIT): 26:02 min
- Step 7 (post-JIT): 24:03 min
- Step 8 (post-JIT): 24:46 min
- Step 9 (post-JIT): 25:39 min

Stabilised post-JIT rate: **~25 min/step** (not the 96–180s estimated from GCSmall scaling).
For 28 AR steps (7 days): full run ≈ **27 min JIT + 28 × 25 min ≈ 12–14 hours total**.

#### Output Validation

Output shape: `(1 batch, 2 lead_times, 721 lat, 1440 lon)` — float32.
lead_time coords: `[0h, 6h]` (t+0h = init state copy, t+6h = first forecast step).

| Variable | Raw range | Units | Notes |
|---|---|---|---|
| tp06 | -0.0002 to 0.0872 | metres/6h | Negative min = float noise → clamp ≥ 0; max 87.2 mm/6h = 14.5 mm/hr |
| t2m | 221.1 to 315.8 | Kelvin | → -52.1 to +42.6 °C after −273.15 |
| u10m | -20.7 to +23.8 | m/s | Zonal wind at 10m |
| v10m | -21.0 to +22.5 | m/s | Meridional wind at 10m |

All 4 target variables present in output. No NaN values. Physically plausible global ranges.

#### Required Post-processing Conversions

```python
tp06_mm_per_hr = np.clip(tp06_raw * 1000 / 6, 0, None)      # metres → mm/hr, clamp ≥ 0
t2m_celsius    = t2m_raw - 273.15                             # K → °C
wind_speed_kt  = np.sqrt(u10m**2 + v10m**2) * 1.94384        # m/s → knots
wind_dir_deg   = (270 - np.degrees(np.arctan2(v10m, u10m))) % 360  # °FROM
```

#### R3 Verdict: **PASS**

GraphCastOperational runs successfully on M4 MacBook Air 24 GB:
- No OOM (peak RSS 1.99 GB post-compilation; ~5 GB during compilation)
- All 4 output variables present and physically plausible
- JAX CPU backend functional; XLA ARM64 compiles correctly
- One-time JIT cost: **34 min per cold start** (significant — see note below)

**JIT Cold-Start Note**: Every new Python process recompiles the XLA kernel (~34 min).
For daily production use, enable JAX persistent compilation cache by setting
`XLA_FLAGS=--xla_gpu_enable_xla_runtime_executable` and `XLA_PYTHON_CLIENT_MEM_FRACTION`
or use `jax.config.update("jax_compilation_cache_dir", "/path/to/cache")` before model load.
With a warm cache, subsequent cold starts reduce to kernel loading time (seconds).

**Measured 7-day run time** (R4 pipeline, 2026-08-16, no persistent cache):
- Step 1 (init/ARCO feed): ~2s
- Step 2 (XLA JIT + 1st forward): ~27 min (1634s)
- Steps 3–29 (post-JIT, measured): ~25 min/step average
- Total estimated: **~27 min + 28 × 25 min ≈ 12–14 hours**

Note: First aborted run (2026-08-13) showed 35–48 min/step, likely due to thermal
throttling or background load. Current run on a quieter machine shows ~25 min/step.

JAX persistent cache status: cache directory exists but remained empty after prior
runs. Warm-start benefit not yet confirmed. Requires investigation.

**Gate R018**: R3 PASS. Phase R4 (7-day pipeline implementation) approved and in progress (2026-08-16).

---

## 11. Phase R4 — Schema and Design Decisions (2026-08-16)

### ADR-019: Schema v4.0 Artifact Design

**Date**: 2026-08-16
**Status**: APPROVED — locked as canonical schema for GCOp 4-variable 168h 0.25° artifacts.

#### Decision

The forecast artifact set for the new target architecture uses `schema_version: "4.0"`.
This schema is **incompatible** with schema v3.0 (GraphCastSmall, 48h, 2 variables, 1.0°).

#### Schema Version Rationale

The Spec Kit (constitution, spec, plan, tasks) consistently uses "schema v4.0" for the GCOp
new target. The implementation file `generate_forecast.py` was temporarily written to emit
`"5.0"` during an interrupted session. This is corrected: the canonical version is `"4.0"`.
`generate_forecast.py` will be updated to write `schema_version: "4.0"` in Phase R4b.

The existing `data/demo/forecast.json` emits `"4.0"` but with an inconsistent payload
(GraphCastSmall model, 1.0° grid, 2 variables). It will be regenerated correctly in Phase R4b
(task RS11–RS14) and MUST NOT be patched manually.

#### Artifact Set

Four float32 binary files + one JSON metadata file:

| File | Content | Shape | Size |
|------|---------|-------|------|
| `precipitation.bin` | mm/hr (clamped ≥ 0) | [29 × 81 × 41] float32 | ~376 KB |
| `temperature.bin` | °C | [29 × 81 × 41] float32 | ~376 KB |
| `wind_speed.bin` | knots | [29 × 81 × 41] float32 | ~376 KB |
| `wind_direction.bin` | °FROM [0, 360) | [29 × 81 × 41] float32 | ~376 KB |
| `forecast.json` | Metadata and provenance | — | ~5 KB |

Total binary payload: **4 × 29 × 81 × 41 × 4 bytes = 1,540,944 bytes ≈ 1.54 MB**

#### forecast.json Top-Level Fields (schema v4.0)

```
schema_version          "4.0"
model                   "GraphCastOperational"
model_version           "1.0"
model_checkpoint        <checkpoint filename from GCS>
model_source            "gs://dm_graphcast/graphcast"
initialization_source   "ARCO/ERA5"
initialization_time     "2021-01-01T00:00:00Z"
forecast_generated_at   <ISO8601 timestamp>
forecast_horizon_hours  168
native_timestep_hours   6
n_times                 29
spatial_resolution_deg  0.25
display_resolution_deg  0.05
region                  "Myanmar"
bbox                    {lat_min:9.0, lat_max:29.0, lon_min:92.0, lon_max:102.0}
grid                    {n_lat:81, n_lon:41}
lat                     [9.0, 9.25, ..., 29.0]  (81 values)
lon                     [92.0, 92.25, ..., 102.0]  (41 values)
times_utc               ["2021-01-01T00:00:00Z", ..., "2021-01-08T00:00:00Z"]  (29 values)
variables               {precipitation: {...}, temperature: {...}, wind_speed: {...}, wind_direction: {...}}
data_source_attribution <ERA5/ARCO attribution string>
model_attribution       <GCOp/DeepMind attribution string>
earth2studio_version    ">=0.17.0"
inference_config        {device, jax_backend, rss_peak_gb, jit_cold_start_seconds,
                          jax_cache_hit, arco_fetch_seconds, per_step_seconds_mean,
                          per_step_seconds_by_step, total_inference_seconds,
                          total_pipeline_seconds}
is_demo                 false
```

#### Per-Variable metadata block (inside `variables`)

Each variable entry contains:
```
display_name            Human-readable name
units                   Display unit string
source_variable         Native model variable name(s)
temporal_resolution     "6-hourly"
temporal_semantics      Description of what the value represents at each frame
temporal_disclosure     (for precipitation/wind_direction) Clarification note for UI
transformation_provenance  {source_variable, source_unit, conversion, output_unit,
                             accumulation_period_hours (precipitation only),
                             log_transform_applied, exp_transform_applied, pipeline}
file                    Binary filename
fill_value              null
```

#### Binary Layout

Each binary file is a flat C-order (row-major) float32 array:
```
index = frame * n_lat * n_lon + lat_idx * n_lon + lon_idx
```
- `frame`: 0 = t+0h init, 1 = t+6h, ..., 28 = t+168h
- `lat_idx`: 0 = 9.0°N (southernmost), 80 = 29.0°N (northernmost) — ascending
- `lon_idx`: 0 = 92.0°E, 40 = 102.0°E — ascending

#### New Target vs Legacy Comparison

| Property | v3.0 (legacy, deployed) | v4.0 (new target) |
|----------|------------------------|-------------------|
| model | GraphCastSmall | GraphCastOperational |
| schema_version | "3.0" | "4.0" |
| spatial_resolution_deg | 1.0 | 0.25 |
| n_times | 9 | 29 |
| forecast_horizon_hours | 48 | 168 |
| grid | 21 × 11 | 81 × 41 |
| variables | precipitation, temperature | precipitation, temperature, wind_speed, wind_direction |
| precip units | mm / 6h | mm/hr |
| binary files | 2 | 4 |
| total payload | ~16.6 KB | ~1.54 MB |

---

### ADR-020: Wind Direction Visualization — Vector-Component Interpolation

**Date**: 2026-08-16
**Status**: APPROVED

#### Problem

Wind direction is a **circular** (or angular) quantity defined on [0°, 360°). Direct
bilinear interpolation of degree values produces mathematically incorrect results near
the 0°/360° boundary:

Example: two adjacent grid points at 10° and 350°.
- Direct bilinear: (10 + 350) / 2 = 180° — WRONG (the true average is 0°/360°)
- Correct circular average: atan2d(sin(10°)+sin(350°), cos(10°)+cos(350°)) ≈ 0°

The existing `renderWithInterpolation` function in `colorscales.ts` performs standard
scalar bilinear interpolation. Applying it to wind direction data would produce visible
artefacts across the 0°/360° seam.

#### Decision

Wind direction rendering MUST use **vector-component bilinear interpolation**:

1. At each display pixel, identify the four surrounding model grid points (i0,j0), (i0,j1), (i1,j0), (i1,j1)
2. Compute unit vector components at each corner:
   ```
   sin_dir[corner] = sin(direction_rad[corner])
   cos_dir[corner] = cos(direction_rad[corner])
   ```
3. Bilinearly interpolate sin_dir and cos_dir separately using fractional weights (tx, ty)
4. Reconstruct interpolated direction:
   ```
   dir_interp = atan2d(interp_sin, interp_cos) mod 360
   ```
5. Apply color wheel LUT to dir_interp

This is mathematically correct for circular quantities and produces smooth, artefact-free
rendering across the 0°/360° seam.

#### Color Wheel LUT Design

Map direction [0°, 360°) → HSL color wheel:
- 0° (N): red
- 90° (E): yellow-green
- 180° (S): cyan
- 270° (W): blue-violet
- 360° (N): red (same as 0° — continuous wrap)

Saturation: 70%; Lightness: 55% (sufficient contrast on dark basemap).

The legend for wind direction shows compass labels (N / NE / E / SE / S / SW / W / NW)
positioned at 0°/45°/90°/135°/180°/225°/270°/315°, with numeric degree values secondary.

#### Frontend Implementation Note

A dedicated render path for wind direction is required in `WeatherMap.tsx` and/or
`colorscales.ts`. This path MUST NOT call the generic `renderWithInterpolation` with
wind direction degree values. A new function `renderWindDirection` (or equivalent) MUST
implement the vector-component method above.

This implementation is deferred to Phase R7 and MUST NOT begin until Phase R6 (data
layer) is complete and the Spec Kit update is approved.

---

### ADR-021: GCOp Precipitation Conversion — tp06 to mm/hr

**Date**: 2026-08-16
**Status**: APPROVED — confirmed from ADR-018 R3 smoke test and Earth2Studio source inspection.

#### Native Variable Semantics

GCOp `tp06` is **6-hour accumulated total precipitation** in **physical metres**.
- Confirmed from R3 smoke test output: range [-0.0002, 0.0872] metres/6h
- The small negative values (-0.0002) are float32 numerical noise, not physical rain
- No log transform is applied (unlike Aurora1p5's tp1h which uses log-space)
- GCOp does NOT require tp06 as an input variable (zeros are injected internally by Earth2Studio)

#### Conversion Formula

```python
# tp06 in metres (6-hour accumulation)
tp06_mm_per_hr = max(tp06_metres * 1000.0 / 6.0, 0.0)
```

Steps:
1. `tp06_metres × 1000` → mm (6-hour accumulation in mm)
2. `÷ 6` → mm/hr (average rate over the 6-hour step)
3. `clamp ≥ 0` → remove float noise (negative values are non-physical)

Negative raw statistics MUST be recorded BEFORE clamping for provenance.

#### Display Semantics

The result is an **average rainfall rate** over the 6-hour forecast step, expressed
as mm per hour. It is NOT an instantaneous measurement.

Required UI disclosure:
> "Precipitation values show the estimated average rainfall rate (mm/hr) during the
> 6-hour period ending at the displayed forecast time. This is derived from the model's
> 6-hour accumulated total, not a momentary rate."

#### Validation Bounds

| Check | Threshold |
|-------|----------|
| Min (after clamp) | ≥ 0 mm/hr |
| Max sanity | ≤ 300 mm/hr (extreme but physically possible) |
| NaN count | 0 |
| Inf count | 0 |
| Zero-rain fraction | Record (expected high; many grid points will be dry) |
| Negative count (pre-clamp) | Record for provenance |

---

### ADR-022: Payload Loading Policy — All Variables at Startup

**Date**: 2026-08-16
**Status**: APPROVED

#### Context

Schema v4.0 produces 4 binary files totalling ~1.54 MB:
- 4 variables × 29 frames × 81 × 41 grid points × 4 bytes (float32)
- Exceeds the ~1 MB threshold noted in constitution §VIII

#### Options Considered

| Strategy | Load time (25 Mbps) | UX on variable switch | Complexity |
|----------|--------------------|-----------------------|-----------|
| Load all 4 at startup | ~0.5s | Instant (<200ms) | Minimal |
| Load active variable only | ~0.13s | Re-fetch ~0.13s on each switch | Moderate |
| Load in background after first | ~0.5s total, staggered | Risk of partial data on fast switch | Moderate |

#### Decision

**Load all 4 variable binaries in parallel at startup.**

Rationale:
1. 1.54 MB / 3.1 MB/s (25 Mbps) ≈ 0.5s — well within the 5-second total load budget
2. Variable switching must feel instant (<200ms per §VIII); re-fetching on switch fails this
3. All 4 arrays fit comfortably in browser memory (1.54 MB is negligible for modern browsers)
4. Parallel fetch via `Promise.all([...])` amortises latency to the largest single file (~376 KB)
5. Implementation simplicity: no lazy-load state machine, no race conditions

#### Implementation

`ForecastLoader.ts` MUST fetch all 4 binary files using `Promise.all` and return all 4
`Float32Array` objects. `ForecastStore.ts` MUST hold all 4 arrays in state from startup.

Loading indicator (spinner) remains visible until all 4 arrays are loaded and validated.

#### Constraint

If a future schema version increases the payload materially beyond ~5 MB, this decision
MUST be revisited and a lazy-loading or streaming strategy designed.

---

### ADR-023: Phase R5 — ERA5 Verification Methodology

**Status**: CLOSED — 2026-08-17. verify_forecast.py exit 0; verification.json schema v2.0 written.

**Decision**: Use ERA5 via ARCO as the reference dataset for verification of the
2021-01-01 GCOp 7-day forecast. Employ empirically confirmed ARCO tp semantics (1-hour
accumulation), 0.25° exact grid alignment, and the locked metric suite documented below.

---

#### Dataset

- **Source**: `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
- **Variables fetched**:
  - `2m_temperature` — 29 × 6-hourly timestamps (t+0h through t+168h)
  - `10m_u_component_of_wind` — 29 × 6-hourly timestamps
  - `10m_v_component_of_wind` — 29 × 6-hourly timestamps
  - `total_precipitation` — 168 hourly timestamps (t+1h through t+168h)
- **Time resolution**: 1-hourly
- **Spatial resolution**: 0.25° global

---

#### Grid Alignment

GCOp output and ERA5 ARCO both use a 0.25° regular lat/lon grid.
Myanmar bbox: lat 9.0°N–29.0°N (81 pts), lon 92.0°E–102.0°E (41 pts).
Verified grid alignment error: **lat_err = 0.000000°, lon_err = 0.000000°** — exact match.
No spatial interpolation was performed or required.

ARCO latitude is descending (90°N → 90°S). The Myanmar lat slice is extracted and flipped
to ascending (9°N → 29°N) to match the forecast binary ordering.

---

#### ERA5 total_precipitation Semantics — Empirically Confirmed

Prior to implementation, the ARCO `total_precipitation` field was empirically inspected
for a 48-hour sequence spanning multiple 00Z and 12Z boundaries (2021-07-01 monsoon period,
which has non-zero rainfall unlike the dry verification period).

**Confirmed representation**: Each `tp[T]` value is the **1-hour precipitation accumulation
ending at timestamp T** (the amount that fell in the window [T−1h, T]). This is ERA5's
standard hourly accumulation convention.

Evidence:
- Values are **NOT monotonically increasing** within 00Z→12Z or 12Z→00Z windows
  (all six runs tested failed the monotonicity check)
- Values at 00Z and 12Z are **non-zero** (e.g., 0.000031 m at 2021-07-01T00Z —
  the rain that fell in [23Z, 00Z])
- Six consecutive hourly values sum to physically plausible 6-hour monsoon totals:
  - [00Z→06Z]: 2.795 mm; [06Z→12Z]: 7.856 mm; [12Z→18Z]: 0.906 mm

**Consequence**: No seam handling is required at 00Z or 12Z. Each hourly value is an
independent accumulation and can be summed directly across any 6-hour window without
regard to forecast-run boundaries.

**Negative values**: 11.73% of hourly tp values in the January 2021 Myanmar domain are
negative (max magnitude: −0.000004 mm). These are spectral diffusion artifacts in the
ERA5 model. All are clamped to ≥ 0 before aggregation.

---

#### ERA5 Precipitation Aggregation — GCOp Alignment

GCOp `tp06` at lead time T represents accumulated precipitation over the 6-hour window
ending at T. The ERA5 equivalent is constructed by summing six consecutive 1-hour
accumulations spanning the same window:

```
For GCOp frame at lead_time = 6k hours  (k = 1..28):
    window_hours   = [6(k−1)+1, 6(k−1)+2, ..., 6k]      # 6 hourly indices into t+1h..t+168h
    hourly_clamped = max(0, tp[h])  for h in window_hours # per-hour clamp ≥ 0
    sum_6h_metres  = sum(hourly_clamped)                  # 6-hour total in metres
    sum_6h_metres  = max(0, sum_6h_metres)                # aggregate clamp ≥ 0
    era5_mm_hr     = sum_6h_metres × 1000 / 6             # → mm/hr average rate
```

Locked mapping:
| GCOp frame | Window | ARCO hourly indices |
|---|---|---|
| +6h | t+0h → t+6h | tp[t+1h] … tp[t+6h] |
| +12h | t+6h → t+12h | tp[t+7h] … tp[t+12h] |
| … | … | … |
| +168h | t+162h → t+168h | tp[t+163h] … tp[t+168h] |

- No timestamp subtraction performed
- No seam detection or forecast-run boundary logic
- t+0h precipitation excluded from all metrics (GCOp pipeline convention: tp06=0 at init)
- 28 precipitation verification frames total (t+6h through t+168h)

---

#### Variable Transformations

| Variable | GCOp binary | ERA5 raw | Transformation |
|---|---|---|---|
| Temperature | °C (already converted) | K | K − 273.15 → °C |
| Wind speed | kt (already converted) | u10m, v10m m/s | √(u²+v²) × 1.94384 → kt |
| Wind direction | °FROM (already computed) | u10m, v10m m/s | (atan2(−u, −v) × 180/π + 360) % 360 → °FROM |
| Precipitation | mm/hr (already converted) | metres/hr hourly | 6-hour sum × 1000/6, clamp ≥ 0 → mm/hr |

---

#### Metrics

**Temperature, wind speed** (all 29 frames):
- MAE, RMSE, Bias — over all 3,321 grid points per frame; summary = mean over all frames

**Wind direction** (all 29 frames):
- Circular MAE: `diff = ((fcst_dir − era5_dir + 180) % 360) − 180`; MAE = mean(|diff|)
- Calm exclusion: grid points where ERA5 wind speed < 2.0 kt excluded from circular MAE
- `n_points_active` and `n_points_calm_excluded` recorded per lead time

**Precipitation** (28 frames, t+6h through t+168h):
- MAE, RMSE, Bias on continuous mm/hr values
- Categorical threshold: 0.1 mm/hr (rain vs no-rain)
- Per-lead-time: hits, misses, false_alarms, POD, FAR, CSI
- Summary POD/FAR/CSI: computed from **aggregated totals** across all 28 frames × 3,321 points —
  NOT by averaging the 28 per-frame ratios

---

#### R5 Verification Results — 2021-01-01 GCOp Forecast vs ERA5

Run date: 2026-08-17. exit code: **0**. All pre/post checks PASS.
verification.json schema v2.0 written to `data/verification/`.

| Variable | MAE | RMSE | Bias |
|---|---|---|---|
| Temperature (29 frames) | **1.3137°C** | 1.6975°C | −0.7595°C |
| Wind speed (29 frames) | **1.0548 kt** | 1.4027 kt | −0.3911 kt |
| Wind direction (29 frames) | circular MAE **16.9113°** | — | — |
| Precipitation (28 frames) | **0.0172 mm/hr** | 0.0645 | +0.0110 |

Precipitation categorical scores (0.1 mm/hr threshold, totals from 28 frames × 3,321 points):

| POD | FAR | CSI | Hits | Misses | False Alarms |
|---|---|---|---|---|---|
| 0.6343 | 0.7279 | 0.2352 | 1,148 | 662 | 3,071 |

---

#### Caveats

1. ERA5 is a numerical reanalysis product, not direct station observations
2. GCOp was trained on ERA5; comparison against ERA5 may be optimistic relative to
   independent observations
3. Single 7-day forecast cycle — skill metrics are not statistically robust
4. January 2021 is the dry season in Myanmar; precipitation metrics reflect near-zero
   rainfall conditions (high FAR expected due to sparse rain events)
5. GCOp and ERA5 use identical 0.25° grids — no spatial interpolation bias is present
6. ERA5 precipitation aggregated from six 1-hour accumulations per 6h window

---

#### Relation to Other ADRs

- Precipitation conversion formula: ADR-021
- Wind direction circular convention: ADR-020
- Schema for verification.json v2.0: this ADR
- Forecast binary layout verified: ADR-019 (385,236 bytes / variable confirmed)

---

### ADR-024 — Authoritative Production Dataset: data/forecast_v4/; RS11–RS14 Deferred

**Date**: 2026-08-17
**Status**: ACCEPTED
**Context**: Phase R4 produced a complete 7-day GCOp forecast (2021-01-01T00Z init, 29 frames, 4 variables, 0.25° Myanmar grid) stored in `data/forecast_v4/`. Phase R5 verified it against ERA5 reanalysis, producing `data/verification/verification.json` (schema v2.0). Phase R6 migrated the frontend to schema v4.0 with all four variables. Phase R9 committed all data artifacts, updated the deployment workflow to serve `data/forecast_v4/`, and confirmed a successful GitHub Pages deployment. Throughout R6–R9, the frontend operated exclusively on the real R4 dataset (`is_demo=false`); no synthetic demo data was required.

**Decision**: `data/forecast_v4/` — the validated R4 GCOp output — is the authoritative production frontend dataset. It is committed to the repository and served by the GitHub Pages deployment workflow. Tasks RS11–RS14 (regeneration of `data/demo/` to schema v4.0 using `generate_demo_data.py`) are DEFERRED and OPTIONAL. They are not on the critical path for any currently deployed functionality.

**Rationale**:
1. The deployment workflow prefers `data/forecast_v4/`; `data/demo/` is only a fallback if `data/forecast_v4/` is absent or incomplete.
2. `data/forecast_v4/` is present, complete (5 files, all 385,236 bytes each), and validated (schema v4.0, `is_demo=false`).
3. The existing `data/demo/` directory contains schema v3.0 (2-variable) artifacts. They are not loaded by the current frontend and are not served to production.
4. Generating schema v4.0 demo data requires a complete rewrite of `generate_demo_data.py` — meaningful effort with no current benefit, as production already serves real data.
5. The primary value of demo data (offline development, CI without real forecast) is already satisfied by copying `data/forecast_v4/` artifacts to `frontend/public/data/` (done in R9).

**Consequences**:
- RS11–RS14 may be revisited if: (a) a new real-data forecast pipeline run is not available, (b) CI requires a lightweight synthetic fixture, or (c) a contributor explicitly needs offline development data independent of the committed real forecast.
- `data/demo/` currently contains stale v3.0 artifacts; they should not be confused with schema v4.0 demo data.
- Any future reauthorization of RS11–RS14 must update `generate_demo_data.py` for 4 variables, 81×41 grid, 29 frames, and `is_demo=true`.

#### Relation to Other ADRs

- Schema v4.0 canonical definition: ADR-019
- Deployment workflow structure: task R100 (deploy-pages.yml)
- Verification schema v2.0: ADR-023
