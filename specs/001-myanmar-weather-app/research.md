# Research: Myanmar Weather Forecast App — Earth2Studio Discovery

**Feature**: 001-myanmar-weather-app
**Date**: 2026-08-11 (revised — Aurora1p5 → GraphCastSmall → 48h+temperature migration)
**Phase**: Phase 0 Research (verified against live Earth2Studio docs and source code)

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
