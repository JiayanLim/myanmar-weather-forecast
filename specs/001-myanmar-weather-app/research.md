# Research: Myanmar Weather Forecast App — Earth2Studio Discovery

**Feature**: 001-myanmar-weather-app
**Date**: 2026-08-11 (revised — Aurora1p5 → GraphCastSmall migration)
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
- JAX backend (not PyTorch) — different memory management; footprint unknown but bounded
- Native `tp06` output: 6-hour accumulated total precipitation is a first-class model output
- No diagnostic model required (no PrecipitationAFNO)
- Scientifically valid global medium-range forecast model (DeepMind/Google)
- Model weights: freely available via Google Cloud Storage
- Checkpoint: `params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz`
- Package source: `gs://dm_graphcast/graphcast`

**What remains unknown**:
- Whether GraphCastSmall actually fits on T4 (16 GB) — MUST be tested experimentally
- Rec VRAM badge states 40 GB; T4 has 16 GB; compatibility is unverified

**Constraints accepted vs. Aurora1p5**:
- 1.0° resolution (coarser than Aurora1p5's 0.25°)
- 6h native timestep (not hourly)
- 24h forecast horizon (vs. 168h)
- tp06 (6h accumulated) rather than tp1h (1h accumulated)
- No sea ice concentration (sic) gap to handle — GraphCastSmall inputs do not require sic

**Variable specification** (verified from Earth2Studio source):
- 83 total input variables covering atmospheric and surface fields
- tp06 (6h total precipitation in metres) is included in input AND output
- No log transform is applied to tp06 — it is already in physical space
- Myanmar at 1.0° grid: 21 lat points (9–29°N) × 11 lon points (92–102°E)

---

### ADR-002: Initialization Source — ARCO (primary) / IFS (operational)

**Decision**: ARCO as primary historical/development source; IFS for operational real-time.
NCAR_ERA5 is NOT compatible. GFS is NOT a verified compatible source.

**Why ARCO**:
- Provides all 83 GraphCastSmall input variables including tp06
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
- Produced natively by GraphCastSmall at each 6h lead time (t+6h, t+12h, t+18h, t+24h)
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

**PrecipitationAFNO status**:
Not required. GraphCastSmall natively outputs tp06. Do not chain a diagnostic precipitation
model.

---

### ADR-004: Temporal Resolution — Native 6h Steps (No Interpolation)

**Decision**: GraphCastSmall produces one forecast step every 6 hours. The MVP covers a 24-hour
horizon with 4 forecast steps plus the t+0h initialization frame = 5 total frames.

**Native steps**: t+0h, t+6h, t+12h, t+18h, t+24h

**Interpolation is PROHIBITED** (Constitution §VI):
- Synthesizing intermediate hourly frames is forbidden
- Only the 5 native 6h frames are displayed
- Timeline slider steps in 6h increments

**Lead time display**:
- t+0h: initialization state (from analysis)
- t+6h: "+6 h"
- t+12h: "+12 h"
- t+18h: "+18 h"
- t+24h: "+24 h"

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

### ADR-006: Forecast Data Format — GraphCastSmall Dimensions

**Decision**: Float32 binary arrays + `forecast.json` metadata.

**Artifact layout**:
```
data/
├── demo/ or forecast/
│   ├── forecast.json          # All metadata
│   └── precipitation.bin      # [5 × 21 × 11] float32
```

**Grid dimensions** (Myanmar 1.0° bbox):
- lat: 9.0°N to 29.0°N → 21 points (step 1.0°)
- lon: 92.0°E to 102.0°E → 11 points (step 1.0°)
- times: t+0h to t+24h inclusive at 6h steps → 5 frames (indices 0, 1, 2, 3, 4)

**Size estimate**: 5 × 21 × 11 × 4 bytes × 1 variable = 4,620 bytes ≈ 4.6 KB total

**Note on t+0h**: The t+0h frame stores the initialization state from the analysis source.
GraphCastSmall starts producing forecasts at t+6h. The t+0h frame is included for reference.

---

### ADR-007: Hardware Transparency and VRAM Staging

**Decision**: VRAM requirements for GraphCastSmall on T4 (16 GB) MUST be established
experimentally through a staged test protocol. Do NOT assume the 40 GB badge means 40 GB
is strictly required.

**Staged test protocol** (Constitution §XI):
1. Measure GPU memory baseline after Colab/CUDA initialization
2. Load GraphCastSmall weights and measure peak VRAM
3. Run single-step inference (t−6h + t+0h → t+6h) and measure peak VRAM
4. If step 3 succeeds: run full 24h forecast
5. Report GPU type, total VRAM, and peak allocated VRAM at each stage

If step 3 OOM: stop immediately, report, seek user approval for any alternative approach.

---

## 2. GraphCastSmall Variable Specification (Verified)

### Input Variables (83 total)

Includes atmospheric variables at 13 pressure levels plus surface variables including:
- `tp06` (6-hour total precipitation — required as initialization input)
- Surface meteorological fields (temperature, wind, humidity, pressure, etc.)
- Pressure-level fields at 13 levels

GraphCastSmall uses the WeatherBench2 (WB2) lexicon for variable mapping.

### Output Variables

GraphCastSmall outputs the same 83 variables it ingests as input at the next 6h lead time.
`tp06` is both an input (required for initialization) and an output (forecast).

### Variables used in frontend

| Variable | GraphCast name | Unit (raw output) | Transformation | Frontend unit |
|----------|---------------|-------------------|---------------|---------------|
| 6h precipitation | `tp06` | metres (physical) | × 1000 | mm / 6h |

No temperature is displayed in the MVP. Frontend is precipitation-only.

---

## 3. Data Source Analysis

### earth2studio.data.ARCO

**Access**: Free via Google Cloud (zarr) — no credentials required
**Coverage**: 1959–2023 (historical only)
**Resolution**: 0.25° (upsampled to 1.0° by GraphCastSmall's Earth2Studio wrapper)
**Variables**: Full ERA5 including tp06
**Compatible with GraphCastSmall**: YES — all 83 required variables including tp06
**Use case**: Development runs, historical validation, Colab notebook testing

### earth2studio.data.IFS

**Access**: ECMWF open data — no credentials required
**Resolution**: 0.25°
**Coverage**: Near-real-time (4× daily: 00, 06, 12, 18 UTC, ~6h latency)
**Variables**: Full IFS variable set including tp06
**Compatible with GraphCastSmall**: YES — all 83 required variables confirmed
**Note**: Unlike the Aurora1p5 use case, IFS for GraphCastSmall does NOT have a sic gap
  problem. GraphCastSmall's 83 input variables do not include sic.
**Use case**: Near-real-time production forecasts

### earth2studio.data.NCAR_ERA5

**Access**: Free via AWS Open Data
**Coverage**: Historical + near-real-time
**Compatible with GraphCastSmall**: NO — missing `tp06` in lexicon
**Use case**: NOT suitable for GraphCastSmall initialization

### GFS

**Compatible with GraphCastSmall**: UNVERIFIED — do not use
**Constitution requirement**: Must be explicitly validated before use

---

## 4. Pipeline Architecture (Final)

```
ARCO (earth2studio.data.ARCO) or IFS (earth2studio.data.IFS)
    ↓ [fetch t-6h AND t+0h — two consecutive timesteps required]
GraphCastSmall (earth2studio.models.px.GraphCastSmall)
    │ Native 6h auto-regressive rollout: t+6h, t+12h, t+18h, t+24h
    │ 83 output variables per step
    ↓
ZarrBackend (earth2studio.io.ZarrBackend)
    ↓
xarray post-processing:
    └── extract tp06 → × 1000 → mm / 6h (clamp to ≥ 0)
    ↓
myanmar_subset: .sel(lat=slice(9,29), lon=slice(92,102))
    ↓
artifact_writer: Float32 binary [5 × 21 × 11] + forecast.json
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

**No secrets, tokens, or API keys are required for the MVP data path.**

---

## 6. Relevant Examples

- Deterministic workflow: https://nvidia.github.io/earth2studio/examples/01_getting_started/01_deterministic_workflow.html
- GraphCast example (if available): check https://nvidia.github.io/earth2studio/examples/

---

## 7. Summary: What Changed from Aurora1p5

| Item | Aurora1p5 | GraphCastSmall |
|------|-----------|----------------|
| Resolution | 0.25° | 1.0° |
| Native timestep | 1h | 6h |
| Forecast horizon | 168h (7 days) | 24h |
| Total frames | 169 | 5 |
| Myanmar grid | 81 × 41 | 21 × 11 |
| Precipitation variable | tp1h | tp06 |
| Precipitation transform | exp() × 1000 (log untransform) | × 1000 only (no log) |
| Precipitation unit | mm / 1h | mm / 6h |
| Precipitation accumulation | 1-hour | 6-hour |
| Sea ice gap (sic) | YES — IFS missing sic, must patch | NO — sic not required by GraphCastSmall |
| Init timesteps required | 1 (t+0h only) | 2 (t-6h AND t+0h) |
| Compatible init sources | IFS (with sic patch), NCAR_ERA5, ARCO | ARCO, IFS — NOT NCAR_ERA5 (missing tp06) |
| Backend | PyTorch | JAX + Haiku |
| Rec VRAM | 48 GB | 40 GB (T4 16 GB unverified) |
| Temperature output | YES (t2m, displayed) | Not displayed in MVP |
| Variable switcher | YES (temp + precip) | NO (precip only) |
| Pipeline complexity | Higher (sic patch, log untransform) | Lower |
| Payload size | ~4.3 MB | ~4.6 KB |
