# Research: Myanmar Weather Forecast App — Earth2Studio Discovery

**Feature**: 001-myanmar-weather-app
**Date**: 2026-08-09 (revised)
**Phase**: Phase 0 Research (verified against live Earth2Studio docs)

---

## ARCHITECTURE DECISION RECORD

### ADR-001: Model Selection — Aurora1p5

**Decision**: Use `earth2studio.models.px.Aurora1p5` as the primary prognostic model.

**Why Aurora1p5**:
- 0.25° global resolution (appropriate for Myanmar synoptic-scale visualization)
- **Native hourly rollout**: produces t+1h through t+168h at 1-hour resolution without interpolation
- **Native `tp1h` output**: 1-hour accumulated total precipitation is a first-class model output
- Fine-tuned on IFS operational analyses → best skill when initialized with IFS
- Model weights: HuggingFace `hf://microsoft/aurora@c171214768997594e1a3fc6b8d9bbb489e9d21ab`
- Checkpoint: `aurora-0.25-v1.5.ckpt` (deterministic), `aurora-0.25-v1.5-ensemble.ckpt` (stochastic)
- VRAM requirement: 48 GB GPU

**Why NOT Aurora (base)**:
- Aurora base outputs only 4 surface variables; Aurora1p5 outputs 18 surface variables
- Aurora1p5 has richer diagnostic outputs including `tp1h`

**Why NOT AIFS**:
- AIFS requires IFS operational analysis with flux accumulation variables (tp06, cp06, ssrd06, strd06, etc.)
- Higher complexity initialization; same VRAM requirement (40 GB)
- Aurora1p5 is the stated primary candidate

---

### ADR-002: Initialization Source — IFS (with NCAR_ERA5 fallback)

**Decision**: IFS open-data analysis as primary; NCAR_ERA5 as development/historical fallback.

**Why IFS is required**:
From Aurora1p5 source docstring (verified):
> "Aurora v1.5 was pretrained on ERA5 and fine-tuned on IFS operational analyses and as such recommended to be initialized with IFS analyses."
> "GFS is not supported due to missing surface variables."

**Why NOT GFS**:
Aurora1p5 requires 18 surface variables (INPUT_VARIABLES surface set):
`msl, u10m, v10m, t2m, d2m, tcwv, tcc, u100m, v100m, sp, lcc, mcc, hcc, skt, stl1, swvl1, sic, sd`

GFS lexicon does NOT provide: `d2m` (dew point 2m), `u100m`, `v100m`, `lcc`, `mcc`, `hcc`, `skt`, `stl1`, `swvl1`, `sic`, `sd`.

**IFS open-data limitation — Sea Ice Concentration (sic)**:
From Aurora1p5 source (verified):
> "The open-data IFS does not publish sea ice concentration (sic). earth2studio.data.NCAR_ERA5 or earth2studio.data.ARCO (which provide all required variables) may be used instead."

**sic gap handling strategy**:
Myanmar sits at 9°N–29°N. The initialization requires a globally-gridded `sic` field.
For the tropical initialization region surrounding Myanmar, sea ice concentration is physically 0.
For the global model field (required by Aurora for the full 720×1440 input):
- **Option A (Recommended for MVP)**: Pad `sic` with climatological zero for non-polar regions; use ARCO/NCAR_ERA5 `sic` for polar grid cells. Documented in pipeline metadata.
- **Option B**: Use NCAR_ERA5 (historical, full variable set) for development runs, IFS for near-real-time
- **Option C**: Use ARCO as initialization (ERA5 historical, cloud-optimized, up to 2023)

**Implementation decision**:
- Production (real-time): `earth2studio.data.IFS` + sic patch from ARCO (or climatological 0)
- Development/demo: `earth2studio.data.NCAR_ERA5` (full variable coverage, historical)

**Access requirements**:
- IFS open-data: No credentials required (ECMWF open data initiative)
- NCAR_ERA5: Free via AWS (may incur transfer costs)
- ARCO: Free via Google Cloud (zarr)

---

### ADR-003: Precipitation — Aurora1p5 Native tp1h (No PrecipitationAFNO)

**Decision**: Use Aurora1p5's native `tp1h` output directly. PrecipitationAFNO/v2 is NOT used.

**Why this is a significant change from the initial plan**:
The initial plan assumed Aurora1p5 had no precipitation output and required PrecipitationAFNOv2 as a separate diagnostic model. This was incorrect.

**Verified Aurora1p5 output variables** (from source code):
- All 83 INPUT_VARIABLES (passed through)
- 7 additional diagnostic outputs: `i10fg`, `blh`, `uvb1h`, `ssrd1h`, `ttr1h`, **`tp1h`**, `sf1h`
- `tp1h` and `sf1h` require **log untransform** (`exp()`) before use

**What tp1h represents**:
- 1-hour accumulated total precipitation
- Produced at each hourly lead time (t+1h, t+2h, ..., t+168h)
- Units after log untransform + conversion: mm (per 1-hour accumulation period)
- Display convention: mm/h, understood as "mm accumulated in this 1-hour period"

**PrecipitationAFNO status (for reference)**:
- PrecipitationAFNO: Original version
- PrecipitationAFNOv2: "Improved" version, labeled "Improved Precipitation AFNO diagnostic model"
- Neither is deprecated in current docs
- Neither is needed for this application

---

### ADR-004: Temporal Resolution — Native Hourly (No Interpolation)

**Decision**: Aurora1p5 produces genuine hourly output. No interpolation required or permitted.

**Mechanism** (verified from source):
Aurora1p5's 6-hour auto-regressive step internally produces intermediate hourly predictions.
The `_forward_sub_steps` method iterates through `lead_time_hours=[1, 2, 3, 4, 5, 6]` per AR step,
calling `model.forward()` at each hour without additional model evaluations.
This produces 168 distinct model predictions (t+1h through t+168h) for a 7-day forecast.

**Implication**: The constitution's §VI requirement for hourly navigation is satisfied by native
model output. The initial plan's interpolation approach is obsolete and must NOT be implemented.

---

### ADR-005: Precipitation Display — 1-Hour Accumulation

**Decision**: Display precipitation as `mm/h` with clear disclosure that values represent
1-hour accumulated totals, not instantaneous rates.

**Rationale**:
- tp1h is a 1-hour accumulated precipitation variable
- Displaying as mm/h is common meteorological convention for hourly accumulated precip
- The UI MUST include a tooltip/info note: "mm/h = total precipitation accumulated over 1 hour"
- No 6-hour accumulation period applies; this is not a PrecipitationAFNO output
- No need for accumulation period display (e.g., "14:00–20:00 UTC") since each frame is 1-hour

---

### ADR-006: Forecast Data Format

**Decision**: Float32 binary arrays + `forecast.json` metadata.

**Artifact layout**:
```
data/
├── demo/ or forecast/
│   ├── forecast.json          # All metadata
│   ├── temperature.bin        # [169 × 81 × 41] float32 (t+0h to t+168h)
│   └── precipitation.bin      # [169 × 81 × 41] float32 (t+0h to t+168h)
```

**Grid dimensions** (Myanmar 0.25° bbox):
- lat: 9.0°N to 29.0°N → 81 points (29.0 - 9.0) / 0.25 + 1 = 81
- lon: 92.0°E to 102.0°E → 41 points (102.0 - 92.0) / 0.25 + 1 = 41
- times: 0h to 168h inclusive → 169 frames

**Size estimate**: 169 × 81 × 41 × 4 bytes × 2 variables ≈ 4.5 MB total

**Note on t+0h**: The t+0h frame stores the initialization state (from GFS/IFS analysis data).
Aurora1p5 starts producing forecasts at t+1h. The t+0h frame is included for reference continuity.

---

## 2. Aurora1p5 Variable Specification (Verified)

### Input Variables (83 total)

**Atmospheric (65)** — z, q, t, u, v at 13 pressure levels:
1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa

**Surface (18)**:
`msl, u10m, v10m, t2m, d2m, tcwv, tcc, u100m, v100m, sp, lcc, mcc, hcc, skt, stl1, swvl1, sic, sd`

### Output Variables (90 total)

All 83 input variables + 7 diagnostic-only outputs:
`i10fg` (instantaneous 10m wind gust), `blh` (boundary layer height),
`uvb1h` (UV radiation 1h), `ssrd1h` (surface solar radiation 1h),
`ttr1h` (top thermal radiation 1h), **`tp1h`** (total precipitation 1h),
`sf1h` (snowfall 1h)

Note: `tp1h` and `sf1h` are in **log space** in raw model output → must apply `exp()`.

### Variables used in frontend

| Variable | Aurora name | Unit (raw) | Transformation | Frontend unit |
|----------|------------|------------|---------------|---------------|
| 2m temperature | `t2m` | K | subtract 273.15 | °C |
| 1h precipitation | `tp1h` | log(m) | exp(), × 1000 | mm/h |

---

## 3. IFS Data Source Analysis

### earth2studio.data.IFS

**Access**: ECMWF open data — no credentials required
**Endpoint**: ECMWF open data public S3/HTTPS
**Resolution**: 0.25° lat-lon
**Coverage**: Global
**Update frequency**: 4× daily (00, 06, 12, 18 UTC), approximately 6h latency
**Variables provided**: Full IFS variable set (surface + pressure levels)
**Missing**: `sic` (sea ice concentration) — not published in ECMWF open data

**Lexicon**: `earth2studio/lexicon/ecmwf.py` → `IFSLexicon`

### earth2studio.data.NCAR_ERA5

**Access**: Free via AWS Open Data (NCAR sponsorship program)
**Coverage**: 1940 to present (near-real-time ERA5)
**Resolution**: 0.25°
**Variables**: Full ERA5 variable set including `sic`
**Use case**: Development, historical validation, sic source for patching

### earth2studio.data.ARCO

**Access**: Free via Google Cloud (zarr)
**Coverage**: 1959–2023 (historical only)
**Resolution**: 0.25°
**Variables**: Full ERA5 including `sic`
**Use case**: Historical development runs only; NOT for real-time production

---

## 4. PrecipitationAFNO Reference (Not Used, But Documented)

For reference if Aurora1p5 is ever replaced by a model without native precipitation:

| Version | Description | Status |
|---------|-------------|--------|
| PrecipitationAFNO | Original AFNO precipitation diagnostic | Available |
| PrecipitationAFNOv2 | "Improved Precipitation AFNO" | Available, current recommendation |
| OrbitGlobalPrecip | Global precipitation downscaling (9.5m, 126m variants) | Available |

If used in future: PrecipitationAFNOv2 requires 20 input variables including `r500`, `r850`, `tcwv`, `sp` — a variable gap that would require bridging from Aurora1p5 outputs.

---

## 5. Pipeline Architecture (Final)

```
IFS open data (earth2studio.data.IFS)
    ↓ [sic gap: patch from ARCO or pad with 0]
Aurora1p5 (earth2studio.models.px.Aurora1p5)
    │ Native hourly rollout: t+1h ... t+168h
    │ 90 output variables per hour
    ↓
ZarrBackend (earth2studio.io.ZarrBackend)
    ↓
xarray post-processing:
    ├── extract t2m → subtract 273.15 → °C
    └── extract tp1h → exp() → × 1000 → mm/h
    ↓
myanmar_subset.py: .sel(lat=slice(9,29), lon=slice(92,102))
    ↓
artifact_writer.py: Float32 binary + forecast.json
    ↓
data/forecast/ (or data/demo/)
```

**No interpolation. No PrecipitationAFNO. No derived variable bridging.**

---

## 6. Model Weights and Licensing

**Aurora1p5 weights**:
- HuggingFace: `hf://microsoft/aurora@c171214768997594e1a3fc6b8d9bbb489e9d21ab`
- Deterministic: `aurora-0.25-v1.5.ckpt`
- Statics: `aurora-0.25-v1.5-static.pickle`
- License: Microsoft Research License (check HuggingFace repository for current terms)
- Download: Handled automatically by `Aurora1p5.load_default_package()`

**IFS open data**:
- ECMWF open data license (Creative Commons-compatible, check ECMWF terms for current version)
- No commercial redistribution of raw analysis data without attribution

**No secrets, tokens, or API keys are required for the MVP data path.**

---

## 7. Relevant Examples

- Deterministic workflow: https://nvidia.github.io/earth2studio/examples/01_getting_started/01_deterministic_workflow.html
- Diagnostic workflow: https://nvidia.github.io/earth2studio/examples/01_getting_started/02_diagnostic_workflow.html
- No Aurora1p5-specific example exists in the current gallery

---

## 8. Summary: What Changed from Initial Plan

| Item | Initial Plan | Corrected Plan |
|------|-------------|----------------|
| Initialization source | GFS (free) | IFS (free, but sic gap to handle) |
| Precipitation model | PrecipitationAFNOv2 (separate) | Aurora1p5 native tp1h |
| Temporal resolution | 6h native → linear interp to 1h | Native 1h (no interpolation) |
| Precip units | mm/h (divided from 6h accum) | mm/h (1-hour accumulation, genuine) |
| Variable bridging | DerivedRH + DerivedTCWV needed | Not needed |
| Pipeline complexity | High | Significantly reduced |
| VRAM needed | 48GB (Aurora) + 40GB (PrecipAFNO) | 48GB (Aurora1p5 only) |
