"""
generate_forecast.py — GraphCastSmall Myanmar 24-hour precipitation forecast.

Model: earth2studio.models.px.GraphCastSmall
  Resolution : 1.0° global (181 × 360 lat-lon grid, pole-inclusive)
  Timestep   : 6h per auto-regressive step
  Backend    : JAX + Haiku (DeepMind GraphCast wrapped by Earth2Studio)
  Checkpoint : gs://dm_graphcast/graphcast
               params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 -
               pressure levels 13 - mesh 2to5 - precipitation input and output.npz

Initialization (two-timestep requirement — verified from input_coords):
  GraphCastSmall.input_coords["lead_time"] = [−6h, 0h]
  Both t−6h and t+0h must be fetched from the data source before inference.
  Earth2Studio handles this automatically when data source is passed to
  e2run.deterministic().

  --source arco  (default) — ARCO ERA5 reanalysis, 1959–2023, no credentials,
                             all 83 GraphCastSmall input variables including tp06
  --source ifs              — IFS HRES open data, near-real-time, no credentials,
                             all 83 GraphCastSmall input variables including tp06

  NOT compatible: NCAR_ERA5 (missing tp06 in lexicon), GFS (unverified)

Precipitation (tp06):
  tp06 is native GraphCastSmall output — physical metres, NO log transform.
  This is NOT Aurora's tp1h which required exp() before use.
  Conversion  : tp06_metres × 1000 = mm / 6-hour accumulation
  Semantics   : total precipitation accumulated over the 6h period ending at
                the forecast valid time. Not an instantaneous rate.
  Physical constraint: clamp to ≥ 0 (no negative precipitation).

Output:
  data/forecast/
    precipitation.bin  — float32 [5 × 21 × 11] C-order (t+0h .. t+24h, step 6h)
    forecast.json      — complete provenance metadata (schema v2.0)

Myanmar grid at 1.0°:
  lat : 9°N to 29°N → 21 points (1° spacing, ascending south-to-north)
  lon : 92°E to 102°E → 11 points (1° spacing)

Inference runs globally; only the output is subset to Myanmar.
GraphCastSmall requires a full 181×360 global input — never crop model inputs.

T4 VRAM staged test (Constitution §XI):
  Documented badge: "gpu:40gb". T4 has 16 GB. Compatibility is unverified.
  This script performs a staged test before the full run:
    Stage 1 — GPU baseline (before model load)
    Stage 2 — Model loaded (after weights downloaded to JAX)
    Stage 3 — Single 6h inference step (smoke test)
    Stage 4 — Full 24h forecast (4 steps) if Stage 3 passes

  On OOM at any stage: print a structured report and exit with code 1.
  The data is never silently modified or substituted with demo values.

Usage:
    uv run python scripts/generate_forecast.py \\
        --source arco \\
        --init-time 2022-01-01T00:00:00Z \\
        [--output-dir data/forecast/] \\
        [--smoke-test-only] \\
        [--precip-max-mm 500] \\
        [--debug]

Note on JAX GPU memory:
  GraphCastSmall's GPU compute is managed by JAX, not PyTorch's CUDA allocator.
  torch.cuda.memory_allocated() reflects only PyTorch's data-transport tensors.
  JAX pre-allocates its own memory pool. Set XLA_PYTHON_CLIENT_PREALLOCATE=false
  (done below) to prevent JAX from reserving all available GPU memory up-front.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Environment variables ────────────────────────────────────────────────────
# Must be set before any JAX or Earth2Studio imports to take effect.

# Prevent JAX from pre-allocating ~90% of GPU memory on first use.
# Without this, JAX grabs most VRAM before any PyTorch allocations,
# making VRAM budgeting unpredictable on a 16 GB T4.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Cap JAX's GPU memory fraction. 0.85 leaves ~2.4 GB for PyTorch tensors on a T4.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
# PyTorch: reduce fragmentation in the data-transport tensor allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import xarray as xr

# ── Myanmar spatial constants ─────────────────────────────────────────────────
MYANMAR_LAT_MIN = 9.0
MYANMAR_LAT_MAX = 29.0
MYANMAR_LON_MIN = 92.0
MYANMAR_LON_MAX = 102.0

# ── GraphCastSmall temporal constants (verified from source) ──────────────────
GC_STEP_HOURS = 6          # Native AR step — each call produces +6h
GC_HORIZON_HOURS = 24      # MVP horizon
GC_N_STEPS = GC_HORIZON_HOURS // GC_STEP_HOURS   # 4 AR steps
GC_N_FRAMES = GC_N_STEPS + 1                      # 5 total frames (t+0 .. t+24)

# Myanmar grid at 1.0° resolution
GC_N_LAT = 21  # 9°N to 29°N inclusive (step 1°)
GC_N_LON = 11  # 92°E to 102°E inclusive (step 1°)

# Default physical sanity threshold for tp06 in mm / 6h.
# Extreme tropical 6-hour rainfall records are ~300–400 mm; 500 mm is a conservative
# upper bound that should never be exceeded in a valid GraphCastSmall output.
DEFAULT_PRECIP_MAX_MM = 500.0


# ── VRAM utilities ────────────────────────────────────────────────────────────

def _torch_vram_gb() -> tuple[float, float]:
    """Return (allocated_gb, reserved_gb) from PyTorch's CUDA allocator.
    Returns (0, 0) on CPU or if torch is not imported yet.
    Note: JAX manages its own separate memory pool not reflected here.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0, 0.0
        return (
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved() / 1e9,
        )
    except Exception:
        return 0.0, 0.0


def _jax_vram_gb() -> float | None:
    """Attempt to read JAX GPU memory usage in GB. Returns None if unavailable."""
    try:
        import jax
        devices = jax.devices("gpu")
        if not devices:
            return None
        stats = devices[0].memory_stats()
        if stats is None:
            return None
        return stats.get("bytes_in_use", 0) / 1e9
    except Exception:
        return None


def print_vram_report(label: str) -> None:
    """Print a VRAM usage snapshot to stdout."""
    torch_alloc, torch_res = _torch_vram_gb()
    jax_used = _jax_vram_gb()
    print(f"  [{label}]")
    print(f"    PyTorch allocated : {torch_alloc:.2f} GB  reserved: {torch_res:.2f} GB")
    if jax_used is not None:
        print(f"    JAX bytes_in_use  : {jax_used:.2f} GB")
    else:
        print(f"    JAX bytes_in_use  : (not available — JAX GPU memory pool is opaque)")


# ── Precipitation sanity check ────────────────────────────────────────────────

def check_precipitation_sanity(tp06: np.ndarray, sanity_max_mm: float) -> bool:
    """
    Print a structured sanity report for the tp06 array and return True if PASS.

    tp06 units entering this function: mm / 6h accumulation (after ×1000 conversion).

    Checks (never modifies the array):
      - No NaN values
      - No infinite values
      - All values ≥ 0 (physical constraint: precipitation cannot be negative)
      - Max ≤ sanity_max_mm (configurable physical upper bound)

    Returns True on PASS, False on any FAIL.
    """
    n_nan = int(np.sum(np.isnan(tp06)))
    n_inf = int(np.sum(np.isinf(tp06)))
    n_neg = int(np.sum(tp06 < 0.0))
    n_zero = int(np.sum(tp06 == 0.0))
    n_total = tp06.size

    valid = tp06[np.isfinite(tp06)]
    vmin    = float(valid.min())    if valid.size else float("nan")
    vmax    = float(valid.max())    if valid.size else float("nan")
    vmedian = float(np.median(valid)) if valid.size else float("nan")
    p95     = float(np.percentile(valid, 95)) if valid.size else float("nan")
    p99     = float(np.percentile(valid, 99)) if valid.size else float("nan")
    zero_frac = n_zero / n_total * 100.0

    print()
    print("  PRECIPITATION SANITY CHECK (tp06)")
    print("  " + "-" * 42)
    print(f"  Variable          : tp06 (GraphCastSmall native output)")
    print(f"  Unit              : mm / 6h accumulation")
    print(f"  Accumulation      : 6-hour total, ending at forecast valid time")
    print(f"  Transform applied : metres × 1000 (NO exp/log transform)")
    print(f"  Shape             : {tp06.shape}")
    print(f"  Min               : {vmin:.4f} mm")
    print(f"  Median            : {vmedian:.4f} mm")
    print(f"  P95               : {p95:.4f} mm")
    print(f"  P99               : {p99:.4f} mm")
    print(f"  Max               : {vmax:.4f} mm")
    print(f"  Zero-rain fraction: {zero_frac:.1f}%  (includes synthetic t+0h zero frame)")

    failures = []
    if n_nan > 0:
        failures.append(f"NaN values: {n_nan} (check data source / model)")
    if n_inf > 0:
        failures.append(f"Infinite values: {n_inf}")
    if n_neg > 0:
        failures.append(f"Negative values after clamping: {n_neg} (clamping is active — this should not happen)")
    if not np.isnan(vmax) and vmax > sanity_max_mm:
        failures.append(
            f"Max {vmax:.2f} mm/6h exceeds physical sanity threshold {sanity_max_mm:.0f} mm/6h. "
            f"Extreme tropical 6h rainfall records are ~300–400 mm/6h. "
            f"If this is a valid intense convective event, increase --precip-max-mm."
        )

    if failures:
        print(f"  Status            : FAIL")
        for msg in failures:
            print(f"    FAIL: {msg}")
        print()
        return False

    print(f"  Status            : PASS")
    print()
    return True


# ── Initialization time utilities ─────────────────────────────────────────────

def find_latest_ifs_init() -> datetime:
    """Return the most recent IFS HRES analysis time likely available on open data.
    IFS publishes 00Z and 12Z runs with ~6h latency.
    """
    now = datetime.now(timezone.utc)
    candidates = []
    for delta_days in [0, 1, 2]:
        day = now - timedelta(days=delta_days)
        for hour in [12, 0]:
            candidates.append(day.replace(hour=hour, minute=0, second=0, microsecond=0))
    for t in candidates:
        if (now - t).total_seconds() >= 6 * 3600:
            return t
    return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    init_time: datetime,
    source: str,
    output_dir: Path,
    smoke_test_only: bool,
    debug: bool,
    sanity_max_mm: float,
) -> int:
    """
    Run the GraphCastSmall 24h Myanmar precipitation forecast pipeline.

    Returns 0 on success, 1 on error.
    """
    import torch
    import earth2studio.run as e2run
    from earth2studio.io import ZarrBackend
    from earth2studio.models.px import GraphCastSmall

    output_dir.mkdir(parents=True, exist_ok=True)
    t_pipeline_start = time_module.time()

    print("=" * 64)
    print("GraphCastSmall Myanmar Forecast Pipeline")
    print(f"  Init time    : {init_time.isoformat()}")
    print(f"  Source       : {source.upper()}")
    print(f"  Horizon      : {GC_HORIZON_HOURS}h ({GC_N_STEPS} steps × {GC_STEP_HOURS}h)")
    print(f"  Frames       : {GC_N_FRAMES} total (t+0h through t+{GC_HORIZON_HOURS}h)")
    print(f"  Myanmar grid : {GC_N_LAT} lat × {GC_N_LON} lon at 1.0°")
    print(f"  Output       : {output_dir.resolve()}")
    print(f"  Smoke only   : {smoke_test_only}")
    print("=" * 64)

    # ── Stage 1: Device detection ─────────────────────────────────────────────
    print("\n[STAGE 1/4] GPU detection and VRAM baseline")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        vram_total_gb = props.total_memory / 1e9
        device_label = f"{gpu_name} ({vram_total_gb:.0f} GB VRAM)"
        print(f"  GPU    : {gpu_name}")
        print(f"  VRAM   : {vram_total_gb:.1f} GB total")
        if vram_total_gb < 16.0:
            print(f"  WARNING: Less than 16 GB VRAM. GraphCastSmall (Rec 40 GB) may OOM.")
        elif vram_total_gb < 40.0:
            print(f"  NOTE: {vram_total_gb:.0f} GB < 40 GB Rec VRAM badge. "
                  f"T4 compatibility is unverified — proceeding with staged test.")
    else:
        device = torch.device("cpu")
        gpu_name = "CPU"
        vram_total_gb = 0.0
        device_label = "CPU (no GPU — inference will be very slow)"
        print(f"  WARNING: No CUDA GPU detected. Running on {device_label}.")

    print_vram_report("baseline — before model load")

    # ── Stage 2: Data source ──────────────────────────────────────────────────
    print(f"\n[STAGE 2/4] Initializing {source.upper()} data source")

    if source == "arco":
        from earth2studio.data import ARCO
        data = ARCO()
        source_label = "ARCO ERA5 reanalysis (Google Cloud, no credentials)"
        source_attribution = (
            "ERA5 via ARCO (Analysis-Ready Cloud-Optimized ERA5). "
            "Hosted on Google Cloud. ERA5 © ECMWF/Copernicus."
        )
        print(f"  Source: {source_label}")
        print(f"  Note: ARCO covers 1959–2023. Init time must be within this range.")
    elif source == "ifs":
        from earth2studio.data import IFS
        data = IFS()
        source_label = "IFS HRES open data (ECMWF, no credentials)"
        source_attribution = (
            "IFS HRES analysis — ECMWF open data. "
            "CC BY 4.0. https://confluence.ecmwf.int/display/DAC/ECMWF+open+data"
        )
        print(f"  Source: {source_label}")
        print(f"  Note: IFS requires a recent init time (last ~48h from operational run).")
    else:
        print(f"  ERROR: Unknown source '{source}'. Use 'arco' or 'ifs'.")
        return 1

    # ── Stage 3: Load GraphCastSmall ──────────────────────────────────────────
    print(f"\n[STAGE 3/4] Loading GraphCastSmall weights")
    t_load_start = time_module.time()

    try:
        package = GraphCastSmall.load_default_package()
        model = GraphCastSmall.load_model(package)
        load_time = time_module.time() - t_load_start
        print(f"  Model loaded in {load_time:.1f}s")
        print(f"  Checkpoint: GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
              f"pressure levels 13 - mesh 2to5 - precipitation input and output.npz")
        print(f"  Backend: JAX + Haiku (bfloat16 cast applied internally by model)")
        print(f"  NOTE: Do NOT call model.to(bfloat16) — it uses bfloat16 internally via JAX.")
    except Exception as e:
        print(f"  ERROR: GraphCastSmall load failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print_vram_report("after model load")

    # ── Stage 4a: Smoke test — single 6h step ────────────────────────────────
    print(f"\n[STAGE 4/4] Staged inference")
    print(f"\n  [4a] Smoke test — single 6h step (nsteps=1)")
    print(f"       Purpose: verify GraphCastSmall fits within available VRAM.")

    io_smoke = ZarrBackend()
    t_smoke_start = time_module.time()
    try:
        with torch.inference_mode():
            io_smoke = e2run.deterministic(
                time=[init_time],
                nsteps=1,
                prognostic=model,
                data=data,
                io=io_smoke,
                device=device,
                verbose=True,
            )
        smoke_time = time_module.time() - t_smoke_start
        print(f"  [4a] Smoke test PASSED in {smoke_time:.1f}s")
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "outofmemory" in type(e).__name__.lower():
            print(f"\n  SMOKE TEST FAILED — OUT OF MEMORY")
            print(f"  Error: {e}")
            print_vram_report("at OOM")
            print(f"\n  GraphCastSmall could not complete a single 6h inference step.")
            print(f"  Documented Rec VRAM badge: 40 GB. This GPU has {vram_total_gb:.0f} GB.")
            print(f"  Do NOT attempt workarounds without explicit user approval (Constitution §XI).")
            return 1
        print(f"  ERROR: Smoke test failed with unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"  ERROR: Smoke test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print_vram_report("after smoke test (1 step)")

    if smoke_test_only:
        print(f"\n  --smoke-test-only specified. Stopping after successful 1-step inference.")
        print(f"  Re-run without --smoke-test-only to generate the full 24h forecast.")
        return 0

    # ── Stage 4b: Full 24h forecast (4 steps) ────────────────────────────────
    print(f"\n  [4b] Full 24h forecast (nsteps={GC_N_STEPS})")

    io_full = ZarrBackend()
    t_infer_start = time_module.time()
    try:
        with torch.inference_mode():
            io_full = e2run.deterministic(
                time=[init_time],
                nsteps=GC_N_STEPS,
                prognostic=model,
                data=data,
                io=io_full,
                device=device,
                verbose=True,
            )
        inference_time = time_module.time() - t_infer_start
        print(f"  [4b] Full forecast PASSED in {inference_time:.1f}s ({inference_time/60:.1f} min)")
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "outofmemory" in type(e).__name__.lower():
            print(f"\n  FULL FORECAST FAILED — OUT OF MEMORY (single-step was OK)")
            print(f"  Error: {e}")
            print_vram_report("at OOM (full run)")
            print(f"  The smoke test (1 step) succeeded but the full 4-step run OOM'd.")
            print(f"  This suggests memory accumulates across AR steps in JAX's JIT cache.")
            return 1
        print(f"  ERROR: Full forecast failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"  ERROR: Full forecast failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print_vram_report("after full forecast (4 steps)")

    # Free model from memory before post-processing
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Post-processing ───────────────────────────────────────────────────────
    print(f"\nPost-processing GraphCastSmall output")
    ds = xr.open_zarr(io_full.store)

    if debug:
        print(f"  Zarr dataset vars      : {list(ds.data_vars)}")
        print(f"  Zarr dataset dims      : {dict(ds.dims)}")
        print(f"  Zarr dataset coords    : {list(ds.coords)}")

    if "tp06" not in ds:
        print(f"  ERROR: 'tp06' not found in GraphCastSmall zarr output.")
        print(f"  Available variables: {list(ds.data_vars)}")
        return 1

    tp06_raw = ds["tp06"]
    if debug:
        print(f"  tp06 raw shape/dims    : {dict(tp06_raw.sizes)}")
        print(f"  tp06 lat range         : {float(tp06_raw.lat.min()):.1f} to {float(tp06_raw.lat.max()):.1f}")
        print(f"  tp06 lon range         : {float(tp06_raw.lon.min()):.1f} to {float(tp06_raw.lon.max()):.1f}")

    # ── Spatial subset to Myanmar ─────────────────────────────────────────────
    # Model output lat is DESCENDING (90 to −90, verified from output_coords source).
    # For descending lat, slice(LAT_MAX, LAT_MIN) selects the correct north-to-south band.
    tp06_myanmar = tp06_raw.sel(
        lat=slice(MYANMAR_LAT_MAX, MYANMAR_LAT_MIN),
        lon=slice(MYANMAR_LON_MIN, MYANMAR_LON_MAX),
    )

    # Squeeze out any size-1 dimensions (batch=1, time=1 init time)
    tp06_myanmar = tp06_myanmar.squeeze()

    if debug:
        print(f"  tp06 after Myanmar subset + squeeze: {dict(tp06_myanmar.sizes)}")

    # Ensure lat is ASCENDING (south-to-north) for the binary artifact.
    # The frontend indexes lat from south: lat[0]=9°N, lat[20]=29°N.
    if tp06_myanmar.coords["lat"].values[0] > tp06_myanmar.coords["lat"].values[-1]:
        tp06_myanmar = tp06_myanmar.isel(lat=slice(None, None, -1))

    lats_out = tp06_myanmar.coords["lat"].values.astype(np.float64)
    lons_out = tp06_myanmar.coords["lon"].values.astype(np.float64)
    n_lat_out, n_lon_out = len(lats_out), len(lons_out)

    # Validate expected grid dimensions
    if n_lat_out != GC_N_LAT:
        print(f"  WARNING: Expected {GC_N_LAT} lat points, got {n_lat_out}. "
              f"Lat values: {lats_out}")
    if n_lon_out != GC_N_LON:
        print(f"  WARNING: Expected {GC_N_LON} lon points, got {n_lon_out}. "
              f"Lon values: {lons_out}")

    # ── tp06 unit conversion ──────────────────────────────────────────────────
    # GraphCastSmall tp06 output is in physical METRES (verified from model source).
    # NO log or exponential transform is applied — unlike Aurora's tp1h.
    # Conversion: metres × 1000 = mm per 6-hour accumulation period.
    tp06_m = tp06_myanmar.values.astype(np.float32)
    tp06_mm = np.maximum(tp06_m * 1000.0, 0.0).astype(np.float32)

    if debug:
        print(f"  tp06 metres: min={tp06_m.min():.6f}, max={tp06_m.max():.4f}")
    print(f"  tp06 (mm/6h): shape={tp06_mm.shape}, "
          f"range=[{tp06_mm.min():.4f}, {tp06_mm.max():.2f}]")

    # ── Prepend t+0h initialization frame ────────────────────────────────────
    # Earth2Studio run.deterministic() produces forecast steps t+6h through t+24h.
    # We prepend a synthetic t+0h frame (precipitation = 0) to complete the 5-frame series.
    # The t+0h frame represents the analysis state; no forecast accumulation has occurred.
    tp06_full = np.concatenate(
        [np.zeros((1, n_lat_out, n_lon_out), dtype=np.float32), tp06_mm],
        axis=0,
    )
    n_frames = tp06_full.shape[0]

    if n_frames != GC_N_FRAMES:
        print(f"  WARNING: Expected {GC_N_FRAMES} frames, got {n_frames}.")

    print(f"  Final array: [{n_frames}, {n_lat_out}, {n_lon_out}] = [frames, lat, lon]")

    # ── Timestamps ───────────────────────────────────────────────────────────
    times_utc = [
        (init_time + timedelta(hours=i * GC_STEP_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(n_frames)  # i=0→t+0h, i=1→t+6h, ..., i=4→t+24h
    ]

    # ── Write precipitation.bin ───────────────────────────────────────────────
    precip_path = output_dir / "precipitation.bin"
    precip_path.write_bytes(tp06_full.astype("<f4").tobytes())
    file_size_bytes = precip_path.stat().st_size
    print(f"  precipitation.bin: {file_size_bytes} bytes ({file_size_bytes/1024:.1f} KB)")

    expected_bytes = n_frames * n_lat_out * n_lon_out * 4
    if file_size_bytes != expected_bytes:
        print(f"  WARNING: Expected {expected_bytes} bytes, got {file_size_bytes}")

    # ── Sanity check ──────────────────────────────────────────────────────────
    precip_ok = check_precipitation_sanity(tp06_full, sanity_max_mm)
    if not precip_ok:
        print("  WARNING: Sanity check FAILED. Data written as-is (never silently modified).")

    # ── Write forecast.json ───────────────────────────────────────────────────
    total_time = time_module.time() - t_pipeline_start
    torch_alloc_final, torch_res_final = _torch_vram_gb()
    jax_used_final = _jax_vram_gb()

    meta = {
        "schema_version": "2.0",
        "model": "GraphCastSmall",
        "model_version": "1.0",
        "model_checkpoint": (
            "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
            "pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
        ),
        "model_source": "gs://dm_graphcast/graphcast",
        "model_attribution": (
            "GraphCast by DeepMind/Google. "
            "Lam et al. (2023), Science. https://arxiv.org/abs/2212.12794. "
            "Earth2Studio wrapper by NVIDIA."
        ),
        "initialization_source": source.upper(),
        "initialization_time": times_utc[0],
        "forecast_generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_horizon_hours": GC_HORIZON_HOURS,
        "native_timestep_hours": GC_STEP_HOURS,
        "n_times": n_frames,
        "spatial_resolution_deg": 1.0,
        "display_resolution_deg": None,
        "region": "Myanmar",
        "bbox": {
            "lat_min": float(lats_out.min()),
            "lat_max": float(lats_out.max()),
            "lon_min": float(lons_out.min()),
            "lon_max": float(lons_out.max()),
        },
        "grid": {
            "n_lat": n_lat_out,
            "n_lon": n_lon_out,
        },
        "lat": lats_out.tolist(),
        "lon": lons_out.tolist(),
        "times_utc": times_utc,
        "variables": {
            "precipitation": {
                "display_name": "Precipitation",
                "units": "mm / 6h",
                "source_variable": "tp06",
                "temporal_resolution": "6-hourly",
                "temporal_semantics": (
                    "Total precipitation accumulated over the 6-hour forecast "
                    "period ending at the displayed valid time."
                ),
                "temporal_disclosure": (
                    "Precipitation values represent total rainfall accumulated "
                    "during the 6-hour forecast period ending at the displayed time. "
                    "These are not instantaneous rainfall rates."
                ),
                "transformation_provenance": {
                    "source_variable": "tp06",
                    "source_unit": "metres",
                    "conversion": "metres × 1000",
                    "output_unit": "mm",
                    "accumulation_period_hours": GC_STEP_HOURS,
                    "log_transform_applied": False,
                    "exp_transform_applied": False,
                    "pipeline": (
                        "GraphCastSmall native tp06 output (physical metres, no log transform) "
                        "→ metres × 1000 → mm / 6h accumulation "
                        "→ clamp to ≥ 0 (physical constraint)"
                    ),
                },
                "t0_note": (
                    "t+0h frame (index 0) is synthetic: precipitation set to 0.0. "
                    "It represents the analysis state; no forecast accumulation has occurred."
                ),
                "native_output": True,
                "file": "precipitation.bin",
                "fill_value": None,
            },
        },
        "data_source_attribution": source_attribution,
        "earth2studio_version": ">=0.17.0",
        "inference_config": {
            "device": device_label,
            "vram_total_gb": round(vram_total_gb, 1),
            "jax_env": {
                "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
                "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            },
            "torch_alloc_gb_at_end": round(torch_alloc_final, 2),
            "torch_reserved_gb_at_end": round(torch_res_final, 2),
            "jax_bytes_in_use_gb_at_end": round(jax_used_final, 2) if jax_used_final is not None else None,
            "model_load_time_seconds": round(load_time),
            "inference_time_seconds": round(inference_time),
            "total_pipeline_time_seconds": round(total_time),
        },
        "is_demo": False,
    }

    json_path = output_dir / "forecast.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  forecast.json written")

    print(f"\n{'='*64}")
    print(f"Pipeline complete in {total_time/60:.1f} min")
    print(f"")
    print(f"  Model     : GraphCastSmall")
    print(f"  Source    : {source.upper()}")
    print(f"  GPU       : {device_label}")
    print(f"  Frames    : {n_frames} ({', '.join(times_utc)})")
    print(f"  Grid      : {n_lat_out} lat × {n_lon_out} lon")
    print(f"  Output    : {output_dir.resolve()}")
    print(f"")
    print(f"Validate with:")
    print(f"  uv run python scripts/validate_forecast.py --data-dir {output_dir}")
    print(f"{'='*64}")

    return 0


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate GraphCastSmall 24h Myanmar precipitation forecast",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Development run (historical date, ARCO ERA5)
  uv run python scripts/generate_forecast.py --source arco --init-time 2022-01-01T00:00:00Z

  # Smoke test only (verify VRAM before committing to full run)
  uv run python scripts/generate_forecast.py --source arco --init-time 2022-01-01T00:00:00Z --smoke-test-only

  # Operational run (IFS open data, recent init time)
  uv run python scripts/generate_forecast.py --source ifs
""",
    )
    parser.add_argument(
        "--init-time",
        default=None,
        metavar="YYYY-MM-DDTHH:MM:SSZ",
        help=(
            "Forecast initialization time (ISO 8601). "
            "For --source arco: must be within 1959–2023 (e.g. 2022-01-01T00:00:00Z). "
            "For --source ifs: defaults to latest available IFS analysis."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["arco", "ifs"],
        default="arco",
        help=(
            "Initialization data source. "
            "'arco' (default): ARCO ERA5 reanalysis, historical, no credentials. "
            "'ifs': IFS HRES open data, near-real-time, no credentials."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/forecast",
        help="Output directory for precipitation.bin and forecast.json (default: data/forecast)",
    )
    parser.add_argument(
        "--smoke-test-only",
        action="store_true",
        help=(
            "Run only the staged 1-step smoke test (6h inference) without "
            "producing the full 24h forecast. Use to verify VRAM compatibility."
        ),
    )
    parser.add_argument(
        "--precip-max-mm",
        type=float,
        default=DEFAULT_PRECIP_MAX_MM,
        metavar="MM",
        help=(
            f"Physical sanity threshold for maximum tp06 (mm / 6h). "
            f"A FAIL is reported if the max exceeds this value; data is never modified. "
            f"Default: {DEFAULT_PRECIP_MAX_MM}. Extreme tropical 6h records: ~300–400 mm."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print verbose debug output including zarr structure and raw value ranges.",
    )

    args = parser.parse_args()

    # Resolve init_time
    if args.init_time:
        try:
            init_time = datetime.fromisoformat(args.init_time.replace("Z", "+00:00"))
        except ValueError as e:
            print(f"ERROR: Invalid --init-time format: {e}")
            print(f"       Expected ISO 8601, e.g. 2022-01-01T00:00:00Z")
            return 1
    else:
        if args.source == "ifs":
            init_time = find_latest_ifs_init()
            print(f"Auto-detected IFS init time: {init_time.isoformat()}")
        else:
            print(
                "ERROR: --init-time is required for --source arco. "
                "Specify a date within the ARCO ERA5 range (1959–2023). "
                "Example: --init-time 2022-01-01T00:00:00Z"
            )
            return 1

    # Warn if init_time appears to be outside ARCO coverage
    if args.source == "arco":
        arco_cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
        if init_time >= arco_cutoff:
            print(
                f"WARNING: init_time {init_time.date()} may be outside ARCO's coverage "
                f"(ERA5 reanalysis ends approximately 2023). "
                f"Use --source ifs for recent dates."
            )

    return run_pipeline(
        init_time=init_time,
        source=args.source,
        output_dir=Path(args.output_dir),
        smoke_test_only=args.smoke_test_only,
        debug=args.debug,
        sanity_max_mm=args.precip_max_mm,
    )


if __name__ == "__main__":
    sys.exit(main())
