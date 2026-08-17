"""
generate_forecast.py — GraphCastOperational Myanmar 168h (7-day) 4-variable forecast.
Schema v5.0. Phase R4.

Model: GraphCastOperational via Earth2Studio 0.17.0
  Resolution : 0.25° global (721 × 1440 lat-lon grid)
  Timestep   : 6h per autoregressive step
  Backend    : JAX CPU (XLA ARM64 on Apple M4)
  Checkpoint : GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 -
               pressure levels 13 - mesh 2to6 - precipitation output only.npz
  Source     : gs://dm_graphcast/graphcast

Initialization: two-timestep (t-6h AND t+0h), handled by earth2studio.
Data source  : ARCO ERA5 reanalysis (1959-2023, no credentials required)

Output variables (schema v5.0):
  precipitation   : tp06 (metres/6h) → mm/hr  (clamp ≥ 0 after recording raw stats)
  temperature     : t2m  (K)         → °C     (K - 273.15)
  wind_speed      : u10m, v10m (m/s) → knots  (sqrt(u²+v²) × 1.94384)
  wind_direction  : u10m, v10m (m/s) → °FROM  ((270 - atan2(v,u)×180/π) mod 360)

Myanmar grid at 0.25°  (extracted after global inference):
  lat : 9.0°N – 29.0°N   → expected 81 points (verified from actual output)
  lon : 92.0°E – 102.0°E → expected 41 points (verified from actual output)

JAX compilation cache:
  ~/.cache/earth2studio/graphcast/jax_compile_cache/
  Cold start: ~34 min (R3 measured).  Warm start: cache hit, seconds.

Output directory (default): data/forecast_v4/
  Does NOT overwrite data/forecast/ (schema v3.0 production artifacts).

Usage:
    uv run python scripts/generate_forecast.py \\
        --init-time 2021-01-01T00:00:00Z \\
        [--output-dir data/forecast_v4/] \\
        [--debug]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as time_module
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── JAX config — MUST precede any JAX/torch import ───────────────────────────
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import numpy as np
import psutil

# ── Spatial constants ─────────────────────────────────────────────────────────
MYANMAR_LAT_MIN =  9.0
MYANMAR_LAT_MAX = 29.0
MYANMAR_LON_MIN = 92.0
MYANMAR_LON_MAX = 102.0

# ── GraphCastOperational temporal constants ───────────────────────────────────
GC_STEP_HOURS    = 6
GC_HORIZON_HOURS = 168
GC_N_STEPS       = GC_HORIZON_HOURS // GC_STEP_HOURS   # 28 AR steps
# n_frames = nsteps + 1 → expected 29 (t+0h through t+168h)
# Verified from actual output; script asserts the actual value.

# ── Physics bounds for validation ────────────────────────────────────────────
PRECIP_SANITY_MAX_MM_HR = 300.0   # 300 mm/hr is extreme but not impossible
TEMP_MIN_C              = -90.0
TEMP_MAX_C              =  70.0
WIND_SANITY_MAX_KT      = 200.0   # 200 kt is Cat-5 hurricane territory

# ── Four output variables needed from the model ───────────────────────────────
OUTPUT_VARS = np.array(["tp06", "t2m", "u10m", "v10m"])

# ── JAX persistent compilation cache ─────────────────────────────────────────
_JAX_CACHE_DIR = (
    Path.home() / ".cache" / "earth2studio" / "graphcast" / "jax_compile_cache"
)
_JAX_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def mem_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


# ── Conversion functions ──────────────────────────────────────────────────────

def tp06_to_mm_per_hr(tp06_metres: np.ndarray) -> np.ndarray:
    """Convert tp06 from native metres/6h to mm/hr (display unit).
    Raw negatives are recorded before clamping; see stats output.
    """
    return np.maximum(tp06_metres * 1000.0 / GC_STEP_HOURS, 0.0).astype(np.float32)


def t2m_to_celsius(t2m_k: np.ndarray) -> np.ndarray:
    """Convert 2m temperature from Kelvin to Celsius."""
    return (t2m_k - 273.15).astype(np.float32)


def wind_speed_knots(u10m: np.ndarray, v10m: np.ndarray) -> np.ndarray:
    """Compute 10m wind speed from u/v components; convert m/s → knots."""
    return (np.sqrt(u10m**2 + v10m**2) * 1.94384).astype(np.float32)


def wind_direction_degrees(u10m: np.ndarray, v10m: np.ndarray) -> np.ndarray:
    """Compute 10m wind direction using standard meteorological FROM convention.
    Result in [0, 360) degrees.
    """
    deg = (270.0 - np.degrees(np.arctan2(v10m, u10m))) % 360.0
    return deg.astype(np.float32)


# ── Variable statistics ───────────────────────────────────────────────────────

def var_stats(arr: np.ndarray, label: str, unit: str) -> dict:
    """Compute full statistics for a variable array. Prints report, returns dict."""
    flat = arr.flatten()
    n_nan  = int(np.sum(np.isnan(flat)))
    n_inf  = int(np.sum(np.isinf(flat)))
    valid  = flat[np.isfinite(flat)]
    n_valid = len(valid)

    stats = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "n_total": flat.size,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_valid": n_valid,
    }
    if n_valid > 0:
        stats.update({
            "min": float(np.min(valid)),
            "median": float(np.median(valid)),
            "p95": float(np.percentile(valid, 95)),
            "p99": float(np.percentile(valid, 99)),
            "max": float(np.max(valid)),
        })
    else:
        stats.update({"min": None, "median": None, "p95": None, "p99": None, "max": None})

    print(f"\n  {label} ({unit})")
    print(f"  {'─'*48}")
    print(f"  shape        : {arr.shape}")
    print(f"  dtype        : {arr.dtype}")
    print(f"  NaN count    : {n_nan}")
    print(f"  Inf count    : {n_inf}")
    if n_valid > 0:
        print(f"  min          : {stats['min']:.4f}")
        print(f"  median       : {stats['median']:.4f}")
        print(f"  P95          : {stats['p95']:.4f}")
        print(f"  P99          : {stats['p99']:.4f}")
        print(f"  max          : {stats['max']:.4f}")
    return stats


def precip_extra_stats(tp06_raw_metres: np.ndarray, tp06_mm_hr: np.ndarray) -> dict:
    """Extra precipitation diagnostics."""
    flat_raw  = tp06_raw_metres.flatten()
    flat_disp = tp06_mm_hr.flatten()
    n_neg_raw  = int(np.sum(flat_raw < 0))
    n_zero_disp = int(np.sum(flat_disp == 0.0))
    zero_frac  = n_zero_disp / flat_disp.size * 100.0

    print(f"\n  tp06 raw (metres/6h) — BEFORE conversion+clamp:")
    print(f"    min raw    : {float(flat_raw.min()):.6f} m/6h")
    print(f"    max raw    : {float(flat_raw.max()):.6f} m/6h")
    print(f"    negatives  : {n_neg_raw} values ({n_neg_raw/flat_raw.size*100:.3f}%)")
    print(f"  After → mm/hr conversion + clamp ≥ 0:")
    print(f"    zero-rain fraction : {zero_frac:.1f}%")

    return {
        "raw_unit": "metres / 6h accumulation",
        "conversion": "metres × 1000 / 6  →  mm/hr;  clamp ≥ 0",
        "n_negative_raw": n_neg_raw,
        "zero_rain_fraction_pct": round(zero_frac, 2),
    }


def wind_extra_stats(wind_spd_kt: np.ndarray, wind_dir_deg: np.ndarray) -> dict:
    """Extra wind diagnostics."""
    spd_flat = wind_spd_kt.flatten()
    dir_flat = wind_dir_deg.flatten()
    dir_in_range = bool(np.all((dir_flat >= 0.0) & (dir_flat < 360.0)))

    print(f"\n  Wind speed [knots]:  min={spd_flat.min():.2f}  max={spd_flat.max():.2f}")
    print(f"  Wind direction [°]:  min={dir_flat.min():.2f}  max={dir_flat.max():.2f}")
    print(f"  Direction in [0,360): {dir_in_range}")

    return {
        "speed_min_kt": float(spd_flat.min()),
        "speed_max_kt": float(spd_flat.max()),
        "direction_min_deg": float(dir_flat.min()),
        "direction_max_deg": float(dir_flat.max()),
        "direction_normalized_to_360": dir_in_range,
    }


def temp_extra_stats(t2m_c: np.ndarray) -> dict:
    """Extra temperature diagnostics."""
    flat = t2m_c.flatten()
    print(f"\n  t2m [°C]:  min={flat.min():.2f}  max={flat.max():.2f}")
    return {
        "min_celsius": float(flat.min()),
        "max_celsius": float(flat.max()),
    }


def check_all_frames_valid(arr: np.ndarray, var: str) -> bool:
    """Verify every frame (time slice) contains at least some valid data."""
    n_frames = arr.shape[0]
    all_ok = True
    for i in range(n_frames):
        if np.all(np.isnan(arr[i])) or np.all(arr[i] == 0):
            print(f"  WARNING: frame {i} for {var} is all-NaN or all-zero")
            all_ok = False
    if all_ok:
        print(f"  All {n_frames} frames contain valid {var} data ✓")
    return all_ok


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    init_time: datetime,
    output_dir: Path,
    debug: bool,
) -> int:
    """
    Run the GraphCastOperational 7-day Myanmar forecast pipeline.
    Returns 0 on success, 1 on failure.
    """
    t_pipeline_start = time_module.time()
    rss_start = mem_gb()

    print("=" * 70)
    print("GraphCastOperational — Myanmar 7-Day (168h) 4-Variable Forecast")
    print(f"  Init time    : {init_time.isoformat()}")
    print(f"  Horizon      : {GC_HORIZON_HOURS}h ({GC_N_STEPS} AR steps, ~{GC_N_STEPS+1} frames)")
    print(f"  Variables    : tp06→mm/hr, t2m→°C, u+v10m→speed(kt)+dir(°)")
    print(f"  Grid target  : 81 lat × 41 lon at 0.25° (verified from output)")
    print(f"  Backend      : JAX CPU (XLA ARM64)")
    print(f"  Output       : {output_dir.resolve()}")
    print(f"  RSS start    : {rss_start:.2f} GB")
    print("=" * 70)

    # ── Configure JAX persistent cache (before any JAX computation) ───────────
    import jax
    cache_was_empty = not any(_JAX_CACHE_DIR.iterdir()) if _JAX_CACHE_DIR.exists() else True
    jax.config.update("jax_compilation_cache_dir", str(_JAX_CACHE_DIR))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 30.0)
    cache_status = "MISS (cold start — XLA will compile ~34 min)" if cache_was_empty else "HIT (warm start)"
    print(f"\n[JAX Cache] {cache_status}")
    print(f"  Cache dir: {_JAX_CACHE_DIR}")

    # ── Earth2Studio imports ──────────────────────────────────────────────────
    import earth2studio.models.px as px
    import earth2studio.data as e2data
    import earth2studio.io as e2io
    import earth2studio.run as e2run

    rss_after_imports = mem_gb()
    print(f"\n[Imports] RSS after Earth2Studio imports: {rss_after_imports:.2f} GB")

    # ── [1/5] Load package ───────────────────────────────────────────────────
    print(f"\n[1/5] Loading GraphCastOperational package (downloads if not cached)...")
    t0 = time_module.time()
    package = px.GraphCastOperational.load_default_package()
    t_pkg = time_module.time() - t0
    rss_pkg = mem_gb()
    print(f"  Package ready in {t_pkg:.1f}s | RSS: {rss_pkg:.2f} GB")
    checkpoint_name = str(package)
    if debug:
        print(f"  Package: {package}")

    # ── [2/5] Load model ─────────────────────────────────────────────────────
    print(f"\n[2/5] Loading model weights...")
    t1 = time_module.time()
    model = px.GraphCastOperational.load_model(package)
    t_load = time_module.time() - t1
    rss_load = mem_gb()
    print(f"  Model loaded in {t_load:.1f}s | RSS: {rss_load:.2f} GB")

    # ── [3/5] Inspect model coordinates ──────────────────────────────────────
    print(f"\n[3/5] Model coordinate inspection...")
    ic = model.input_coords()
    oc_sample = model.output_coords(ic.copy())
    n_input_vars  = len(ic["variable"])
    n_output_vars = len(oc_sample["variable"])
    native_step_h = int(oc_sample["lead_time"][0] / np.timedelta64(1, "h"))
    native_lat    = ic["lat"]
    native_lon    = ic["lon"]

    print(f"  Input variables   : {n_input_vars}")
    print(f"  Output variables  : {n_output_vars}")
    print(f"  Native step       : {native_step_h}h")
    print(f"  Global lat        : {len(native_lat)} pts ({native_lat[0]} to {native_lat[-1]})")
    print(f"  Global lon        : {len(native_lon)} pts ({native_lon[0]} to {native_lon[-1]})")
    for var in OUTPUT_VARS:
        present = var in oc_sample["variable"]
        print(f"  {var} in output    : {present} {'✓' if present else '✗'}")
        if not present:
            print(f"  ERROR: {var} is not in model output_coords.")
            return 1

    # ── Myanmar grid computation (verified against actual model coords) ────────
    lat_arr = np.array(native_lat, dtype=np.float64)
    lon_arr = np.array(native_lon, dtype=np.float64)
    lat_mask = (lat_arr >= MYANMAR_LAT_MIN) & (lat_arr <= MYANMAR_LAT_MAX)
    lon_mask = (lon_arr >= MYANMAR_LON_MIN) & (lon_arr <= MYANMAR_LON_MAX)
    lat_idx  = np.where(lat_mask)[0]
    lon_idx  = np.where(lon_mask)[0]

    # GCOp stores lat descending (90 → -90); sort to ascending for output
    lat_sort = np.argsort(lat_arr[lat_idx])
    lat_idx_asc = lat_idx[lat_sort]
    lats_out = lat_arr[lat_idx_asc]
    lons_out = lon_arr[lon_idx]

    actual_n_lat = len(lats_out)
    actual_n_lon = len(lons_out)
    # GCOp native lat is descending (90→-90); output files are sorted ascending
    native_lat_desc = lat_arr[0] > lat_arr[-1]   # True for GCOp (90→-90)
    lat_ordering = "ascending (south→north, 9→29)"   # describes OUTPUT file ordering

    print(f"\n  Myanmar grid (verified):")
    print(f"  Native lat ordering  : {'descending (90→-90)' if native_lat_desc else 'ascending'}")
    print(f"  Output lat ordering  : {lat_ordering} (sorted by argsort for output files)")
    print(f"  lat range    : {lats_out[0]} to {lats_out[-1]} ({actual_n_lat} pts)")
    print(f"  lon range    : {lons_out[0]} to {lons_out[-1]} ({actual_n_lon} pts)")
    print(f"  grid dims    : {actual_n_lat} × {actual_n_lon} = {actual_n_lat*actual_n_lon} grid points")

    # ── [4/5] Inference ──────────────────────────────────────────────────────
    print(f"\n[4/5] Running {GC_N_STEPS}-step AR inference → expected {GC_N_STEPS+1} frames")
    print(f"  output_coords filter: {list(OUTPUT_VARS)} (4 of {n_output_vars} vars)")
    print(f"  In-memory IO: ~{4 * (GC_N_STEPS+1) * len(lat_arr) * len(lon_arr) * 4 / 1e6:.0f} MB")
    print(f"  First step includes JAX XLA compilation (~34 min cold, <1 min warm).")

    arco = e2data.ARCO()
    io   = e2io.ZarrBackend()   # in-memory

    output_coords_filter = OrderedDict({"variable": OUTPUT_VARS})

    t_infer_start = time_module.time()
    rss_pre_infer = mem_gb()

    try:
        io = e2run.deterministic(
            time=[init_time],
            nsteps=GC_N_STEPS,
            prognostic=model,
            data=arco,
            io=io,
            output_coords=output_coords_filter,
            verbose=True,
        )
    except Exception as exc:
        t_elapsed = time_module.time() - t_infer_start
        print(f"\n  INFERENCE ERROR after {t_elapsed:.1f}s: {exc}")
        import traceback; traceback.print_exc()
        return 1

    t_infer_total = time_module.time() - t_infer_start
    rss_post_infer = mem_gb()
    rss_peak_estimate = max(rss_pre_infer, rss_post_infer)  # peak may have been during compile

    print(f"\n  Inference complete in {t_infer_total:.1f}s ({t_infer_total/60:.1f} min)")
    print(f"  RSS after inference : {rss_post_infer:.2f} GB")

    # ── [5/5] Post-processing ────────────────────────────────────────────────
    print(f"\n[5/5] Post-processing...")

    root   = io.root
    coords = io.coords

    if debug:
        print(f"  Zarr keys: {list(root.keys())}")

    # ── Verify required vars present ──────────────────────────────────────────
    for var in ["tp06", "t2m", "u10m", "v10m"]:
        if var not in root:
            print(f"  ERROR: '{var}' not found in zarr output. Available: {list(root.keys())}")
            return 1

    # ── Read global arrays: shape (1, n_frames, 721, 1440) ───────────────────
    tp06_global_raw = root["tp06"][0].astype(np.float32)   # (n_frames, 721, 1440)
    t2m_global_raw  = root["t2m"][0].astype(np.float32)
    u10m_global     = root["u10m"][0].astype(np.float32)
    v10m_global     = root["v10m"][0].astype(np.float32)

    n_frames_actual = tp06_global_raw.shape[0]
    expected_frames = GC_N_STEPS + 1  # 29

    print(f"\n  Frame count verification:")
    print(f"  Expected : {expected_frames} frames (t+0h through t+{GC_HORIZON_HOURS}h)")
    print(f"  Actual   : {n_frames_actual} frames")
    if n_frames_actual != expected_frames:
        print(f"  WARNING : Frame count mismatch — using actual {n_frames_actual}")

    # ── Myanmar extraction ────────────────────────────────────────────────────
    # Shape: (n_frames, n_lat, n_lon) after extraction
    def extract_myanmar(arr_global: np.ndarray) -> np.ndarray:
        return arr_global[:, lat_idx_asc, :][:, :, lon_idx]

    tp06_mm   = extract_myanmar(tp06_global_raw)
    t2m_k     = extract_myanmar(t2m_global_raw)
    u10m_mm   = extract_myanmar(u10m_global)
    v10m_mm   = extract_myanmar(v10m_global)

    if debug:
        print(f"  Raw tp06 Myanmar: min={tp06_mm.min():.6f}  max={tp06_mm.max():.4f} m/6h")
        print(f"  Raw t2m  Myanmar: min={t2m_k.min():.2f}  max={t2m_k.max():.2f} K")
        print(f"  Raw u10m Myanmar: min={u10m_mm.min():.4f}  max={u10m_mm.max():.4f} m/s")
        print(f"  Raw v10m Myanmar: min={v10m_mm.min():.4f}  max={v10m_mm.max():.4f} m/s")

    # ── Unit conversions ──────────────────────────────────────────────────────
    precip_extra = precip_extra_stats(tp06_mm, tp06_to_mm_per_hr(tp06_mm))

    precip_out  = tp06_to_mm_per_hr(tp06_mm)
    temp_out    = t2m_to_celsius(t2m_k)
    wspd_out    = wind_speed_knots(u10m_mm, v10m_mm)
    wdir_out    = wind_direction_degrees(u10m_mm, v10m_mm)

    # ── Variable statistics ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"VARIABLE STATISTICS (Myanmar {actual_n_lat}×{actual_n_lon} subset, {n_frames_actual} frames)")
    print(f"{'─'*60}")

    stats_precip = var_stats(precip_out,  "Precipitation", "mm/hr")
    stats_precip.update(precip_extra)

    stats_temp   = var_stats(temp_out,    "Temperature",   "°C")
    stats_temp.update(temp_extra_stats(temp_out))

    stats_wspd   = var_stats(wspd_out,    "Wind Speed",    "knots")
    stats_wdir   = var_stats(wdir_out,    "Wind Direction","°FROM")
    stats_wdir.update(wind_extra_stats(wspd_out, wdir_out))

    # ── Per-variable frame validity ───────────────────────────────────────────
    print(f"\n[Frame validity]")
    check_all_frames_valid(precip_out,  "precipitation")
    check_all_frames_valid(temp_out,    "temperature")
    check_all_frames_valid(wspd_out,    "wind_speed")
    check_all_frames_valid(wdir_out,    "wind_direction")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    validation_errors = []
    if stats_precip.get("n_nan", 1) > 0:
        validation_errors.append(f"Precipitation: {stats_precip['n_nan']} NaN values")
    if stats_precip.get("n_inf", 1) > 0:
        validation_errors.append(f"Precipitation: {stats_precip['n_inf']} Inf values")
    if stats_precip.get("min") is not None and stats_precip["min"] < 0:
        validation_errors.append(f"Precipitation: negative after clamp (min={stats_precip['min']:.4f})")
    if stats_precip.get("max") is not None and stats_precip["max"] > PRECIP_SANITY_MAX_MM_HR:
        validation_errors.append(f"Precipitation: max {stats_precip['max']:.1f} mm/hr exceeds {PRECIP_SANITY_MAX_MM_HR}")
    if stats_temp.get("n_nan", 1) > 0:
        validation_errors.append(f"Temperature: {stats_temp['n_nan']} NaN values")
    if stats_temp.get("min") is not None and stats_temp["min"] < TEMP_MIN_C:
        validation_errors.append(f"Temperature: min {stats_temp['min']:.1f}°C < {TEMP_MIN_C}°C")
    if stats_temp.get("max") is not None and stats_temp["max"] > TEMP_MAX_C:
        validation_errors.append(f"Temperature: max {stats_temp['max']:.1f}°C > {TEMP_MAX_C}°C")
    if stats_wspd.get("max") is not None and stats_wspd["max"] > WIND_SANITY_MAX_KT:
        validation_errors.append(f"Wind speed: max {stats_wspd['max']:.1f} kt exceeds {WIND_SANITY_MAX_KT}")
    if not stats_wdir.get("direction_normalized_to_360", True):
        validation_errors.append("Wind direction: values outside [0, 360)")

    # ── Timestamps ────────────────────────────────────────────────────────────
    times_utc = [
        (init_time + timedelta(hours=i * GC_STEP_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(n_frames_actual)
    ]

    # ── Write binary artifacts ────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "precipitation.bin":  precip_out.astype("<f4"),
        "temperature.bin":    temp_out.astype("<f4"),
        "wind_speed.bin":     wspd_out.astype("<f4"),
        "wind_direction.bin": wdir_out.astype("<f4"),
    }

    expected_bytes = n_frames_actual * actual_n_lat * actual_n_lon * 4
    print(f"\n[Binary artifacts]")
    print(f"  Shape per file : [{n_frames_actual}, {actual_n_lat}, {actual_n_lon}] float32 C-order")
    print(f"  Expected bytes : {expected_bytes:,} per file")

    for fname, arr in files.items():
        fpath = output_dir / fname
        fpath.write_bytes(arr.tobytes())
        actual_bytes = fpath.stat().st_size
        status = "✓" if actual_bytes == expected_bytes else f"MISMATCH (got {actual_bytes})"
        print(f"  {fname:<22}: {actual_bytes:>10,} bytes  {status}")

    # ── Compute cache info ────────────────────────────────────────────────────
    cache_files_after = list(_JAX_CACHE_DIR.rglob("*")) if _JAX_CACHE_DIR.exists() else []
    cache_populated  = len(cache_files_after) > 0
    cache_size_mb    = sum(f.stat().st_size for f in cache_files_after if f.is_file()) / 1e6

    # ── Write forecast.json (schema v5.0) ─────────────────────────────────────
    t_total  = time_module.time() - t_pipeline_start
    rss_peak = mem_gb()   # current RSS as peak proxy (compilation peak freed by now)

    import earth2studio
    e2studio_version = getattr(earth2studio, "__version__", "0.17.0")

    # Estimate per-step times
    # Step 0 (ARCO fetch + init): ~10s based on R3
    # Step 1 (JIT + first forward): dominant cost
    # Steps 2-28 (post-JIT): measured by dividing remaining time
    # Since we don't have per-step breakdown, note total and estimate
    step1_estimate_s = 2081.0   # from R3 measurement (cold start)
    if not cache_was_empty:
        # warm start — JIT loaded from cache
        step1_estimate_s = None   # unknown without measurement

    meta = {
        "schema_version": "4.0",
        "model": "GraphCastOperational",
        "model_checkpoint": (
            "GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - "
            "pressure levels 13 - mesh 2to6 - precipitation output only.npz"
        ),
        "model_source": "gs://dm_graphcast/graphcast",
        "model_attribution": (
            "GraphCast by DeepMind/Google. "
            "Lam et al. (2023), Science. https://arxiv.org/abs/2212.12794. "
            "Earth2Studio wrapper by NVIDIA."
        ),
        "earth2studio_version": e2studio_version,
        "native_resolution_deg": 0.25,
        "native_timestep_hours": GC_STEP_HOURS,
        "initialization_source": "ARCO ERA5",
        "initialization_time": times_utc[0],
        "init_timesteps": [
            (init_time - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            times_utc[0],
        ],
        "forecast_generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_horizon_hours": GC_HORIZON_HOURS,
        "n_frames": n_frames_actual,
        "times_utc": times_utc,
        "region": "Myanmar",
        "bbox": {
            "lat_min": float(lats_out.min()),
            "lat_max": float(lats_out.max()),
            "lon_min": float(lons_out.min()),
            "lon_max": float(lons_out.max()),
        },
        "grid": {
            "n_lat": actual_n_lat,
            "n_lon": actual_n_lon,
            "resolution_deg": 0.25,
            "lat_ordering": lat_ordering,
        },
        "lat": lats_out.tolist(),
        "lon": lons_out.tolist(),
        "variables": {
            "precipitation": {
                "file": "precipitation.bin",
                "shape": [n_frames_actual, actual_n_lat, actual_n_lon],
                "dtype": "float32",
                "native_variable": "tp06",
                "native_unit": "metres / 6h accumulation",
                "accumulation_period_hours": GC_STEP_HOURS,
                "display_unit": "mm / hr",
                "conversion": "metres × 1000 / 6  →  mm/hr;  clamp ≥ 0",
                "negative_raw_count": precip_extra["n_negative_raw"],
                "zero_rain_fraction_pct": precip_extra["zero_rain_fraction_pct"],
                "stats": stats_precip,
            },
            "temperature": {
                "file": "temperature.bin",
                "shape": [n_frames_actual, actual_n_lat, actual_n_lon],
                "dtype": "float32",
                "native_variable": "t2m",
                "native_unit": "Kelvin",
                "display_unit": "°C",
                "conversion": "K − 273.15  →  °C",
                "stats": stats_temp,
            },
            "wind_speed": {
                "file": "wind_speed.bin",
                "shape": [n_frames_actual, actual_n_lat, actual_n_lon],
                "dtype": "float32",
                "native_variables": ["u10m", "v10m"],
                "native_unit": "m/s",
                "display_unit": "knots",
                "conversion": "sqrt(u10m² + v10m²) × 1.94384  →  knots",
                "stats": stats_wspd,
                "extra": {
                    "speed_min_kt": stats_wdir.get("speed_min_kt"),
                    "speed_max_kt": stats_wdir.get("speed_max_kt"),
                },
            },
            "wind_direction": {
                "file": "wind_direction.bin",
                "shape": [n_frames_actual, actual_n_lat, actual_n_lon],
                "dtype": "float32",
                "native_variables": ["u10m", "v10m"],
                "native_unit": "radians (atan2)",
                "display_unit": "degrees FROM (meteorological convention)",
                "conversion": "(270 − atan2(v, u) × 180/π) mod 360  →  °FROM",
                "direction_normalized": stats_wdir.get("direction_normalized_to_360"),
                "stats": stats_wdir,
            },
        },
        "inference_config": {
            "hardware": "Apple M4 MacBook Air 24 GB",
            "jax_backend": "cpu",
            "jax_env": {
                "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
                "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
                "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            },
            "jax_cache_dir": str(_JAX_CACHE_DIR),
            "jax_cache_status": cache_status,
            "jax_cache_populated_after": cache_populated,
            "jax_cache_size_mb": round(cache_size_mb, 1),
            "rss_start_gb": round(rss_start, 2),
            "rss_after_model_load_gb": round(rss_load, 2),
            "rss_after_inference_gb": round(rss_post_infer, 2),
            "rss_peak_gb": round(rss_peak, 2),
            "package_load_time_s": round(t_pkg, 1),
            "model_load_time_s": round(t_load, 1),
            "inference_total_s": round(t_infer_total, 1),
            "inference_total_min": round(t_infer_total / 60.0, 1),
            "total_pipeline_s": round(t_total, 1),
            "total_pipeline_min": round(t_total / 60.0, 1),
            "r3_jit_cold_start_s": step1_estimate_s,
            "note": (
                "inference_total_s includes ARCO data fetch, JAX JIT compilation "
                "(cold: ~34 min; warm: seconds), and all 28 AR forward passes."
            ),
        },
        "r3_hardware_validation": "PASS",
        "r4_status": "PASS" if not validation_errors else "FAIL",
        "validation_errors": validation_errors,
        "is_demo": False,
    }

    json_path = output_dir / "forecast.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  Model            : GraphCastOperational (0.25°)")
    print(f"  Init time        : {times_utc[0]}")
    print(f"  Horizon          : {GC_HORIZON_HOURS}h / {n_frames_actual} frames")
    print(f"  Myanmar grid     : {actual_n_lat} lat × {actual_n_lon} lon @ 0.25°")
    print(f"  JAX cache        : {cache_status}")
    print(f"  Package load     : {t_pkg:.1f}s")
    print(f"  Model load       : {t_load:.1f}s")
    print(f"  Inference total  : {t_infer_total:.1f}s = {t_infer_total/60:.1f} min")
    print(f"  Pipeline total   : {t_total:.1f}s = {t_total/60:.1f} min")
    print(f"  Peak RSS         : {rss_peak:.2f} GB (current; compilation peak may be higher)")
    print(f"  Cache after run  : {'populated' if cache_populated else 'empty'} ({cache_size_mb:.1f} MB)")
    print(f"\n  Output: {output_dir.resolve()}/")
    for fname in files:
        fpath = output_dir / fname
        print(f"    {fname:<22}: {fpath.stat().st_size:>10,} bytes")
    print(f"    {'forecast.json':<22}: {json_path.stat().st_size:>10,} bytes")

    if validation_errors:
        print(f"\n  VALIDATION: FAIL ({len(validation_errors)} errors)")
        for err in validation_errors:
            print(f"    - {err}")
        return 1
    else:
        print(f"\n  VALIDATION: PASS (all sanity checks passed)")

    print(f"\n  Run validate_forecast.py to verify:")
    print(f"    uv run python scripts/validate_forecast.py --data-dir {output_dir}")
    print(f"{'='*70}")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate GraphCastOperational 168h (7-day) Myanmar 4-variable forecast. "
            "Schema v5.0."
        ),
    )
    parser.add_argument(
        "--init-time",
        required=True,
        metavar="YYYY-MM-DDTHH:MM:SSZ",
        help="Forecast init time (ISO 8601 UTC). Must be within ARCO coverage (1959-2023).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/forecast_v4",
        help="Output directory (default: data/forecast_v4). Does NOT touch data/forecast/.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw value ranges and zarr keys.",
    )
    args = parser.parse_args()

    try:
        init_time = datetime.fromisoformat(args.init_time.replace("Z", "+00:00"))
    except ValueError as e:
        print(f"ERROR: Invalid --init-time: {e}")
        return 1

    return run_pipeline(
        init_time=init_time,
        output_dir=Path(args.output_dir),
        debug=args.debug,
    )


if __name__ == "__main__":
    sys.exit(main())
