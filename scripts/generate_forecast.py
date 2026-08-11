"""
generate_forecast.py — GraphCastSmall Myanmar 48-hour temperature + precipitation forecast.

Local M4 CPU inference. No GPU required.

Model: GraphCastSmall via Earth2Studio 0.17.0
  Resolution : 1.0° global (181 × 360 lat-lon grid)
  Timestep   : 6h per autoregressive step
  Backend    : JAX CPU (XLA ARM64 on Apple M4)
  Checkpoint : gs://dm_graphcast/graphcast/params/
               GraphCast_small - ERA5 1979-2015 - resolution 1.0 -
               pressure levels 13 - mesh 2to5 - precipitation input and output.npz

Initialization (two-timestep requirement):
  GraphCastSmall requires t-6h AND t+0h at initialization.
  Earth2Studio handles this automatically via e2run.deterministic().

Data sources:
  --source arco  (default) — ARCO ERA5 reanalysis, 1959-2023, no credentials
  --source ifs              — IFS HRES open data, near-real-time, no credentials

Variables:
  t2m  : 2m temperature (K from model → °C via -273.15)
  tp06 : 6h accumulated precipitation (physical metres → mm/6h via ×1000, clamp ≥ 0)

Output (schema v3.0):
  data/forecast/
    temperature.bin   — float32 [9 × 21 × 11] (t+0h through t+48h, °C)
    precipitation.bin — float32 [9 × 21 × 11] (t+0h frame = 0.0, mm/6h)
    forecast.json     — complete provenance metadata

Myanmar grid at 1.0°:
  lat : 9°N to 29°N → 21 points (ascending, south-to-north)
  lon : 92°E to 102°E → 11 points

Usage:
    uv run python scripts/generate_forecast.py \\
        --source arco \\
        --init-time 2022-07-01T00:00:00Z \\
        [--output-dir data/forecast/] \\
        [--debug]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path

# JAX CPU — must be set before any JAX import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import psutil

# ── Myanmar spatial constants ─────────────────────────────────────────────────
MYANMAR_LAT_MIN = 9.0
MYANMAR_LAT_MAX = 29.0
MYANMAR_LON_MIN = 92.0
MYANMAR_LON_MAX = 102.0

# ── GraphCastSmall temporal constants ─────────────────────────────────────────
GC_STEP_HOURS = 6
GC_HORIZON_HOURS = 48
GC_N_STEPS = GC_HORIZON_HOURS // GC_STEP_HOURS   # 8 AR steps
GC_N_FRAMES = GC_N_STEPS + 1                      # 9 total (t+0h through t+48h)

GC_N_LAT = 21   # 9°N to 29°N at 1.0°
GC_N_LON = 11   # 92°E to 102°E at 1.0°

DEFAULT_PRECIP_MAX_MM = 500.0    # mm/6h physical upper bound
TEMP_MIN_C = -90.0               # physical lower bound (absolute zero ~ -273°C)
TEMP_MAX_C = 70.0                # generous upper bound; Myanmar tropics ~ 20-42°C


def mem_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


# ── Temperature sanity check ──────────────────────────────────────────────────

def check_temperature_sanity(t2m_c: np.ndarray) -> bool:
    """Print a sanity report for t2m (°C). Returns True on PASS."""
    valid = t2m_c[np.isfinite(t2m_c)]
    n_nan = int(np.sum(np.isnan(t2m_c)))
    n_inf = int(np.sum(np.isinf(t2m_c)))
    vmin = float(valid.min()) if valid.size else float("nan")
    vmax = float(valid.max()) if valid.size else float("nan")

    print()
    print("  TEMPERATURE SANITY CHECK (t2m)")
    print("  " + "-" * 42)
    print(f"  Variable          : t2m (GraphCastSmall native output)")
    print(f"  Unit              : °C  (converted from K via -273.15)")
    print(f"  Shape             : {t2m_c.shape}")
    print(f"  Min               : {vmin:.2f} °C")
    print(f"  Max               : {vmax:.2f} °C")
    print(f"  Mean              : {float(valid.mean()) if valid.size else float('nan'):.2f} °C")

    failures = []
    if n_nan > 0:
        failures.append(f"NaN values: {n_nan}")
    if n_inf > 0:
        failures.append(f"Infinite values: {n_inf}")
    if not np.isnan(vmin) and vmin < TEMP_MIN_C:
        failures.append(f"Min {vmin:.2f}°C < physical lower bound {TEMP_MIN_C}°C")
    if not np.isnan(vmax) and vmax > TEMP_MAX_C:
        failures.append(f"Max {vmax:.2f}°C > upper bound {TEMP_MAX_C}°C")

    if failures:
        print(f"  Status            : FAIL")
        for msg in failures:
            print(f"    FAIL: {msg}")
        print()
        return False

    print(f"  Status            : PASS")
    print()
    return True


# ── Precipitation sanity check ────────────────────────────────────────────────

def check_precipitation_sanity(tp06: np.ndarray, sanity_max_mm: float) -> bool:
    """Print a structured sanity report for tp06 (mm/6h). Returns True on PASS."""
    n_nan = int(np.sum(np.isnan(tp06)))
    n_inf = int(np.sum(np.isinf(tp06)))
    n_neg = int(np.sum(tp06 < 0.0))
    n_zero = int(np.sum(tp06 == 0.0))
    n_total = tp06.size

    valid = tp06[np.isfinite(tp06)]
    vmin    = float(valid.min())     if valid.size else float("nan")
    vmax    = float(valid.max())     if valid.size else float("nan")
    vmedian = float(np.median(valid)) if valid.size else float("nan")
    p95     = float(np.percentile(valid, 95)) if valid.size else float("nan")
    p99     = float(np.percentile(valid, 99)) if valid.size else float("nan")
    zero_frac = n_zero / n_total * 100.0

    print()
    print("  PRECIPITATION SANITY CHECK (tp06)")
    print("  " + "-" * 42)
    print(f"  Variable          : tp06 (GraphCastSmall native output)")
    print(f"  Unit              : mm / 6h accumulation")
    print(f"  Transform applied : metres × 1000 (NO exp/log transform)")
    print(f"  Shape             : {tp06.shape}")
    print(f"  Min               : {vmin:.4f} mm")
    print(f"  Median            : {vmedian:.4f} mm")
    print(f"  P95               : {p95:.4f} mm")
    print(f"  P99               : {p99:.4f} mm")
    print(f"  Max               : {vmax:.4f} mm")
    print(f"  Zero-rain fraction: {zero_frac:.1f}%")

    failures = []
    if n_nan > 0:
        failures.append(f"NaN values: {n_nan}")
    if n_inf > 0:
        failures.append(f"Infinite values: {n_inf}")
    if n_neg > 0:
        failures.append(f"Negative values after clamping: {n_neg} (should never happen)")
    if not np.isnan(vmax) and vmax > sanity_max_mm:
        failures.append(
            f"Max {vmax:.2f} mm/6h exceeds physical sanity threshold {sanity_max_mm:.0f} mm/6h."
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
    debug: bool,
    sanity_max_mm: float,
) -> int:
    """
    Run the GraphCastSmall 48h Myanmar temperature + precipitation forecast.

    Returns 0 on success, 1 on error.
    """
    import torch
    import earth2studio.run as e2run
    from earth2studio.io import ZarrBackend
    from earth2studio.models.px import GraphCastSmall

    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time_module.time()
    mem_start = mem_gb()

    print("=" * 64)
    print("GraphCastSmall Myanmar 48h Forecast — Local M4 CPU")
    print(f"  Init time  : {init_time.isoformat()}")
    print(f"  Source     : {source.upper()}")
    print(f"  Horizon    : {GC_HORIZON_HOURS}h ({GC_N_STEPS} steps × {GC_STEP_HOURS}h)")
    print(f"  Frames     : {GC_N_FRAMES} (t+0h through t+{GC_HORIZON_HOURS}h)")
    print(f"  Variables  : temperature (t2m → °C) + precipitation (tp06 → mm/6h)")
    print(f"  Grid       : {GC_N_LAT} lat × {GC_N_LON} lon at 1.0°")
    print(f"  Backend    : JAX CPU (XLA ARM64)")
    print(f"  Output     : {output_dir.resolve()}")
    print(f"  RSS start  : {mem_start:.2f} GB")
    print("=" * 64)

    # ── [1/4] Data source ─────────────────────────────────────────────────────
    print(f"\n[1/4] Initializing {source.upper()} data source")

    if source == "arco":
        from earth2studio.data import ARCO
        data = ARCO()
        source_label = "ARCO ERA5 reanalysis (Google Cloud, no credentials)"
        source_attribution = (
            "ERA5 via ARCO (Analysis-Ready Cloud-Optimized ERA5). "
            "Hosted on Google Cloud. ERA5 © ECMWF/Copernicus."
        )
        print(f"  Source: {source_label}")
        print(f"  Note: ARCO covers 1959-2023. Init time must be within this range.")
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

    # ── [2/4] Load GraphCastSmall ─────────────────────────────────────────────
    print(f"\n[2/4] Loading GraphCastSmall checkpoint")
    mem_before_load = mem_gb()
    t_load_start = time_module.time()

    try:
        package = GraphCastSmall.load_default_package()
        model = GraphCastSmall.load_model(package)
        load_time = time_module.time() - t_load_start
        mem_after_load = mem_gb()
        print(f"  Loaded in {load_time:.1f}s")
        print(f"  RSS: {mem_after_load:.2f} GB (+{mem_after_load - mem_before_load:.2f} GB)")
        print(f"  Backend: JAX + Haiku, bfloat16 cast applied internally.")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 1

    # ── [3/4] Inference ───────────────────────────────────────────────────────
    print(f"\n[3/4] Running {GC_N_STEPS}-step inference (→ {GC_N_FRAMES} frames)")
    print(f"  First call includes JAX JIT compilation (~10-15s) + ARCO data fetch.")

    io = ZarrBackend()
    t_infer_start = time_module.time()
    mem_pre_infer = mem_gb()

    try:
        with torch.inference_mode():
            io = e2run.deterministic(
                time=[init_time],
                nsteps=GC_N_STEPS,
                prognostic=model,
                data=data,
                io=io,
                device=torch.device("cpu"),
                verbose=True,
            )
        infer_time = time_module.time() - t_infer_start
        mem_post_infer = mem_gb()
        print(f"  Inference complete in {infer_time:.1f}s ({infer_time/60:.1f} min)")
        print(f"  RSS: {mem_post_infer:.2f} GB (+{mem_post_infer - mem_pre_infer:.2f} GB)")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 1

    del model

    # ── [4/4] Post-processing ─────────────────────────────────────────────────
    print(f"\n[4/4] Post-processing zarr output")

    root = io.root
    coords = io.coords

    if debug:
        print(f"  Zarr keys  : {list(root.keys())}")
        print(f"  Coord keys : {list(coords.keys())}")

    # Verify required variables are present
    for var in ("tp06", "t2m"):
        if var not in root:
            print(f"  ERROR: '{var}' not found in zarr output.")
            print(f"  Available keys: {list(root.keys())}")
            return 1

    # Raw arrays — shape is (n_init_times=1, n_lead_times, n_lat=181, n_lon=360)
    tp06_raw = root["tp06"][:]
    t2m_raw  = root["t2m"][:]

    if debug:
        print(f"  tp06 raw shape : {tp06_raw.shape}  dtype: {tp06_raw.dtype}")
        print(f"  t2m  raw shape : {t2m_raw.shape}  dtype: {t2m_raw.dtype}")

    # Drop init-time dimension (index 0)
    tp06_global = tp06_raw[0]   # (n_frames, 181, 360)
    t2m_global  = t2m_raw[0]

    n_lead = tp06_global.shape[0]
    if n_lead != GC_N_FRAMES:
        print(f"  WARNING: Expected {GC_N_FRAMES} lead frames, got {n_lead}.")

    # Get lat/lon coordinate arrays from zarr store
    lat_arr = np.array(coords["lat"], dtype=np.float64)   # (181,)
    lon_arr = np.array(coords["lon"], dtype=np.float64)   # (360,)

    if debug:
        print(f"  lat range: {lat_arr.min():.1f} to {lat_arr.max():.1f}")
        print(f"  lon range: {lon_arr.min():.1f} to {lon_arr.max():.1f}")

    # Myanmar lat/lon index masks
    lat_mask = (lat_arr >= MYANMAR_LAT_MIN) & (lat_arr <= MYANMAR_LAT_MAX)
    lon_mask = (lon_arr >= MYANMAR_LON_MIN) & (lon_arr <= MYANMAR_LON_MAX)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    if len(lat_idx) != GC_N_LAT:
        print(f"  WARNING: Expected {GC_N_LAT} lat pts, got {len(lat_idx)}.")
    if len(lon_idx) != GC_N_LON:
        print(f"  WARNING: Expected {GC_N_LON} lon pts, got {len(lon_idx)}.")

    # Sort lat ascending (south→north) regardless of zarr storage order
    sort_order = np.argsort(lat_arr[lat_idx])
    lat_idx_asc = lat_idx[sort_order]
    lats_out = lat_arr[lat_idx_asc]
    lons_out = lon_arr[lon_idx]

    # Subset to Myanmar: (n_frames, n_lat, n_lon)
    tp06_myanmar = tp06_global[:, lat_idx_asc, :][:, :, lon_idx]
    t2m_myanmar  = t2m_global[:, lat_idx_asc, :][:, :, lon_idx]

    # ── Unit conversions ──────────────────────────────────────────────────────
    # tp06: physical metres → mm/6h; clamp ≥ 0; set t+0h to 0 (init frame)
    tp06_mm = (tp06_myanmar * 1000.0).astype(np.float32)
    tp06_mm = np.maximum(tp06_mm, 0.0)
    tp06_mm[0] = 0.0   # t+0h: no forecast accumulation at init hour

    # t2m: Kelvin → Celsius
    t2m_c = (t2m_myanmar - 273.15).astype(np.float32)

    if debug:
        print(f"  tp06 raw metres: min={tp06_myanmar.min():.6f}  max={tp06_myanmar.max():.4f}")
        print(f"  t2m  raw K     : min={t2m_myanmar.min():.2f}  max={t2m_myanmar.max():.2f}")

    print(f"  t2m  (°C)  : shape={t2m_c.shape}  range=[{t2m_c.min():.2f}, {t2m_c.max():.2f}]")
    print(f"  tp06 (mm/6h): shape={tp06_mm.shape}  range=[{tp06_mm.min():.4f}, {tp06_mm.max():.2f}]")

    # ── Timestamps ────────────────────────────────────────────────────────────
    times_utc = [
        (init_time + timedelta(hours=i * GC_STEP_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(GC_N_FRAMES)
    ]

    # ── Write binary artifacts ────────────────────────────────────────────────
    temp_path   = output_dir / "temperature.bin"
    precip_path = output_dir / "precipitation.bin"

    temp_path.write_bytes(t2m_c.astype("<f4").tobytes())
    precip_path.write_bytes(tp06_mm.astype("<f4").tobytes())

    expected_bytes = GC_N_FRAMES * len(lats_out) * len(lons_out) * 4
    print(f"  temperature.bin   : {temp_path.stat().st_size:,} bytes")
    print(f"  precipitation.bin : {precip_path.stat().st_size:,} bytes")
    print(f"  Expected          : {expected_bytes:,} bytes each ({GC_N_FRAMES}×{len(lats_out)}×{len(lons_out)}×4)")

    if temp_path.stat().st_size != expected_bytes:
        print(f"  WARNING: temperature.bin size mismatch")
    if precip_path.stat().st_size != expected_bytes:
        print(f"  WARNING: precipitation.bin size mismatch")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    temp_ok   = check_temperature_sanity(t2m_c)
    precip_ok = check_precipitation_sanity(tp06_mm, sanity_max_mm)

    if not temp_ok or not precip_ok:
        print("  WARNING: One or more sanity checks FAILED. Data written as-is.")

    # ── Write forecast.json (schema v3.0) ─────────────────────────────────────
    total_time = time_module.time() - t_start
    peak_rss   = mem_gb()

    meta = {
        "schema_version": "3.0",
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
        "n_times": GC_N_FRAMES,
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
            "n_lat": int(len(lats_out)),
            "n_lon": int(len(lons_out)),
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
                        "GraphCastSmall native tp06 (physical metres, no log transform) "
                        "→ metres × 1000 → mm/6h → clamp ≥ 0"
                    ),
                },
                "t0_note": (
                    "t+0h frame (index 0) is set to 0.0 mm/6h. "
                    "Represents the analysis state; no forecast accumulation at init hour."
                ),
                "native_output": True,
                "file": "precipitation.bin",
                "fill_value": None,
            },
            "temperature": {
                "display_name": "Temperature",
                "units": "°C",
                "source_variable": "t2m",
                "temporal_resolution": "6-hourly",
                "temporal_semantics": "2m temperature at the forecast valid time.",
                "transformation_provenance": {
                    "source_variable": "t2m",
                    "source_unit": "K",
                    "conversion": "K - 273.15",
                    "output_unit": "°C",
                    "log_transform_applied": False,
                    "exp_transform_applied": False,
                    "pipeline": (
                        "GraphCastSmall native t2m (Kelvin) → K - 273.15 → °C"
                    ),
                },
                "t0_note": (
                    "t+0h frame (index 0) is the analysis temperature from the data source, "
                    "not a forecast."
                ),
                "native_output": True,
                "file": "temperature.bin",
                "fill_value": None,
            },
        },
        "data_source_attribution": source_attribution,
        "earth2studio_version": ">=0.17.0",
        "inference_config": {
            "device": "Apple M4 CPU",
            "jax_backend": "cpu",
            "jax_env": {
                "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
                "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
                "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
            },
            "rss_start_gb": round(mem_start, 2),
            "rss_after_load_gb": round(mem_after_load, 2),
            "rss_after_infer_gb": round(mem_post_infer, 2),
            "rss_peak_gb": round(peak_rss, 2),
            "model_load_time_seconds": round(load_time),
            "inference_time_seconds": round(infer_time),
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
    print(f"  Model    : GraphCastSmall")
    print(f"  Source   : {source.upper()}")
    print(f"  Device   : M4 CPU (JAX XLA ARM64)")
    print(f"  Frames   : {GC_N_FRAMES} ({', '.join(times_utc)})")
    print(f"  Grid     : {len(lats_out)} lat × {len(lons_out)} lon at 1.0°")
    print(f"  Peak RSS : {peak_rss:.2f} GB")
    print(f"  Output   : {output_dir.resolve()}")
    print(f"")
    print(f"Validate with:")
    print(f"  uv run python scripts/validate_forecast.py --data-dir {output_dir}")
    print(f"{'='*64}")

    return 0


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate GraphCastSmall 48h Myanmar temperature + precipitation forecast",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Historical date via ARCO (default)
  uv run python scripts/generate_forecast.py --source arco --init-time 2022-07-01T00:00:00Z

  # Near-real-time via IFS
  uv run python scripts/generate_forecast.py --source ifs
""",
    )
    parser.add_argument(
        "--init-time",
        default=None,
        metavar="YYYY-MM-DDTHH:MM:SSZ",
        help=(
            "Forecast initialization time (ISO 8601 UTC). "
            "For --source arco: must be within 1959-2023. "
            "For --source ifs: defaults to latest available run."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["arco", "ifs"],
        default="arco",
        help="Initialization data source (default: arco)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/forecast",
        help="Output directory for artifacts (default: data/forecast)",
    )
    parser.add_argument(
        "--precip-max-mm",
        type=float,
        default=DEFAULT_PRECIP_MAX_MM,
        metavar="MM",
        help=f"Sanity threshold for tp06 max (mm/6h). Default: {DEFAULT_PRECIP_MAX_MM}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print verbose debug output including raw value ranges.",
    )

    args = parser.parse_args()

    if args.init_time:
        try:
            init_time = datetime.fromisoformat(args.init_time.replace("Z", "+00:00"))
        except ValueError as e:
            print(f"ERROR: Invalid --init-time: {e}")
            return 1
    else:
        if args.source == "ifs":
            init_time = find_latest_ifs_init()
            print(f"Auto-detected IFS init time: {init_time.isoformat()}")
        else:
            print(
                "ERROR: --init-time required for --source arco. "
                "Example: --init-time 2022-07-01T00:00:00Z"
            )
            return 1

    if args.source == "arco":
        arco_cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
        if init_time >= arco_cutoff:
            print(
                f"WARNING: init_time {init_time.date()} may be outside ARCO coverage "
                f"(ERA5 ends ~2023). Use --source ifs for recent dates."
            )

    return run_pipeline(
        init_time=init_time,
        source=args.source,
        output_dir=Path(args.output_dir),
        debug=args.debug,
        sanity_max_mm=args.precip_max_mm,
    )


if __name__ == "__main__":
    sys.exit(main())
