"""
generate_forecast.py — Full Aurora1p5 + IFS forecast pipeline.

Requires:
  - earth2studio >= 0.17.0 with --extra data --extra aurora installed
  - IFS open data accessible (ECMWF; Google Cloud mirror default, no credentials)
  - GPU: CUDA with >= 20 GB VRAM recommended (A100 40/80 GB ideal)

Usage:
    uv run python scripts/generate_forecast.py \
        [--init-time YYYY-MM-DDTHH:MM:SSZ]   # defaults to latest IFS 00Z
        [--output-dir data/forecast/] \
        [--n-hours 168] \
        [--ifs-source google|aws|ecmwf] \
        [--debug]

IFS variables patched (not available in IFS open data):
    sic  — zero globally (no sea ice; valid for tropical Myanmar init)
    lcc  — zero (IFS open data provides tcc but not individual cloud layer fractions)
    mcc  — zero (same)
    hcc  — zero (same)

tp1h post-processing:
    Aurora raw output -> Earth2Studio inverse log transform -> physical metres in zarr -> x1000 -> mm / 1h

    Earth2Studio's aurora1p5.py applies aurora_log_untransform() before yielding,
    converting Aurora's internal scaled_tp_1h (log-space) to physical metres.
    So zarr["tp1h"] stores physical metres directly.
    Physical quantity: zarr_metres * 1000 -> mm / 1-hour accumulation
    Negative values (numerical noise) are clamped to 0.
    A configurable sanity threshold (--precip-max-mm) warns if the max exceeds a physical
    upper bound. The data is never silently modified by this check.

Output:
    data/forecast/forecast.json
    data/forecast/temperature.bin  — float32 [n_times, n_lat, n_lon] C-order
    data/forecast/precipitation.bin — float32 [n_times, n_lat, n_lon] C-order
"""

from __future__ import annotations

import argparse
import json
import sys
import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

MYANMAR_LAT_MIN, MYANMAR_LAT_MAX = 9.0, 29.0
MYANMAR_LON_MIN, MYANMAR_LON_MAX = 92.0, 102.0

# Variables required by Aurora1p5 that are NOT in IFS open data (patched to zero)
IFS_PATCH_VARS = {
    "sic": "zero",   # Sea ice concentration: 0 globally (no sea ice, tropical init)
    "lcc": "zero",   # Low cloud cover: 0 (IFS open data only has tcc)
    "mcc": "zero",   # Medium cloud cover: 0 (same)
    "hcc": "zero",   # High cloud cover: 0 (same)
}


def make_ifs_with_patches(source: str = "google"):
    """
    Return a DataSource-compatible callable wrapping IFS with patches for
    the 4 variables not available in IFS open data.

    Complies with earth2studio DataSource protocol:
      __call__(time, variable) -> xr.DataArray [time, variable, lat, lon]
    """
    from earth2studio.data import IFS as IFSBase

    ifs_base = IFSBase(source=source)

    class IFSWithPatches:
        def __call__(self, time, variable):
            var_list = [variable] if isinstance(variable, str) else list(variable)
            patch_vars = [v for v in var_list if v in IFS_PATCH_VARS]
            ifs_vars = [v for v in var_list if v not in IFS_PATCH_VARS]

            if ifs_vars:
                da = ifs_base(time, ifs_vars)
            else:
                # Only patch vars requested — fetch t2m to get grid shape
                ref = ifs_base(time, ["t2m"])
                da = ref.isel(variable=slice(0, 0))

            if patch_vars:
                lat = da.coords["lat"].values
                lon = da.coords["lon"].values
                zeros = np.zeros(
                    (len(da.coords["time"]), len(patch_vars), len(lat), len(lon)),
                    dtype=np.float32,
                )
                patch_da = xr.DataArray(
                    zeros,
                    dims=["time", "variable", "lat", "lon"],
                    coords={
                        "time": da.coords["time"],
                        "variable": patch_vars,
                        "lat": lat,
                        "lon": lon,
                    },
                )
                if ifs_vars:
                    da = xr.concat([da, patch_da], dim="variable")
                    da = da.sel(variable=var_list)
                else:
                    da = patch_da

            return da

    return IFSWithPatches()


def check_precipitation_sanity(tp1h: np.ndarray, sanity_max_mm: float) -> bool:
    """
    Print a structured sanity report for the precipitation array and return True if PASS.

    Checks:
      - min >= 0 (no negative precipitation)
      - no NaN values
      - no infinite values
      - max <= sanity_max_mm threshold

    The array is never modified. If the threshold is exceeded a WARNING is printed
    and False is returned so the caller can decide how to proceed.
    """
    n_nan = int(np.sum(np.isnan(tp1h)))
    n_inf = int(np.sum(np.isinf(tp1h)))
    n_neg = int(np.sum(tp1h < 0.0))
    n_zero = int(np.sum(tp1h == 0.0))
    n_total = tp1h.size

    tp1h_valid = tp1h[np.isfinite(tp1h)]
    vmin = float(tp1h_valid.min()) if tp1h_valid.size else float("nan")
    vmax = float(tp1h_valid.max()) if tp1h_valid.size else float("nan")
    vmedian = float(np.median(tp1h_valid)) if tp1h_valid.size else float("nan")
    p95 = float(np.percentile(tp1h_valid, 95)) if tp1h_valid.size else float("nan")
    p99 = float(np.percentile(tp1h_valid, 99)) if tp1h_valid.size else float("nan")
    zero_frac = n_zero / n_total * 100.0

    print()
    print("  PRECIPITATION SANITY CHECK")
    print("  " + "-" * 38)
    print(f"  Unit              : mm / 1h accumulation")
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
        failures.append(f"Negative values: {n_neg}")
    if not np.isnan(vmax) and vmax > sanity_max_mm:
        failures.append(
            f"Max {vmax:.2f} mm exceeds physical sanity threshold {sanity_max_mm:.0f} mm "
            f"(check transformation pipeline)"
        )

    if failures:
        print(f"  Status            : FAIL")
        for msg in failures:
            print(f"    WARNING: {msg}")
        print()
        return False

    print(f"  Status            : PASS")
    print()
    return True


def find_latest_ifs_init() -> datetime:
    """Return the most recent IFS analysis available on the open data portal.
    IFS runs at 00Z and 12Z; data is available with ~6h delay.
    """
    now = datetime.now(timezone.utc)
    # Try 00Z and 12Z of today and yesterday in reverse order
    candidates = []
    for delta_days in [0, 1, 2]:
        day = now - timedelta(days=delta_days)
        for hour in [12, 0]:
            candidates.append(day.replace(hour=hour, minute=0, second=0, microsecond=0))

    # Return the most recent that's at least 6h in the past
    for t in candidates:
        if (now - t).total_seconds() >= 6 * 3600:
            return t

    # Fallback: yesterday 00Z
    return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def run_pipeline(
    init_time: datetime,
    n_hours: int,
    output_dir: Path,
    ifs_source: str,
    debug: bool,
    sanity_max_mm: float = 300.0,
) -> int:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time_module.time()

    print("=" * 60)
    print("Aurora1p5 + IFS Forecast Pipeline")
    print(f"Init time  : {init_time.isoformat()}")
    print(f"Horizon    : {n_hours}h ({n_hours // 24} days)")
    print(f"Output dir : {output_dir.resolve()}")
    print(f"IFS source : {ifs_source}")
    print(f"Patched    : {list(IFS_PATCH_VARS.keys())}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1: Device selection
    # ------------------------------------------------------------------ #
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        device_desc = f"CUDA — {gpu_name} ({vram_gb:.0f} GB VRAM)"
    else:
        device = torch.device("cpu")
        device_desc = "CPU (very slow)"
        vram_gb = 0.0

    print(f"\n[1/5] Device: {device_desc}")

    # ------------------------------------------------------------------ #
    # Step 2: IFS data source
    # ------------------------------------------------------------------ #
    print(f"\n[2/5] Initializing IFS data source (source={ifs_source})...")
    ifs = make_ifs_with_patches(ifs_source)
    sic_handling = (
        "sic set to 0.0 globally — IFS open data does not publish sic. "
        "Zero is physically correct for tropical Myanmar initialization. "
        f"Also patched lcc/mcc/hcc to 0.0 (IFS provides tcc but not individual layer fractions)."
    )
    print(f"  Patch variables: {list(IFS_PATCH_VARS.keys())} → zero-filled")

    # Quick connectivity test
    print(f"  Testing IFS connectivity...")
    try:
        test = ifs(init_time, ["t2m"])
        t2m_mean = float(test.sel(variable="t2m").mean()) - 273.15
        print(f"  IFS OK — t2m global mean: {t2m_mean:.1f}°C")
    except Exception as e:
        print(f"  ERROR: IFS connectivity failed: {e}")
        return 1

    # ------------------------------------------------------------------ #
    # Step 3: Load Aurora1p5
    # ------------------------------------------------------------------ #
    import earth2studio.run as e2run
    from earth2studio.io import ZarrBackend
    from earth2studio.models.px import Aurora1p5

    print(f"\n[3/5] Loading Aurora1p5 (1.26B params)...")
    t1 = time_module.time()
    try:
        package = Aurora1p5.load_default_package()
        model = Aurora1p5.load_model(package)
        print(f"  Model loaded in {time_module.time()-t1:.1f}s")
        print(f"  Checkpoint: aurora-0.25-v1.5.ckpt")
    except Exception as e:
        print(f"  ERROR: Aurora1p5 load failed: {e}")
        return 1

    # ------------------------------------------------------------------ #
    # Step 4: Run inference
    # ------------------------------------------------------------------ #
    print(f"\n[4/5] Running {n_hours}h forecast on {device}...")
    print(f"  Note: Aurora1p5 needs IFS at t-6h and t+0h for initialization.")
    print(f"  Produces t+1h through t+{n_hours}h native hourly output (no interpolation).")

    io = ZarrBackend()
    t2 = time_module.time()
    try:
        io = e2run.deterministic(
            time=[init_time],
            nsteps=n_hours,
            prognostic=model,
            data=ifs,
            io=io,
            device=device,
            verbose=True,
        )
    except Exception as e:
        print(f"  ERROR: Inference failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    inference_time = time_module.time() - t2
    print(f"  Inference completed in {inference_time:.0f}s ({inference_time/60:.1f} min)")

    # ------------------------------------------------------------------ #
    # Step 5: Post-process and write artifacts
    # ------------------------------------------------------------------ #
    print(f"\n[5/5] Post-processing and writing artifacts...")
    ds = xr.open_zarr(io.store)

    if debug:
        print(f"  Raw output variables: {list(ds.data_vars)}")
        print(f"  Lead times: {list(ds.coords.get('lead_time', ['N/A']))[:6]}...")

    # --- t2m ---
    t2m_raw = ds["t2m"]  # Units: K, shape [batch, time, lead_time, lat, lon]

    # Subset to Myanmar bbox (lat descending in Aurora output)
    t2m_mm = t2m_raw.sel(
        lat=slice(MYANMAR_LAT_MAX, MYANMAR_LAT_MIN),
        lon=slice(MYANMAR_LON_MIN, MYANMAR_LON_MAX),
    ).squeeze()

    # Ensure lat is ascending (south-to-north for our frontend)
    if t2m_mm.coords["lat"].values[0] > t2m_mm.coords["lat"].values[-1]:
        t2m_mm = t2m_mm.isel(lat=slice(None, None, -1))

    t2m_celsius = (t2m_mm.values - 273.15).astype(np.float32)
    n_lead, n_lat, n_lon = t2m_celsius.shape
    print(f"  t2m: shape={t2m_celsius.shape}, range=[{t2m_celsius.min():.1f}, {t2m_celsius.max():.1f}] °C")

    # --- tp1h ---
    if "tp1h" not in ds:
        print("  ERROR: tp1h not in Aurora1p5 output. Check earth2studio version.")
        return 1

    tp1h_raw = ds["tp1h"]
    tp1h_mm_ds = tp1h_raw.sel(
        lat=slice(MYANMAR_LAT_MAX, MYANMAR_LAT_MIN),
        lon=slice(MYANMAR_LON_MIN, MYANMAR_LON_MAX),
    ).squeeze()

    if tp1h_mm_ds.coords["lat"].values[0] > tp1h_mm_ds.coords["lat"].values[-1]:
        tp1h_mm_ds = tp1h_mm_ds.isel(lat=slice(None, None, -1))

    # Earth2Studio aurora1p5.py applies aurora_log_untransform() internally before
    # writing to zarr, so zarr["tp1h"] is already in physical metres (not log-space).
    # aurora_log_untransform: 0.001 * (exp(scaled_tp_1h) - 1) → metres
    # Correct inverse: just multiply by 1000 to convert m → mm.
    tp1h_m = tp1h_mm_ds.values.astype(np.float32)
    tp1h_mm = np.maximum(tp1h_m * 1000.0, 0.0).astype(np.float32)
    print(f"  tp1h: range=[{tp1h_mm.min():.4f}, {tp1h_mm.max():.2f}] mm / 1-hour accumulation")
    print(f"  tp1h unit: mm per 1-hour accumulation period (not instantaneous rate)")

    # Build lat/lon coordinate arrays from the actual output
    lats_out = t2m_mm.coords["lat"].values.astype(np.float32)
    lons_out = t2m_mm.coords["lon"].values.astype(np.float32)
    n_lat_out, n_lon_out = len(lats_out), len(lons_out)

    # Prepend t+0 (analysis/init state)
    # The Aurora output starts at t+1h. We prepend t+0 from IFS.
    print("  Fetching t+0 IFS temperature for prepend...")
    try:
        init_da = ifs(init_time, ["t2m"])
        init_t2m = init_da.sel(
            variable="t2m",
            lat=slice(MYANMAR_LAT_MAX, MYANMAR_LAT_MIN),
            lon=slice(MYANMAR_LON_MIN, MYANMAR_LON_MAX),
        ).squeeze()
        if init_t2m.coords["lat"].values[0] > init_t2m.coords["lat"].values[-1]:
            init_t2m = init_t2m.isel(lat=slice(None, None, -1))
        # Interpolate to same lat/lon as forecast output
        init_t2m_interp = init_t2m.interp(lat=lats_out, lon=lons_out)
        t0_celsius = (init_t2m_interp.values - 273.15).astype(np.float32)
    except Exception as e:
        print(f"  Warning: could not fetch IFS t+0: {e}. Using first forecast frame.")
        t0_celsius = t2m_celsius[0]

    # Full arrays: [t+0, t+1h, ..., t+n_hours]  shape [n_hours+1, n_lat, n_lon]
    t2m_full = np.concatenate([t0_celsius[np.newaxis], t2m_celsius], axis=0)
    tp1h_full = np.concatenate([np.zeros((1, n_lat_out, n_lon_out), dtype=np.float32), tp1h_mm], axis=0)
    n_times = t2m_full.shape[0]  # should be n_hours + 1 = 169

    print(f"  Final arrays: [{n_times}, {n_lat_out}, {n_lon_out}] = [times, lat, lon]")
    print(f"  t2m full: [{t2m_full.min():.1f}, {t2m_full.max():.1f}] °C")
    print(f"  tp1h full: [{tp1h_full.min():.4f}, {tp1h_full.max():.4f}] mm / 1h accumulation")

    # Write binary artifacts
    temp_path = output_dir / "temperature.bin"
    precip_path = output_dir / "precipitation.bin"
    temp_path.write_bytes(t2m_full.astype("<f4").tobytes())
    precip_path.write_bytes(tp1h_full.astype("<f4").tobytes())
    print(f"  temperature.bin: {temp_path.stat().st_size / 1024:.0f} KB")
    print(f"  precipitation.bin: {precip_path.stat().st_size / 1024:.0f} KB")

    # Timestamps
    times_utc = [
        (init_time + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for h in range(n_times)
    ]

    # Temperature NaN check
    t2m_nan = int(np.sum(np.isnan(t2m_full)))
    if t2m_nan > 0:
        print(f"  WARNING: NaN values in t2m: {t2m_nan}")
    else:
        print(f"  t2m NaN check: PASS")

    # Precipitation sanity check
    precip_ok = check_precipitation_sanity(tp1h_full, sanity_max_mm)
    if not precip_ok:
        print(
            "  WARNING: Precipitation sanity check FAILED. "
            "The data has been written as-is. "
            "Review the transformation pipeline before publishing."
        )

    # Build metadata
    total_time = time_module.time() - t_start
    meta = {
        "schema_version": "1.0",
        "model": "Aurora1p5",
        "model_version": "1.5",
        "model_checkpoint": "aurora-0.25-v1.5.ckpt",
        "model_source": "hf://microsoft/aurora@c171214768997594e1a3fc6b8d9bbb489e9d21ab",
        "initialization_source": "IFS",
        "initialization_time": times_utc[0],
        "forecast_generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sic_handling": sic_handling,
        "forecast_horizon_hours": n_hours,
        "n_times": n_times,
        "spatial_resolution_deg": 0.25,
        "display_resolution_deg": 0.05,
        "spatial_interpolation": "bilinear",
        "boundary_mask": "Myanmar administrative boundary (GeoJSON scanline rasterization)",
        "region": "Myanmar",
        "bbox": {
            "lat_min": float(lats_out.min()),
            "lat_max": float(lats_out.max()),
            "lon_min": float(lons_out.min()),
            "lon_max": float(lons_out.max()),
        },
        "grid": {"n_lat": n_lat_out, "n_lon": n_lon_out},
        "lat": lats_out.tolist(),
        "lon": lons_out.tolist(),
        "times_utc": times_utc,
        "variables": {
            "temperature_2m": {
                "display_name": "2m Temperature",
                "units": "degC",
                "source_variable": "t2m",
                "temporal_resolution": "hourly",
                "temporal_semantics": "Point forecast at each hour from Aurora1p5 native hourly rollout",
                "transformation": "K - 273.15",
                "file": "temperature.bin",
                "fill_value": -9999.0,
            },
            "precipitation": {
                "display_name": "Precipitation",
                "units": "mm / 1-hour accumulation",
                "source_variable": "tp1h",
                "temporal_resolution": "hourly",
                "temporal_semantics": "Total precipitation accumulated over 1-hour forecast period",
                "temporal_disclosure": (
                    "Precipitation values represent total rainfall accumulated during each "
                    "1-hour forecast period. These are not instantaneous rainfall rates."
                ),
                "transformation_provenance": {
                    "variable": "tp1h",
                    "unit": "mm",
                    "accumulation_period": "1h",
                    "source_unit": "m",
                    "conversion": "metres * 1000",
                    "temporal_semantics": "1-hour accumulation over the associated forecast interval",
                    "pipeline": (
                        "Aurora raw output (scaled_tp_1h, log-space)"
                        " -> Earth2Studio aurora_log_untransform: 0.001*(exp(x)-1) -> physical metres in zarr"
                        " -> metres * 1000 -> mm / 1-hour accumulation"
                    ),
                },
                "native_output": True,
                "file": "precipitation.bin",
                "fill_value": -9999.0,
            },
        },
        "data_source_attribution": (
            f"IFS HRES analysis — ECMWF open data ({ifs_source} mirror). "
            "CC BY 4.0. https://confluence.ecmwf.int/display/DAC/ECMWF+open+data"
        ),
        "model_attribution": "Microsoft Research Aurora v1.5 — https://github.com/microsoft/aurora",
        "earth2studio_version": "0.17.0",
        "inference_config": {
            "device": device_desc,
            "ifs_source": ifs_source,
            "patched_vars": list(IFS_PATCH_VARS.keys()),
            "inference_time_seconds": round(inference_time),
            "total_pipeline_time_seconds": round(total_time),
        },
        "is_demo": False,
    }

    json_path = output_dir / "forecast.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  forecast.json written")

    print(f"\nPipeline complete in {total_time/60:.1f} min")
    print(f"Validate with:")
    print(f"  uv run python scripts/validate_forecast.py --data-dir {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Aurora1p5 + IFS forecast")
    parser.add_argument(
        "--init-time",
        default=None,
        help="Initialization time ISO 8601 (e.g. 2026-08-09T00:00:00Z). "
             "Defaults to latest available IFS analysis.",
    )
    parser.add_argument("--output-dir", default="data/forecast", help="Output directory")
    parser.add_argument(
        "--ifs-source",
        choices=["google", "aws", "ecmwf"],
        default="google",
        help="IFS open data mirror (default: google; AWS may be rate-limited)",
    )
    parser.add_argument("--n-hours", type=int, default=168, help="Forecast hours (default: 168)")
    parser.add_argument(
        "--precip-max-mm",
        type=float,
        default=300.0,
        help=(
            "Physical sanity threshold for precipitation maximum (mm / 1h). "
            "A WARNING is printed if the max exceeds this value; data is not modified. "
            "Global extreme observed values are ~300 mm/h (default: 300)."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Verbose debug output")
    args = parser.parse_args()

    if args.init_time:
        init_time = datetime.fromisoformat(args.init_time.replace("Z", "+00:00"))
    else:
        init_time = find_latest_ifs_init()
        print(f"Auto-detected init time: {init_time.isoformat()}")

    return run_pipeline(
        init_time=init_time,
        n_hours=args.n_hours,
        output_dir=Path(args.output_dir),
        ifs_source=args.ifs_source,
        debug=args.debug,
        sanity_max_mm=args.precip_max_mm,
    )


if __name__ == "__main__":
    sys.exit(main())
