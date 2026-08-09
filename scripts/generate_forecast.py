"""
generate_forecast.py — Full Aurora1p5 + IFS forecast pipeline.

Requires:
  - GPU with >= 48 GB VRAM
  - Earth2Studio >= 0.17.0 installed
  - IFS open data accessible (ECMWF; no credentials for public stream)

Usage:
    uv run python scripts/generate_forecast.py \
        --init-time 2026-08-09T00:00:00Z \
        [--output-dir data/forecast/] \
        [--sic-method zero|arco] \
        [--n-hours 168]

Output:
    data/forecast/forecast.json
    data/forecast/temperature.bin
    data/forecast/precipitation.bin
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

MYANMAR_LAT_MIN, MYANMAR_LAT_MAX = 9.0, 29.0
MYANMAR_LON_MIN, MYANMAR_LON_MAX = 92.0, 102.0


def run_pipeline(
    init_time: datetime,
    n_hours: int,
    output_dir: Path,
    sic_method: str,
) -> int:
    import numpy as np
    import xarray as xr

    try:
        import earth2studio.data as e2_data
        import earth2studio.run as e2_run
        from earth2studio.io import ZarrBackend
        from earth2studio.models.px import Aurora1p5
    except ImportError:
        print("ERROR: earth2studio not installed. Run: uv sync")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1: Build initialization data source with sic handling
    # ------------------------------------------------------------------ #
    print(f"\n[1/5] Building IFS initialization source (sic_method={sic_method})...")
    ifs = e2_data.IFS()
    sic_handling_desc = ""

    if sic_method == "zero":
        # Monkey-patch: after IFS fetch, inject sic=0 array
        # This is valid for tropical initialization (Myanmar region)
        sic_handling_desc = (
            "sic set to 0.0 globally. "
            "IFS open data does not publish sic. "
            "Zero is physically appropriate for tropical Myanmar initialization."
        )
        print(f"  sic handling: {sic_handling_desc}")

        class IFSWithZeroSIC:
            """Wraps IFS and injects sic=0 when requested."""

            def __call__(self, time, variables):
                needs_sic = "sic" in variables
                vars_no_sic = [v for v in variables if v != "sic"]

                if vars_no_sic:
                    times_out, data_out = ifs(time, vars_no_sic)
                else:
                    # Only sic was requested — return a zero array with correct shape
                    # Shape determined from a quick t2m fetch
                    _, ref_data = ifs(time, ["t2m"])
                    times_out = np.array([time])
                    data_out = np.zeros((1, 1, *ref_data.shape[-2:]), dtype=np.float32)
                    return times_out, data_out

                if needs_sic:
                    # Append zeros for sic
                    sic_zeros = np.zeros(
                        (data_out.shape[0], 1, *data_out.shape[-2:]), dtype=np.float32
                    )
                    data_out = np.concatenate([data_out, sic_zeros], axis=1)
                    # Note: the variable ordering must match — Earth2Studio handles this
                    # via its lexicon system; the ZarrBackend tracks variable names

                return times_out, data_out

        init_source = IFSWithZeroSIC()

    elif sic_method == "arco":
        from earth2studio.data import ARCO
        arco = ARCO()
        sic_handling_desc = f"sic patched from ARCO ERA5 at {init_time.isoformat()}"
        print(f"  sic handling: {sic_handling_desc}")

        class IFSWithARCOSIC:
            def __call__(self, time, variables):
                needs_sic = "sic" in variables
                vars_no_sic = [v for v in variables if v != "sic"]

                if vars_no_sic:
                    times_out, data_out = ifs(time, vars_no_sic)
                else:
                    _, ref = ifs(time, ["t2m"])
                    return np.array([time]), np.zeros((1, 1, *ref.shape[-2:]), dtype=np.float32)

                if needs_sic:
                    _, sic_data = arco(time, ["sic"])
                    data_out = np.concatenate([data_out, sic_data], axis=1)

                return times_out, data_out

        init_source = IFSWithARCOSIC()
    else:
        raise ValueError(f"Unknown sic_method: {sic_method}")

    # ------------------------------------------------------------------ #
    # Step 2: Load Aurora1p5
    # ------------------------------------------------------------------ #
    print("\n[2/5] Loading Aurora1p5 model weights (HuggingFace)...")
    print("  This may take several minutes on first run (~5 GB download)...")
    package = Aurora1p5.load_default_package()
    model = Aurora1p5.load_model(package)
    print("  Aurora1p5 loaded.")

    # ------------------------------------------------------------------ #
    # Step 3: Run inference
    # ------------------------------------------------------------------ #
    print(f"\n[3/5] Running {n_hours}h Aurora1p5 forecast from {init_time.isoformat()}...")
    print("  Native hourly rollout: produces t+1h through t+{n_hours}h without interpolation")
    io = ZarrBackend()
    io = e2_run.deterministic(
        [init_time.strftime("%Y-%m-%dT%H:%M:%S")],
        n_hours,
        model,
        init_source,
        io,
    )
    print("  Inference complete.")

    # ------------------------------------------------------------------ #
    # Step 4: Extract and post-process
    # ------------------------------------------------------------------ #
    print("\n[4/5] Extracting t2m and tp1h, subsetting Myanmar bbox...")
    ds = xr.open_zarr(io.store)

    # Extract t2m (temperature)
    t2m_raw = ds["t2m"]  # [batch, time, lead_time, lat, lon] or similar
    # Subset to Myanmar
    t2m_mm = t2m_raw.sel(
        lat=slice(MYANMAR_LAT_MAX, MYANMAR_LAT_MIN),
        lon=slice(MYANMAR_LON_MIN, MYANMAR_LON_MAX),
    )
    # Handle potential descending lat order
    if t2m_mm.lat.values[0] > t2m_mm.lat.values[-1]:
        t2m_mm = t2m_mm.isel(lat=slice(None, None, -1))

    # Convert K to C
    t2m_celsius = (t2m_mm.values.squeeze() - 273.15).astype(np.float32)
    print(f"  t2m shape after subset: {t2m_celsius.shape}, range: [{t2m_celsius.min():.1f}, {t2m_celsius.max():.1f}] °C")

    # Extract tp1h (precipitation) — apply log untransform
    tp1h_raw = ds["tp1h"]
    tp1h_mm_ds = tp1h_raw.sel(
        lat=slice(MYANMAR_LAT_MAX, MYANMAR_LAT_MIN),
        lon=slice(MYANMAR_LON_MIN, MYANMAR_LON_MAX),
    )
    if tp1h_mm_ds.lat.values[0] > tp1h_mm_ds.lat.values[-1]:
        tp1h_mm_ds = tp1h_mm_ds.isel(lat=slice(None, None, -1))

    tp1h_log = tp1h_mm_ds.values.squeeze()
    tp1h_mm = np.exp(tp1h_log) * 1000.0  # log(m) -> m -> mm
    tp1h_mm = np.maximum(tp1h_mm, 0.0).astype(np.float32)
    print(f"  tp1h shape after subset: {tp1h_mm.shape}, range: [{tp1h_mm.min():.4f}, {tp1h_mm.max():.2f}] mm/h")

    # Build full time array including t+0 (init state)
    # t2m_celsius shape: [n_lead, n_lat, n_lon] (lead = 1..n_hours)
    # Prepend t+0 from initialization data (fetch t2m from IFS)
    print("  Fetching t+0 initialization frame...")
    try:
        _, init_t2m = init_source(init_time, ["t2m"])
        # init_t2m shape: [1, 1, n_lat_global, n_lon_global]
        # Subset manually
        lat_arr = np.linspace(90, -90, 720)
        lon_arr = np.linspace(0, 359.75, 1440)
        lat_mask = (lat_arr >= MYANMAR_LAT_MIN) & (lat_arr <= MYANMAR_LAT_MAX)
        lon_mask = (lon_arr >= MYANMAR_LON_MIN) & (lon_arr <= MYANMAR_LON_MAX)
        t0_frame = (init_t2m.squeeze()[np.ix_(lat_mask, lon_mask)] - 273.15).astype(np.float32)
        t0_frame = t0_frame[::-1]  # south-to-north
    except Exception as e:
        print(f"  Warning: could not fetch t+0 init frame: {e}. Using first forecast frame.")
        t0_frame = t2m_celsius[0]

    # Prepend t+0 frames
    t2m_full = np.concatenate([t0_frame[np.newaxis], t2m_celsius], axis=0)  # [169, 81, 41]
    tp1h_full = np.concatenate([np.zeros_like(tp1h_mm[:1]), tp1h_mm], axis=0)  # t+0 = 0 precip

    lats_out = np.linspace(MYANMAR_LAT_MIN, MYANMAR_LAT_MAX, t2m_full.shape[1], dtype=np.float32)
    lons_out = np.linspace(MYANMAR_LON_MIN, MYANMAR_LON_MAX, t2m_full.shape[2], dtype=np.float32)

    print(f"  Final array shape: {t2m_full.shape} = [n_times={t2m_full.shape[0]}, n_lat={t2m_full.shape[1]}, n_lon={t2m_full.shape[2]}]")

    # ------------------------------------------------------------------ #
    # Step 5: Write artifacts
    # ------------------------------------------------------------------ #
    print(f"\n[5/5] Writing artifacts to {output_dir.resolve()}...")

    (output_dir / "temperature.bin").write_bytes(t2m_full.astype("<f4").tobytes())
    print(f"  temperature.bin written ({t2m_full.nbytes / 1024:.0f} KB)")

    (output_dir / "precipitation.bin").write_bytes(tp1h_full.astype("<f4").tobytes())
    print(f"  precipitation.bin written ({tp1h_full.nbytes / 1024:.0f} KB)")

    times_utc = [
        (init_time + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for h in range(t2m_full.shape[0])
    ]

    meta = {
        "schema_version": "1.0",
        "model": "Aurora1p5",
        "model_version": "1.5",
        "model_checkpoint": "aurora-0.25-v1.5.ckpt",
        "model_source": "hf://microsoft/aurora",
        "initialization_source": "IFS",
        "initialization_time": times_utc[0],
        "forecast_generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sic_handling": sic_handling_desc,
        "forecast_horizon_hours": n_hours,
        "n_times": t2m_full.shape[0],
        "spatial_resolution_deg": 0.25,
        "region": "Myanmar",
        "bbox": {
            "lat_min": MYANMAR_LAT_MIN,
            "lat_max": MYANMAR_LAT_MAX,
            "lon_min": MYANMAR_LON_MIN,
            "lon_max": MYANMAR_LON_MAX,
        },
        "grid": {"n_lat": t2m_full.shape[1], "n_lon": t2m_full.shape[2]},
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
                "units": "mm/h",
                "source_variable": "tp1h",
                "temporal_resolution": "hourly",
                "temporal_semantics": "Total precipitation accumulated over 1-hour forecast period",
                "temporal_disclosure": (
                    "Precipitation values represent total rainfall accumulated during each "
                    "1-hour forecast period. These are not instantaneous rainfall rates."
                ),
                "transformation": "exp(raw_model_output) * 1000 — log-untransform then m to mm",
                "native_output": True,
                "file": "precipitation.bin",
                "fill_value": -9999.0,
            },
        },
        "data_source_attribution": "IFS HRES analysis — ECMWF open data (https://data.ecmwf.int/)",
        "model_attribution": "Microsoft Research Aurora v1.5 (https://github.com/microsoft/aurora)",
        "earth2studio_version": ">=0.17.0",
        "is_demo": False,
    }

    with open(output_dir / "forecast.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  forecast.json written")

    print(f"\nForecast complete. Validate with:")
    print(f"  uv run python scripts/validate_forecast.py --data-dir {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Aurora1p5 + IFS forecast (GPU required)")
    parser.add_argument(
        "--init-time",
        required=True,
        help="Initialization time (ISO 8601, e.g. 2026-08-09T00:00:00Z)",
    )
    parser.add_argument("--output-dir", default="data/forecast", help="Output directory")
    parser.add_argument(
        "--sic-method",
        choices=["zero", "arco"],
        default="zero",
        help="How to handle missing sic from IFS open data (default: zero)",
    )
    parser.add_argument("--n-hours", type=int, default=168, help="Forecast hours (default: 168)")
    args = parser.parse_args()

    init_time = datetime.fromisoformat(args.init_time.replace("Z", "+00:00"))
    return run_pipeline(
        init_time=init_time,
        n_hours=args.n_hours,
        output_dir=Path(args.output_dir),
        sic_method=args.sic_method,
    )


if __name__ == "__main__":
    sys.exit(main())
