"""
generate_demo_data.py — Generate deterministic demo forecast artifacts (no GPU required).

Produces the same schema as production forecasts. Used for:
  - Frontend development without a GPU
  - CI/CD pipeline on GitHub Actions
  - Integration testing

The synthetic data is physically plausible but not a real forecast.
The forecast.json sets is_demo: true.

Usage:
    uv run python scripts/generate_demo_data.py [--output-dir data/demo/]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

MYANMAR_LAT_MIN, MYANMAR_LAT_MAX = 9.0, 29.0
MYANMAR_LON_MIN, MYANMAR_LON_MAX = 92.0, 102.0
N_LAT = 81   # 9N to 29N step 0.25
N_LON = 41   # 92E to 102E step 0.25
N_TIMES = 169  # 0h to 168h inclusive

# Fixed init time for reproducibility
DEMO_INIT_TIME = datetime(2026, 8, 9, 0, 0, 0, tzinfo=timezone.utc)


def make_lat_lon_grids() -> tuple[np.ndarray, np.ndarray]:
    lats = np.linspace(MYANMAR_LAT_MIN, MYANMAR_LAT_MAX, N_LAT, dtype=np.float32)
    lons = np.linspace(MYANMAR_LON_MIN, MYANMAR_LON_MAX, N_LON, dtype=np.float32)
    return lats, lons


def generate_temperature(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Synthetic 2m temperature in degrees C.
    Pattern: base temp decreasing with latitude, diurnal cycle, slow synoptic change.
    All values physically plausible for Myanmar (20–40°C).
    """
    times = np.arange(N_TIMES, dtype=np.float32)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")  # [81, 41]

    # Base temperature: ~35°C at south, ~25°C at north
    base = 35.0 - 0.5 * (lat_grid - MYANMAR_LAT_MIN)

    # Diurnal cycle: +/- 5°C over 24h
    hour_of_day = (times % 24) / 24.0  # [0, 1)
    diurnal = 5.0 * np.sin(2 * np.pi * (hour_of_day - 0.25))  # peak at ~14:00 local

    # Slow synoptic signal: ±3°C drift over 7 days
    synoptic = 3.0 * np.sin(2 * np.pi * times / (168.0 * 1.5))

    # Spatial variability: inland hotter, coast cooler
    lon_factor = (lon_grid - MYANMAR_LON_MIN) / (MYANMAR_LON_MAX - MYANMAR_LON_MIN)
    spatial_mod = 2.0 * (lon_factor - 0.5)

    # Build [n_times, n_lat, n_lon]
    t = (
        base[np.newaxis, :, :]
        + diurnal[:, np.newaxis, np.newaxis]
        + synoptic[:, np.newaxis, np.newaxis]
        + spatial_mod[np.newaxis, :, :]
    )
    return t.astype(np.float32)


def generate_precipitation(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Synthetic 1-hour accumulated precipitation in mm/h.
    Pattern: convective cells moving northwest, diurnally triggered in afternoon.
    All values >= 0.
    """
    times = np.arange(N_TIMES, dtype=np.float32)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")  # [81, 41]

    rng = np.random.default_rng(seed=42)  # deterministic

    precip = np.zeros((N_TIMES, N_LAT, N_LON), dtype=np.float32)

    # Afternoon convective cells (peak 14-18 UTC for Myanmar ~ 20-24 local)
    afternoon_hours = [h for h in range(N_TIMES) if (h % 24) in range(14, 20)]

    # Fixed convective cell centres
    cell_centres = [
        (18.0, 96.0, 12.0),   # lat, lon, max_intensity mm/h
        (21.0, 97.5, 8.0),
        (15.0, 98.0, 15.0),
        (25.0, 95.0, 5.0),
        (12.0, 99.0, 20.0),
    ]

    for h in afternoon_hours:
        # Cells drift slowly northwestward
        drift_lat = 0.002 * h
        drift_lon = -0.001 * h

        for (clat, clon, intensity) in cell_centres:
            clat_t = clat + drift_lat + rng.normal(0, 0.1)
            clon_t = clon + drift_lon + rng.normal(0, 0.1)

            dist_sq = (lat_grid - clat_t) ** 2 + (lon_grid - clon_t) ** 2
            cell_precip = intensity * np.exp(-dist_sq / 2.0)
            # Scale by hour-of-day factor (peak at hour 16)
            hour_factor = max(0, np.sin(np.pi * ((h % 24) - 12) / 10))
            precip[h] += cell_precip * hour_factor

    precip = np.maximum(precip, 0.0)
    return precip.astype(np.float32)


def write_binary(arr: np.ndarray, path: Path) -> None:
    """Write [n_times, n_lat, n_lon] float32 array as little-endian binary."""
    path.write_bytes(arr.astype("<f4").tobytes())


def write_forecast_json(output_dir: Path, lats: np.ndarray, lons: np.ndarray) -> None:
    times_utc = [
        (DEMO_INIT_TIME + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for h in range(N_TIMES)
    ]

    meta = {
        "schema_version": "1.0",
        "model": "Aurora1p5",
        "model_version": "1.5",
        "model_checkpoint": "aurora-0.25-v1.5.ckpt",
        "model_source": "hf://microsoft/aurora",
        "initialization_source": "DEMO",
        "initialization_time": times_utc[0],
        "forecast_generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sic_handling": "demo mode — no real IFS data; sic not applicable",
        "forecast_horizon_hours": 168,
        "n_times": N_TIMES,
        "spatial_resolution_deg": 0.25,
        "display_resolution_deg": 0.05,
        "spatial_interpolation": "bilinear",
        "boundary_mask": "Myanmar administrative boundary (GeoJSON scanline rasterization)",
        "region": "Myanmar",
        "bbox": {
            "lat_min": MYANMAR_LAT_MIN,
            "lat_max": MYANMAR_LAT_MAX,
            "lon_min": MYANMAR_LON_MIN,
            "lon_max": MYANMAR_LON_MAX,
        },
        "grid": {"n_lat": N_LAT, "n_lon": N_LON},
        "lat": lats.tolist(),
        "lon": lons.tolist(),
        "times_utc": times_utc,
        "variables": {
            "temperature_2m": {
                "display_name": "2m Temperature",
                "units": "degC",
                "source_variable": "t2m",
                "temporal_resolution": "hourly",
                "temporal_semantics": "Point forecast at each hour",
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
                "transformation": "demo uses synthetic physical values in mm / 1-hour accumulation (no log transform applied)",
                "native_output": True,
                "file": "precipitation.bin",
                "fill_value": -9999.0,
            },
        },
        "data_source_attribution": "DEMO DATA — synthetic, not real forecast",
        "model_attribution": "Microsoft Research Aurora v1.5 (schema only; weights not used in demo)",
        "earth2studio_version": ">=0.17.0",
        "is_demo": True,
    }

    with open(output_dir / "forecast.json", "w") as f:
        json.dump(meta, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate demo forecast artifacts (no GPU)")
    parser.add_argument("--output-dir", default="data/demo", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating demo forecast artifacts → {output_dir.resolve()}")
    print(f"Grid: {N_TIMES} times × {N_LAT} lat × {N_LON} lon")

    lats, lons = make_lat_lon_grids()

    print("  Generating temperature...", end=" ", flush=True)
    temp = generate_temperature(lats, lons)
    write_binary(temp, output_dir / "temperature.bin")
    print(f"done ({temp.nbytes / 1024:.0f} KB, range [{temp.min():.1f}, {temp.max():.1f}] °C)")

    print("  Generating precipitation...", end=" ", flush=True)
    precip = generate_precipitation(lats, lons)
    write_binary(precip, output_dir / "precipitation.bin")
    print(f"done ({precip.nbytes / 1024:.0f} KB, range [{precip.min():.2f}, {precip.max():.2f}] mm/h)")

    print("  Writing forecast.json...", end=" ", flush=True)
    write_forecast_json(output_dir, lats, lons)
    print("done")

    print(f"\nDemo artifacts written to {output_dir.resolve()}")
    print("Validate with:  uv run python scripts/validate_forecast.py --data-dir", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
