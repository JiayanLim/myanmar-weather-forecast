"""
generate_demo_data.py — Generate deterministic demo forecast artifacts (no GPU required).

Produces the same schema as GraphCastSmall production forecasts (schema v4.0).
Used for:
  - Frontend development without a GPU
  - CI/CD pipeline on GitHub Actions
  - Integration testing

The synthetic data is NOT from GraphCastSmall. It is physically plausible but
clearly marked is_demo: true. Do not use for operational decisions.

Temporal semantics:
  tp06 = 6-hour accumulated total precipitation (same contract as production).
  t2m  = 2m temperature in °C (same contract as production).
  t+0h precipitation is set to 0.0 (init frame; no forecast accumulation).
  t+0h temperature is the analysis state temperature.

Usage:
    uv run python scripts/generate_demo_data.py [--output-dir data/demo/]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# ── Grid constants (must match production GraphCastSmall contract) ────────────
MYANMAR_LAT_MIN, MYANMAR_LAT_MAX = 9.0, 29.0
MYANMAR_LON_MIN, MYANMAR_LON_MAX = 92.0, 102.0
N_LAT = 21          # 9°N to 29°N at 1.0° spacing
N_LON = 11          # 92°E to 102°E at 1.0° spacing
N_FRAMES = 29       # t+0h through t+168h at 6h steps (28 AR steps + init)
STEP_HOURS = 6      # Native GraphCastSmall timestep

# Fixed init time for reproducibility
DEMO_INIT_TIME = datetime(2026, 8, 9, 0, 0, 0, tzinfo=timezone.utc)


def make_lat_lon_grids() -> tuple[np.ndarray, np.ndarray]:
    lats = np.linspace(MYANMAR_LAT_MIN, MYANMAR_LAT_MAX, N_LAT, dtype=np.float64)
    lons = np.linspace(MYANMAR_LON_MIN, MYANMAR_LON_MAX, N_LON, dtype=np.float64)
    return lats, lons


def generate_precipitation(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Synthetic tp06: 6-hour accumulated precipitation (mm / 6h).

    Pattern: convective cells that intensify and then weaken.
    All values >= 0. t+0h is all zeros (init frame; no forecast accumulation).

    Returns array of shape [N_FRAMES, N_LAT, N_LON] in float32.
    """
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    rng = np.random.default_rng(seed=42)

    precip = np.zeros((N_FRAMES, N_LAT, N_LON), dtype=np.float32)

    # Cell centres: (lat, lon, peak_intensity_mm_per_6h, spread_deg)
    cell_centres = [
        (18.0,  96.0,  45.0, 1.5),  # central Myanmar, moderate
        (21.0,  97.5,  30.0, 1.0),  # northeastern
        (15.0,  98.0,  60.0, 2.0),  # southern coast, intense
        (25.0,  95.0,  20.0, 0.8),  # northern hills, light
        (12.0,  99.0,  80.0, 1.8),  # Tanintharyi coast, heavy
        (17.0,  94.5,  35.0, 1.2),  # Ayeyarwady delta
    ]

    # Frame scales: t+0h always 0; t+6h through t+168h build, peak, and decay
    # 29 entries: index 0 = t+0h (init, always 0), indices 1-28 = t+6h through t+168h
    frame_scales = [
        0.0,                                   # t+0h: init, no accumulation
        0.3, 0.6, 1.0, 0.9, 0.8, 0.7, 0.6,   # t+6h–t+42h: build and peak (day 1)
        0.5, 0.7, 0.9, 1.0, 0.8, 0.7, 0.5,   # t+48h–t+84h: second peak (day 2-3)
        0.4, 0.5, 0.6, 0.7, 0.5, 0.4, 0.3,   # t+90h–t+126h: weakening (day 4-5)
        0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1,   # t+132h–t+168h: dissipating (day 6-7)
    ]

    for frame_idx in range(N_FRAMES):
        if frame_scales[frame_idx] == 0.0:
            continue

        scale = frame_scales[frame_idx]
        drift_lat = 0.3 * (frame_idx - 3)
        drift_lon = -0.2 * (frame_idx - 3)

        for (clat, clon, intensity, spread) in cell_centres:
            clat_t = clat + drift_lat + rng.normal(0, 0.3)
            clon_t = clon + drift_lon + rng.normal(0, 0.3)
            dist_sq = (lat_grid - clat_t) ** 2 + (lon_grid - clon_t) ** 2
            cell = intensity * scale * np.exp(-dist_sq / (2 * spread ** 2))
            precip[frame_idx] += cell

    precip = np.maximum(precip, 0.0)
    return precip.astype(np.float32)


def generate_temperature(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Synthetic t2m: 2m temperature in °C.

    Pattern:
      - Base temperature varies with latitude (cooler north, warmer south/coast)
      - Diurnal cycle: warmer in the afternoon frames, cooler at night
      - Myanmar range: ~22-40°C in summer; higher inland, cooler mountains

    Returns array of shape [N_FRAMES, N_LAT, N_LON] in float32.
    """
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    # Base temperature: latitudinal gradient (°C)
    # Southern Myanmar coast (9°N): ~32°C; northern hills (29°N): ~24°C
    t_base = 36.0 - 0.4 * (lat_grid - 9.0)

    # Elevation cooling: approximate (northern highlands slightly cooler)
    t_base -= 0.1 * np.maximum(lat_grid - 20.0, 0.0)

    rng = np.random.default_rng(seed=99)

    temp = np.zeros((N_FRAMES, N_LAT, N_LON), dtype=np.float32)

    # Diurnal pattern for each 6h frame — repeating 24h cycle (4 steps per day)
    # Frame 0 = 00Z, 1 = 06Z (warm), 2 = 12Z (hottest), 3 = 18Z (cooling), repeat
    base_diurnal = [0.0, 2.0, 5.0, 2.0]
    diurnal_offsets = [base_diurnal[i % 4] for i in range(N_FRAMES)]

    for frame_idx in range(N_FRAMES):
        offset = diurnal_offsets[frame_idx]
        # Small spatial noise for realism
        noise = rng.normal(0, 0.3, size=t_base.shape)
        temp[frame_idx] = (t_base + offset + noise).astype(np.float32)

    return temp.astype(np.float32)


def write_binary(arr: np.ndarray, path: Path) -> None:
    """Write [n_frames, n_lat, n_lon] float32 array as little-endian binary."""
    path.write_bytes(arr.astype("<f4").tobytes())


def write_forecast_json(
    output_dir: Path,
    lats: np.ndarray,
    lons: np.ndarray,
) -> None:
    times_utc = [
        (DEMO_INIT_TIME + timedelta(hours=i * STEP_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(N_FRAMES)
    ]

    meta = {
        "schema_version": "4.0",
        "model": "GraphCastSmall",
        "model_version": "1.0",
        "model_checkpoint": (
            "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
            "pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
        ),
        "model_source": "gs://dm_graphcast/graphcast",
        "initialization_source": "DEMO",
        "initialization_time": times_utc[0],
        "forecast_generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_horizon_hours": (N_FRAMES - 1) * STEP_HOURS,
        "native_timestep_hours": STEP_HOURS,
        "n_times": N_FRAMES,
        "spatial_resolution_deg": 1.0,
        "display_resolution_deg": None,
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
                    "accumulation_period_hours": STEP_HOURS,
                    "log_transform_applied": False,
                    "exp_transform_applied": False,
                    "pipeline": (
                        "DEMO synthetic data: mm/6h values generated directly. "
                        "Production: GraphCastSmall native tp06 → metres × 1000 → mm/6h."
                    ),
                },
                "t0_note": (
                    "t+0h frame (index 0) is synthetic: precipitation set to 0.0. "
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
                    "pipeline": "DEMO synthetic data: °C values generated directly.",
                },
                "t0_note": (
                    "t+0h frame (index 0) is the synthetic analysis temperature."
                ),
                "native_output": True,
                "file": "temperature.bin",
                "fill_value": None,
            },
        },
        "data_source_attribution": "DEMO DATA — synthetic, not a real GraphCastSmall forecast",
        "model_attribution": (
            "GraphCast by DeepMind/Google (schema only; model not executed in demo). "
            "Earth2Studio wrapper by NVIDIA."
        ),
        "earth2studio_version": ">=0.17.0",
        "inference_config": {
            "device": "DEMO (no inference)",
            "jax_backend": None,
            "rss_peak_gb": None,
            "model_load_time_seconds": None,
            "inference_time_seconds": None,
            "total_pipeline_time_seconds": None,
        },
        "is_demo": True,
    }

    with open(output_dir / "forecast.json", "w") as f:
        json.dump(meta, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate GraphCastSmall-compatible demo forecast artifacts (no GPU)"
    )
    parser.add_argument("--output-dir", default="data/demo", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating demo forecast artifacts → {output_dir.resolve()}")
    print(f"Schema : v4.0 (GraphCastSmall 168h/7-day, temperature + precipitation)")
    print(f"Grid   : {N_FRAMES} frames × {N_LAT} lat × {N_LON} lon")
    print(f"Steps  : {STEP_HOURS}h × {N_FRAMES - 1} = {(N_FRAMES-1)*STEP_HOURS}h horizon")

    lats, lons = make_lat_lon_grids()
    expected_bytes = N_FRAMES * N_LAT * N_LON * 4  # 29 × 21 × 11 × 4 = 26,796

    print("  Generating tp06 precipitation (mm/6h)...", end=" ", flush=True)
    precip = generate_precipitation(lats, lons)
    write_binary(precip, output_dir / "precipitation.bin")
    actual_bytes = (output_dir / "precipitation.bin").stat().st_size
    assert actual_bytes == expected_bytes, f"Size mismatch: {actual_bytes} != {expected_bytes}"
    print(f"done ({actual_bytes} bytes, range [{precip.min():.2f}, {precip.max():.2f}] mm/6h)")

    print("  Generating t2m temperature (°C)...", end=" ", flush=True)
    temp = generate_temperature(lats, lons)
    write_binary(temp, output_dir / "temperature.bin")
    actual_temp_bytes = (output_dir / "temperature.bin").stat().st_size
    assert actual_temp_bytes == expected_bytes, f"Size mismatch: {actual_temp_bytes} != {expected_bytes}"
    print(f"done ({actual_temp_bytes} bytes, range [{temp.min():.2f}, {temp.max():.2f}] °C)")

    print("  Writing forecast.json...", end=" ", flush=True)
    write_forecast_json(output_dir, lats, lons)
    print("done")

    print()
    print(f"Demo artifacts written to {output_dir.resolve()}")
    print(f"  precipitation.bin : {N_FRAMES} × {N_LAT} × {N_LON} × float32 = {expected_bytes} bytes")
    print(f"  temperature.bin   : {N_FRAMES} × {N_LAT} × {N_LON} × float32 = {expected_bytes} bytes")
    print(f"  forecast.json     : schema v4.0, is_demo=true, model=GraphCastSmall")
    print()
    print(f"Validate with:")
    print(f"  uv run python scripts/validate_forecast.py --data-dir {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
