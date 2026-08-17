"""
validate_forecast.py — Validate schema v5.0 forecast artifacts (GraphCastOperational, 7-day, 4-variable).

Checks:
  - Required files: forecast.json + 4 binary files
  - forecast.json schema v5.0 completeness
  - Binary file size matches metadata (n_frames × n_lat × n_lon × 4 bytes)
  - Precipitation  : NaN, Inf, negative count, zero-rain fraction, min/med/P95/P99/max
  - Temperature    : NaN, Inf, range in °C
  - Wind speed     : NaN, Inf, min/max in knots
  - Wind direction : NaN, Inf, range [0, 360)
  - All 29 frames contain valid data (no all-NaN frames)
  - Grid: n_lat expected 81, n_lon expected 41 at 0.25°
  - n_frames: expected 29

Usage:
    uv run python scripts/validate_forecast.py --data-dir data/forecast_v4/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXPECTED_N_LAT      = 81
EXPECTED_N_LON      = 41
EXPECTED_N_FRAMES   = 29
EXPECTED_TIMESTEP_H = 6
EXPECTED_HORIZON_H  = 168
EXPECTED_RESOLUTION = 0.25

PRECIP_SANITY_MAX_MM_HR = 300.0
TEMP_MIN_C = -90.0
TEMP_MAX_C =  70.0
WIND_MAX_KT = 200.0

REQUIRED_FILES = [
    "forecast.json",
    "precipitation.bin",
    "temperature.bin",
    "wind_speed.bin",
    "wind_direction.bin",
]


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def percentile_stats(arr: np.ndarray) -> dict:
    valid = arr.flatten()
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        return {}
    return {
        "min": float(np.min(valid)),
        "median": float(np.median(valid)),
        "p95": float(np.percentile(valid, 95)),
        "p99": float(np.percentile(valid, 99)),
        "max": float(np.max(valid)),
    }


def validate(data_dir: str) -> int:
    path   = Path(data_dir)
    errors = 0

    print(f"\nValidating schema v5.0 forecast artifacts: {path.resolve()}")
    print("=" * 65)

    # ── File existence ────────────────────────────────────────────────────────
    print("\n[Files]")
    for fname in REQUIRED_FILES:
        fpath = path / fname
        if fpath.exists():
            ok(f"{fname:<22} present ({fpath.stat().st_size:,} bytes)")
        else:
            fail(f"{fname} MISSING")
            errors += 1

    if errors > 0:
        print("\nCannot continue — required files missing.")
        return errors

    # ── forecast.json ─────────────────────────────────────────────────────────
    print("\n[forecast.json — schema v5.0]")
    with open(path / "forecast.json") as f:
        meta = json.load(f)

    sv = meta.get("schema_version", "")
    if sv == "5.0":
        ok(f"schema_version == '5.0'")
    else:
        fail(f"schema_version '{sv}' != '5.0'")
        errors += 1

    required_top = [
        "model", "model_checkpoint", "earth2studio_version",
        "native_resolution_deg", "native_timestep_hours",
        "initialization_source", "initialization_time", "init_timesteps",
        "forecast_generated_at", "forecast_horizon_hours",
        "n_frames", "times_utc", "region", "bbox", "grid", "lat", "lon",
        "variables", "inference_config", "is_demo",
    ]
    for field in required_top:
        if field in meta:
            ok(f"{field:<30}: {str(meta[field])[:50]}")
        else:
            fail(f"Missing field: {field}")
            errors += 1

    # Model check
    model_name = meta.get("model", "")
    if model_name == "GraphCastOperational":
        ok(f"model = '{model_name}'")
    else:
        fail(f"model = '{model_name}' (expected 'GraphCastOperational')")
        errors += 1

    res = meta.get("native_resolution_deg", -1)
    if res == EXPECTED_RESOLUTION:
        ok(f"native_resolution_deg = {res}°")
    else:
        fail(f"native_resolution_deg = {res} (expected {EXPECTED_RESOLUTION})")
        errors += 1

    is_demo = meta.get("is_demo")
    if is_demo is False:
        ok("is_demo = false")
    else:
        fail(f"is_demo = {is_demo!r} (must be false for real forecast)")
        errors += 1

    # ── Grid validation ───────────────────────────────────────────────────────
    print("\n[Grid]")
    grid = meta.get("grid", {})
    n_lat    = grid.get("n_lat", -1)
    n_lon    = grid.get("n_lon", -1)
    n_frames = meta.get("n_frames", -1)

    for actual, expected, name in [
        (n_lat,    EXPECTED_N_LAT,    "n_lat"),
        (n_lon,    EXPECTED_N_LON,    "n_lon"),
        (n_frames, EXPECTED_N_FRAMES, "n_frames"),
    ]:
        if actual == expected:
            ok(f"{name} = {actual}")
        else:
            fail(f"{name} = {actual} (expected {expected})")
            errors += 1

    timestep_h = meta.get("native_timestep_hours", -1)
    horizon_h  = meta.get("forecast_horizon_hours", -1)
    if timestep_h == EXPECTED_TIMESTEP_H:
        ok(f"native_timestep_hours = {timestep_h}h")
    else:
        fail(f"native_timestep_hours = {timestep_h} (expected {EXPECTED_TIMESTEP_H})")
        errors += 1
    if horizon_h == EXPECTED_HORIZON_H:
        ok(f"forecast_horizon_hours = {horizon_h}h")
    else:
        fail(f"forecast_horizon_hours = {horizon_h} (expected {EXPECTED_HORIZON_H})")
        errors += 1

    # ── Variables metadata ────────────────────────────────────────────────────
    print("\n[Variables metadata]")
    variables = meta.get("variables", {})
    for vname in ["precipitation", "temperature", "wind_speed", "wind_direction"]:
        if vname in variables:
            ok(f"'{vname}' present in variables")
        else:
            fail(f"'{vname}' MISSING from variables")
            errors += 1

    precip_meta = variables.get("precipitation", {})
    if precip_meta.get("native_variable") == "tp06":
        ok("precipitation.native_variable = 'tp06'")
    else:
        fail(f"precipitation.native_variable = {precip_meta.get('native_variable')!r}")
        errors += 1
    if "mm / hr" in precip_meta.get("display_unit", ""):
        ok(f"precipitation.display_unit = '{precip_meta.get('display_unit')}'")
    else:
        fail(f"precipitation.display_unit = '{precip_meta.get('display_unit')}' (expected 'mm / hr')")
        errors += 1

    temp_meta = variables.get("temperature", {})
    if temp_meta.get("native_variable") == "t2m":
        ok("temperature.native_variable = 't2m'")
    else:
        fail(f"temperature.native_variable = {temp_meta.get('native_variable')!r}")
        errors += 1

    wspd_meta = variables.get("wind_speed", {})
    if sorted(wspd_meta.get("native_variables", [])) == ["u10m", "v10m"]:
        ok("wind_speed.native_variables = ['u10m', 'v10m']")
    else:
        fail(f"wind_speed.native_variables = {wspd_meta.get('native_variables')}")
        errors += 1

    wdir_meta = variables.get("wind_direction", {})
    if "FROM" in wdir_meta.get("display_unit", "") or "FROM" in wdir_meta.get("conversion", ""):
        ok("wind_direction uses FROM convention")
    else:
        fail("wind_direction.display_unit does not mention FROM convention")
        errors += 1

    # ── Binary file validation ────────────────────────────────────────────────
    expected_elements = n_frames * n_lat * n_lon
    expected_bytes    = expected_elements * 4  # float32

    def validate_binary(fname: str, label: str) -> tuple[int, dict]:
        """Returns (new_errors, stats_dict)."""
        errs = 0
        print(f"\n[{label}]")
        fpath = path / fname
        raw = np.frombuffer(fpath.read_bytes(), dtype="<f4")

        if len(raw) == expected_elements:
            ok(f"Size: {len(raw)} elements = {n_frames}×{n_lat}×{n_lon} ({expected_bytes:,} bytes)")
        else:
            fail(f"Size: {len(raw)} elements (expected {expected_elements})")
            errs += 1
            return errs, {}

        arr = raw.reshape(n_frames, n_lat, n_lon)
        n_nan = int(np.sum(np.isnan(arr)))
        n_inf = int(np.sum(np.isinf(arr)))
        s     = percentile_stats(arr)

        if n_nan == 0:
            ok("No NaN values")
        else:
            fail(f"{n_nan} NaN values")
            errs += 1
        if n_inf == 0:
            ok("No Inf values")
        else:
            fail(f"{n_inf} Inf values")
            errs += 1

        if s:
            info(f"min={s['min']:.4f}  median={s['median']:.4f}  P95={s['p95']:.4f}  P99={s['p99']:.4f}  max={s['max']:.4f}")

        # Per-frame validity
        bad_frames = [i for i in range(n_frames) if np.all(np.isnan(arr[i]))]
        if bad_frames:
            fail(f"All-NaN frames: {bad_frames}")
            errs += 1
        else:
            ok(f"All {n_frames} frames contain valid data")

        return errs, {"arr": arr, **s, "n_nan": n_nan, "n_inf": n_inf}

    # Precipitation
    e, p_data = validate_binary("precipitation.bin", "Precipitation (mm/hr)")
    errors += e
    if p_data:
        arr_p = p_data["arr"]
        n_neg = int(np.sum(arr_p < 0.0))
        n_zero = int(np.sum(arr_p == 0.0))
        zero_frac = n_zero / arr_p.size * 100.0
        info(f"Negative values after clamp: {n_neg} (expected 0)")
        if n_neg > 0:
            fail(f"{n_neg} negative precipitation values after clamp")
            errors += 1
        else:
            ok("No negative precipitation values")
        info(f"Zero-rain fraction: {zero_frac:.1f}%")
        if p_data.get("max") is not None and p_data["max"] > PRECIP_SANITY_MAX_MM_HR:
            fail(f"Max {p_data['max']:.1f} mm/hr exceeds sanity threshold {PRECIP_SANITY_MAX_MM_HR}")
            errors += 1
        else:
            ok(f"Max {p_data.get('max', 0):.2f} mm/hr ≤ {PRECIP_SANITY_MAX_MM_HR} sanity threshold")

    # Temperature
    e, t_data = validate_binary("temperature.bin", "Temperature (°C)")
    errors += e
    if t_data:
        if t_data.get("min") is not None:
            if t_data["min"] >= TEMP_MIN_C:
                ok(f"Min {t_data['min']:.2f}°C ≥ {TEMP_MIN_C}°C lower bound")
            else:
                fail(f"Min {t_data['min']:.2f}°C < {TEMP_MIN_C}°C lower bound")
                errors += 1
            if t_data["max"] <= TEMP_MAX_C:
                ok(f"Max {t_data['max']:.2f}°C ≤ {TEMP_MAX_C}°C upper bound")
            else:
                fail(f"Max {t_data['max']:.2f}°C > {TEMP_MAX_C}°C upper bound")
                errors += 1

    # Wind speed
    e, ws_data = validate_binary("wind_speed.bin", "Wind Speed (knots)")
    errors += e
    if ws_data:
        if ws_data.get("min") is not None and ws_data["min"] >= 0.0:
            ok(f"Wind speed min {ws_data['min']:.2f} kt ≥ 0")
        elif ws_data.get("min") is not None:
            fail(f"Negative wind speed: min = {ws_data['min']:.4f} kt")
            errors += 1
        if ws_data.get("max") is not None and ws_data["max"] > WIND_MAX_KT:
            fail(f"Wind speed max {ws_data['max']:.1f} kt > {WIND_MAX_KT} sanity threshold")
            errors += 1

    # Wind direction
    e, wd_data = validate_binary("wind_direction.bin", "Wind Direction (°FROM)")
    errors += e
    if wd_data:
        arr_dir = wd_data["arr"]
        in_range = bool(np.all((arr_dir[np.isfinite(arr_dir)] >= 0.0) &
                               (arr_dir[np.isfinite(arr_dir)] < 360.0)))
        if in_range:
            ok(f"All direction values in [0, 360)")
        else:
            fail("Some direction values outside [0, 360)")
            errors += 1
        if wd_data.get("min") is not None:
            info(f"Direction range: {wd_data['min']:.2f}° to {wd_data['max']:.2f}°")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    if errors == 0:
        print(f"VALIDATION PASSED — all checks passed for {path}")
    else:
        print(f"VALIDATION FAILED — {errors} error(s) in {path}")
    print("=" * 65)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate schema v5.0 GraphCastOperational 7-day 4-variable forecast artifacts"
    )
    parser.add_argument("--data-dir", required=True, help="Path to forecast artifact directory")
    args = parser.parse_args()
    return min(validate(args.data_dir), 1)


if __name__ == "__main__":
    sys.exit(main())
