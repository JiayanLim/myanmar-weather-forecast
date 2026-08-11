"""
validate_forecast.py — Validate generated forecast artifacts before frontend use.

Schema v3.0: GraphCastSmall 48h, temperature + precipitation, 9 frames, local M4 CPU.

Checks:
  - Required files present (forecast.json, precipitation.bin, temperature.bin)
  - forecast.json schema v3.0 completeness
  - Binary file dimensions match metadata [n_times × n_lat × n_lon]
  - No NaN values in either variable
  - tp06 >= 0 (no negative precipitation)
  - tp06 < 500 mm/6h (extreme upper bound for tropical rainfall)
  - t2m in plausible range [-90°C, 70°C]
  - Timestamps 6h apart and monotonically increasing
  - Grid dimensions: n_lat == 21, n_lon == 11
  - n_times == 9, native_timestep_hours == 6, forecast_horizon_hours == 48
  - transformation_provenance checks for both variables
  - is_demo flag present

Usage:
    uv run python scripts/validate_forecast.py --data-dir data/demo/
    uv run python scripts/validate_forecast.py --data-dir data/forecast/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


EXPECTED_N_LAT      = 21     # 9N to 29N at 1.0°
EXPECTED_N_LON      = 11     # 92E to 102E at 1.0°
EXPECTED_N_TIMES    = 9      # t+0h through t+48h
EXPECTED_TIMESTEP_H = 6
EXPECTED_HORIZON_H  = 48
PRECIP_MAX_THRESHOLD = 500.0  # mm/6h
TEMP_MIN_C = -90.0
TEMP_MAX_C = 70.0


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def info(msg: str) -> None:
    print(f"  [    ] {msg}")


def validate(data_dir: str) -> int:
    path = Path(data_dir)
    errors = 0

    print(f"\nValidating forecast artifacts in: {path.resolve()}")
    print("=" * 60)

    # --- File existence ---
    print("\n[Files]")
    required_files = ["forecast.json", "precipitation.bin", "temperature.bin"]
    for fname in required_files:
        fpath = path / fname
        if fpath.exists():
            ok(f"{fname} present ({fpath.stat().st_size:,} bytes)")
        else:
            fail(f"{fname} MISSING")
            errors += 1

    if errors > 0:
        print("\nCannot continue: required files missing.")
        return errors

    # --- forecast.json ---
    print("\n[forecast.json schema]")
    with open(path / "forecast.json") as f:
        meta = json.load(f)

    required_fields = [
        "schema_version", "model", "model_version",
        "initialization_source", "initialization_time",
        "forecast_generated_at", "forecast_horizon_hours",
        "native_timestep_hours", "n_times",
        "spatial_resolution_deg", "bbox", "grid",
        "lat", "lon", "times_utc", "variables",
        "is_demo", "earth2studio_version",
    ]
    for field in required_fields:
        if field in meta:
            ok(f"{field}: {str(meta[field])[:60]}")
        else:
            fail(f"Missing field: {field}")
            errors += 1

    sv = meta.get("schema_version", "")
    if sv == "3.0":
        ok(f"schema_version == '3.0'")
    else:
        fail(f"schema_version '{sv}' != '3.0'")
        errors += 1

    # --- Grid dimensions ---
    print("\n[Grid dimensions]")
    grid = meta.get("grid", {})
    n_lat = grid.get("n_lat", -1)
    n_lon = grid.get("n_lon", -1)
    n_times = meta.get("n_times", -1)

    for actual, expected, name in [
        (n_lat, EXPECTED_N_LAT, "n_lat"),
        (n_lon, EXPECTED_N_LON, "n_lon"),
        (n_times, EXPECTED_N_TIMES, "n_times"),
    ]:
        if actual == expected:
            ok(f"{name} = {actual} (expected {expected})")
        else:
            fail(f"{name} = {actual} (expected {expected})")
            errors += 1

    # --- Timestep / horizon ---
    print("\n[Timestep & horizon]")
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

    res = meta.get("spatial_resolution_deg", -1)
    if res == 1.0:
        ok(f"spatial_resolution_deg = {res}")
    else:
        fail(f"spatial_resolution_deg = {res} (expected 1.0)")
        errors += 1

    # --- Timestamps ---
    print("\n[Timestamps]")
    times = meta.get("times_utc", [])

    if len(times) == n_times:
        ok(f"{len(times)} timestamps in times_utc")
    else:
        fail(f"{len(times)} timestamps (expected {n_times})")
        errors += 1

    if len(times) >= 2:
        t_parsed = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in times]

        expected_dt_s = EXPECTED_TIMESTEP_H * 3600
        bad_gaps = []
        for i in range(1, len(t_parsed)):
            gap_s = (t_parsed[i] - t_parsed[i - 1]).total_seconds()
            if gap_s != expected_dt_s:
                bad_gaps.append((i, gap_s / 3600))

        if not bad_gaps:
            ok(f"All timestamps {EXPECTED_TIMESTEP_H}h apart")
        else:
            for idx, gap_h in bad_gaps:
                fail(f"Gap at index {idx-1}→{idx}: {gap_h:.1f}h (expected {EXPECTED_TIMESTEP_H}h)")
                errors += 1

        if all(t_parsed[i] > t_parsed[i - 1] for i in range(1, len(t_parsed))):
            ok("Timestamps monotonically increasing")
        else:
            fail("Timestamps not monotonically increasing")
            errors += 1

    # --- precipitation.bin ---
    print("\n[precipitation.bin]")
    arr_p = np.frombuffer((path / "precipitation.bin").read_bytes(), dtype="<f4")
    expected_size = n_times * n_lat * n_lon
    expected_bytes = expected_size * 4

    if len(arr_p) == expected_size:
        ok(f"Size: {len(arr_p)} elements = {n_times}×{n_lat}×{n_lon} ({expected_bytes} bytes)")
    else:
        fail(f"Size mismatch: {len(arr_p)} elements (expected {expected_size})")
        errors += 1
        print("\n" + "=" * 60)
        print(f"VALIDATION FAILED — {errors} error(s) in {path}")
        print("=" * 60)
        return errors

    data_p = arr_p.reshape(n_times, n_lat, n_lon)
    nan_count = int(np.sum(np.isnan(data_p)))
    if nan_count == 0:
        ok("No NaN values")
    else:
        fail(f"{nan_count} NaN values")
        errors += 1

    vmin_p = float(np.nanmin(data_p))
    vmax_p = float(np.nanmax(data_p))
    info(f"Value range: [{vmin_p:.3f}, {vmax_p:.3f}] mm/6h")

    if vmin_p >= 0:
        ok(f"tp06 >= 0 (min = {vmin_p:.4f} mm/6h)")
    else:
        fail(f"Negative precipitation: min = {vmin_p:.4f} mm/6h")
        errors += 1

    if vmax_p < PRECIP_MAX_THRESHOLD:
        ok(f"tp06 max = {vmax_p:.2f} mm/6h (< {PRECIP_MAX_THRESHOLD:.0f} threshold)")
    else:
        fail(f"tp06 max = {vmax_p:.1f} mm/6h exceeds threshold {PRECIP_MAX_THRESHOLD:.0f}")
        errors += 1

    t0_max_p = float(np.nanmax(np.abs(data_p[0])))
    info(f"t+0h frame: max abs = {t0_max_p:.4f} mm/6h (expected 0.0 for init frame)")

    # --- temperature.bin ---
    print("\n[temperature.bin]")
    arr_t = np.frombuffer((path / "temperature.bin").read_bytes(), dtype="<f4")

    if len(arr_t) == expected_size:
        ok(f"Size: {len(arr_t)} elements = {n_times}×{n_lat}×{n_lon} ({expected_bytes} bytes)")
    else:
        fail(f"Size mismatch: {len(arr_t)} elements (expected {expected_size})")
        errors += 1
        print("\n" + "=" * 60)
        print(f"VALIDATION FAILED — {errors} error(s) in {path}")
        print("=" * 60)
        return errors

    data_t = arr_t.reshape(n_times, n_lat, n_lon)
    nan_count_t = int(np.sum(np.isnan(data_t)))
    if nan_count_t == 0:
        ok("No NaN values")
    else:
        fail(f"{nan_count_t} NaN values")
        errors += 1

    vmin_t = float(np.nanmin(data_t))
    vmax_t = float(np.nanmax(data_t))
    info(f"Value range: [{vmin_t:.2f}, {vmax_t:.2f}] °C")

    if vmin_t >= TEMP_MIN_C:
        ok(f"t2m min = {vmin_t:.2f}°C >= {TEMP_MIN_C}°C lower bound")
    else:
        fail(f"t2m min = {vmin_t:.2f}°C < {TEMP_MIN_C}°C lower bound")
        errors += 1

    if vmax_t <= TEMP_MAX_C:
        ok(f"t2m max = {vmax_t:.2f}°C <= {TEMP_MAX_C}°C upper bound")
    else:
        fail(f"t2m max = {vmax_t:.2f}°C > {TEMP_MAX_C}°C upper bound")
        errors += 1

    # --- Variable metadata ---
    print("\n[Variable metadata — precipitation]")
    variables = meta.get("variables", {})

    if "precipitation" not in variables:
        fail("Missing 'precipitation' variable metadata")
        errors += 1
    else:
        vmeta = variables["precipitation"]
        required_var_fields = [
            "display_name", "units", "source_variable",
            "temporal_resolution", "temporal_semantics",
            "temporal_disclosure", "transformation_provenance",
            "native_output", "file", "fill_value",
        ]
        for field in required_var_fields:
            if field in vmeta:
                ok(f"precipitation.{field}: {str(vmeta[field])[:60]}")
            else:
                fail(f"precipitation.{field} MISSING")
                errors += 1

        units = vmeta.get("units", "")
        if "6" in units and ("mm" in units.lower() or "millim" in units.lower()):
            ok(f"units = '{units}' (contains '6' and 'mm')")
        else:
            fail(f"units = '{units}' does not match mm/6h contract")
            errors += 1

        prov = vmeta.get("transformation_provenance", {})
        if prov.get("log_transform_applied") is False:
            ok("log_transform_applied = false")
        else:
            fail(f"log_transform_applied = {prov.get('log_transform_applied')} (must be false)")
            errors += 1

        if prov.get("exp_transform_applied") is False:
            ok("exp_transform_applied = false")
        else:
            fail(f"exp_transform_applied = {prov.get('exp_transform_applied')} (must be false)")
            errors += 1

        src_var = prov.get("source_variable", "")
        if src_var == "tp06":
            ok(f"source_variable = 'tp06'")
        else:
            fail(f"source_variable = '{src_var}' (expected 'tp06')")
            errors += 1

    print("\n[Variable metadata — temperature]")
    if "temperature" not in variables:
        fail("Missing 'temperature' variable metadata")
        errors += 1
    else:
        tmeta = variables["temperature"]
        required_temp_fields = [
            "display_name", "units", "source_variable",
            "temporal_resolution", "temporal_semantics",
            "transformation_provenance", "native_output", "file", "fill_value",
        ]
        for field in required_temp_fields:
            if field in tmeta:
                ok(f"temperature.{field}: {str(tmeta[field])[:60]}")
            else:
                fail(f"temperature.{field} MISSING")
                errors += 1

        t_units = tmeta.get("units", "")
        if "°C" in t_units or "C" in t_units:
            ok(f"units = '{t_units}' (°C)")
        else:
            fail(f"units = '{t_units}' does not indicate °C")
            errors += 1

        t_src = tmeta.get("source_variable", "")
        if t_src == "t2m":
            ok(f"source_variable = 't2m'")
        else:
            fail(f"source_variable = '{t_src}' (expected 't2m')")
            errors += 1

        t_prov = tmeta.get("transformation_provenance", {})
        if t_prov.get("log_transform_applied") is False:
            ok("log_transform_applied = false")
        else:
            fail(f"log_transform_applied = {t_prov.get('log_transform_applied')} (must be false)")
            errors += 1

    # --- is_demo flag ---
    print("\n[Demo flag]")
    is_demo = meta.get("is_demo")
    if isinstance(is_demo, bool):
        ok(f"is_demo = {is_demo}")
    else:
        fail(f"is_demo = {is_demo!r} (must be bool)")
        errors += 1

    # --- Summary ---
    print("\n" + "=" * 60)
    if errors == 0:
        print(f"VALIDATION PASSED — all checks passed for {path}")
    else:
        print(f"VALIDATION FAILED — {errors} error(s) in {path}")
    print("=" * 60)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate GraphCastSmall forecast artifacts (schema v3.0)"
    )
    parser.add_argument("--data-dir", required=True, help="Path to forecast artifact directory")
    args = parser.parse_args()
    return min(validate(args.data_dir), 1)


if __name__ == "__main__":
    sys.exit(main())
