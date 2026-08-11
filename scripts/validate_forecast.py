"""
validate_forecast.py — Validate generated forecast artifacts before frontend use.

Checks:
  - Required files present (forecast.json, precipitation.bin)
  - forecast.json schema completeness (schema v2.0)
  - Binary file dimensions match metadata [n_times × n_lat × n_lon]
  - No NaN values
  - tp06 >= 0 (physically impossible to have negative precipitation)
  - tp06 < 500 mm/6h (extreme upper bound for tropical rainfall)
  - Timestamps 6h apart and monotonically increasing
  - Grid dimensions match Myanmar bbox at 1.0 deg
  - n_times == 5, n_lat == 21, n_lon == 11
  - native_timestep_hours == 6, forecast_horizon_hours == 24
  - transformation_provenance.log_transform_applied == false
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


EXPECTED_N_LAT = 21    # 9N to 29N at 1.0 deg
EXPECTED_N_LON = 11    # 92E to 102E at 1.0 deg
EXPECTED_N_TIMES = 5   # t+0h, t+6h, t+12h, t+18h, t+24h
EXPECTED_TIMESTEP_H = 6
EXPECTED_HORIZON_H = 24
PRECIP_MAX_THRESHOLD = 500.0  # mm/6h — physical upper bound for tropical rainfall


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
    for fname in ["forecast.json", "precipitation.bin"]:
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

    # schema_version check
    sv = meta.get("schema_version", "")
    if sv == "2.0":
        ok(f"schema_version == '2.0'")
    else:
        fail(f"schema_version '{sv}' != '2.0'")
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
    horizon_h = meta.get("forecast_horizon_hours", -1)

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

    # spatial resolution
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

        # Uniform 6h spacing
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
                fail(f"Gap at index {idx - 1}→{idx}: {gap_h:.1f}h (expected {EXPECTED_TIMESTEP_H}h)")
                errors += 1

        # Monotonic check
        if all(t_parsed[i] > t_parsed[i - 1] for i in range(1, len(t_parsed))):
            ok("Timestamps monotonically increasing")
        else:
            fail("Timestamps not monotonically increasing")
            errors += 1

    # --- Binary file ---
    print("\n[precipitation.bin]")
    arr = np.frombuffer((path / "precipitation.bin").read_bytes(), dtype="<f4")
    expected_size = n_times * n_lat * n_lon
    expected_bytes = expected_size * 4

    if len(arr) == expected_size:
        ok(f"Size: {len(arr)} elements = {n_times} × {n_lat} × {n_lon} ({expected_bytes} bytes)")
    else:
        fail(f"Size mismatch: {len(arr)} elements (expected {expected_size} = {expected_bytes} bytes)")
        errors += 1
        # Cannot reshape — bail early
        print("\n" + "=" * 60)
        print(f"VALIDATION FAILED — {errors} error(s) in {path}")
        print("=" * 60)
        return errors

    data = arr.reshape(n_times, n_lat, n_lon)

    # NaN check
    nan_count = int(np.sum(np.isnan(data)))
    if nan_count == 0:
        ok("No NaN values")
    else:
        fail(f"{nan_count} NaN values detected")
        errors += 1

    # Range checks
    vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
    info(f"Value range: [{vmin:.3f}, {vmax:.3f}] mm/6h")

    if vmin >= 0:
        ok(f"tp06 >= 0 (min = {vmin:.4f} mm/6h)")
    else:
        fail(f"Negative precipitation detected: min = {vmin:.4f} mm/6h")
        errors += 1

    if vmax < PRECIP_MAX_THRESHOLD:
        ok(f"tp06 max = {vmax:.2f} mm/6h (< {PRECIP_MAX_THRESHOLD:.0f} threshold)")
    else:
        fail(f"tp06 max = {vmax:.1f} mm/6h exceeds threshold {PRECIP_MAX_THRESHOLD:.0f}")
        errors += 1

    # t+0h frame (should be 0 for both demo and production init frame)
    t0_max = float(np.nanmax(np.abs(data[0])))
    info(f"t+0h frame: max abs = {t0_max:.4f} mm/6h (expected 0.0 for init frame)")

    # --- Variable metadata ---
    print("\n[Variable metadata]")
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

        # units check
        units = vmeta.get("units", "")
        if "6" in units and ("mm" in units.lower() or "millim" in units.lower()):
            ok(f"units = '{units}' (contains '6' and 'mm')")
        else:
            fail(f"units = '{units}' does not match mm/6h contract")
            errors += 1

        # transformation_provenance checks
        prov = vmeta.get("transformation_provenance", {})
        if prov.get("log_transform_applied") is False:
            ok("log_transform_applied = false")
        else:
            fail(f"log_transform_applied = {prov.get('log_transform_applied')} (must be false for tp06)")
            errors += 1

        if prov.get("exp_transform_applied") is False:
            ok("exp_transform_applied = false")
        else:
            fail(f"exp_transform_applied = {prov.get('exp_transform_applied')} (must be false for tp06)")
            errors += 1

        src_var = prov.get("source_variable", "")
        if src_var == "tp06":
            ok(f"source_variable = '{src_var}'")
        else:
            fail(f"source_variable = '{src_var}' (expected 'tp06')")
            errors += 1

    # is_demo check
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
    parser = argparse.ArgumentParser(description="Validate GraphCastSmall forecast artifacts (schema v2.0)")
    parser.add_argument("--data-dir", required=True, help="Path to forecast artifact directory")
    args = parser.parse_args()
    return min(validate(args.data_dir), 1)


if __name__ == "__main__":
    sys.exit(main())
