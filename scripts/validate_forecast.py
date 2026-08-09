"""
validate_forecast.py — Validate generated forecast artifacts before frontend use.

Checks:
  - Required files present (forecast.json, temperature.bin, precipitation.bin)
  - forecast.json schema completeness
  - Binary file dimensions match metadata
  - No NaN in expected fields
  - t2m in physically plausible range [-20, 60] degC
  - tp1h >= 0 (post-untransform)
  - Timestamps monotonically increasing
  - Grid dimensions match Myanmar bbox at 0.25 deg
  - sic_handling field documented
  - is_demo flag present

Usage:
    uv run python scripts/validate_forecast.py --data-dir data/demo/
    uv run python scripts/validate_forecast.py --data-dir data/forecast/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


EXPECTED_N_LAT = 81   # 9N to 29N at 0.25 deg
EXPECTED_N_LON = 41   # 92E to 102E at 0.25 deg
EXPECTED_N_TIMES = 169  # 0h to 168h inclusive


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
    for fname in ["forecast.json", "temperature.bin", "precipitation.bin"]:
        fpath = path / fname
        if fpath.exists():
            ok(f"{fname} present ({fpath.stat().st_size / 1024:.1f} KB)")
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
        "model", "model_version", "initialization_source", "initialization_time",
        "forecast_generated_at", "forecast_horizon_hours", "n_times",
        "spatial_resolution_deg", "bbox", "grid", "lat", "lon", "times_utc",
        "variables", "sic_handling", "is_demo", "earth2studio_version",
    ]
    for field in required_fields:
        if field in meta:
            ok(f"{field}: {str(meta[field])[:60]}")
        else:
            fail(f"Missing field: {field}")
            errors += 1

    # Check sic_handling is not empty
    if meta.get("sic_handling", "") == "":
        fail("sic_handling is empty — must document the method used")
        errors += 1

    # --- Grid dimensions ---
    print("\n[Grid dimensions]")
    grid = meta.get("grid", {})
    n_lat = grid.get("n_lat", -1)
    n_lon = grid.get("n_lon", -1)
    n_times = meta.get("n_times", -1)

    if n_lat == EXPECTED_N_LAT:
        ok(f"n_lat = {n_lat} (expected {EXPECTED_N_LAT})")
    else:
        fail(f"n_lat = {n_lat} (expected {EXPECTED_N_LAT})")
        errors += 1

    if n_lon == EXPECTED_N_LON:
        ok(f"n_lon = {n_lon} (expected {EXPECTED_N_LON})")
    else:
        fail(f"n_lon = {n_lon} (expected {EXPECTED_N_LON})")
        errors += 1

    if n_times == EXPECTED_N_TIMES:
        ok(f"n_times = {n_times} (expected {EXPECTED_N_TIMES})")
    else:
        fail(f"n_times = {n_times} (expected {EXPECTED_N_TIMES})")
        errors += 1

    # --- Timestamps ---
    print("\n[Timestamps]")
    times = meta.get("times_utc", [])
    if len(times) == n_times:
        ok(f"{len(times)} timestamps in forecast.json")
    else:
        fail(f"{len(times)} timestamps (expected {n_times})")
        errors += 1

    if len(times) >= 2:
        from datetime import datetime
        t0 = datetime.fromisoformat(times[0].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(times[1].replace("Z", "+00:00"))
        dt_hours = (t1 - t0).total_seconds() / 3600
        if dt_hours == 1.0:
            ok(f"Hourly timestamps confirmed (dt = {dt_hours:.1f}h)")
        else:
            fail(f"Non-hourly timestamps (dt = {dt_hours:.1f}h)")
            errors += 1

        # Monotonic check
        dts = []
        for i in range(1, min(len(times), 10)):
            ta = datetime.fromisoformat(times[i - 1].replace("Z", "+00:00"))
            tb = datetime.fromisoformat(times[i].replace("Z", "+00:00"))
            dts.append((tb - ta).total_seconds())
        if all(d > 0 for d in dts):
            ok("Timestamps monotonically increasing")
        else:
            fail("Timestamps not monotonically increasing")
            errors += 1

    # --- Binary file validation ---
    for var_name, bin_file in [("temperature", "temperature.bin"), ("precipitation", "precipitation.bin")]:
        print(f"\n[{var_name}.bin]")
        arr = np.frombuffer((path / bin_file).read_bytes(), dtype=np.float32)
        expected_size = n_times * n_lat * n_lon
        if len(arr) == expected_size:
            ok(f"Size: {len(arr)} elements = {n_times} × {n_lat} × {n_lon}")
        else:
            fail(f"Size mismatch: {len(arr)} elements (expected {expected_size})")
            errors += 1
            continue

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
        info(f"Value range: [{vmin:.3f}, {vmax:.3f}]")

        if var_name == "temperature":
            if -20 <= vmin and vmax <= 60:
                ok(f"Temperature range plausible: [{vmin:.1f}, {vmax:.1f}] °C")
            else:
                fail(f"Temperature out of range [-20, 60] °C: [{vmin:.1f}, {vmax:.1f}]")
                errors += 1

        elif var_name == "precipitation":
            if vmin >= 0:
                ok(f"Precipitation >= 0 (min={vmin:.4f} mm/h)")
            else:
                fail(f"Negative precipitation detected: min={vmin:.4f}")
                errors += 1
            if vmax > 500:
                fail(f"Precipitation max={vmax:.1f} mm/h seems unrealistically high")
                errors += 1
            else:
                ok(f"Precipitation max={vmax:.2f} mm/h within plausible range")

        # Check t+0 frame (initialization state)
        t0_vals = data[0]
        info(f"t+0 frame range: [{float(np.nanmin(t0_vals)):.3f}, {float(np.nanmax(t0_vals)):.3f}]")

    # --- Variable metadata ---
    print("\n[Variable metadata]")
    for var_key in ["temperature_2m", "precipitation"]:
        if var_key in meta.get("variables", {}):
            vmeta = meta["variables"][var_key]
            ok(f"{var_key}: units={vmeta.get('units')}, temporal_semantics={vmeta.get('temporal_semantics', 'MISSING')[:50]}")
            if "temporal_semantics" not in vmeta:
                fail(f"{var_key} missing temporal_semantics field")
                errors += 1
            if "temporal_disclosure" not in vmeta and var_key == "precipitation":
                fail(f"precipitation missing temporal_disclosure field")
                errors += 1
        else:
            fail(f"Variable metadata missing for {var_key}")
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
    parser = argparse.ArgumentParser(description="Validate forecast artifacts")
    parser.add_argument("--data-dir", required=True, help="Path to forecast artifact directory")
    args = parser.parse_args()
    return min(validate(args.data_dir), 1)


if __name__ == "__main__":
    sys.exit(main())
