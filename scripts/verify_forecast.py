"""
verify_forecast.py — GraphCastOperational Myanmar 7-day forecast verification vs ERA5.
Schema v2.0. Phase R5.

ERA5/ARCO precipitation confirmed as 1-hour accumulation ending at each timestamp
(empirically verified 2026-08-16). Six consecutive hourly values summed per 6h GCOp
window. No cumulative subtraction. No 00Z/12Z seam handling.

Usage:
    uv run python scripts/verify_forecast.py \\
        [--forecast-dir data/forecast_v4] \\
        [--output-dir data/verification]

Exit codes:
    0 — all checks pass; verification.json written to output-dir
    1 — any validation failure; verification.json NOT written
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ARCO_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
EXPECTED_BYTES = 385_236           # 29 × 81 × 41 × 4
PRECIP_THRESHOLD_MM_HR = 0.1      # rain/no-rain categorical threshold
CALM_THRESHOLD_KT = 2.0           # exclude calm ERA5 points from wind-dir MAE
GRID_TOL = 0.001                  # degrees: lat/lon alignment tolerance


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def _valid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ~(np.isnan(a) | np.isnan(b))


def calc_mae(fcst: np.ndarray, ref: np.ndarray) -> float:
    m = _valid(fcst, ref)
    return float(np.mean(np.abs(fcst[m] - ref[m])))


def calc_rmse(fcst: np.ndarray, ref: np.ndarray) -> float:
    m = _valid(fcst, ref)
    return float(np.sqrt(np.mean((fcst[m] - ref[m]) ** 2)))


def calc_bias(fcst: np.ndarray, ref: np.ndarray) -> float:
    m = _valid(fcst, ref)
    return float(np.mean(fcst[m] - ref[m]))


def calc_circular_mae(
    fcst_dir: np.ndarray,
    ref_dir: np.ndarray,
    calm_mask: np.ndarray,
) -> tuple[float, int, int]:
    """
    Circular MAE for wind direction (degrees).
    Excludes points where ERA5 wind speed < CALM_THRESHOLD_KT (calm_mask=True).
    Returns (circular_mae_deg, n_active, n_calm_excluded).
    """
    active = ~calm_mask & ~np.isnan(fcst_dir) & ~np.isnan(ref_dir)
    n_calm = int(calm_mask.sum())
    n_active = int(active.sum())
    if n_active == 0:
        return float("nan"), 0, n_calm
    diff = fcst_dir[active] - ref_dir[active]
    diff = ((diff + 180.0) % 360.0) - 180.0  # wrap to [-180, +180]
    return float(np.mean(np.abs(diff))), n_active, n_calm


def contingency_counts(
    fcst: np.ndarray,
    ref: np.ndarray,
    threshold: float,
) -> tuple[int, int, int]:
    """Returns (hits, misses, false_alarms) for exceedance of threshold."""
    m = _valid(fcst, ref)
    f = fcst[m] >= threshold
    o = ref[m] >= threshold
    hits        = int(np.sum(f & o))
    misses      = int(np.sum(~f & o))
    false_alarms = int(np.sum(f & ~o))
    return hits, misses, false_alarms


def pod_far_csi(
    hits: int, misses: int, false_alarms: int
) -> tuple[float, float, float]:
    pod = hits / (hits + misses) if (hits + misses) > 0 else float("nan")
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else float("nan")
    denom = hits + misses + false_alarms
    csi = hits / denom if denom > 0 else float("nan")
    return pod, far, csi


# ---------------------------------------------------------------------------
# ARCO fetch
# ---------------------------------------------------------------------------

def _open_arco():
    import gcsfs
    import xarray as xr
    fs = gcsfs.GCSFileSystem(token="anon")
    store = gcsfs.mapping.GCSMap(ARCO_ZARR, gcs=fs)
    return xr.open_zarr(store, chunks=None)


def _myanmar_indices(ds):
    """
    Returns (lat_slice, lon_slice, lat_asc, lon_asc) for the Myanmar bbox.
    ARCO lat is descending (90→-90); selection is flipped to ascending after load.
    """
    lats_full = ds.latitude.values   # descending
    lons_full = ds.longitude.values  # ascending

    lat_29_idx = int(np.argmin(np.abs(lats_full - 29.0)))
    lat_9_idx  = int(np.argmin(np.abs(lats_full - 9.0)))
    lat_slice  = slice(lat_29_idx, lat_9_idx + 1)  # 29°N → 9°N in descending array

    lon_92_idx  = int(np.argmin(np.abs(lons_full - 92.0)))
    lon_102_idx = int(np.argmin(np.abs(lons_full - 102.0)))
    lon_slice   = slice(lon_92_idx, lon_102_idx + 1)

    lat_asc = lats_full[lat_slice][::-1]   # flip to ascending: 9°N → 29°N
    lon_asc = lons_full[lon_slice]          # already ascending: 92°E → 102°E
    return lat_slice, lon_slice, lat_asc, lon_asc


def fetch_era5_6h_vars(ds, init_dt: datetime, lat_slice, lon_slice):
    """
    Fetch 2m_temperature, 10m_u_component_of_wind, 10m_v_component_of_wind
    at 29 × 6-hourly timestamps aligned to the GCOp forecast frames.
    Returns dict {'t2m': [29,81,41], 'u10m': [29,81,41], 'v10m': [29,81,41]} in
    native units (K, m/s, m/s), lat in ascending order.
    """
    time_vals = ds.time.values

    # Build 29 timestamps: t+0h, t+6h, ..., t+168h
    ts_np = np.array([
        np.datetime64((init_dt + timedelta(hours=6 * i)).replace(tzinfo=None).isoformat(), "ns")
        for i in range(29)
    ])
    t_indices = [int(np.argmin(np.abs(time_vals - ts))) for ts in ts_np]

    # Verify timestamps matched correctly (within 1 minute)
    for i, (ts, ti) in enumerate(zip(ts_np, t_indices)):
        matched = time_vals[ti]
        diff_ns = abs(int(matched) - int(ts))
        if diff_ns > 60 * 1e9:  # 60 seconds in nanoseconds
            raise ValueError(
                f"ARCO time mismatch at frame {i}: "
                f"wanted {ts}, got {matched} (diff {diff_ns/1e9:.0f}s)"
            )

    var_map = {
        "t2m":  "2m_temperature",
        "u10m": "10m_u_component_of_wind",
        "v10m": "10m_v_component_of_wind",
    }
    result = {}
    for short, arco_name in var_map.items():
        print(f"    Fetching ARCO {short} ({arco_name}, 29 timestamps)...", flush=True)
        arr = ds[arco_name].isel(
            latitude=lat_slice, longitude=lon_slice, time=t_indices
        ).load().values  # [29, n_lat_desc, n_lon]
        arr = arr[:, ::-1, :]  # flip lat to ascending
        result[short] = arr.astype(np.float64)

    return result


def fetch_era5_hourly_tp(ds, init_dt: datetime, lat_slice, lon_slice):
    """
    Fetch total_precipitation hourly from t+1h through t+168h (168 contiguous values).
    Returns ndarray [168, 81, 41] in raw metres (1h accumulation), lat ascending.
    """
    time_vals = ds.time.values
    t_start_np = np.datetime64(
        (init_dt + timedelta(hours=1)).replace(tzinfo=None).isoformat(), "ns"
    )
    t_end_np = np.datetime64(
        (init_dt + timedelta(hours=168)).replace(tzinfo=None).isoformat(), "ns"
    )

    mask = (time_vals >= t_start_np) & (time_vals <= t_end_np)
    t_indices = np.where(mask)[0]

    if len(t_indices) != 168:
        raise ValueError(
            f"Expected 168 consecutive hourly tp timestamps; got {len(t_indices)}. "
            "ARCO coverage may differ from expected — stopping as required."
        )

    # Verify start and end timestamps
    actual_start = time_vals[t_indices[0]]
    actual_end   = time_vals[t_indices[-1]]
    print(
        f"    Fetching ARCO total_precipitation "
        f"({actual_start!s:.16} → {actual_end!s:.16}, 168 h)...",
        flush=True,
    )

    arr = ds["total_precipitation"].isel(
        latitude=lat_slice, longitude=lon_slice, time=t_indices
    ).load().values  # [168, n_lat_desc, n_lon]
    arr = arr[:, ::-1, :]  # flip lat to ascending
    return arr.astype(np.float64)


def build_era5_precip_6h(tp_hourly: np.ndarray) -> np.ndarray:
    """
    Aggregate 168 hourly ERA5 tp values into 28 six-hour windows (mm/hr).

    Locked mapping:
        GCOp +6h   ← ERA5 t+1h … t+6h   → tp_hourly[0:6]
        GCOp +12h  ← ERA5 t+7h … t+12h  → tp_hourly[6:12]
        ...
        GCOp +168h ← ERA5 t+163h…t+168h → tp_hourly[162:168]

    Per-hour clamp ≥ 0 before sum; aggregate clamp ≥ 0 after sum.
    Conversion: sum_6h_metres × 1000 / 6 → mm/hr.
    """
    n_windows = 28
    result = np.zeros((n_windows,) + tp_hourly.shape[1:], dtype=np.float64)
    for k in range(n_windows):
        window     = tp_hourly[6 * k : 6 * k + 6]       # [6, 81, 41] metres
        clamped    = np.maximum(window, 0.0)              # per-hour clamp ≥ 0
        sum_6h_m   = np.sum(clamped, axis=0)             # [81, 41] metres
        sum_6h_m   = np.maximum(sum_6h_m, 0.0)          # aggregate clamp ≥ 0
        result[k]  = sum_6h_m * 1000.0 / 6.0            # → mm/hr
    return result


# ---------------------------------------------------------------------------
# Main verification logic
# ---------------------------------------------------------------------------

def run_verification(forecast_dir: Path, output_dir: Path) -> int:
    print("=" * 70)
    print("GraphCastOperational Myanmar 7-day Forecast Verification vs ERA5")
    print("Phase R5  |  verification.json schema v2.0")
    print("=" * 70)

    # ── Pre-computation checks ──────────────────────────────────────────────
    print("\n[Pre-computation checks]")

    meta_path = forecast_dir / "forecast.json"
    bin_paths = {
        "temperature":    forecast_dir / "temperature.bin",
        "precipitation":  forecast_dir / "precipitation.bin",
        "wind_speed":     forecast_dir / "wind_speed.bin",
        "wind_direction": forecast_dir / "wind_direction.bin",
    }

    if not meta_path.exists():
        print(f"  [FAIL] {meta_path} not found")
        return 1
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  [PASS] forecast.json readable")
    except Exception as e:
        print(f"  [FAIL] forecast.json parse error: {e}")
        return 1

    if meta.get("is_demo", True):
        print("  [FAIL] is_demo=True — verification requires real forecast data")
        return 1
    print("  [PASS] is_demo=False")

    for varname, path in bin_paths.items():
        if not path.exists():
            print(f"  [FAIL] {path.name} not found")
            return 1
        sz = path.stat().st_size
        if sz != EXPECTED_BYTES:
            print(f"  [FAIL] {path.name}: {sz:,} bytes (expected {EXPECTED_BYTES:,})")
            return 1
        print(f"  [PASS] {path.name}: {sz:,} bytes")

    # Extract metadata dynamically
    n_frames  = meta["n_frames"]
    n_lat     = meta["grid"]["n_lat"]
    n_lon     = meta["grid"]["n_lon"]
    step_h    = meta["native_timestep_hours"]
    horizon   = meta["forecast_horizon_hours"]
    lats      = np.array(meta["lat"], dtype=np.float64)
    lons      = np.array(meta["lon"], dtype=np.float64)
    init_str  = meta["initialization_time"]
    init_dt   = datetime.fromisoformat(init_str.replace("Z", "+00:00"))

    if n_frames != 29 or n_lat != 81 or n_lon != 41:
        print(
            f"  [FAIL] Unexpected dims: n_frames={n_frames}, "
            f"n_lat={n_lat}, n_lon={n_lon} (expected 29/81/41)"
        )
        return 1
    print(f"  [PASS] Grid: {n_frames} frames × {n_lat}×{n_lon} @ 0.25°")
    print(f"  [INFO] Model: {meta.get('model')} | Init: {init_str} | Horizon: {horizon}h")
    print(f"  [INFO] forecast.json schema_version: {meta.get('schema_version')}")

    # Load forecast arrays [n_frames, n_lat, n_lon]
    shape = (n_frames, n_lat, n_lon)

    def load_bin(p: Path) -> np.ndarray:
        return np.frombuffer(p.read_bytes(), dtype="<f4").reshape(shape).astype(np.float64)

    fcst_temp   = load_bin(bin_paths["temperature"])    # °C
    fcst_precip = load_bin(bin_paths["precipitation"])  # mm/hr
    fcst_wspeed = load_bin(bin_paths["wind_speed"])     # kt
    fcst_wdir   = load_bin(bin_paths["wind_direction"]) # °FROM

    # ── ERA5 fetch ──────────────────────────────────────────────────────────
    print(f"\n[ERA5 fetch]")
    print(f"  Dataset: {ARCO_ZARR}")

    ds = _open_arco()
    lat_slice, lon_slice, era5_lat, era5_lon = _myanmar_indices(ds)

    print(f"  lat: {era5_lat[0]:.2f}°N → {era5_lat[-1]:.2f}°N ({len(era5_lat)} pts)")
    print(f"  lon: {era5_lon[0]:.2f}°E → {era5_lon[-1]:.2f}°E ({len(era5_lon)} pts)")

    # Grid alignment check — stop if misaligned
    lat_err = float(np.max(np.abs(era5_lat - lats)))
    lon_err = float(np.max(np.abs(era5_lon - lons)))
    if lat_err > GRID_TOL:
        print(f"  [FAIL] Lat grid mismatch: max error {lat_err:.5f}° > {GRID_TOL}°")
        return 1
    if lon_err > GRID_TOL:
        print(f"  [FAIL] Lon grid mismatch: max error {lon_err:.5f}° > {GRID_TOL}°")
        return 1
    print(f"  [PASS] Grid aligned: lat_err≤{lat_err:.6f}°, lon_err≤{lon_err:.6f}°")

    # Fetch 6-hourly vars (t2m, u10m, v10m)
    era5_6h = fetch_era5_6h_vars(ds, init_dt, lat_slice, lon_slice)
    era5_t2m_K = era5_6h["t2m"]   # [29, 81, 41] K
    era5_u10m  = era5_6h["u10m"]  # [29, 81, 41] m/s
    era5_v10m  = era5_6h["v10m"]  # [29, 81, 41] m/s

    # Fetch hourly tp [168, 81, 41] metres
    tp_hourly = fetch_era5_hourly_tp(ds, init_dt, lat_slice, lon_slice)

    # Report tp raw stats
    n_neg = int((tp_hourly < 0).sum())
    print(
        f"  [INFO] ERA5 tp hourly: {n_neg}/{tp_hourly.size} negative "
        f"({n_neg/tp_hourly.size*100:.2f}%) — spectral artifacts, clamped to 0"
    )
    print(
        f"  [INFO] ERA5 tp range: "
        f"min={tp_hourly.min():.8f} m, max={tp_hourly.max():.6f} m"
    )

    # ERA5 sanity checks — stop if out of range
    t2m_min, t2m_max = float(era5_t2m_K.min()), float(era5_t2m_K.max())
    if t2m_min < 200 or t2m_max > 335:
        print(f"  [FAIL] ERA5 t2m out of plausible range: [{t2m_min:.1f}, {t2m_max:.1f}] K")
        return 1
    print(f"  [PASS] ERA5 t2m: [{t2m_min:.1f}, {t2m_max:.1f}] K")

    u_max = float(np.abs(era5_u10m).max())
    v_max = float(np.abs(era5_v10m).max())
    if u_max > 150 or v_max > 150:
        print(f"  [FAIL] ERA5 wind out of range: |u|≤{u_max:.1f} m/s, |v|≤{v_max:.1f} m/s")
        return 1
    print(f"  [PASS] ERA5 wind: |u|≤{u_max:.2f} m/s, |v|≤{v_max:.2f} m/s")

    # ── Transformations ─────────────────────────────────────────────────────
    print("\n[Transformations]")

    # Temperature: K → °C
    era5_temp = era5_t2m_K - 273.15  # [29, 81, 41]

    # Wind speed: m/s → kt
    era5_wspeed = np.sqrt(era5_u10m**2 + era5_v10m**2) * 1.94384  # [29, 81, 41]

    # Wind direction: meteorological FROM (°)
    # (atan2(-u, -v) + 360) % 360  — equivalent to (270 - atan2d(v, u)) % 360
    era5_wdir = (np.degrees(np.arctan2(-era5_u10m, -era5_v10m)) + 360.0) % 360.0

    # Calm mask: exclude where ERA5 speed < CALM_THRESHOLD_KT
    era5_calm = era5_wspeed < CALM_THRESHOLD_KT  # [29, 81, 41] bool

    # Precipitation: hourly → 28 × 6h windows → mm/hr
    era5_precip_28 = build_era5_precip_6h(tp_hourly)  # [28, 81, 41]

    print(f"  [INFO] ERA5 temp:   [{era5_temp.min():.2f}, {era5_temp.max():.2f}] °C")
    print(f"  [INFO] ERA5 speed:  [{era5_wspeed.min():.3f}, {era5_wspeed.max():.3f}] kt")
    print(f"  [INFO] ERA5 precip: [{era5_precip_28.min():.4f}, {era5_precip_28.max():.4f}] mm/hr (28 windows)")

    # ── Metrics ─────────────────────────────────────────────────────────────
    print("\n[Computing metrics]")

    temp_by_lt   = []
    wspeed_by_lt = []
    wdir_by_lt   = []
    precip_by_lt = []

    # Contingency count accumulators for summary POD/FAR/CSI
    total_hits, total_misses, total_fa = 0, 0, 0

    for i in range(n_frames):
        lead_h     = i * step_h
        valid_time = (init_dt + timedelta(hours=lead_h)).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Temperature — all 29 frames
        mae_t  = calc_mae(fcst_temp[i], era5_temp[i])
        rmse_t = calc_rmse(fcst_temp[i], era5_temp[i])
        bias_t = calc_bias(fcst_temp[i], era5_temp[i])
        temp_by_lt.append({
            "lead_time_hours": lead_h,
            "valid_time": valid_time,
            "mae":  round(mae_t,  4),
            "rmse": round(rmse_t, 4),
            "bias": round(bias_t, 4),
            "n_points": n_lat * n_lon,
        })

        # Wind speed — all 29 frames
        mae_ws  = calc_mae(fcst_wspeed[i], era5_wspeed[i])
        rmse_ws = calc_rmse(fcst_wspeed[i], era5_wspeed[i])
        bias_ws = calc_bias(fcst_wspeed[i], era5_wspeed[i])
        wspeed_by_lt.append({
            "lead_time_hours": lead_h,
            "valid_time": valid_time,
            "mae":  round(mae_ws,  4),
            "rmse": round(rmse_ws, 4),
            "bias": round(bias_ws, 4),
            "n_points": n_lat * n_lon,
        })

        # Wind direction — all 29 frames; calm points excluded
        c_mae, n_act, n_calm = calc_circular_mae(
            fcst_wdir[i], era5_wdir[i], era5_calm[i]
        )
        wdir_by_lt.append({
            "lead_time_hours": lead_h,
            "valid_time": valid_time,
            "circular_mae": round(c_mae, 4) if not np.isnan(c_mae) else None,
            "n_points_active": n_act,
            "n_points_calm_excluded": n_calm,
        })

        # Precipitation — frames 1..28 only (t+6h → t+168h); t+0h excluded
        if i > 0:
            p_idx = i - 1   # index into era5_precip_28 [0..27]
            mae_p  = calc_mae(fcst_precip[i], era5_precip_28[p_idx])
            rmse_p = calc_rmse(fcst_precip[i], era5_precip_28[p_idx])
            bias_p = calc_bias(fcst_precip[i], era5_precip_28[p_idx])
            h, m, fa = contingency_counts(
                fcst_precip[i], era5_precip_28[p_idx], PRECIP_THRESHOLD_MM_HR
            )
            pod_lt, far_lt, csi_lt = pod_far_csi(h, m, fa)
            total_hits   += h
            total_misses += m
            total_fa     += fa
            precip_by_lt.append({
                "lead_time_hours": lead_h,
                "valid_time": valid_time,
                "mae":  round(mae_p,  4),
                "rmse": round(rmse_p, 4),
                "bias": round(bias_p, 4),
                "pod": round(pod_lt, 4) if not np.isnan(pod_lt) else None,
                "far": round(far_lt, 4) if not np.isnan(far_lt) else None,
                "csi": round(csi_lt, 4) if not np.isnan(csi_lt) else None,
                "n_hits": h,
                "n_misses": m,
                "n_false_alarms": fa,
                "n_points": n_lat * n_lon,
            })

    # ── Summary metrics ──────────────────────────────────────────────────────
    sum_temp = {
        "mae":      round(float(np.mean([r["mae"]  for r in temp_by_lt])), 4),
        "rmse":     round(float(np.mean([r["rmse"] for r in temp_by_lt])), 4),
        "bias":     round(float(np.mean([r["bias"] for r in temp_by_lt])), 4),
        "n_frames": n_frames,
    }
    sum_wspeed = {
        "mae":      round(float(np.mean([r["mae"]  for r in wspeed_by_lt])), 4),
        "rmse":     round(float(np.mean([r["rmse"] for r in wspeed_by_lt])), 4),
        "bias":     round(float(np.mean([r["bias"] for r in wspeed_by_lt])), 4),
        "n_frames": n_frames,
    }
    wdir_maes = [r["circular_mae"] for r in wdir_by_lt if r["circular_mae"] is not None]
    sum_wdir = {
        "circular_mae": round(float(np.mean(wdir_maes)), 4) if wdir_maes else None,
        "n_frames": n_frames,
    }
    # Summary POD/FAR/CSI from total contingency counts (not averaged per-frame ratios)
    sum_pod, sum_far, sum_csi = pod_far_csi(total_hits, total_misses, total_fa)
    sum_precip = {
        "mae":               round(float(np.mean([r["mae"]  for r in precip_by_lt])), 4),
        "rmse":              round(float(np.mean([r["rmse"] for r in precip_by_lt])), 4),
        "bias":              round(float(np.mean([r["bias"] for r in precip_by_lt])), 4),
        "pod":               round(sum_pod, 4) if not np.isnan(sum_pod) else None,
        "far":               round(sum_far, 4) if not np.isnan(sum_far) else None,
        "csi":               round(sum_csi, 4) if not np.isnan(sum_csi) else None,
        "total_hits":        total_hits,
        "total_misses":      total_misses,
        "total_false_alarms": total_fa,
        "n_frames":          28,
    }

    # ── Post-computation checks ──────────────────────────────────────────────
    print("\n[Post-computation checks]")
    ok = True

    def chk(cond: bool, msg: str) -> None:
        nonlocal ok
        if not cond:
            print(f"  [FAIL] {msg}")
            ok = False
        else:
            print(f"  [PASS] {msg}")

    chk(sum_temp["mae"]  >= 0, f"Temperature MAE ≥ 0 ({sum_temp['mae']})")
    chk(sum_temp["rmse"] >= sum_temp["mae"],
        f"Temperature RMSE ≥ MAE ({sum_temp['rmse']} ≥ {sum_temp['mae']})")

    chk(sum_wspeed["mae"]  >= 0, f"Wind speed MAE ≥ 0 ({sum_wspeed['mae']})")
    chk(sum_wspeed["rmse"] >= sum_wspeed["mae"],
        f"Wind speed RMSE ≥ MAE ({sum_wspeed['rmse']} ≥ {sum_wspeed['mae']})")

    chk(sum_precip["mae"]  >= 0, f"Precipitation MAE ≥ 0 ({sum_precip['mae']})")
    chk(sum_precip["rmse"] >= sum_precip["mae"],
        f"Precipitation RMSE ≥ MAE ({sum_precip['rmse']} ≥ {sum_precip['mae']})")

    if sum_precip["pod"] is not None:
        chk(0.0 <= sum_precip["pod"] <= 1.0, f"POD ∈ [0,1] ({sum_precip['pod']})")
        chk(0.0 <= sum_precip["far"] <= 1.0, f"FAR ∈ [0,1] ({sum_precip['far']})")
        chk(0.0 <= sum_precip["csi"] <= 1.0, f"CSI ∈ [0,1] ({sum_precip['csi']})")

    if sum_wdir["circular_mae"] is not None:
        chk(0.0 <= sum_wdir["circular_mae"] <= 180.0,
            f"Circular MAE ∈ [0°,180°] ({sum_wdir['circular_mae']}°)")

    # Per-lead-time invariant check
    for r in temp_by_lt:
        if r["mae"] < 0 or r["rmse"] < r["mae"]:
            print(f"  [FAIL] Temperature invariant at lead +{r['lead_time_hours']}h")
            ok = False
    for r in wspeed_by_lt:
        if r["mae"] < 0 or r["rmse"] < r["mae"]:
            print(f"  [FAIL] Wind speed invariant at lead +{r['lead_time_hours']}h")
            ok = False
    for r in precip_by_lt:
        if r["mae"] < 0 or r["rmse"] < r["mae"]:
            print(f"  [FAIL] Precipitation invariant at lead +{r['lead_time_hours']}h")
            ok = False

    # NaN check
    all_scalar_metrics = (
        [r["mae"] for r in temp_by_lt]
        + [r["rmse"] for r in temp_by_lt]
        + [r["mae"] for r in wspeed_by_lt]
        + [r["mae"] for r in precip_by_lt]
    )
    if any(np.isnan(v) for v in all_scalar_metrics):
        print("  [FAIL] NaN found in metric values")
        ok = False
    else:
        print("  [PASS] No NaN in metric values")

    # Sanity warnings (non-fatal)
    if sum_temp["mae"] > 15.0:
        print(f"  [WARN] Temperature MAE {sum_temp['mae']:.4f}°C is unusually high")
    if sum_wspeed["mae"] > 50.0:
        print(f"  [WARN] Wind speed MAE {sum_wspeed['mae']:.4f} kt is unusually high")
    if sum_precip["mae"] > 50.0:
        print(f"  [WARN] Precipitation MAE {sum_precip['mae']:.4f} mm/hr is unusually high")

    if not ok:
        print("\n  Validation FAILED — verification.json NOT written")
        return 1

    print("  [PASS] All post-computation checks passed")

    # ── Build verification.json ──────────────────────────────────────────────
    verification = {
        "schema_version": "2.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_metadata": {
            "model":                    meta.get("model"),
            "model_checkpoint":         meta.get("model_checkpoint"),
            "init_time":                init_str,
            "forecast_schema_version":  meta.get("schema_version"),
            "earth2studio_version":     meta.get("earth2studio_version"),
            "n_times":                  n_frames,
            "native_timestep_hours":    step_h,
            "forecast_horizon_hours":   horizon,
            "spatial_resolution_deg":   meta.get("native_resolution_deg"),
            "lat_bbox":                 [float(lats[0]), float(lats[-1])],
            "lon_bbox":                 [float(lons[0]), float(lons[-1])],
            "grid_dims":                [n_lat, n_lon],
            "is_demo":                  meta.get("is_demo"),
        },
        "reference_data": {
            "source":               "ERA5 via ARCO",
            "dataset":              ARCO_ZARR,
            "type":                 "reanalysis",
            "period_start":         init_str,
            "period_end":           (init_dt + timedelta(hours=168)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "spatial_resolution_deg": 0.25,
            "n_points":             n_lat * n_lon,
            "tp_variable":          "total_precipitation",
            "tp_representation": (
                "1-hour accumulation ending at each valid timestamp. "
                "Empirically confirmed from ARCO data on 2026-08-16. "
                "No cumulative forecast-run semantics. No seam handling."
            ),
            "caveats": [
                "ERA5 is a numerical reanalysis product, not direct observations.",
                "GCOp was trained on ERA5; comparison may be optimistic relative to independent observations.",
                "Single forecast cycle — skill metrics are not statistically robust.",
                "GCOp and ERA5 use 0.25° grids; no spatial interpolation was performed.",
                "ERA5 precipitation aggregated from 6 consecutive 1-hour accumulations per 6h window.",
                "January 2021 is dry season in Myanmar; precipitation metrics reflect low-rain conditions.",
            ],
        },
        "precipitation_threshold_mm_hr": PRECIP_THRESHOLD_MM_HR,
        "wind_direction_calm_threshold_kt": CALM_THRESHOLD_KT,
        "variables": {
            "temperature": {
                "unit":              "°C",
                "n_frames_verified": n_frames,
                "by_lead_time":      temp_by_lt,
                "summary":           sum_temp,
            },
            "precipitation": {
                "unit":   "mm/hr",
                "note": (
                    "t+0h excluded by GCOp pipeline convention (tp06=0 at init frame). "
                    "28 frames verified: t+6h through t+168h."
                ),
                "n_frames_verified": 28,
                "by_lead_time":      precip_by_lt,
                "summary":           sum_precip,
            },
            "wind_speed": {
                "unit":              "kt",
                "n_frames_verified": n_frames,
                "by_lead_time":      wspeed_by_lt,
                "summary":           sum_wspeed,
            },
            "wind_direction": {
                "unit":    "degrees",
                "method":  "meteorological FROM: (atan2(-u, -v) + 360) % 360",
                "calm_exclusion": (
                    f"Points where ERA5 speed < {CALM_THRESHOLD_KT} kt "
                    "excluded from circular MAE"
                ),
                "n_frames_verified": n_frames,
                "by_lead_time":      wdir_by_lt,
                "summary":           sum_wdir,
            },
        },
    }

    # ── Write output — only if all checks passed ─────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    json_out = output_dir / "verification.json"
    with open(json_out, "w") as f:
        json.dump(verification, f, indent=2)
    print(f"\n  Wrote {json_out} ({json_out.stat().st_size:,} bytes)")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("VERIFICATION COMPLETE — exit 0")
    print(f"{'=' * 70}")
    wdir_str = (
        f"{sum_wdir['circular_mae']:.4f}°"
        if sum_wdir["circular_mae"] is not None
        else "N/A"
    )
    print(
        f"\n  Temperature (29 frames):\n"
        f"    MAE={sum_temp['mae']:.4f}°C  "
        f"RMSE={sum_temp['rmse']:.4f}°C  "
        f"bias={sum_temp['bias']:+.4f}°C\n"
        f"\n  Wind speed (29 frames):\n"
        f"    MAE={sum_wspeed['mae']:.4f} kt  "
        f"RMSE={sum_wspeed['rmse']:.4f} kt  "
        f"bias={sum_wspeed['bias']:+.4f} kt\n"
        f"\n  Wind direction (29 frames, calm excluded):\n"
        f"    circular MAE={wdir_str}\n"
        f"\n  Precipitation (28 frames, t+6h→t+168h):\n"
        f"    MAE={sum_precip['mae']:.4f} mm/hr  "
        f"RMSE={sum_precip['rmse']:.4f}  "
        f"bias={sum_precip['bias']:+.4f}\n"
        f"    POD={sum_precip['pod']}  "
        f"FAR={sum_precip['far']}  "
        f"CSI={sum_precip['csi']}\n"
        f"    ({total_hits} hits / {total_misses} misses / {total_fa} false alarms "
        f"from {total_hits+total_misses+total_fa:,} total contingency events)"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify GCOp Myanmar 7-day forecast vs ERA5 (Phase R5)"
    )
    parser.add_argument(
        "--forecast-dir",
        default="data/forecast_v4",
        help="Directory containing forecast.json and 4 binary files (default: data/forecast_v4)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/verification",
        help="Output directory for verification.json (default: data/verification)",
    )
    args = parser.parse_args()
    return run_verification(Path(args.forecast_dir), Path(args.output_dir))


if __name__ == "__main__":
    sys.exit(main())
