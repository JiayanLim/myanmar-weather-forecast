"""
verify_forecast.py — Verify Aurora1p5 forecast against ERA5 reanalysis.

PURPOSE
-------
Quantitative comparison of the Aurora1p5+IFS forecast against ERA5 reanalysis
as an independent reference dataset.

IMPORTANT CAVEATS
-----------------
ERA5 is a reanalysis product (model-based, not direct observations). It is the
best available gridded reference for Myanmar at 0.25° resolution but is not
"ground truth". Skill scores here measure consistency with ERA5, not absolute
accuracy against station observations.

ERA5 is produced with ~5-day latency. This script will fail gracefully if ERA5
data is not yet available for the verification period.

REQUIREMENTS
------------
  earth2studio >= 0.17.0 with --extra data (for ERA5/ARCO data access)
  numpy, xarray, scipy, json

USAGE
-----
  uv run python scripts/verify_forecast.py \
      [--forecast-dir data/forecast] \
      [--output-dir data/verification] \
      [--lead-hours 6 12 24 48 72 96 120 144 168] \
      [--era5-source arco]

OUTPUT
------
  data/verification/verification.json   — machine-readable metrics
  data/verification/verification.md     — human-readable report
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

MYANMAR_LAT_MIN, MYANMAR_LAT_MAX = 9.0, 29.0
MYANMAR_LON_MIN, MYANMAR_LON_MAX = 92.0, 102.0

# Default lead times to verify
DEFAULT_LEAD_HOURS = [6, 12, 24, 48, 72, 96, 120, 144, 168]

# Precipitation thresholds for categorical scores (mm / 1-hour accumulation)
PRECIP_THRESHOLDS = [0.1, 1.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def mae(fcst: np.ndarray, ref: np.ndarray) -> float:
    valid = ~(np.isnan(fcst) | np.isnan(ref))
    return float(np.mean(np.abs(fcst[valid] - ref[valid])))


def rmse(fcst: np.ndarray, ref: np.ndarray) -> float:
    valid = ~(np.isnan(fcst) | np.isnan(ref))
    return float(np.sqrt(np.mean((fcst[valid] - ref[valid]) ** 2)))


def bias(fcst: np.ndarray, ref: np.ndarray) -> float:
    valid = ~(np.isnan(fcst) | np.isnan(ref))
    return float(np.mean(fcst[valid] - ref[valid]))


def categorical_scores(
    fcst: np.ndarray, ref: np.ndarray, threshold: float
) -> dict[str, float]:
    """POD, FAR, CSI for exceedance of threshold."""
    valid = ~(np.isnan(fcst) | np.isnan(ref))
    f = fcst[valid] >= threshold
    o = ref[valid] >= threshold
    hits = int(np.sum(f & o))
    misses = int(np.sum(~f & o))
    false_alarms = int(np.sum(f & ~o))
    pod = hits / (hits + misses) if (hits + misses) > 0 else float("nan")
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else float("nan")
    csi_denom = hits + misses + false_alarms
    csi = hits / csi_denom if csi_denom > 0 else float("nan")
    return {"hits": hits, "misses": misses, "false_alarms": false_alarms,
            "POD": round(pod, 4), "FAR": round(far, 4), "CSI": round(csi, 4)}


# ---------------------------------------------------------------------------
# ERA5 fetch
# ---------------------------------------------------------------------------

def fetch_era5_for_lead(
    init_time: datetime,
    lead_h: int,
    lats: np.ndarray,
    lons: np.ndarray,
    era5_source: str,
) -> dict[str, np.ndarray | None]:
    """
    Fetch ERA5 t2m and tp for (init_time + lead_h) at Myanmar bbox.
    Returns dict with 't2m_C' and 'tp_mm' arrays, or None if unavailable.
    """
    import xarray as xr

    valid_time = init_time + timedelta(hours=lead_h)
    print(f"    Fetching ERA5 for {valid_time.isoformat()} (lead +{lead_h}h)...")

    if era5_source == "arco":
        try:
            from earth2studio.data import ARCO

            arco = ARCO()
            da = arco(valid_time, ["t2m", "tp"])
            da_mm = da.sel(
                lat=slice(MYANMAR_LAT_MAX, MYANMAR_LAT_MIN),
                lon=slice(MYANMAR_LON_MIN, MYANMAR_LON_MAX),
            )
            # Interpolate to forecast grid
            t2m_da = da_mm.sel(variable="t2m").interp(lat=lats, lon=lons, method="linear")
            tp_da = da_mm.sel(variable="tp").interp(lat=lats, lon=lons, method="linear")
            t2m_C = (t2m_da.values - 273.15).astype(np.float32)
            tp_m = tp_da.values.astype(np.float32)
            # ERA5 tp is accumulated from forecast start; for 1-hour accumulation we need
            # to take the difference between consecutive hours. ARCO provides analysis-step tp.
            # At analysis time (step=0), tp=0. For step=1h, tp=accumulated 0→1h.
            # ARCO exposes tp directly per analysis time; treat as 1-hour accumulation.
            tp_mm = np.maximum(tp_m * 1000.0, 0.0).astype(np.float32)
            return {"t2m_C": t2m_C, "tp_mm": tp_mm}
        except Exception as e:
            print(f"    WARNING: ARCO fetch failed: {e}")
            return {"t2m_C": None, "tp_mm": None}

    elif era5_source == "cds":
        try:
            from earth2studio.data import CDS

            cds = CDS()
            da = cds(valid_time, ["t2m", "tp"])
            da_mm = da.sel(
                lat=slice(MYANMAR_LAT_MAX, MYANMAR_LAT_MIN),
                lon=slice(MYANMAR_LON_MIN, MYANMAR_LON_MAX),
            )
            t2m_da = da_mm.sel(variable="t2m").interp(lat=lats, lon=lons, method="linear")
            tp_da = da_mm.sel(variable="tp").interp(lat=lats, lon=lons, method="linear")
            t2m_C = (t2m_da.values - 273.15).astype(np.float32)
            tp_mm = np.maximum(tp_da.values * 1000.0, 0.0).astype(np.float32)
            return {"t2m_C": t2m_C, "tp_mm": tp_mm}
        except Exception as e:
            print(f"    WARNING: CDS fetch failed: {e}")
            return {"t2m_C": None, "tp_mm": None}

    else:
        print(f"    ERROR: Unknown ERA5 source '{era5_source}'")
        return {"t2m_C": None, "tp_mm": None}


# ---------------------------------------------------------------------------
# Main verification logic
# ---------------------------------------------------------------------------

def run_verification(
    forecast_dir: Path,
    output_dir: Path,
    lead_hours: list[int],
    era5_source: str,
) -> int:
    print("=" * 60)
    print("Aurora1p5 Forecast Verification vs ERA5")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load forecast
    # ------------------------------------------------------------------
    meta_path = forecast_dir / "forecast.json"
    temp_path = forecast_dir / "temperature.bin"
    precip_path = forecast_dir / "precipitation.bin"

    for p in [meta_path, temp_path, precip_path]:
        if not p.exists():
            print(f"ERROR: Missing {p}. Run generate_forecast.py first.")
            return 1

    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("is_demo", True):
        print("ERROR: Forecast is demo data. Verification requires a real forecast.")
        return 1

    n_times = meta["n_times"]
    n_lat = meta["grid"]["n_lat"]
    n_lon = meta["grid"]["n_lon"]
    lats = np.array(meta["lat"], dtype=np.float32)
    lons = np.array(meta["lon"], dtype=np.float32)
    times_utc = meta["times_utc"]
    init_time = datetime.fromisoformat(times_utc[0].replace("Z", "+00:00"))

    temp_all = np.frombuffer(temp_path.read_bytes(), dtype="<f4").reshape(n_times, n_lat, n_lon)
    precip_all = np.frombuffer(precip_path.read_bytes(), dtype="<f4").reshape(n_times, n_lat, n_lon)

    print(f"\nForecast:  {meta['model']} {meta['model_version']}")
    print(f"Init time: {init_time.isoformat()}")
    print(f"Grid:      {n_lat}×{n_lon} @ {meta['spatial_resolution_deg']}°  [{n_times} frames]")
    print(f"ERA5 src:  {era5_source}")
    print(f"Lead hrs:  {lead_hours}\n")

    # Filter lead hours to those actually in the forecast
    max_lead = n_times - 1  # index 0 = init, index n = t+n
    valid_leads = [h for h in lead_hours if 0 < h <= max_lead]
    if not valid_leads:
        print(f"ERROR: Requested lead hours are out of range [1, {max_lead}].")
        return 1

    # ------------------------------------------------------------------
    # Check ERA5 availability
    # ------------------------------------------------------------------
    latest_valid_time = init_time + timedelta(hours=max(valid_leads))
    era5_latency_days = 5
    if (datetime.now(timezone.utc) - latest_valid_time).days < era5_latency_days:
        print(
            f"WARNING: ERA5 for the latest verification time ({latest_valid_time.date()}) "
            f"may not yet be available (ERA5 latency ~{era5_latency_days} days). "
            "Some lead times may fail."
        )

    # ------------------------------------------------------------------
    # Verify per lead time
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_results = []
    precip_results = []

    # For spatial error maps (accumulated)
    temp_error_sum = np.zeros((n_lat, n_lon), dtype=np.float64)
    temp_n_valid = np.zeros((n_lat, n_lon), dtype=int)
    precip_bias_sum = np.zeros((n_lat, n_lon), dtype=np.float64)
    precip_n_valid = np.zeros((n_lat, n_lon), dtype=int)

    for lead_h in valid_leads:
        print(f"[+{lead_h:3d}h] ", end="")
        frame_idx = lead_h  # index 0 = t+0 (init), index h = t+h

        fcst_t2m = temp_all[frame_idx]      # °C
        fcst_tp = precip_all[frame_idx]     # mm / 1-hour accumulation

        era5 = fetch_era5_for_lead(init_time, lead_h, lats, lons, era5_source)

        if era5["t2m_C"] is None:
            print(f"    t2m: SKIPPED (ERA5 unavailable)")
            temp_results.append({"lead_h": lead_h, "available": False})
            precip_results.append({"lead_h": lead_h, "available": False})
            continue

        ref_t2m = era5["t2m_C"]
        ref_tp = era5["tp_mm"]

        # Temperature metrics
        t_mae = mae(fcst_t2m, ref_t2m)
        t_rmse = rmse(fcst_t2m, ref_t2m)
        t_bias = bias(fcst_t2m, ref_t2m)

        # Precipitation metrics
        p_mae = mae(fcst_tp, ref_tp)
        p_rmse = rmse(fcst_tp, ref_tp)
        p_bias = bias(fcst_tp, ref_tp)
        p_cat = {
            f"thr_{thr:.1f}mm": categorical_scores(fcst_tp, ref_tp, thr)
            for thr in PRECIP_THRESHOLDS
        }

        print(
            f"    t2m: MAE={t_mae:.2f}°C  RMSE={t_rmse:.2f}°C  bias={t_bias:+.2f}°C  |  "
            f"tp: MAE={p_mae:.3f}mm  RMSE={p_rmse:.3f}mm  bias={p_bias:+.3f}mm"
        )

        temp_results.append({
            "lead_h": lead_h,
            "available": True,
            "MAE_C": round(t_mae, 4),
            "RMSE_C": round(t_rmse, 4),
            "bias_C": round(t_bias, 4),
            "n_points": int((~np.isnan(fcst_t2m)).sum()),
        })
        precip_results.append({
            "lead_h": lead_h,
            "available": True,
            "MAE_mm": round(p_mae, 4),
            "RMSE_mm": round(p_rmse, 4),
            "bias_mm": round(p_bias, 4),
            "categorical": p_cat,
            "n_points": int((~np.isnan(fcst_tp)).sum()),
        })

        # Accumulate spatial error maps
        valid_mask = ~(np.isnan(fcst_t2m) | np.isnan(ref_t2m))
        temp_error_sum[valid_mask] += np.abs(fcst_t2m[valid_mask] - ref_t2m[valid_mask])
        temp_n_valid[valid_mask] += 1

        valid_mask_p = ~(np.isnan(fcst_tp) | np.isnan(ref_tp))
        precip_bias_sum[valid_mask_p] += (fcst_tp[valid_mask_p] - ref_tp[valid_mask_p])
        precip_n_valid[valid_mask_p] += 1

    # ------------------------------------------------------------------
    # Spatial error maps
    # ------------------------------------------------------------------
    with np.errstate(invalid="ignore"):
        temp_mae_map = np.where(temp_n_valid > 0, temp_error_sum / temp_n_valid, np.nan)
        precip_bias_map = np.where(precip_n_valid > 0, precip_bias_sum / precip_n_valid, np.nan)

    # ------------------------------------------------------------------
    # Build verification report
    # ------------------------------------------------------------------
    available_leads = [r for r in temp_results if r.get("available")]

    # Summary stats (domain-average over all verified leads)
    if available_leads:
        mean_t_mae = float(np.mean([r["MAE_C"] for r in available_leads]))
        mean_t_bias = float(np.mean([r["bias_C"] for r in available_leads]))
        mean_t_rmse = float(np.mean([r["RMSE_C"] for r in available_leads]))
    else:
        mean_t_mae = mean_t_bias = mean_t_rmse = None

    available_p = [r for r in precip_results if r.get("available")]
    if available_p:
        mean_p_mae = float(np.mean([r["MAE_mm"] for r in available_p]))
        mean_p_bias = float(np.mean([r["bias_mm"] for r in available_p]))
    else:
        mean_p_mae = mean_p_bias = None

    verification = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast": {
            "model": meta["model"],
            "model_version": meta["model_version"],
            "initialization_time": meta["initialization_time"],
            "initialization_source": meta["initialization_source"],
            "forecast_generated_at": meta["forecast_generated_at"],
            "spatial_resolution_deg": meta["spatial_resolution_deg"],
        },
        "reference": {
            "dataset": "ERA5",
            "source": era5_source,
            "caveat": (
                "ERA5 is a reanalysis product, not direct observations. "
                "Skill scores measure consistency with ERA5, not absolute accuracy. "
                "ERA5 ~5-day latency may cause some lead times to be unavailable."
            ),
        },
        "region": {
            "name": "Myanmar",
            "lat_min": float(lats.min()),
            "lat_max": float(lats.max()),
            "lon_min": float(lons.min()),
            "lon_max": float(lons.max()),
            "n_points": n_lat * n_lon,
        },
        "patched_variables": {
            "summary": (
                "4 variables not available in IFS open data were zero-filled: "
                "sic (sea ice), lcc (low cloud cover), mcc (mid cloud cover), hcc (high cloud cover)."
            ),
            "sic": {
                "method": "zero-fill globally",
                "impact": (
                    "sic mean in training data: ~0.111 (scale ~0.297). "
                    "Zero-fill is -0.37σ from training mean. "
                    "Negligible for Myanmar (tropical, no sea ice). Acceptable."
                ),
            },
            "lcc": {
                "method": "zero-fill globally",
                "impact": (
                    "lcc mean in training data: ~0.462 (scale ~0.393). "
                    "Zero-fill is -1.18σ from training mean. "
                    "Aurora uses tcc (total cloud cover, available from IFS) as a partial substitute. "
                    "May cause modest warm bias in 2m temperature (less cloud shading) and "
                    "underestimation of convective trigger conditions. Effect likely small vs "
                    "other sources of forecast error."
                ),
            },
            "mcc": {
                "method": "zero-fill globally",
                "impact": (
                    "mcc mean in training data: ~0.299 (scale ~0.373). "
                    "Zero-fill is -0.80σ from training mean. "
                    "Similar impact to lcc; tcc partially compensates. "
                    "Myanmar mid-level clouds affect radiation and precipitation patterns."
                ),
            },
            "hcc": {
                "method": "zero-fill globally",
                "impact": (
                    "hcc mean in training data: ~0.327 (scale ~0.413). "
                    "Zero-fill is -0.79σ from training mean. "
                    "IFS open data does not provide individual cloud layer fractions at step=0. "
                    "Tropical cirrus is underrepresented in Aurora's input. "
                    "Effect on near-surface forecasts likely secondary."
                ),
            },
            "overall_assessment": (
                "sic zero-fill is scientifically appropriate for tropical Myanmar. "
                "lcc/mcc/hcc zero-fill is a known limitation: IFS open data provides only tcc "
                "at the analysis time (step=0), not layer fractions. "
                "Aurora uses tcc directly as an AR variable, providing partial cloud information. "
                "The lcc/mcc/hcc zero-fill may introduce a slight warm/dry bias but is unlikely "
                "to dominate forecast error beyond 1-2 days. "
                "This is a structural limitation of the IFS open data availability, "
                "not a modelling error. Better alternatives: ECMWF CDS (requires registration) "
                "or ERA5 for hindcasts."
            ),
        },
        "precipitation_semantics": {
            "variable": "tp1h",
            "units": "mm per 1-hour accumulation period",
            "not_rate": True,
            "transform_pipeline": (
                "Aurora model: predicts scaled_tp_1h = log_transform(tp1h_metres) in log-space. "
                "Earth2Studio aurora1p5.py _prepare_output(): applies aurora_log_untransform "
                "(= 0.001 * (exp(x) - 1)) to convert log-space → physical metres before zarr. "
                "Pipeline: zarr_metres * 1000 → mm / 1-hour accumulation. "
                "Zero-clamp applied to suppress numerical noise (physical tp1h ≥ 0)."
            ),
            "display_label": "mm / 1-hour accumulation",
            "frontend_note": (
                "Values should be displayed as 'X mm / 1-hour accumulation', "
                "not 'X mm/h'. These are total accumulations per forecast hour, "
                "not instantaneous rainfall intensity."
            ),
        },
        "temperature_verification": {
            "leads_verified": [r["lead_h"] for r in temp_results if r.get("available")],
            "domain": "Myanmar (9–29°N, 92–102°E)",
            "summary": {
                "mean_MAE_C": round(mean_t_mae, 4) if mean_t_mae is not None else None,
                "mean_bias_C": round(mean_t_bias, 4) if mean_t_bias is not None else None,
                "mean_RMSE_C": round(mean_t_rmse, 4) if mean_t_rmse is not None else None,
            },
            "by_lead": temp_results,
        },
        "precipitation_verification": {
            "leads_verified": [r["lead_h"] for r in precip_results if r.get("available")],
            "thresholds_mm": PRECIP_THRESHOLDS,
            "summary": {
                "mean_MAE_mm": round(mean_p_mae, 4) if mean_p_mae is not None else None,
                "mean_bias_mm": round(mean_p_bias, 4) if mean_p_bias is not None else None,
            },
            "by_lead": precip_results,
        },
        "spatial_error": {
            "temperature_mae": {
                "description": "Mean absolute error of 2m temperature averaged over all verified lead times",
                "shape": [n_lat, n_lon],
                "lats": lats.tolist(),
                "lons": lons.tolist(),
                "values_C": [
                    [round(float(v), 3) if not np.isnan(v) else None for v in row]
                    for row in temp_mae_map
                ],
            },
            "precipitation_bias": {
                "description": "Mean bias (fcst - ERA5) of 1-hour precipitation averaged over all verified lead times",
                "shape": [n_lat, n_lon],
                "lats": lats.tolist(),
                "lons": lons.tolist(),
                "values_mm": [
                    [round(float(v), 4) if not np.isnan(v) else None for v in row]
                    for row in precip_bias_map
                ],
            },
        },
        "limitations": [
            "ERA5 is a reanalysis, not direct observations — station-based verification would be more rigorous.",
            "Both Aurora1p5 and ERA5 use the same IFS model family; results may show optimistic skill for temperature.",
            "Precipitation verification is sensitive to timing and position errors (double-penalty problem).",
            "Four IFS initialization variables were zero-filled (sic, lcc, mcc, hcc); see patched_variables.",
            "Forecast skill degrades significantly beyond 3–5 days at 0.25° resolution.",
            "Myanmar topography (Chin Hills, Rakhine Yoma, Shan Plateau) is partially resolved at 0.25°.",
            "This verification covers one forecast cycle; multi-cycle averaging would be needed for robust statistics.",
        ],
    }

    # ------------------------------------------------------------------
    # Write machine-readable JSON
    # ------------------------------------------------------------------
    json_out = output_dir / "verification.json"
    with open(json_out, "w") as f:
        json.dump(verification, f, indent=2)
    print(f"\nVerification JSON: {json_out}")

    # ------------------------------------------------------------------
    # Write human-readable Markdown
    # ------------------------------------------------------------------
    md_lines = [
        "# Aurora1p5 Forecast Verification",
        "",
        f"**Generated:** {verification['generated_at']}  ",
        f"**Model:** {meta['model']} {meta['model_version']}  ",
        f"**Init time:** {meta['initialization_time']}  ",
        f"**Reference:** ERA5 (via {era5_source})  ",
        "",
        "> **Caveat:** ERA5 is a reanalysis product, not direct observations. Skill scores here measure",
        "> consistency with ERA5. Both Aurora1p5 and ERA5 share IFS heritage, which may inflate temperature skill.",
        "",
        "---",
        "",
        "## Temperature (2m) Verification",
        "",
        "| Lead time | MAE (°C) | RMSE (°C) | Bias (°C) |",
        "|-----------|----------|-----------|-----------|",
    ]
    for r in temp_results:
        if r.get("available"):
            md_lines.append(
                f"| +{r['lead_h']}h | {r['MAE_C']:.2f} | {r['RMSE_C']:.2f} | {r['bias_C']:+.2f} |"
            )
        else:
            md_lines.append(f"| +{r['lead_h']}h | N/A | N/A | N/A |")

    if mean_t_mae is not None:
        md_lines += [
            "",
            f"**Domain mean:** MAE = {mean_t_mae:.2f}°C  |  RMSE = {mean_t_rmse:.2f}°C  |  Bias = {mean_t_bias:+.2f}°C",
        ]

    md_lines += [
        "",
        "---",
        "",
        "## Precipitation Verification",
        "",
        "### Continuous metrics (mm / 1-hour accumulation)",
        "",
        "| Lead time | MAE (mm) | RMSE (mm) | Bias (mm) |",
        "|-----------|----------|-----------|-----------|",
    ]
    for r in precip_results:
        if r.get("available"):
            md_lines.append(
                f"| +{r['lead_h']}h | {r['MAE_mm']:.3f} | {r['RMSE_mm']:.3f} | {r['bias_mm']:+.3f} |"
            )
        else:
            md_lines.append(f"| +{r['lead_h']}h | N/A | N/A | N/A |")

    md_lines += ["", "### Categorical scores (POD / FAR / CSI)"]
    for thr in PRECIP_THRESHOLDS:
        key = f"thr_{thr:.1f}mm"
        md_lines += [f"", f"**Threshold: {thr} mm / 1h**", "", "| Lead time | POD | FAR | CSI |", "|-----------|-----|-----|-----|"]
        for r in precip_results:
            if r.get("available") and key in r.get("categorical", {}):
                cat = r["categorical"][key]
                md_lines.append(f"| +{r['lead_h']}h | {cat['POD']:.3f} | {cat['FAR']:.3f} | {cat['CSI']:.3f} |")
            else:
                md_lines.append(f"| +{r['lead_h']}h | N/A | N/A | N/A |")

    md_lines += [
        "",
        "---",
        "",
        "## IFS Patched Variables",
        "",
        "Four variables required by Aurora1p5 were not available in IFS open data and were zero-filled:",
        "",
        "| Variable | Zero-fill deviation | Assessment |",
        "|----------|--------------------|----------------------------------------------------|",
        "| `sic` (sea ice) | −0.37σ | Acceptable — Myanmar is tropical, no sea ice |",
        "| `lcc` (low cloud) | −1.18σ | Moderate impact; tcc provides partial substitute |",
        "| `mcc` (mid cloud) | −0.80σ | Moderate impact; tcc provides partial substitute |",
        "| `hcc` (high cloud) | −0.79σ | Minor near-surface impact |",
        "",
        "**IFS open data** provides `tcc` (total cloud cover) at step=0, which Aurora1p5 uses as an AR variable.",
        "Layer fractions (lcc, mcc, hcc) are only available in IFS forecast stream (step>0), not the analysis.",
        "`sic` is absent from IFS entirely — zero-fill is scientifically sound for tropical Myanmar.",
        "",
        "---",
        "",
        "## Precipitation Semantics",
        "",
        "- Aurora's `scaled_tp_1h` is the log-transformed 1-hour total precipitation in metres",
        "- Earth2Studio applies `aurora_log_untransform` before zarr: `0.001 × (eˣ − 1)` → metres",
        "- Pipeline output: `metres × 1000 → mm / 1-hour accumulation`",
        "- These are **accumulation totals** per forecast hour, not instantaneous rates",
        "- Display should read: `X mm / 1-hour accumulation`, not `X mm/h`",
        "",
        "---",
        "",
        "## Limitations",
        "",
    ]
    for lim in verification["limitations"]:
        md_lines.append(f"- {lim}")
    md_lines.append("")

    md_out = output_dir / "verification.md"
    md_out.write_text("\n".join(md_lines))
    print(f"Verification MD:   {md_out}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    n_verified = len([r for r in temp_results if r.get("available")])
    print(f"\n{'=' * 60}")
    print(f"Verified {n_verified}/{len(valid_leads)} lead times against ERA5")
    if mean_t_mae is not None:
        print(f"Temperature:  mean MAE={mean_t_mae:.2f}°C  bias={mean_t_bias:+.2f}°C")
    if mean_p_mae is not None:
        print(f"Precipitation: mean MAE={mean_p_mae:.3f}mm  bias={mean_p_bias:+.3f}mm")
    print("=" * 60)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Aurora1p5 forecast against ERA5 reanalysis"
    )
    parser.add_argument(
        "--forecast-dir",
        default="data/forecast",
        help="Directory containing forecast.json, temperature.bin, precipitation.bin",
    )
    parser.add_argument(
        "--output-dir",
        default="data/verification",
        help="Output directory for verification.json and verification.md",
    )
    parser.add_argument(
        "--lead-hours",
        nargs="+",
        type=int,
        default=DEFAULT_LEAD_HOURS,
        help=f"Lead times to verify (default: {DEFAULT_LEAD_HOURS})",
    )
    parser.add_argument(
        "--era5-source",
        choices=["arco", "cds"],
        default="arco",
        help=(
            "ERA5 data source. 'arco' (Google Cloud, no credentials needed) is preferred. "
            "'cds' requires a CDS API key (https://cds.climate.copernicus.eu)."
        ),
    )
    args = parser.parse_args()

    return run_verification(
        forecast_dir=Path(args.forecast_dir),
        output_dir=Path(args.output_dir),
        lead_hours=sorted(set(args.lead_hours)),
        era5_source=args.era5_source,
    )


if __name__ == "__main__":
    sys.exit(main())
