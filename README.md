# Myanmar Weather Forecast

AI-powered 7-day weather forecast for Myanmar, built with NVIDIA Earth2Studio and Aurora 1.5.

**Status**: Specification complete. Implementation in progress.

---

## Overview

```
IFS Open Data (ECMWF)
        |
   Aurora 1.5 (0.25 degree global)
        |
   Myanmar spatial subset (9N-29N, 92E-102E)
        |
   temperature + precipitation (hourly, 168 frames)
        |
   React + TypeScript + MapLibre GL JS
        |
   GitHub Pages (static, no server required)
```

Forecast variables:
- **2m Temperature** — hourly, degrees C
- **Precipitation** — hourly 1-hour accumulated total (mm), from Aurora1p5 native `tp1h` output

Forecast horizon: 7 days (168 hourly steps)
Spatial resolution: 0.25 degrees (~28 km)
Model: [Aurora 1.5](https://github.com/microsoft/aurora) by Microsoft Research

---

## Architecture

### Forecast pipeline (offline, GPU required)

```
scripts/generate_forecast.py
  - Initializes Aurora1p5 from IFS HRES analysis (ECMWF open data)
  - Runs 168-hour forecast with native hourly rollout
  - Extracts t2m (temperature) and tp1h (precipitation)
  - Subsets to Myanmar bounding box
  - Writes binary artifacts to data/forecast/
```

### Frontend (static, GitHub Pages)

```
frontend/
  - React 18 + TypeScript 5 + Vite 5
  - MapLibre GL JS for interactive map
  - Tailwind CSS for styling
  - Zustand for state management
  - Reads pre-generated binary artifacts from data/
```

---

## Repository Structure

```
myanmar-weather-forecast/
|-- .github/workflows/deploy-pages.yml   # GitHub Pages CI/CD
|-- specs/001-myanmar-weather-app/       # Spec Kit artifacts
|   |-- research.md                      # Earth2Studio discovery + ADRs
|   |-- spec.md                          # Feature specification
|   |-- plan.md                          # Technical plan
|   `-- tasks.md                         # Implementation task list
|-- frontend/                            # React application (to be built)
|-- forecast/                            # Pipeline modules (to be built)
|-- scripts/                             # CLI scripts (to be built)
|-- data/demo/                           # Demo artifacts (committed)
|-- data/forecast/                       # Production artifacts (gitignored)
|-- docs/                                # Documentation
`-- pyproject.toml                       # Python dependencies
```

---

## GPU Requirements

Production forecast generation requires:
- CUDA-capable GPU with **48 GB VRAM** (e.g., A100 80GB, H100)
- Aurora 1.5 model weights (~5 GB, auto-downloaded from HuggingFace)

Demo mode (development/testing) runs on any CPU — no GPU required.

---

## Quick Start

### Demo mode (no GPU)

```bash
# Install Python dependencies
uv sync

# Generate demo data
uv run python scripts/generate_demo_data.py

# Run frontend with demo data
cd frontend && npm install && npm run dev
```

### Production forecast (GPU required)

```bash
# Generate real forecast (requires IFS access + 48GB GPU)
uv run python scripts/generate_forecast.py --init-time 2026-08-09T00:00:00Z

# Validate output
uv run python scripts/validate_forecast.py --data-dir data/forecast/

# Serve frontend with production data
cd frontend && npm run dev
```

---

## Initialization Data Source

Aurora 1.5 is initialized from **IFS HRES analysis** (ECMWF open data).

- No credentials required for ECMWF open data
- GFS is **not supported** as an initialization source (missing required surface variables)
- The IFS open data stream lacks sea ice concentration (`sic`); the pipeline handles this gap (see `docs/forecasting.md`)

---

## Deployment

The frontend deploys automatically to GitHub Pages on push to `main`.

Deployed URL: `https://JiayanLim.github.io/myanmar-weather-forecast/`

The forecast pipeline runs **offline** on a GPU machine. Generated artifacts are
committed to `data/demo/` for the demo deployment, or uploaded separately for
production use.

---

## Model Attribution

- **Forecast model**: [Aurora 1.5](https://github.com/microsoft/aurora) — Microsoft Research
- **Initialization data**: IFS HRES — European Centre for Medium-Range Weather Forecasts (ECMWF)
- **Framework**: [Earth2Studio](https://github.com/NVIDIA/earth2studio) — NVIDIA
- **Basemap**: OpenStreetMap contributors
- **Boundary data**: Natural Earth (public domain)

---

## Limitations

- Spatial resolution 0.25 degrees (~28 km) — not suitable for neighborhood-scale prediction
- Precipitation represents 1-hour accumulation totals, not instantaneous rates
- Forecast skill degrades beyond 3-5 days
- Sea ice concentration gap in IFS open data handled by climatological padding (documented in forecast.json)
- Model weights are subject to Microsoft Research license terms

---

## License

Code: MIT. Model weights and forecast data subject to their respective upstream licenses.
