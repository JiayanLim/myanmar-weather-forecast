# Myanmar Weather Forecast

AI-powered 7-day weather forecast for Myanmar using Microsoft Aurora 1.5 and IFS open data.

**Live site**: https://jiayanlim.github.io/myanmar-weather-forecast/

---

## Running Aurora Forecasts with Google Colab

This project uses a free Google Colab GPU to generate real forecasts and publish them to GitHub Pages. No paid cloud services required.

### Prerequisites

1. A Google account with access to [Google Colab](https://colab.research.google.com/)
2. A GitHub Personal Access Token (Fine-grained PAT) with **Contents: Read and Write** on this repository
3. Basic familiarity with running Jupyter notebooks

### Step 1 — Create a GitHub Personal Access Token

1. Go to GitHub → Settings → Developer Settings → Personal Access Tokens → Fine-grained tokens
2. Click **Generate new token**
3. Set: Token name = `myanmar-weather-colab`, Repository access = this repo only, Permissions → Contents: **Read and Write**
4. Copy the token (you will not see it again)

### Step 2 — Add Token to Colab Secrets

1. Open the notebook in Colab (see Step 3)
2. Click the 🔑 **Secrets** icon in the left sidebar
3. Add secret: **Name** = `GITHUB_TOKEN`, **Value** = your token
4. Enable **Notebook access** toggle

> Colab secrets are encrypted and never leave your Google account. The token is never printed or logged.

### Step 3 — Open the Notebook in Colab

Open `notebooks/aurora_myanmar_forecast.ipynb` directly in Colab:

```
https://colab.research.google.com/github/JiayanLim/myanmar-weather-forecast/blob/main/notebooks/aurora_myanmar_forecast.ipynb
```

### Step 4 — Select a GPU Runtime

Runtime → Change runtime type → Hardware accelerator: **T4 GPU** (free) or **A100** (Colab Pro)

> **T4 (16 GB VRAM)**: Slower (~60–90 min), may run out of memory on the full 168h forecast.
> **A100 (40/80 GB VRAM)**: Recommended (~20–40 min), more reliable for Aurora1p5.

### Step 5 — Run the Notebook

Run cells **in order** from top to bottom:

| Section | What it does | Approx. time |
|---------|-------------|--------------|
| 1. GPU Verification | Checks CUDA is available, shows GPU/VRAM info | < 10 sec |
| 2. Install Dependencies | `eccodes` + `earth2studio[aurora,data]` | 3–5 min |
| 3. Clone Repository | Clones/pulls repo, configures git auth | < 30 sec |
| 4. IFS Data Test | Verifies IFS open data connectivity | < 30 sec |
| 5. Aurora Checkpoint | Downloads model weights from HuggingFace (~5 GB) | 5–10 min |
| 6. Smoke Test | 6-hour forecast to verify pipeline end-to-end | 1–8 min |
| 7. Full Forecast | **168-hour production forecast** | **20–90 min** |
| 8. Validation | Checks dimensions, NaN, physical plausibility, file sizes | < 10 sec |
| 9. Publish | `git commit` + `git push` → triggers GitHub Pages deploy | < 30 sec |
| 10. Summary | Final report with all details | < 5 sec |

**Do not close the browser tab during Section 7 (inference).**

### Step 6 — Wait for GitHub Pages Deployment

After the publish step succeeds:
1. Go to [Actions](https://github.com/JiayanLim/myanmar-weather-forecast/actions) to watch the deploy
2. GitHub Pages rebuilds automatically in ~1–2 minutes
3. Open https://jiayanlim.github.io/myanmar-weather-forecast/
4. You will see the real forecast (no DEMO DATA banner)

### Expected GPU Requirements

| GPU | VRAM | 168h Forecast Time | Notes |
|-----|------|--------------------|-------|
| T4 (Colab free) | 16 GB | 60–90 min | May OOM; use for testing |
| A100 SXM4 (Colab Pro) | 40 GB | 20–30 min | Recommended |
| A100 80GB | 80 GB | 15–20 min | Fastest |

### Generated Artifact Sizes

| File | Size |
|------|------|
| `forecast.json` | ~80–100 KB |
| `temperature.bin` | ~2.2 MB |
| `precipitation.bin` | ~2.2 MB |
| **Total** | **~4.5 MB** |

All three files are committed to `data/forecast/` — no Git LFS needed.

### Troubleshooting

| Error | Fix |
|-------|-----|
| "CUDA not available" | Runtime → Change runtime type → T4 GPU |
| "Out of memory" | Switch to Colab Pro (A100) |
| IFS fetch fails (429/503) | Wait a few minutes, retry — rate limits reset quickly |
| Git push rejected | Token expired or missing Contents write permission — regenerate at GitHub |
| Smoke test fails | Review error above. Common cause: IFS data not yet available for init time |

---

> **Important**: Google Colab free GPU availability and runtime duration are not guaranteed.
> Free tier sessions may be interrupted and GPU access unavailable at peak times.
> This workflow is intended for **demonstration and research use**, not guaranteed operational forecasting.
> For reliable daily forecasting, use a dedicated CUDA GPU machine.

---

## Architecture

```
notebooks/aurora_myanmar_forecast.ipynb  (runs in Colab)
    ↓ calls
scripts/generate_forecast.py             (Aurora1p5 + IFS → Myanmar artifacts)
    ↓ git commit + push
.github/workflows/deploy-pages.yml       (prefers data/forecast/, falls back to data/demo/)
    ↓
GitHub Pages → https://jiayanlim.github.io/myanmar-weather-forecast/
```

**Frontend** (React 18 + TypeScript + Vite + MapLibre GL JS + Zustand + Tailwind):
- No GPU, no Python, no server required to view
- Bilinear interpolation from 0.25° model grid to 0.05° display grid
- Myanmar boundary mask via GeoJSON scanline rasterization
- Hourly playback, variable switcher, info panel

---

## Repository Structure

```
myanmar-weather-forecast/
├── .github/workflows/
│   └── deploy-pages.yml              # CI/CD: prefers data/forecast/, falls back to data/demo/
├── notebooks/
│   └── aurora_myanmar_forecast.ipynb # Colab notebook for real forecast generation
├── scripts/
│   ├── generate_forecast.py          # Production pipeline
│   ├── generate_demo_data.py         # Demo data (CPU, no GPU)
│   ├── validate_aurora.py            # Validate Aurora+IFS (--dry-run for CPU check)
│   └── validate_forecast.py          # Validate binary artifacts
├── frontend/                         # React application
├── data/
│   ├── demo/                         # Synthetic demo artifacts (always committed)
│   └── forecast/                     # Real forecast artifacts (committed after Colab run)
└── pyproject.toml                    # Python dependencies
```

---

## Local Development (Demo Mode — No GPU)

```bash
uv sync
uv run python scripts/generate_demo_data.py
cd frontend && npm install && npm run dev
```

## Local Environment Check (No GPU)

```bash
uv run python scripts/validate_aurora.py --dry-run
```

---

## IFS Variable Patches

Aurora1p5 requires 83 input variables. Four are not in IFS open data and are zero-filled:

| Variable | Description | Patch |
|----------|-------------|-------|
| `sic` | Sea ice concentration | 0.0 globally (correct for tropical Myanmar) |
| `lcc` | Low cloud cover | 0.0 (IFS provides `tcc` only) |
| `mcc` | Medium cloud cover | 0.0 |
| `hcc` | High cloud cover | 0.0 |

---

## Model Attribution

- **Forecast model**: [Aurora 1.5](https://github.com/microsoft/aurora) — Microsoft Research
- **Initialization data**: IFS HRES — ECMWF open data (CC BY 4.0)
- **Framework**: [Earth2Studio](https://github.com/NVIDIA/earth2studio) — NVIDIA
- **Basemap**: © OpenStreetMap contributors

---

## Limitations

- 0.25° (~28 km) resolution — not for neighborhood-scale prediction
- Precipitation is 1-hour accumulation totals, not instantaneous rate
- Forecast skill degrades beyond 3–5 days
- `sic`/`lcc`/`mcc`/`hcc` zero-patched (documented in `forecast.json`)
- Model weights subject to [Microsoft Research license](https://github.com/microsoft/aurora/blob/main/LICENSE)

---

## License

Code: MIT. Model weights and forecast data subject to their respective upstream licenses.
