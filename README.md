# Myanmar Weather Forecast

AI-powered 24-hour precipitation forecast for Myanmar using NVIDIA GraphCastSmall and ARCO/IFS reanalysis data.

**Live site**: https://jiayanlim.github.io/myanmar-weather-forecast/

---

## Running GraphCastSmall Forecasts with Google Colab

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

Open `notebooks/graphcast_myanmar_forecast.ipynb` directly in Colab:

```
https://colab.research.google.com/github/JiayanLim/myanmar-weather-forecast/blob/main/notebooks/graphcast_myanmar_forecast.ipynb
```

### Step 4 — Select a GPU Runtime

Runtime → Change runtime type → Hardware accelerator: **T4 GPU** (free) or **A100** (Colab Pro)

> **T4 (16 GB VRAM)**: VRAM compatibility pending experimental verification. The notebook includes a staged VRAM test — if the T4 runs out of memory on the smoke test (Stage 3), the notebook will stop and report clearly.
> **A100 (40 GB VRAM, Colab Pro)**: Recommended for reliable inference.

### Step 5 — Run the Notebook

Run cells **in order** from top to bottom:

| Section | What it does | Approx. time |
|---------|-------------|--------------|
| 1. Install Dependencies | `chex` + `dm-haiku` + `graphcast` + `earth2studio` (kernel restarts on first run) | 2–4 min |
| 2. Imports | Verify all packages loaded correctly | < 10 sec |
| 3. GPU Verification | Checks CUDA, nvidia-smi, torch + JAX device info | < 10 sec |
| 4. JAX Config & Constants | Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, pipeline constants | < 5 sec |
| 5. Initialization Data | ARCO reanalysis fetch for t-6h and t+0h | 1–3 min |
| 6. Load GraphCastSmall | Download model weights, measure VRAM (Stage 2) | 2–5 min |
| 7. Smoke Test | nsteps=1 VRAM gate (Stage 3) — OOM stops here with full report | 1–5 min |
| 8. Full Forecast | 24h forecast, nsteps=4 (Stage 4) | 5–20 min |
| 9. Post-processing | tp06 metres × 1000 → mm/6h, Myanmar subset, prepend t+0h frame | < 10 sec |
| 10. Statistics | Precipitation stats: min/median/P95/P99/max, NaN/inf/neg checks | < 5 sec |
| 11. Write Artifacts | `precipitation.bin` + `forecast.json` | < 10 sec |
| 12. Validate | Runs `scripts/validate_forecast.py` — all 27 checks must PASS | < 10 sec |
| 13. Provenance Report | Full run summary | < 5 sec |
| 14. Git Push | `git commit` + `git push` → triggers GitHub Pages deploy | < 30 sec |

> **Section 1 note**: On first run, the install cell restarts the Colab kernel automatically (SIGKILL). After reconnect, re-run from Section 1 — it will detect packages already installed and continue without reinstalling.

**Do not close the browser tab during Section 8 (inference).**

### Step 6 — Wait for GitHub Pages Deployment

After the push step succeeds:
1. Go to [Actions](https://github.com/JiayanLim/myanmar-weather-forecast/actions) to watch the deploy
2. GitHub Pages rebuilds automatically in ~1–2 minutes
3. Open https://jiayanlim.github.io/myanmar-weather-forecast/
4. You will see the real forecast (no DEMO DATA banner)

### Expected GPU Requirements

| GPU | VRAM | 24h Forecast Time | Notes |
|-----|------|--------------------|-------|
| T4 (Colab free) | 16 GB | TBD | VRAM compatibility unverified — staged test gates inference |
| A100 SXM4 (Colab Pro) | 40 GB | 5–15 min | Recommended |
| A100 80GB | 80 GB | 3–8 min | Fastest |

> GraphCastSmall is rated for 40 GB VRAM. T4 (16 GB) compatibility is not guaranteed and must be determined experimentally via the staged VRAM test in Section 7.

### Generated Artifact Sizes

| File | Size |
|------|------|
| `forecast.json` | ~3 KB |
| `precipitation.bin` | 4,620 bytes exactly |
| **Total** | **< 10 KB** |

Both files are committed to `data/forecast/`. No Git LFS needed.

### Troubleshooting

| Error | Fix |
|-------|-----|
| "CUDA not available" | Runtime → Change runtime type → T4 GPU |
| "Out of memory" (Stage 3) | T4 VRAM insufficient — switch to Colab Pro (A100) |
| `No module named 'chex'` after restart | Re-run Section 1; it detects install and skips reinstall |
| ARCO fetch fails | ARCO dev dataset may be temporarily unavailable; retry or switch `--source ifs` |
| Git push rejected | Token expired or missing Contents write permission — regenerate at GitHub |
| Validation FAIL | Review which check failed; re-run Section 9–11 if post-processing was correct |

---

> **Important**: Google Colab free GPU availability and runtime duration are not guaranteed.
> Free tier sessions may be interrupted and GPU access unavailable at peak times.
> This workflow is intended for **demonstration and research use**, not guaranteed operational forecasting.
> For reliable daily forecasting, use a dedicated CUDA GPU machine.

---

## Architecture

```
notebooks/graphcast_myanmar_forecast.ipynb  (runs in Colab)
    ↓ ARCO/IFS reanalysis → GraphCastSmall inference → tp06 post-processing
scripts/generate_forecast.py                (GraphCastSmall + ARCO/IFS → Myanmar artifacts)
    ↓ git commit + push
.github/workflows/deploy-pages.yml          (prefers data/forecast/, falls back to data/demo/)
    ↓
GitHub Pages → https://jiayanlim.github.io/myanmar-weather-forecast/
```

**Model**: GraphCastSmall (NVIDIA Earth2Studio)
- Global 1.0° grid (181 × 360)
- Native 6-hour timestep; 24-hour forecast horizon (4 autoregressive steps)
- Two-timestep initialization: t−6h and t+0h fetched from ARCO or IFS
- Output variable: `tp06` — 6-hour accumulated total precipitation in metres → × 1000 → mm/6h
- No log/exp transform applied (unlike tp1h from Aurora)
- JAX backend with bfloat16 internal precision; requires `XLA_PYTHON_CLIENT_PREALLOCATE=false`

**Myanmar domain**: lat 9–29°N (21 pts), lon 92–102°E (11 pts) at 1.0°

**Frontend** (React 18 + TypeScript + Vite + MapLibre GL JS + Zustand + Tailwind):
- No GPU, no Python, no server required to view
- Bilinear interpolation from 1.0° model grid to 0.05° display grid
- Myanmar boundary mask via GeoJSON scanline rasterization
- 5-frame timeline at 0h / 6h / 12h / 18h / 24h, with play/pause animation
- Point inspector popup shows mm/6h accumulation

---

## Repository Structure

```
myanmar-weather-forecast/
├── .github/workflows/
│   └── deploy-pages.yml                       # CI/CD: prefers data/forecast/, falls back to data/demo/
├── notebooks/
│   └── graphcast_myanmar_forecast.ipynb       # Colab notebook for real forecast generation
├── scripts/
│   ├── generate_forecast.py                   # Production pipeline (GraphCastSmall + ARCO/IFS)
│   ├── generate_demo_data.py                  # Demo data (CPU, no GPU)
│   └── validate_forecast.py                   # Validate binary artifacts (27 checks)
├── frontend/                                  # React application
├── data/
│   ├── demo/                                  # Synthetic demo artifacts (always committed)
│   └── forecast/                              # Real forecast artifacts (committed after Colab run)
└── pyproject.toml                             # Python dependencies
```

---

## Local Development (Demo Mode — No GPU)

```bash
uv sync
uv run python scripts/generate_demo_data.py
uv run python scripts/validate_forecast.py --data-dir data/demo/
cd frontend && npm install && npm run dev
```

---

## Data Format

### `precipitation.bin`

Raw IEEE 754 float32, C-order (row-major):

```
shape: [5, 21, 11]   # [n_times, n_lat, n_lon]
dtype: float32
size:  4,620 bytes   # 5 × 21 × 11 × 4
```

- **n_times = 5**: frames at t+0h, t+6h, t+12h, t+18h, t+24h
- **n_lat = 21**: 9°N to 29°N at 1.0° spacing (ascending)
- **n_lon = 11**: 92°E to 102°E at 1.0° spacing
- **Units**: mm / 6-hour accumulation (no log/exp transform)
- **Values**: ≥ 0.0 mm/6h (clamped); physical upper bound ~ 500 mm/6h

### `forecast.json`

Schema v2.0. Key fields:

```json
{
  "model": "GraphCastSmall",
  "n_times": 5,
  "native_timestep_hours": 6,
  "forecast_horizon_hours": 24,
  "spatial_resolution_deg": 1.0,
  "variables": {
    "precipitation": {
      "source_variable": "tp06",
      "transformation_provenance": {
        "log_transform_applied": false,
        "exp_transform_applied": false,
        "conversion": "metres * 1000"
      }
    }
  }
}
```

---

## Model Attribution

- **Forecast model**: [GraphCastSmall](https://arxiv.org/abs/2212.12794) — Google DeepMind, via NVIDIA Earth2Studio
- **Initialization data**: ARCO reanalysis (dev dataset) or IFS HRES (ECMWF open data, CC BY 4.0)
- **Framework**: [Earth2Studio](https://github.com/NVIDIA/earth2studio) — NVIDIA
- **Basemap**: © OpenStreetMap contributors

---

## Limitations

- 1.0° (~111 km) resolution — not for neighborhood-scale prediction
- Precipitation is 6-hour accumulation totals, not instantaneous rate
- T4 (16 GB) VRAM compatibility with GraphCastSmall is unverified; 40 GB is the recommended minimum
- ARCO (dev) dataset may have intermittent availability; IFS is the operational fallback

---

## License

Code: MIT. Model weights and forecast data subject to their respective upstream licenses.
