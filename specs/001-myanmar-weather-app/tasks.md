# Tasks: Myanmar Weather Forecast Web Application

**Input**: `specs/001-myanmar-weather-app/` (spec.md v3, plan.md v3, research.md v3)
**Feature**: 001-myanmar-weather-app
**Revised**: 2026-08-11 (GraphCastSmall migration — replaces Aurora1p5 task set)

**Key architectural facts for task authors**:
- GraphCastSmall natively produces tp06 (6h accumulated precipitation) — no PrecipitationAFNO
- ARCO (dev) or IFS (prod) are the only verified init sources; NCAR_ERA5 and GFS are out
- GraphCastSmall requires t-6h AND t+0h at initialization — two timesteps must be fetched
- tp06 requires NO log/exp transform — only metres × 1000 → mm / 6h
- Binary format: [5 × 21 × 11] float32 per variable (~4.6 KB)
- Timeline: 5 steps at 0, 6, 12, 18, 24h
- Myanmar at 1.0°: lat 9–29°N (21 pts), lon 92–102°E (11 pts)
- No temperature, no variable switcher
- VRAM must be tested experimentally; 40 GB is the badge but T4 (16 GB) is unverified

---

## Phase 1: Spec Kit Update

**Status**: COMPLETE (2026-08-11)

- [x] T001 Update `.specify/memory/constitution.md` v1.0.1 → v2.0.0 (GraphCastSmall, 6h steps, tp06)
- [x] T002 Update `specs/001-myanmar-weather-app/research.md` with GraphCastSmall ADRs
- [x] T003 Update `specs/001-myanmar-weather-app/spec.md` for 24h/6h-step/1°/21×11/tp06-only
- [x] T004 Update `specs/001-myanmar-weather-app/plan.md` with new implementation phases
- [x] T005 Update `specs/001-myanmar-weather-app/tasks.md` (this file)

---

## Phase 2: Pipeline Rewrite

**Status**: COMPLETE (2026-08-11)

**Goal**: Rewrite `scripts/generate_forecast.py` for GraphCastSmall + ARCO/IFS.

- [x] T010 Rewrite `scripts/generate_forecast.py`:
  - Uses `earth2studio.models.px.GraphCastSmall`
  - Default source: ARCO (historical); flag `--source [arco|ifs]`
  - Two-timestep init handled by Earth2Studio via input_coords lead_time=[-6h,0h]
  - nsteps=4 (produces t+6h, t+12h, t+18h, t+24h)
  - Extracts tp06 from zarr output; applies ×1000 (no exp); clamps ≥ 0
  - Subsets to Myanmar bbox: lat 9–29°N, lon 92–102°E (21×11 at 1.0°)
  - Handles descending lat (90→-90) by checking and flipping to ascending
  - Prepends t+0h init frame → [5 × 21 × 11] total
  - Writes `precipitation.bin` as float32 C-order
  - Writes `forecast.json` with schema v2.0
  - Records GPU type and VRAM in `inference_config` field
  - JAX env vars set before imports: XLA_PYTHON_CLIENT_PREALLOCATE=false,
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
  - All Aurora1p5, tp1h, sic, log-untransform references removed

- [x] T011 Staged VRAM test:
  - Stage 1: GPU detection + baseline VRAM (PyTorch + JAX separately)
  - Stage 2: Model load + VRAM snapshot
  - Stage 3a: Single-step smoke test (nsteps=1) — OOM → exit code 1
  - Stage 3b: Full 4-step run (nsteps=4) if smoke test passes
  - --smoke-test-only flag: stop after stage 3a
  - VRAM reported via both torch.cuda and jax.devices()[0].memory_stats()

  **Discovery**: JAX uses its own GPU memory pool separate from PyTorch's allocator.
  torch.cuda.memory_allocated() reflects only data-transport tensors, not model weights.
  XLA_PYTHON_CLIENT_PREALLOCATE=false prevents JAX pre-allocating ~90% of VRAM.
  This is documented in forecast.json inference_config.jax_env.

  **NOT applied**: model.to(bfloat16) — GraphCastSmall already uses bfloat16 internally
  via casting.Bfloat16Cast in JAX. Calling .to(bfloat16) on the torch.nn.Module wrapper
  has no effect on JAX weights and must NOT be done.

- [ ] T012 Update `frontend/src/data/types.ts`:
  - `n_times: number` (was hard-coded expectations)
  - `native_timestep_hours: number` (new field)
  - `spatial_resolution_deg: number`
  - `display_resolution_deg: number | null` (new field)
  - `inference_config: { device: string; peak_vram_gb: number | null } | undefined`
  - `transformation_provenance` in precipitation variable metadata
  - Remove any Aurora/sic/tp1h references

**Checkpoint**: `generate_forecast.py --source arco --init-time 2023-01-01T00:00:00Z` produces
`data/forecast/precipitation.bin` (4,620 bytes) and valid `forecast.json` on a CUDA machine.

---

## Phase 3: Demo Data Update

**Goal**: Update `scripts/generate_demo_data.py` for 5 frames, 21×11 grid, tp06-only.

- [ ] T013 Rewrite `scripts/generate_demo_data.py`:
  - N_TIMES = 5 (was 49 for 48h, now 5 for 24h at 6h steps)
  - N_LAT = 21, N_LON = 11 (was 81, 41 for 0.25°)
  - Generate precipitation only (no temperature)
  - Synthetic tp06: Gaussian blobs at realistic Myanmar positions, range 0–80 mm/6h
  - Write `precipitation.bin` [5 × 21 × 11] float32
  - Write `forecast.json` with schema v2.0 matching plan.md
  - `is_demo: true`, `model: "GraphCastSmall"`, `spatial_resolution_deg: 1.0`
  - `native_timestep_hours: 6`, `forecast_horizon_hours: 24`, `n_times: 5`
  - times_utc: 5 entries at 6h spacing
  - Remove temperature generation entirely

- [ ] T014 Run `scripts/generate_demo_data.py` and verify `data/demo/` contains 2 files:
  - `data/demo/forecast.json`
  - `data/demo/precipitation.bin` (4,620 bytes exactly)

- [ ] T015 Commit updated `data/demo/` artifacts to git

**Checkpoint**: `data/demo/precipitation.bin` is 4,620 bytes. `forecast.json` has `n_times: 5`,
`spatial_resolution_deg: 1.0`, `is_demo: true`.

---

## Phase 4: Frontend Updates

**Goal**: Update frontend to match GraphCastSmall artifact format (6h steps, 5 frames, 21×11,
precipitation only, tp06 disclosure).

### Already completed (precipitation-only migration)

- [x] VariableSwitcher.tsx replaced with no-op `export {}`
- [x] ForecastStore.ts: removed `activeVariable`, `setVariable`, temperature state
- [x] ForecastLoader.ts: precipitation only, no temperature fetch
- [x] WeatherMap.tsx: precipitation-only rendering
- [x] Legend.tsx: precipitation-only
- [x] Header.tsx: title updated, staleness logic present

### Remaining frontend updates

- [ ] T020 Update `frontend/src/components/Timeline.tsx`:
  - Slider `max` must come from `metadata.n_times - 1` (not hard-coded 48 or 168)
  - Step buttons advance by 1 index (each index = 6h)
  - Hour markers: show "0h · 6h · 12h · 18h · 24h" (not "0h · 24h · 48h")
  - Lead time display: `currentHour * (metadata.native_timestep_hours ?? 6)` hours
  - Animation loop: step by 1 index; loop from index 4 back to 0

- [ ] T021 Update `frontend/src/components/Header.tsx`:
  - Title: "Myanmar 24h Precipitation" (was "Myanmar 48h Precipitation")
  - Staleness threshold: 48h (was 72h; forecast horizon is now 24h)
  - Resolution display: use `metadata.spatial_resolution_deg` and `metadata.display_resolution_deg`

- [ ] T022 Update `frontend/src/components/InfoPanel.tsx`:
  - Horizon field: read from `metadata.forecast_horizon_hours` (was hard-coded "2 days")
  - Limitations bullet: update resolution reference from 0.25° to read from metadata
  - Limitations bullet: update precipitation accumulation period from "1-hour" to "6-hour"
  - Add `native_timestep_hours` row to metadata grid
  - Remove any remaining Aurora/tp1h/sic references

- [ ] T023 Update `frontend/src/data/ForecastLoader.ts`:
  - Verify `ForecastData` has only `precipitation` (no temperature)
  - Verify loader does not hard-code array dimensions; derive from metadata

- [ ] T024 Verify `frontend/src/map/WeatherMap.tsx`:
  - Legend and popup reference tp06 / mm/6h correctly
  - No Aurora or tp1h strings present

**Checkpoint**: `npm run dev` with updated demo data shows Myanmar map, 5-step timeline (0, 6,
12, 18, 24h markers), and mm/6h legend. Slider max = 4. Header shows "Myanmar 24h Precipitation".

---

## Phase 5: Validation Script Update

**Goal**: Update `scripts/validate_forecast.py` for new dimensions and tp06 semantics.

- [ ] T025 Update `scripts/validate_forecast.py`:
  - Expected shape: [5, 21, 11] (was [169, 81, 41] or [49, 81, 41])
  - Check: `n_times == 5`, `n_lat == 21`, `n_lon == 11`
  - Check: `native_timestep_hours == 6`
  - Check: `forecast_horizon_hours == 24`
  - Check: tp06 ≥ 0 (no negative precipitation)
  - Check: tp06 < 500 mm/6h (extreme upper bound for tropical rainfall)
  - Check: timestamps 6h apart and monotonically increasing
  - Check: `transformation_provenance.log_transform_applied == false`
  - Remove temperature validation, t2m range check, sic_handling check

- [ ] T026 Run `scripts/validate_forecast.py --data-dir data/demo/` and verify all checks PASS

**Checkpoint**: All validation checks PASS for the updated demo data.

---

## Phase 6: Colab Notebook Rewrite

**Goal**: Replace `notebooks/aurora_myanmar_forecast.ipynb` with `notebooks/graphcast_myanmar_forecast.ipynb`.

- [x] T030 Rename `notebooks/aurora_myanmar_forecast.ipynb` → `notebooks/graphcast_myanmar_forecast.ipynb`
  - `git mv` to preserve git history

- [x] T031 Rewrite notebook content for GraphCastSmall:
  - Section 1: Title, architecture summary (GraphCastSmall, tp06, 6h, 24h)
  - Section 2: Install `earth2studio[graphcast]` (or equivalent extras); no cudf uninstall needed
  - Section 3: GPU detection and baseline VRAM measurement
  - Section 4: ARCO data fetch test (t-6h and t+0h) — no IFS fallback complexity for initial test
  - Section 5: Load GraphCastSmall weights, measure VRAM
  - Section 6: Single-step inference VRAM test (staged VRAM check per §XI)
    - If OOM: stop with clear error; do not proceed to full run
  - Section 7: Full 24h forecast run (4 steps)
  - Section 8: Post-process tp06 → mm/6h (×1000, no exp, clamp ≥ 0)
  - Section 9: Myanmar subset (21×11)
  - Section 10: Write artifacts (precipitation.bin, forecast.json)
  - Section 11: Validate artifacts
  - Section 12: Git commit + push to trigger GitHub Actions deploy
  - Remove all Aurora1p5, sic patch, tp1h, exp(), torch references

- [ ] T032 Verify notebook runs from top to bottom on Colab without error on a fresh runtime
  (with GPU; skip full inference in dry run, verify at least up to Section 6)

**Checkpoint**: Notebook renamed. Content updated for GraphCastSmall. Staged VRAM test in place.

---

## Phase 7: Integration Testing

- [ ] T040 Copy demo data to frontend/public/data/ and run `npm run dev`; confirm:
  - 5-frame timeline with 6h labels
  - Precipitation overlay renders on Myanmar map
  - Slider max = 4; steps advance correctly
  - Header shows "Myanmar 24h Precipitation"
  - InfoPanel shows correct GraphCastSmall metadata

- [ ] T041 Run `npm run build` in `frontend/` and confirm no TypeScript errors

- [ ] T042 Profile frame transitions in browser DevTools: confirm < 200ms

- [ ] T043 Push to main; confirm GitHub Actions green; confirm GitHub Pages loads demo data
  with DEMO DATA banner visible

---

## Phase 8: README Update

- [ ] T050 Update `README.md`:
  - Architecture section: GraphCastSmall (1.0°, 6h, 24h, tp06)
  - Remove Aurora1p5 references
  - Update pipeline section: ARCO/IFS, two-timestep init, tp06 → mm/6h
  - Update Colab notebook link to graphcast_myanmar_forecast.ipynb
  - Update data format section: [5 × 21 × 11], 4.6 KB
  - Note: T4 (16 GB) VRAM compatibility pending experimental verification

---

## Dependencies & Execution Order

- **Phase 1** (Spec Kit): COMPLETE
- **Phase 2** (Pipeline): Start now — depends on Phase 1 spec
- **Phase 3** (Demo Data): After Phase 2 (format locked by generate_forecast.py schema)
- **Phase 4** (Frontend): After Phase 3 (needs committed demo data for dev testing)
- **Phase 5** (Validation): After Phase 3 (needs correct dimensions)
- **Phase 6** (Notebook): After Phase 2 (needs working pipeline code to port)
- **Phase 7** (Integration): After Phases 3, 4, 5
- **Phase 8** (README): After Phase 7

### MVP Path (Phases 2–5 + deployment)

Complete Phases 2 → 3 → 4 → 5 → deploy for a working GitHub Pages app with:
- Myanmar 24h precipitation map
- 5-step timeline at 6h increments
- GraphCastSmall demo data with DEMO banner

---

## Acceptance Criteria Verification Map

| Criterion | Tasks |
|-----------|-------|
| Earth2Studio actually used (GraphCastSmall) | T010 |
| Myanmar forecast generated (24h) | T010 |
| 6h native steps (no interpolation) | T010, T020 |
| tp06 precipitation available | T010 |
| No log/exp transform applied | T010, T025 |
| Units correct (mm/6h) | T010, T013 |
| Two-timestep init (t-6h + t+0h) | T010 |
| Staged VRAM test | T011, T031 |
| Timestamps correct (6h spacing) | T013, T025 |
| Resolution documented (1.0°) | T012, T022 |
| Myanmar map visible | T024 |
| Precipitation overlay works | T024 |
| Legend correct (mm/6h) | T024 |
| 5-step slider (0,6,12,18,24h) | T020 |
| Lead time display correct | T020 |
| Play/Pause animation | T020 |
| Point inspector (popup) | T024 |
| tp06 disclosure in UI | T022 |
| Header: 24h title | T021 |
| InfoPanel: 6h accumulation disclosure | T022 |
| Resolution honesty disclosure | T022 |
| TypeScript strict (no errors) | T041 |
| Demo data correct format | T013–T015 |
| Validation all PASS | T026 |
| GitHub Pages deployment | T043 |
| Notebook renamed + rewritten | T030–T032 |
| README updated | T050 |
| No secrets committed | (existing .gitignore) |
