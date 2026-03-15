# Session Notes — Multi-Objective Path Planning Project

A record of what was built, why key decisions were made, and what to know before
continuing work on this repository.

---

## What This Project Is

A 3D aircraft path planning system that uses **NSGA-II (multi-objective genetic
algorithm)** to find Pareto-optimal routes from a source to a destination across
terrain with three storm keep-out zones. Three objectives are traded off
simultaneously: mission time, survival risk, and fuel burn. An interactive
visualization lets a user drag sliders to explore the trade-off in real time.

---

## What Was Built (In Order)

### Phase 1 — Core Optimization

Started from an existing battle simulation (`path_planning.py`) and rebuilt it as
a focused weather-avoidance scenario:

- **`weather_path_planning.py`** — all-in-one module:
  - Synthetic terrain model (bicubic spline over Gaussian hills)
  - 3D path encoding (8 waypoints × (x, y, AGL))
  - `evaluate_path()` — vectorised integration of all 3 objectives
  - `WeatherPathProblem` / `RouteSampling` — pymoo problem class + warm-start sampler
  - `_run_nsga2()` — runs NSGA-II, returns Pareto front (F, X)
  - `launch_interactive()` — full matplotlib interactive figure

### Phase 2 — Interactive Visualization

Built a multi-panel matplotlib figure with:
- 2D top-down map with draggable storm circles
- Altitude AGL profile with stealth corridor markers
- 3D Pareto scatter → then replaced with 3 pairwise 2D scatter plots
- Parallel coordinates plot (PCP) for all 3 objectives simultaneously
- Three weight sliders → ASF selection → live path update

### Phase 3 — Web Application

- **`app.py`** — full Dash/Plotly web app port of the matplotlib viz
- Storm position controls, Re-optimize button, dark theme throughout
- **`requirements.txt`, `Procfile`, `runtime.txt`** for Railway deployment
- Repository pushed to: https://github.com/singhg77/path-planning-moo

### Phase 4 — Documentation

- **`Algo-README.md`** — detailed algorithm explanation (problem formulation, NSGA-II
  mechanics, all objective functions with maths, key implementation decisions)
- **`Visualization-README.md`** — panel-by-panel guide with embedded screenshots
- **`Performance-README.md`** — real benchmarks: 16 configurations (pop 50–200,
  ngen 50–200), measured wall times, hypervolume, convergence generations

---

## Key Design Decisions and Why

### 1. Graded radar detection penalty (not binary)

**Problem:** With a flat binary detection penalty, the optimizer treated "fly high
through storms" as roughly equivalent to "fly low around storms". Setting Risk=1.0
still selected high-altitude paths.

**Fix:** Penalty scales linearly from 0 at 300 m AGL to W_DETECT=3.0 at 600 m.
Every extra metre above 300 m continuously increases risk. W_DETECT=3.0 (3× stronger
than weather exposure) ensures the optimizer firmly stays below 300 m when risk is
prioritised.

```python
detect_excess = max(0, (agl - DETECT_AGL)) / (MAX_AGL - DETECT_AGL)
# 0 at 300 m, 1.0 at 600 m — multiplied by W_DETECT = 3.0
```

### 2. ASF normalisation for sliders

**Problem:** Mission time (~1700 s) is ~10× larger than fuel (~160 kg). The ASF
`max(F_i / w_i)` was always dominated by time regardless of slider positions —
making the fuel and risk sliders useless.

**Fix:** Normalise F to [0, 1] per objective before calling ASF.

```python
F_norm = (F - F.min()) / (F.max() - F.min())   # each objective [0, 1]
return ASF().do(F_norm, 1.0 / w).argmin()
```

### 3. Full AGL range in warm-start sampling

**Problem:** Initial population sampled AGL only in 61–300 m (stealth corridor).
The Pareto front had no high-altitude solutions — the fuel slider appeared to do
nothing.

**Fix:** Sample AGL uniformly across full range 30–600 m. Initial population now
contains both stealthy low-level paths and fuel-efficient high-altitude paths.

### 4. Explicit numpy seed for determinism

**Problem:** pymoo 0.6 does NOT seed numpy's global RNG when you pass `seed=` to
`minimize()`. Our `RouteSampling` uses `np.random.normal` — so every Re-optimize
with identical storm positions produced a different result.

**Fix:** `np.random.seed(seed)` called explicitly before `minimize()`.

### 5. Persistent PCP line (no `cla()` on slider move)

**Problem:** `cla()` + redraw of 150 background lines on every slider move caused
the cyan active line to be visually buried under overlapping faint lines.

**Fix:** Background lines drawn once at init. Only the active line's ydata is
updated on slider change. Added glow effect (thick transparent layer behind the
line).

### 6. pymoo made optional for server deployment

**Problem:** Railway deployment failed because `pymoo==0.6.1.4` had no pre-built
wheel. Even after relaxing the version pin, the build was fragile.

**Fix:** All pymoo imports wrapped in `try/except ImportError`. `_PYMOO_AVAILABLE`
flag controls whether Re-optimize is active. App starts and the full visualization
works using pre-computed `path_results.npz` even without pymoo installed.

### 7. N_WP=8 waypoints (not 5)

**Problem:** With 5 waypoints, the optimizer clipped Storm-Alpha when trying to
thread between Alpha and Bravo — couldn't route precisely around multiple obstacles
simultaneously.

**Fix:** Increased to 8 waypoints. With 8, the optimizer can route north of Alpha,
through the gap, and south of Charlie in a single smooth trajectory.

---

## Known Issues / Limitations

| Issue | Status | Notes |
|-------|--------|-------|
| Railway deployment may fail if pymoo wheel unavailable | Mitigated | App works without pymoo using pre-computed results |
| Re-optimize takes ~15 s (blocks UI in web app) | Open | No background task implemented; consider `dash.long_callback` |
| Storm drag only in desktop viz | Open | Web app uses coordinate inputs instead of drag |
| MILP solver (`milp_path_planning.py`) was started but not fully integrated | Open | File exists but not linked to web app |
| Pareto front is non-convex — weighted sum methods would miss concave regions | By design | NSGA-II handles this correctly |
| `path_results.npz` not in git (gitignored) | By design | Regenerated on first run; too large for git |

---

## Performance Summary

From real benchmarks (see `Performance-README.md`):

- Each path evaluation costs **~0.65 ms** (constant regardless of pop/gen)
- Default run (pop=150, ngen=150): **14.8 s**, 22,500 evaluations
- Population converges to fully non-dominated front at **generation 46**
- Beyond ~20,000 evaluations, HV improvement is <1% per doubling of compute
- Best quick config: pop=100, ngen=100 (~7 s, good quality)

---

## Repository Structure

```
path-planning-moo/
├── weather_path_planning.py   # Core: terrain, objectives, NSGA-II, matplotlib viz
├── viz_sliders.py             # Desktop launcher (loads npz, calls launch_interactive)
├── app.py                     # Dash web application
├── path_results.npz           # Pre-computed Pareto front (gitignored, regenerated on run)
├── requirements.txt           # Python deps (dash, plotly, pymoo>=0.6.0, etc.)
├── Procfile                   # Railway/Heroku: web: gunicorn app:server
├── runtime.txt                # python-3.10.12
├── .gitignore                 # excludes *.npz, *.png, venv/, .env
├── README.md                  # Quick start + deployment instructions
├── Algo-README.md             # Algorithm deep-dive (problem formulation + NSGA-II)
├── Visualization-README.md    # UI guide with screenshots
├── Performance-README.md      # Measured benchmarks (16 configurations)
└── SESSION-NOTES.md           # This file
```

---

## How to Run

### Desktop interactive viz
```bash
pip install -r requirements.txt
python weather_path_planning.py   # runs optimization + opens matplotlib figure
# OR if path_results.npz already exists:
python viz_sliders.py             # skips optimization, opens viz directly
```

### Web app (local)
```bash
python app.py
# Open http://localhost:8050
```

### Web app (Railway deployment)
1. Connect GitHub repo `singhg77/path-planning-moo` to Railway
2. Railway auto-detects `Procfile` → deploys with gunicorn
3. App uses pre-computed `path_results.npz` (committed) or regenerates on startup

---

## Continuing Work — Suggested Next Steps

1. **Background Re-optimize in web app** — use `dash.long_callback` with a progress
   indicator so the UI doesn't freeze during the 15 s NSGA-II run
2. **Storm drag in web app** — implement via `clickData` on the Plotly map figure
   (click near storm → select it; next click → move it)
3. **MILP solver integration** — `milp_path_planning.py` exists; wire it into the
   web app as a second solver tab for comparison with NSGA-II
4. **3D path visualization** — add a rotating 3D Plotly figure (go.Scatter3d) as an
   optional panel
5. **Export selected path** — add a "Download JSON" button that exports the active
   path's waypoints and metrics as a file
