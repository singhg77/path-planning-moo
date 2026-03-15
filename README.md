# Multi-Objective Aircraft Path Planning — Interactive Web App

A browser-based interactive visualisation of multi-objective path planning for a low-level tactical aircraft flying from a western airfield to an eastern landing zone across 140 km of varied terrain with three circular bad-weather storm cells blocking the direct route. The NSGA-II genetic algorithm generates a Pareto front of 150 solutions trading off mission time, survival risk (weather exposure + radar detection + terrain collision), and fuel burn. Three real-time weight sliders let you instantly explore the entire trade-off space: drag a slider and the optimal path, altitude profile, Pareto scatter plots, and parallel coordinates all update simultaneously. Storm positions are adjustable via input fields, and a Re-optimize button re-runs the full NSGA-II optimisation for the new storm layout.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:8050](http://localhost:8050) in your browser.

If `path_results.npz` is not present in the working directory, the app will automatically run a full NSGA-II optimisation on startup (approximately 1–2 minutes). Subsequent launches load the cached results instantly.

## Panel Descriptions

### 2D Mission Map (top-left)
Top-down view of the 140 km × 80 km mission area. The terrain elevation is shown as a heatmap. Blue dashed circles mark the three storm cells (Alpha, Bravo, Charlie). All 150 Pareto-optimal paths are drawn as faint blue lines; the currently selected path is highlighted in bright cyan. Green square = source airfield; red star = destination landing zone. The direct-route reference line is shown as a white dotted line.

### Altitude AGL Profile (top-right)
Altitude Above Ground Level (AGL) along the active path's along-track distance. Colour bands show the three altitude regimes: red = terrain collision risk (below 61 m AGL), green = ideal stealthy corridor (61–300 m), orange = radar-detectable zone (above 300 m). The gold dash-dot line marks the fuel-optimal altitude (500 m). The filled cyan curve shows the selected path's AGL profile.

### Pareto Scatter Plots (middle-right)
Three side-by-side scatter plots showing every Pareto solution projected onto pairs of objectives (Time×Risk, Risk×Fuel, Time×Fuel). The third objective is encoded as the point colour. Point size is proportional to crowding distance — larger points occupy more unique regions of the Pareto front and represent more diverse solutions. Special markers: triangle = best-time solution, diamond = best-risk solution, square = best-fuel solution. The active (currently selected) solution is shown as a cyan star.

### Parallel Coordinates (bottom-right)
All 150 Pareto solutions drawn as polylines across three parallel axes (Time in minutes, Survival Risk, Fuel in kg). Lines are coloured by normalised risk. Filtering by brushing an axis shows which solutions satisfy multiple constraints simultaneously. The layout makes cross-objective trade-offs visible at a glance.

## Objective Sliders

Three sliders at the bottom set the relative importance of each objective:
- **TIME** — minimise total mission time (seconds)
- **RISK** — minimise survival risk (weather exposure + altitude violations)
- **FUEL** — minimise fuel burn (kg)

Weights are normalised before use so only their ratios matter. Setting all three equal gives the balanced Pareto solution closest to the utopia point.

## Storm Repositioning

Adjust the X/Y coordinates of each storm via the input fields below the map. After changing positions, click **Re-optimize** to run a fresh NSGA-II optimisation for the new layout (approximately 1–2 minutes). The **Reset Storms** button restores the original positions.

## Deployment on Railway (free tier)

1. Push this repository to GitHub.
2. Go to [railway.app](https://railway.app) and create a new project.
3. Select **Deploy from GitHub repo** and connect your repository.
4. Railway automatically detects `Procfile` and `requirements.txt`.
5. The app is deployed at a public URL within minutes; every push to `main` triggers an automatic redeploy.
6. Set environment variable `PORT` if Railway does not inject it automatically (Dash reads it via `os.environ.get("PORT", 8050)`).

## Screenshot

![App screenshot placeholder](screenshot.png)

## Project Structure

```
path_planning/
├── app.py                   # Dash web application (this file)
├── weather_path_planning.py # Core domain module (NSGA-II, physics, helpers)
├── requirements.txt         # Python dependencies
├── Procfile                 # Gunicorn entry point for Railway/Heroku
├── runtime.txt              # Python version pin
├── .gitignore               # Excludes *.npz, venv, __pycache__, etc.
└── README.md                # This file
```
