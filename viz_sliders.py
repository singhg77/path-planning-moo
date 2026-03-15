"""
viz_sliders.py — Interactive Pareto Explorer
=============================================
Loads the saved optimisation results (path_results.npz) and opens the
full interactive slider figure without re-running NSGA-II.

Usage:
    python3 viz_sliders.py
"""

import numpy as np
import os

if not os.path.exists("path_results.npz"):
    raise FileNotFoundError(
        "path_results.npz not found — run weather_path_planning.py first."
    )

data = np.load("path_results.npz")
F, X = data["F"], data["X"]

print(f"Loaded {len(F)} Pareto solutions from path_results.npz")
print(f"  Mission time  : {F[:,0].min()/60:.1f} – {F[:,0].max()/60:.1f} min")
print(f"  Survival risk : {F[:,1].min():.1f} – {F[:,1].max():.1f}")
print(f"  Fuel burn     : {F[:,2].min():.1f} – {F[:,2].max():.1f} kg")
print("\nOpening interactive slider figure …")

from weather_path_planning import launch_interactive
launch_interactive(F, X)
