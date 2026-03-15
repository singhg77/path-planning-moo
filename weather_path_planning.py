"""
Aircraft Path Planning — Weather Zone Avoidance with Terrain Following
=======================================================================
A low-level aircraft flies SOURCE (west, 3-D) → DESTINATION (east, 3-D)
across 140 km of varied terrain. Three circular bad-weather cells (storms)
sit directly in the path. The aircraft CAN enter them but accumulates
damage probability proportional to time spent inside.

Flying low hugs the terrain for stealth but raises fuel burn (denser air
= more drag) and risks terrain collision. Flying high cuts drag but
exposes the aircraft to radar detection.

Decision variables  (N_WP × 3)
--------------------------------
    x_i   – waypoint easting   (km)
    y_i   – waypoint northing  (km)
    agl_i – altitude AGL       (m)

Objectives  (all minimised)
-----------------------------
    f1  Mission time   – total 3-D flight time                   (seconds)
    f2  Survival risk  – weather exposure + altitude violations   (s-equiv)
    f3  Fuel burn      – Σ segment_km × fuel_rate(agl)           (kg)

Altitude bands
---------------
    < 61 m AGL   (< 200 ft) : terrain-collision risk  → adds to f2
    61–300 m AGL             : stealthy low-level corridor (safe from radar)
    > 300 m AGL              : radar detectable        → adds to f2
    > 500 m AGL              : fuel-optimal (thin air, low drag)
    Key tension: fuel wants HIGH (≥ 500 m), stealth wants LOW (< 300 m).

Interactive use
---------------
    After optimisation an interactive figure opens with three sliders
    to blend the objective weights in real time — the displayed path
    updates immediately to show the best trade-off match.
"""

from __future__ import annotations
import warnings
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401
from scipy.interpolate import RectBivariateSpline

from pymoo.core.problem  import Problem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm  import PM
from pymoo.optimize   import minimize
from pymoo.decomposition.asf import ASF

warnings.filterwarnings("ignore")


# ── SCENARIO PARAMETERS ────────────────────────────────────────────────────────

# 3-D fixed endpoints: [x_km, y_km, agl_m]
SOURCE = np.array([  5.0, 40.0,  80.0])   # airfield at ~80 m AGL (west)
DEST   = np.array([145.0, 40.0, 120.0])   # elevated landing zone at ~120 m AGL (east)

X_BOUNDS = (0.0, 150.0)   # km east–west
Y_BOUNDS = (0.0,  80.0)   # km north–south

# Aircraft performance
SPEED        = 0.083   # km/s ≈ 300 km/h (low-level tactical)
MIN_AGL      = 30.0    # m  absolute floor
MAX_AGL      = 600.0   # m  planning ceiling
SAFE_AGL     = 61.0    # m ≈ 200 ft — below: terrain-collision risk
DETECT_AGL   = 300.0   # m — above: radar detectable
FUEL_OPT_AGL = 500.0   # m — minimum-drag altitude (above radar detection floor)
FUEL_BASE    = 1.0     # kg/km at FUEL_OPT_AGL
FUEL_LOW_K   = 0.50    # extra fuel fraction at AGL = 0 (dense-air drag)

# Survival-risk weights (per second of exposure)
W_WEATHER     = 1.0
W_TERRAIN_HIT = 0.8    # below SAFE_AGL  (terrain collision)
W_DETECT      = 3.0    # altitude-graded above DETECT_AGL (see evaluate_path)
                        # graded means: at DETECT_AGL → 0, at MAX_AGL → W_DETECT*time
                        # strong value forces stay-low behaviour even when fuel=0
W_UNDERGROUND = 20.0   # below terrain — catastrophic

# Three storms: Bravo is centred exactly on the route; Alpha/Charlie slightly offset.
# In the interactive figure storm circles are draggable (click+drag near their centre).
WEATHER = [
    {"center": np.array([40.0, 44.0]), "radius": 11.0, "name": "Storm-Alpha"},    # offset N
    {"center": np.array([80.0, 40.0]), "radius": 14.0, "name": "Storm-Bravo"},    # ON route
    {"center": np.array([118.0, 36.5]), "radius": 10.0, "name": "Storm-Charlie"}, # offset S
]
# Keep original positions for reset
_WEATHER_ORIG = [{"center": s["center"].copy(), "radius": s["radius"],
                  "name": s["name"]} for s in WEATHER]

N_WP   = 8    # intermediate waypoints (more = finer routing around storms)
N_SAMP = 20   # integration samples per segment


# ── TERRAIN ───────────────────────────────────────────────────────────────────

class Terrain:
    """
    Synthetic terrain along the SOURCE→DEST corridor.
    Features:
      • Rolling base (sinusoidal)
      • Central ridge at x≈60 km — cuts across the direct route
      • Northern hill at (32, 60) — blocks the northern bypass
      • Southern saddle at (95, 22) — lower southern bypass
      • Eastern ridge at (128, 44) — near the destination
      • Valley corridor at (75, 52) — a natural nap-of-earth channel
    """
    RESOLUTION = 120

    def __init__(self) -> None:
        np.random.seed(42)
        x = np.linspace(*X_BOUNDS, self.RESOLUTION)
        y = np.linspace(*Y_BOUNDS, self.RESOLUTION)
        X, Y = np.meshgrid(x, y)

        elev = (
              80 * np.sin(0.07 * X) * np.cos(0.10 * Y)
            + 50 * np.sin(0.16 * X + 1.2) * np.cos(0.22 * Y + 0.8)
            # Central ridge — blocks direct route
            + 420 * np.exp(-((X - 60)**2 + (Y - 40)**2) / 180)
            # Northern hill — makes northern bypass costly
            + 300 * np.exp(-((X - 32)**2 + (Y - 60)**2) / 140)
            # Southern saddle (lower)
            + 160 * np.exp(-((X - 95)**2 + (Y - 22)**2) / 120)
            # Eastern ridge near destination
            + 290 * np.exp(-((X - 128)**2 + (Y - 44)**2) / 160)
            # Valley corridor north of direct path — natural NOE channel
            - 50  * np.exp(-((X - 75)**2 + (Y - 52)**2) / 280)
            + np.random.normal(0, 8, X.shape)
        )
        self._elev   = np.maximum(elev, 0.0)
        self._x, self._y = x, y
        self._interp = RectBivariateSpline(x, y, self._elev.T)

    def at(self, x: float, y: float) -> float:
        return float(self._interp(
            np.clip(x, *X_BOUNDS), np.clip(y, *Y_BOUNDS))[0, 0])

    def at_points(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return self._interp.ev(
            np.clip(xs, *X_BOUNDS), np.clip(ys, *Y_BOUNDS))

    def grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._x, self._y, self._elev


TERRAIN = Terrain()


# ── FUEL MODEL ────────────────────────────────────────────────────────────────

def fuel_rate(agl_arr: np.ndarray) -> np.ndarray:
    """kg/km — penalty for flying below FUEL_OPT_AGL (dense-air drag)."""
    low = np.maximum(0.0, (FUEL_OPT_AGL - agl_arr) / FUEL_OPT_AGL) * FUEL_LOW_K
    return FUEL_BASE + low


# ── PATH CONSTRUCTION ─────────────────────────────────────────────────────────

def build_path(waypoints: np.ndarray) -> np.ndarray:
    """
    Decode (N_WP, 3) waypoints [x_km, y_km, agl_m]
    → (N_WP+2, 3) full path [x_km, y_km, alt_amsl_m].

    SOURCE[:2] / DEST[:2] are fixed xy endpoints.
    SOURCE[2]  / DEST[2]  are the prescribed AGL at those endpoints.
    """
    def to_amsl(xy_agl: np.ndarray) -> np.ndarray:
        h = TERRAIN.at(float(xy_agl[0]), float(xy_agl[1])) + xy_agl[2]
        return np.array([xy_agl[0], xy_agl[1], h])

    path = [to_amsl(SOURCE)]
    for wp in waypoints:
        path.append(to_amsl(wp))
    path.append(to_amsl(DEST))
    return np.array(path, dtype=float)


# ── PATH EVALUATION ───────────────────────────────────────────────────────────

def evaluate_path(path: np.ndarray) -> Tuple[float, float, float]:
    """
    Return (mission_time_s, survival_risk, fuel_kg).
    Vectorised over all segments × N_SAMP integration points.
    """
    t     = np.linspace(0.0, 1.0, N_SAMP)
    p1    = path[:-1]
    p2    = path[1:]
    n_seg = len(p1)

    pts  = p1[:, None, :] + t[None, :, None] * (p2 - p1)[:, None, :]
    xs   = pts[:, :, 0].ravel()
    ys   = pts[:, :, 1].ravel()

    ter      = TERRAIN.at_points(xs, ys).reshape(n_seg, N_SAMP)
    alt_amsl = pts[:, :, 2]
    agl      = alt_amsl - ter

    dp       = p2 - p1
    dp_km    = np.column_stack([dp[:, :2], dp[:, 2] / 1000.0])
    seg3d_km = np.linalg.norm(dp_km, axis=1)
    seg2d_km = np.linalg.norm(dp[:, :2], axis=1)
    seg_time = seg3d_km / SPEED

    # f1 — mission time
    mission_time = float(np.sum(seg_time))

    # f2 — survival risk
    pts2    = pts[:, :, :2]
    wx_frac = np.zeros(n_seg)
    for storm in WEATHER:
        d        = np.linalg.norm(pts2 - storm["center"], axis=2)
        wx_frac += (d < storm["radius"]).mean(axis=1)
    weather_risk = float(np.sum(wx_frac * seg_time * W_WEATHER))

    underground_frac = (agl < 0).mean(axis=1)
    low_frac         = ((agl >= 0) & (agl < SAFE_AGL)).mean(axis=1)

    # Detection penalty is altitude-graded: 0 at DETECT_AGL, W_DETECT at MAX_AGL.
    # This makes flying slightly above 300 m cheap; flying at 500 m+ very costly.
    # Effect: risk-focused paths stay firmly below 300 m rather than trading off
    # detection for storm avoidance (which was the bug when W_DETECT was flat + small).
    detect_excess = np.maximum(0.0, (agl - DETECT_AGL)) / (MAX_AGL - DETECT_AGL)
    alt_risk = float(np.sum((
          underground_frac           * W_UNDERGROUND
        + low_frac                   * W_TERRAIN_HIT
        + detect_excess.mean(axis=1) * W_DETECT
    ) * seg_time))
    survival_risk = weather_risk + alt_risk

    # f3 — fuel burn
    mean_agl  = agl.mean(axis=1)
    fuel_kg   = float(np.sum(seg2d_km * fuel_rate(mean_agl)))

    return mission_time, survival_risk, fuel_kg


# ── PYMOO PROBLEM ─────────────────────────────────────────────────────────────

class WeatherPathProblem(Problem):
    def __init__(self) -> None:
        xl = np.tile([X_BOUNDS[0], Y_BOUNDS[0], MIN_AGL], N_WP)
        xu = np.tile([X_BOUNDS[1], Y_BOUNDS[1], MAX_AGL], N_WP)
        super().__init__(n_var=N_WP*3, n_obj=3, n_ieq_constr=0,
                         xl=xl, xu=xu, type_var=float)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        F = np.zeros((X.shape[0], 3))
        for i in range(X.shape[0]):
            path = build_path(X[i].reshape(N_WP, 3))
            F[i] = evaluate_path(path)
        out["F"] = F


# ── WARM-START SAMPLING ───────────────────────────────────────────────────────

class RouteSampling(Sampling):
    """Seed near straight SOURCE→DEST line; AGL drawn from ideal band."""
    def _do(self, problem: Problem, n_samples: int, **kwargs) -> np.ndarray:
        xl, xu = problem.xl, problem.xu
        X = np.zeros((n_samples, problem.n_var))
        for i in range(n_samples):
            for j in range(N_WP):
                t   = (j + 1) / (N_WP + 1)
                mid = SOURCE[:2] + t * (DEST[:2] - SOURCE[:2])
                x   = float(np.clip(mid[0] + np.random.normal(0, 18),
                                    xl[j*3],   xu[j*3]))
                y   = float(np.clip(mid[1] + np.random.normal(0, 20),
                                    xl[j*3+1], xu[j*3+1]))
                # Sample full AGL range so the Pareto front covers high-altitude
                # (fuel-efficient) solutions as well as low-level stealthy ones.
                agl = float(np.random.uniform(MIN_AGL, MAX_AGL))
                X[i, j*3:j*3+3] = [x, y, agl]
        return X


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _path_agls(path: np.ndarray) -> np.ndarray:
    return np.array([path[k, 2] - TERRAIN.at(path[k, 0], path[k, 1])
                     for k in range(len(path))])

def _path_along_track(path: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(
        np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))))

def _path_2d_length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))

def _best_idx(F: np.ndarray, w: np.ndarray) -> int:
    """Select the Pareto solution that best matches weight vector w.

    ASF operates on normalised objectives (each scaled to [0,1] using the
    min/max of the current Pareto front).  Without normalisation the time
    objective (~1700 s) dominates the fuel (~160 kg) and risk (~0-2200)
    terms regardless of the slider weights — making the sliders useless.
    """
    w = np.maximum(w, 1e-6)
    w = w / w.sum()
    F_min   = F.min(axis=0)
    F_range = np.maximum(F.max(axis=0) - F_min, 1e-9)
    F_norm  = (F - F_min) / F_range          # each objective in [0, 1]
    return int(ASF().do(F_norm, 1.0 / w).argmin())


# ── INTERACTIVE FIGURE ────────────────────────────────────────────────────────

def _run_nsga2(pop: int = 100, ngen: int = 80, seed: int = 42) -> tuple:
    """Run NSGA-II with current WEATHER positions; return (F, X).

    A fixed seed makes repeated Re-optimize calls with the same storm positions
    and parameters deterministic.  NSGA-II is stochastic (random initialisation,
    crossover, mutation) — without a fixed seed each call explores a different
    random trajectory and produces a different Pareto front.
    """
    # pymoo 0.6 does not seed numpy's global RNG — seed it explicitly so that
    # RouteSampling (which calls np.random.normal / np.random.uniform) is also
    # deterministic.  Same storm positions + same seed → identical Pareto front.
    np.random.seed(seed)
    result = minimize(
        WeatherPathProblem(),
        NSGA2(pop_size=pop, sampling=RouteSampling(),
              crossover=SBX(prob=0.9, eta=15),
              mutation=PM(eta=20), eliminate_duplicates=True),
        termination=("n_gen", ngen), seed=seed, verbose=False,
    )
    return result.F, result.X


def launch_interactive(F: np.ndarray, X: np.ndarray) -> None:
    """
    Interactive figure with three weight sliders.
    Drag any slider to re-weight the objectives; the optimal path,
    altitude profile, and Pareto highlight update instantly.
    """
    xg, yg, elev = TERRAIN.grid()
    Xg, Yg = np.meshgrid(xg, yg)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 10), facecolor="#1a1a2e")
    fig.suptitle(
        "Multi-Objective Path Planning — Weather Zone Avoidance\n"
        "Sliders: trade off objectives  |  Click+drag storm ✥ centres to reposition them",
        color="white", fontsize=12, fontweight="bold", y=0.98
    )

    gs_main = gridspec.GridSpec(
        3, 2,
        left=0.04, right=0.98, top=0.92, bottom=0.30,
        hspace=0.42, wspace=0.25,
        height_ratios=[1.5, 1.2, 1.0],
    )
    ax_map  = fig.add_subplot(gs_main[:, 0])          # left column: 2-D map
    ax_alt  = fig.add_subplot(gs_main[0, 1])           # right row-0: altitude profile
    # row-1: pairwise scatter sub-grid (3 plots, one per objective pair)
    gs_pw   = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_main[1, 1], wspace=0.55)
    ax_pw   = [fig.add_subplot(gs_pw[0, i]) for i in range(3)]
    ax_pcp  = fig.add_subplot(gs_main[2, 1])           # right row-2: parallel coordinates

    # Slider axes (add_axes uses figure-fraction coordinates)
    ax_s1  = fig.add_axes([0.18, 0.205, 0.65, 0.028], facecolor="#2d2d4e")
    ax_s2  = fig.add_axes([0.18, 0.145, 0.65, 0.028], facecolor="#2d2d4e")
    ax_s3  = fig.add_axes([0.18, 0.085, 0.65, 0.028], facecolor="#2d2d4e")
    ax_txt = fig.add_axes([0.04, 0.01,  0.92, 0.055], facecolor="#12122a")
    ax_txt.axis("off")

    # ── 2-D Map ───────────────────────────────────────────────────────────────
    tcf = ax_map.contourf(Xg, Yg, elev, levels=22, cmap="terrain", alpha=0.60)
    fig.colorbar(tcf, ax=ax_map, label="Elevation (m AMSL)",
                 shrink=0.75, fraction=0.03)
    ax_map.contour(Xg, Yg, elev, levels=10, colors="white",
                   linewidths=0.25, alpha=0.18)

    # Mutable state — lets Re-optimize fully replace F, X, and cached paths
    _st = {
        "F": F.copy(),
        "X": X.copy(),
        "paths": [build_path(X[j].reshape(N_WP, 3)) for j in range(len(X))],
    }

    def _reeval_risk():
        """Recompute F[:,1] for every Pareto solution with current storm positions."""
        for j, p in enumerate(_st["paths"]):
            _, new_risk, _ = evaluate_path(p)
            _st["F"][j, 1] = new_risk

    # Draggable storm circles — keep patch/annotation refs so we can move them
    _storm_fill, _storm_edge, _storm_ann = [], [], []
    for storm in WEATHER:
        cx, cy = storm["center"]
        _storm_fill.append(ax_map.add_patch(
            plt.Circle((cx, cy), storm["radius"],
                       color="#6688ff", alpha=0.25, zorder=3)))
        _storm_edge.append(ax_map.add_patch(
            plt.Circle((cx, cy), storm["radius"],
                       fill=False, edgecolor="#aabbff",
                       lw=2.0, ls="--", zorder=3)))
        _storm_ann.append(ax_map.annotate(
            storm["name"] + "\n✥",  # ✥ = drag handle hint
            (cx, cy), ha="center", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=4))

    # Direct-route dashed reference
    ax_map.plot([SOURCE[0], DEST[0]], [SOURCE[1], DEST[1]],
                color="white", lw=1.2, ls=":", alpha=0.35,
                label="Direct route (ref)", zorder=2)

    # Source and destination markers
    ax_map.plot(*SOURCE[:2], "gs", ms=14, zorder=8,
                markeredgecolor="lime", markeredgewidth=1.5)
    ax_map.plot(*DEST[:2],   "r*", ms=18, zorder=8,
                markeredgecolor="#ff8888", markeredgewidth=1.0)
    ax_map.annotate(f"SOURCE\n({SOURCE[2]:.0f} m AGL)",
                    SOURCE[:2] + np.array([1.5, -6]),
                    fontsize=8, color="lime", fontweight="bold")
    ax_map.annotate(f"DEST\n({DEST[2]:.0f} m AGL)",
                    DEST[:2] + np.array([-20, -6]),
                    fontsize=8, color="#ff8888", fontweight="bold")

    # Waypoint scatter (intermediate)
    wp_scat = ax_map.scatter([], [], c="cyan", s=45, zorder=7,
                             edgecolors="white", linewidths=0.7, label="Waypoints")

    # Active path line on map
    (path_line_map,) = ax_map.plot([], [], color="#00e5ff", lw=2.8,
                                   marker="o", ms=6, zorder=6, label="Active path",
                                   markeredgecolor="white", markeredgewidth=0.7)

    ax_map.set_xlim(*X_BOUNDS); ax_map.set_ylim(*Y_BOUNDS)
    ax_map.set_xlabel("Easting (km)", color="lightgray")
    ax_map.set_ylabel("Northing (km)", color="lightgray")
    ax_map.set_title("Top-Down View", color="white", fontsize=11, pad=6)
    ax_map.set_aspect("equal")
    ax_map.tick_params(colors="lightgray")
    for sp in ax_map.spines.values():
        sp.set_color("#555")
    ax_map.grid(True, alpha=0.15, color="white")

    storm_patch = mpatches.Patch(color="#6688ff", alpha=0.5,
                                 label="Bad-weather zone")
    ax_map.legend(handles=[path_line_map, wp_scat, storm_patch],
                  loc="upper left", fontsize=8,
                  facecolor="#222240", labelcolor="lightgray", framealpha=0.85)

    # ── Altitude AGL Profile ──────────────────────────────────────────────────
    ax_alt.set_facecolor("#12122a")
    ax_alt.axhspan(0,           SAFE_AGL,   color="red",    alpha=0.12,
                   label=f"Terrain risk  (< {SAFE_AGL:.0f} m AGL)")
    ax_alt.axhspan(SAFE_AGL,    DETECT_AGL, color="#00cc66", alpha=0.07,
                   label=f"Ideal band  ({SAFE_AGL:.0f}–{DETECT_AGL:.0f} m)")
    ax_alt.axhspan(DETECT_AGL,  MAX_AGL+60, color="orange",  alpha=0.08,
                   label=f"Radar detect  (> {DETECT_AGL:.0f} m)")
    ax_alt.axhline(SAFE_AGL,    color="red",     lw=1.2, ls="--", alpha=0.7)
    ax_alt.axhline(DETECT_AGL,  color="orange",  lw=1.2, ls="--", alpha=0.7)
    ax_alt.axhline(FUEL_OPT_AGL, color="gold",   lw=1.1, ls="-.", alpha=0.85,
                   label=f"Fuel-optimal ({FUEL_OPT_AGL:.0f} m)")

    (alt_line,) = ax_alt.plot([], [], color="#00e5ff", lw=2.4,
                              marker="o", ms=5, zorder=5,
                              markeredgecolor="white", markeredgewidth=0.6)

    ax_alt.set_xlim(0, _path_2d_length(build_path(
        np.zeros((N_WP, 3)))) * 1.05)
    ax_alt.set_ylim(0, MAX_AGL + 60)
    ax_alt.set_xlabel("Along-Track Distance (km)", color="lightgray", fontsize=9)
    ax_alt.set_ylabel("Altitude AGL (m)", color="lightgray", fontsize=9)
    ax_alt.set_title("Altitude AGL Profile", color="white", fontsize=10, pad=4)
    ax_alt.tick_params(colors="lightgray", labelsize=8)
    for sp in ax_alt.spines.values():
        sp.set_color("#555")
    ax_alt.grid(True, alpha=0.15, color="white")
    ax_alt.legend(fontsize=7.5, ncol=2, facecolor="#222240",
                  labelcolor="lightgray", framealpha=0.85)

    # ── Pairwise Scatter Matrix  (3 plots × 2 objectives, 3rd = colour) ────────
    # Each panel: axes = two objectives; colour = third objective.
    # Pairs: (Time, Risk | Fuel colour),  (Risk, Fuel | Time colour),
    #        (Time, Fuel | Risk colour)
    _PW_PAIRS = [
        # (x_col, y_col, c_col, x_label,      y_label,    c_label,    cmap)
        (0, 1, 2, "Time (min)", "Risk",       "Fuel (kg)", "plasma"),
        (1, 2, 0, "Risk",       "Fuel (kg)",  "Time (min)","viridis_r"),
        (0, 2, 1, "Time (min)", "Fuel (kg)",  "Risk",      "RdYlGn_r"),
    ]
    # Display transform: column 0 → /60 for minutes, others raw
    def _fw(F, col):
        return F[:, col] / 60 if col == 0 else F[:, col]

    # Crowding distance → normalised [10, 80] for point size
    def _crowd_sizes(Fm, smin=14, smax=80):
        try:
            from pymoo.operators.survival.rank_and_crowding.metrics import (
                calc_crowding_distance)
            cd = calc_crowding_distance(Fm)
        except Exception:
            cd = np.ones(len(Fm))
        cd = np.clip(cd, 0, np.percentile(cd[np.isfinite(cd)], 95))
        cd = np.where(np.isfinite(cd), cd, cd[np.isfinite(cd)].max())
        lo, hi = cd.min(), cd.max()
        if hi == lo:
            return np.full(len(Fm), (smin + smax) / 2)
        return smin + (smax - smin) * (cd - lo) / (hi - lo)

    # Indices of the 3 "extreme" solutions (best in each single objective)
    def _extreme_idx(Fm):
        return [int(Fm[:, c].argmin()) for c in range(3)]

    _EXTREME_MARKERS = ["^", "D", "s"]       # Time-best, Risk-best, Fuel-best
    _EXTREME_COLORS  = ["#ff6b6b", "#6bffb8", "#ffd93d"]
    _EXTREME_LABELS  = ["Best Time", "Best Risk", "Best Fuel"]

    _F0 = _st["F"]
    _pw_clouds   = []
    _pw_stars    = []
    _pw_extremes = []   # list of lists (one per panel, 3 extreme scatter handles)
    _pw_cbars    = []
    _pw_hlines   = []   # horizontal crosshair per panel
    _pw_vlines   = []   # vertical crosshair per panel
    _pw_anns     = []   # colour-axis callout annotation per panel

    for ax_i, (xc, yc, cc, xl, yl, cl, cmap) in zip(ax_pw, _PW_PAIRS):
        ax_i.set_facecolor("#12122a")
        xv, yv, cv = _fw(_F0, xc), _fw(_F0, yc), _fw(_F0, cc)
        sizes = _crowd_sizes(_F0)

        # Main cloud — sized by crowding distance
        sc = ax_i.scatter(xv, yv, c=cv, cmap=cmap, s=sizes, alpha=0.70,
                          linewidths=0, zorder=2)
        cb = fig.colorbar(sc, ax=ax_i, pad=0.02, fraction=0.07, shrink=0.85)
        cb.set_label(cl, color="lightgray", fontsize=6)
        cb.ax.tick_params(colors="lightgray", labelsize=6)
        _pw_cbars.append(cb)

        # Extreme-solution markers (3 corners of the Pareto front)
        ext_handles = []
        for ei, (em, ec, elb) in enumerate(
                zip(_EXTREME_MARKERS, _EXTREME_COLORS, _EXTREME_LABELS)):
            eidx = int(_F0[:, ei].argmin())
            eh = ax_i.scatter(
                [xv[eidx]], [yv[eidx]],
                marker=em, c=ec, s=90, zorder=7,
                edgecolors="white", linewidths=0.6,
                label=elb,
            )
            ext_handles.append(eh)
        _pw_extremes.append(ext_handles)

        # Active-solution star
        star = ax_i.scatter(
            [xv[0]], [yv[0]],
            c="#00e5ff", s=260, marker="*",
            edgecolors="white", linewidths=0.8, zorder=9,
            label="Active ★",
        )
        # Crosshair lines through active star (updated in update())
        hline = ax_i.axhline(yv[0], color="#00e5ff", lw=0.8,
                             ls="--", alpha=0.45, zorder=6)
        vline = ax_i.axvline(xv[0], color="#00e5ff", lw=0.8,
                             ls="--", alpha=0.45, zorder=6)
        # Callout at fixed top-right corner: shows active solution's 3rd-obj value
        ann = ax_i.text(
            0.97, 0.97, f"{cl}:\n{cv[0]:.1f}",
            transform=ax_i.transAxes, ha="right", va="top",
            color="#00e5ff", fontsize=6,
            bbox=dict(fc="#1a1a3a", ec="#00e5ff", lw=0.7,
                      boxstyle="round,pad=0.25", alpha=0.88),
            zorder=10,
        )
        _pw_clouds.append(sc)
        _pw_stars.append(star)
        _pw_hlines.append(hline)
        _pw_vlines.append(vline)
        _pw_anns.append(ann)

        ax_i.set_xlabel(xl, color="lightgray", fontsize=7)
        ax_i.set_ylabel(yl, color="lightgray", fontsize=7)
        ax_i.tick_params(colors="lightgray", labelsize=6)
        for sp in ax_i.spines.values():
            sp.set_color("#333355")
        ax_i.grid(True, alpha=0.12, color="white")

        # "Better ↓" arrows on both axes (smaller = better for all 3 objectives)
        ax_i.annotate("", xy=(0.02, 0.08), xytext=(0.02, 0.22),
                      xycoords="axes fraction",
                      arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.9))
        ax_i.text(0.03, 0.04, "better", transform=ax_i.transAxes,
                  color="#aaaaaa", fontsize=5, va="bottom")
        ax_i.annotate("", xy=(0.08, 0.02), xytext=(0.22, 0.02),
                      xycoords="axes fraction",
                      arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.9))

    # Legend on first panel
    ax_pw[0].legend(
        handles=(
            _pw_extremes[0]
            + [_pw_stars[0],
               plt.scatter([], [], s=14, c="#888", label="Pareto solution\n(size ∝ uniqueness)")]
        ),
        labels=_EXTREME_LABELS + ["Active ★", "Pareto solution\n(size ∝ uniqueness)"],
        fontsize=5.5, facecolor="#1a1a3a", labelcolor="lightgray",
        framealpha=0.85, loc="upper right", markerscale=0.9,
    )
    ax_pw[0].set_title(
        "Pareto Front  •  ▲=BestTime  ◆=BestRisk  ■=BestFuel  ★=Active\n"
        "A point may look dominated in 2D but excels on the colour axis",
        color="#cccccc", fontsize=7, loc="left", pad=3)

    # ── Parallel Coordinates (PCP) — all three objectives ─────────────────────
    # Normalise F to [0,1] per column for equal-scale axes
    def _pcp_norm(Fm):
        lo, hi = Fm.min(axis=0), Fm.max(axis=0)
        hi = np.where(hi == lo, lo + 1e-9, hi)
        return (Fm - lo) / (hi - lo), lo, hi

    ax_pcp.set_facecolor("#12122a")
    _OBJ_LABELS  = ["Time (min)", "Risk", "Fuel (kg)"]
    _OBJ_COLORS  = ["#ff6b6b", "#6bffb8", "#ffd93d"]
    _PCP_X       = [0, 1, 2]

    # Normalise once at init; updated only on re-optimise
    _pcp_Fn, _pcp_lo, _pcp_hi = _pcp_norm(_st["F"])

    # Draw all background lines ONCE (very faint gray so active line pops)
    for row in _pcp_Fn:
        ax_pcp.plot(_PCP_X, row, color="#334466", alpha=0.18, lw=0.6, zorder=1)

    # Persistent active-line handle (glow = thick pale layer + bright thin layer)
    _pcp_glow, = ax_pcp.plot(_PCP_X, _pcp_Fn[0], color="#00e5ff",
                              lw=7, alpha=0.25, zorder=7, solid_capstyle="round")
    _pcp_line, = ax_pcp.plot(_PCP_X, _pcp_Fn[0], color="#00e5ff",
                              lw=2.5, zorder=8, solid_capstyle="round",
                              marker="o", ms=7,
                              markerfacecolor="#00e5ff",
                              markeredgecolor="white", markeredgewidth=1.0)

    # Static decorations
    ax_pcp.set_xticks(_PCP_X)
    ax_pcp.set_xticklabels(_OBJ_LABELS, color="lightgray", fontsize=8)
    ax_pcp.set_ylim(-0.08, 1.12)
    ax_pcp.set_ylabel("Normalised value (0=best, 1=worst)", color="lightgray", fontsize=7)
    ax_pcp.set_title("Parallel Coordinates  —  each line = one Pareto solution  |  cyan = active",
                     color="white", fontsize=8, pad=3)
    ax_pcp.tick_params(axis="y", colors="lightgray", labelsize=7)
    for sp in ax_pcp.spines.values():
        sp.set_color("#555")
    ax_pcp.grid(axis="x", alpha=0.20, color="white")
    for xi, col in zip(_PCP_X, _OBJ_COLORS):
        ax_pcp.axvline(xi, color=col, lw=1.5, alpha=0.7, zorder=2)
    # Min/max labels on each axis
    for xi, (lv, hv, col) in enumerate(
            zip(_pcp_lo / [60, 1, 1], _pcp_hi / [60, 1, 1], _OBJ_COLORS)):
        ax_pcp.text(xi, -0.07, f"best\n{lv:.1f}", ha="center", va="top",
                    fontsize=6, color=col)
        ax_pcp.text(xi,  1.10, f"worst\n{hv:.1f}", ha="center", va="bottom",
                    fontsize=6, color=col)

    # Persistent active-value dot annotations (one per axis)
    _pcp_dots = [
        ax_pcp.text(xi, _pcp_Fn[0, xi], "", ha="left", va="center",
                    color="#00e5ff", fontsize=6.5,
                    bbox=dict(fc="#1a1a3a", ec="#00e5ff", lw=0.6,
                              boxstyle="round,pad=0.2", alpha=0.85),
                    zorder=9)
        for xi in _PCP_X
    ]
    _PCP_DISP_SCALE = np.array([1/60, 1.0, 1.0])  # col 0 → minutes

    def _update_pcp(Fm, Fn, active_idx):
        """Update only the active line — no cla(), no full redraw."""
        yvals = Fn[active_idx]
        _pcp_glow.set_ydata(yvals)
        _pcp_line.set_ydata(yvals)
        for xi, (dot, raw) in enumerate(zip(_pcp_dots, Fm[active_idx])):
            dot.set_position((xi + 0.04, yvals[xi]))
            dot.set_text(f"{raw * _PCP_DISP_SCALE[xi]:.1f}")

    # ── Stats Text ────────────────────────────────────────────────────────────
    stats_txt = ax_txt.text(
        0.5, 0.5, "", transform=ax_txt.transAxes,
        ha="center", va="center", fontsize=10.5, color="white",
        fontfamily="monospace",
        bbox=dict(facecolor="#0d0d1a", edgecolor="#4466cc",
                  boxstyle="round,pad=0.4", alpha=0.9)
    )

    # ── Sliders ───────────────────────────────────────────────────────────────
    slider_kw = dict(color="#5577cc", track_color="#2d2d4e")
    s_time = Slider(ax_s1, "Mission Time weight",  0.0, 1.0,
                    valinit=1/3, **slider_kw)
    s_risk = Slider(ax_s2, "Survival Risk weight", 0.0, 1.0,
                    valinit=1/3, **slider_kw)
    s_fuel = Slider(ax_s3, "Fuel Burn weight",     0.0, 1.0,
                    valinit=1/3, **slider_kw)

    for sl, ax_sl in [(s_time, ax_s1), (s_risk, ax_s2), (s_fuel, ax_s3)]:
        sl.label.set_color("lightgray")
        sl.valtext.set_color("cyan")
        ax_sl.set_title("")

    # Labels to the left of sliders
    for ax_sl, txt, col in [
        (ax_s1, "⏱  TIME",     "#ff6b6b"),
        (ax_s2, "🛡  RISK",     "#6bffb8"),
        (ax_s3, "⛽  FUEL",     "#ffd93d"),
    ]:
        fig.text(ax_sl.get_position().x0 - 0.10,
                 ax_sl.get_position().y0 + 0.012,
                 txt, color=col, fontsize=9.5, fontweight="bold",
                 transform=fig.transFigure)

    # ── Update callback ───────────────────────────────────────────────────────
    def update(_val):
        w   = np.array([s_time.val, s_risk.val, s_fuel.val])
        Fc, Xc = _st["F"], _st["X"]
        idx  = _best_idx(Fc, w)
        wps  = Xc[idx].reshape(N_WP, 3)
        path = build_path(wps)

        # Update 2-D map path
        path_line_map.set_data(path[:, 0], path[:, 1])
        wp_scat.set_offsets(wps[:, :2])

        # Update altitude profile
        dists = _path_along_track(path)
        agls  = _path_agls(path)
        alt_line.set_data(dists, agls)
        ax_alt.set_xlim(0, dists[-1] * 1.05)

        # Update pairwise scatter stars, crosshairs, callouts, extremes
        ext_idxs = _extreme_idx(Fc)
        for pi, (star_i, hl, vl, ann_i, (xc, yc, cc, _xl, _yl, cl, _cm)) in enumerate(
                zip(_pw_stars, _pw_hlines, _pw_vlines, _pw_anns, _PW_PAIRS)):
            xval = _fw(Fc, xc)[idx]
            yval = _fw(Fc, yc)[idx]
            cval = _fw(Fc, cc)[idx]
            star_i.set_offsets([[xval, yval]])
            star_i.set_facecolor(["#00e5ff"])
            star_i.set_sizes([260])
            hl.set_ydata([yval, yval])   # axhline: ydata = [y, y]
            vl.set_xdata([xval, xval])   # axvline: xdata = [x, x]
            ann_i.set_text(f"{cl}:\n{cval:.1f}")
            for ei, eh in enumerate(_pw_extremes[pi]):
                eidx = ext_idxs[ei]
                eh.set_offsets([[_fw(Fc, xc)[eidx], _fw(Fc, yc)[eidx]]])

        # Update parallel coordinates active line only
        _update_pcp(Fc, _pcp_norm(Fc)[0], idx)

        # Normalised weights for display
        wn = w / max(w.sum(), 1e-9)

        # Stats text
        plen = _path_2d_length(path)
        detour = plen - float(np.linalg.norm(DEST[:2] - SOURCE[:2]))
        agl_arr = agls
        time_in_zone = _weather_exposure(path)
        min_risk = Fc[:, 1].min()
        stats_txt.set_text(
            f"Mission time: {Fc[idx,0]/60:.1f} min   |   "
            f"Survival risk: {Fc[idx,1]:.1f}  (Pareto min={min_risk:.1f})   |   "
            f"Fuel burn: {Fc[idx,2]:.1f} kg   |   "
            f"Path: {plen:.0f} km  (+{detour:.0f} km detour)   |   "
            f"AGL range: {agl_arr.min():.0f}–{agl_arr.max():.0f} m   |   "
            f"Weather exp: {time_in_zone:.0f} s   "
            f"[T:{wn[0]:.2f}  R:{wn[1]:.2f}  F:{wn[2]:.2f}]"
        )
        fig.canvas.draw_idle()

    s_time.on_changed(update)
    s_risk.on_changed(update)
    s_fuel.on_changed(update)

    # ── Draggable storm circles ────────────────────────────────────────────────
    _drag = {"idx": None}

    def _refresh_storms():
        for i, storm in enumerate(WEATHER):
            cx, cy = storm["center"]
            _storm_fill[i].center = (cx, cy)
            _storm_edge[i].center = (cx, cy)
            _storm_ann[i].set_position((cx, cy))

    def on_press(event):
        if event.inaxes is not ax_map or event.button != 1:
            return
        for i, storm in enumerate(WEATHER):
            d = np.hypot(event.xdata - storm["center"][0],
                         event.ydata - storm["center"][1])
            if d < storm["radius"] * 0.45:   # click near centre to grab
                _drag["idx"] = i
                return

    def on_motion(event):
        if event.inaxes is not ax_map or _drag["idx"] is None:
            return
        i = _drag["idx"]
        WEATHER[i]["center"][0] = np.clip(event.xdata, *X_BOUNDS)
        WEATHER[i]["center"][1] = np.clip(event.ydata, *Y_BOUNDS)
        _refresh_storms()
        # Re-score every Pareto solution with the new storm geometry so that
        # _best_idx picks the correct path for the current slider weights.
        _reeval_risk()
        update(None)

    def on_release(event):
        _drag["idx"] = None

    fig.canvas.mpl_connect("button_press_event",   on_press)
    fig.canvas.mpl_connect("motion_notify_event",  on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    # Reset Storms button
    ax_reset = fig.add_axes([0.76, 0.035, 0.10, 0.035], facecolor="#2d2d4e")
    btn_reset = Button(ax_reset, "Reset Storms",
                       color="#2d2d4e", hovercolor="#445588")
    btn_reset.label.set_color("lightgray")
    btn_reset.label.set_fontsize(8.5)

    def on_reset(_evt):
        for i, orig in enumerate(_WEATHER_ORIG):
            WEATHER[i]["center"][:] = orig["center"]
        _refresh_storms()
        _reeval_risk()
        update(None)

    btn_reset.on_clicked(on_reset)

    # Re-optimize button — re-runs NSGA-II with CURRENT storm positions.
    # Use after dragging storms to get Pareto paths shaped around the new layout.
    ax_reopt = fig.add_axes([0.87, 0.035, 0.10, 0.035], facecolor="#2d2d4e")
    btn_reopt = Button(ax_reopt, "Re-optimize",
                       color="#2d2d4e", hovercolor="#225522")
    btn_reopt.label.set_color("#88ff88")
    btn_reopt.label.set_fontsize(8.5)

    def on_reoptimize(_evt):
        stats_txt.set_text("Re-optimising for current storm positions … (pop=120, gen=100)")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        new_F, new_X = _run_nsga2(pop=120, ngen=100)
        _st["F"]     = new_F
        _st["X"]     = new_X
        _st["paths"] = [build_path(new_X[j].reshape(N_WP, 3))
                        for j in range(len(new_X))]
        # Refresh pairwise scatter clouds + sizes + extreme markers
        new_sizes = _crowd_sizes(new_F)
        new_ext   = _extreme_idx(new_F)
        for pi, (cloud_i, (xc, yc, cc, *_rest)) in enumerate(
                zip(_pw_clouds, _PW_PAIRS)):
            cloud_i.set_offsets(
                np.column_stack([_fw(new_F, xc), _fw(new_F, yc)]))
            cloud_i.set_array(_fw(new_F, cc))
            cloud_i.set_clim(_fw(new_F, cc).min(), _fw(new_F, cc).max())
            cloud_i.set_sizes(new_sizes)
            for ei, eh in enumerate(_pw_extremes[pi]):
                eidx = new_ext[ei]
                eh.set_offsets([[_fw(new_F, xc)[eidx], _fw(new_F, yc)[eidx]]])
        # Redraw PCP background lines with new Pareto front
        new_Fn, new_lo, new_hi = _pcp_norm(new_F)
        # remove old background lines (all lines except glow+active+axvlines)
        for ln in ax_pcp.lines[:-2]:
            ln.remove()
        for row in new_Fn:
            ax_pcp.plot(_PCP_X, row, color="#334466", alpha=0.18, lw=0.6, zorder=1)
        # move glow/active back to top
        _pcp_glow.set_zorder(7); _pcp_line.set_zorder(8)
        update(None)

    btn_reopt.on_clicked(on_reoptimize)

    update(None)  # draw initial state (balanced weights)
    plt.show()


def _weather_exposure(path: np.ndarray) -> float:
    """Total seconds spent inside any storm cell (used for stats display)."""
    t     = np.linspace(0.0, 1.0, N_SAMP)
    p1, p2 = path[:-1], path[1:]
    pts   = p1[:, None, :] + t[None, :, None] * (p2 - p1)[:, None, :]
    pts2  = pts[:, :, :2]
    dp    = p2 - p1
    seg3d = np.linalg.norm(np.column_stack(
        [dp[:, :2], dp[:, 2] / 1000.0]), axis=1)
    seg_t = seg3d / SPEED
    frac  = np.zeros(len(p1))
    for storm in WEATHER:
        frac += (np.linalg.norm(pts2 - storm["center"], axis=2)
                 < storm["radius"]).mean(axis=1)
    return float(np.sum(frac.clip(0, 1) * seg_t))


# ── STATIC EXPORT PLOTS ───────────────────────────────────────────────────────

_STYLES = [
    dict(color="#e63946", lw=2.5, ls="-",  marker="o", ms=7),   # Fast
    dict(color="#2a9d8f", lw=2.5, ls="--", marker="s", ms=6),   # Safe
    dict(color="#e9c46a", lw=2.5, ls="-.", marker="^", ms=7),   # Efficient
    dict(color="#457b9d", lw=2.5, ls=":",  marker="D", ms=6),   # Balanced
]


def plot_2d(paths: List[np.ndarray], labels: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 8))
    xg, yg, elev = TERRAIN.grid()
    Xg, Yg = np.meshgrid(xg, yg)
    tcf = ax.contourf(Xg, Yg, elev, levels=22, cmap="terrain", alpha=0.55)
    plt.colorbar(tcf, ax=ax, label="Elevation (m AMSL)", shrink=0.75)
    ax.contour(Xg, Yg, elev, levels=10, colors="k", linewidths=0.3, alpha=0.25)

    for storm in WEATHER:
        ax.add_patch(plt.Circle(storm["center"], storm["radius"],
                                color="#5566dd", alpha=0.22, zorder=3))
        ax.add_patch(plt.Circle(storm["center"], storm["radius"],
                                fill=False, edgecolor="#3344aa",
                                lw=2.2, ls="--", zorder=3))
        ax.annotate(storm["name"], storm["center"],
                    ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold", zorder=4)

    ax.plot([SOURCE[0], DEST[0]], [SOURCE[1], DEST[1]],
            "k--", lw=1.3, alpha=0.30, label="Direct route (ref)", zorder=2)

    for st, path, lbl in zip(_STYLES, paths, labels):
        ax.plot(path[:, 0], path[:, 1],
                color=st["color"], lw=st["lw"], ls=st["ls"],
                marker=st["marker"], ms=st["ms"], label=lbl, zorder=6,
                markeredgecolor="white", markeredgewidth=0.8)
        mid = len(path) // 2
        dx, dy = path[mid+1, 0]-path[mid, 0], path[mid+1, 1]-path[mid, 1]
        ax.annotate("", xy=(path[mid, 0]+dx*0.5, path[mid, 1]+dy*0.5),
                    xytext=(path[mid, 0], path[mid, 1]),
                    arrowprops=dict(arrowstyle="-|>", color=st["color"], lw=1.5),
                    zorder=7)

    ax.plot(*SOURCE[:2], "gs", ms=16, zorder=8,
            markeredgecolor="darkgreen", markeredgewidth=1.5, label="Source")
    ax.plot(*DEST[:2],   "r*", ms=20, zorder=8,
            markeredgecolor="darkred",   markeredgewidth=1.0, label="Destination")
    ax.annotate(f"SOURCE\n({SOURCE[2]:.0f} m AGL)", SOURCE[:2]+np.array([1.5, -6]),
                fontsize=9, color="darkgreen", fontweight="bold")
    ax.annotate(f"DEST\n({DEST[2]:.0f} m AGL)",     DEST[:2]+np.array([-20, -6]),
                fontsize=9, color="darkred",   fontweight="bold")

    ax.set_xlim(*X_BOUNDS); ax.set_ylim(*Y_BOUNDS)
    ax.set_xlabel("Easting (km)", fontsize=12)
    ax.set_ylabel("Northing (km)", fontsize=12)
    ax.set_title(
        "Weather-Avoidance Path Planning — 2-D Overview\n"
        "Blue dashed circles = bad-weather cells (enter = cumulative damage risk)",
        fontsize=12, fontweight="bold"
    )
    ax.set_aspect("equal"); ax.grid(True, alpha=0.20)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color="#5566dd", alpha=0.5, label="Bad-weather zone"))
    ax.legend(handles=handles, loc="upper left", fontsize=9,
              framealpha=0.87, ncol=2)
    plt.tight_layout()
    return fig


def plot_altitude_profile(paths: List[np.ndarray], labels: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axhspan(0,          SAFE_AGL,   color="red",    alpha=0.10,
               label=f"Terrain risk (< {SAFE_AGL:.0f} m AGL)")
    ax.axhspan(SAFE_AGL,   DETECT_AGL, color="green",  alpha=0.07,
               label=f"Ideal band ({SAFE_AGL:.0f}–{DETECT_AGL:.0f} m)")
    ax.axhspan(DETECT_AGL, MAX_AGL+60, color="orange", alpha=0.08,
               label=f"Radar detect (> {DETECT_AGL:.0f} m)")
    ax.axhline(SAFE_AGL,    color="red",    lw=1.5, ls="--", alpha=0.75)
    ax.axhline(DETECT_AGL,  color="orange", lw=1.5, ls="--", alpha=0.75)
    ax.axhline(FUEL_OPT_AGL, color="gold", lw=1.3, ls="-.", alpha=0.90,
               label=f"Fuel-optimal ({FUEL_OPT_AGL:.0f} m)")

    for st, path, lbl in zip(_STYLES, paths, labels):
        dists = _path_along_track(path)
        agls  = _path_agls(path)
        ax.plot(dists, agls, color=st["color"], lw=st["lw"], ls=st["ls"],
                marker=st["marker"], ms=st["ms"], label=lbl, zorder=5,
                markeredgecolor="white", markeredgewidth=0.6)

    ax.set_xlabel("Along-Track Distance (km)", fontsize=12)
    ax.set_ylabel("Altitude AGL (m)", fontsize=12)
    ax.set_ylim(0, MAX_AGL + 60)
    ax.set_title(
        "Altitude AGL Profile — Terrain-Following & Weather Avoidance\n"
        "Green = ideal low-level corridor  |  Red = terrain risk  |  Orange = radar",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.87)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


def plot_3d(paths: List[np.ndarray], labels: List[str]) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10))
    ax  = fig.add_subplot(111, projection="3d")
    xg, yg, elev = TERRAIN.grid()
    Xg, Yg = np.meshgrid(xg, yg)
    ax.plot_surface(Xg, Yg, elev, cmap="terrain",
                    alpha=0.40, linewidth=0, antialiased=True)

    theta = np.linspace(0, 2*np.pi, 32)
    for storm in WEATHER:
        cx, cy, r = storm["center"][0], storm["center"][1], storm["radius"]
        Xc = cx + r * np.outer(np.cos(theta), np.ones(2))
        Yc = cy + r * np.outer(np.sin(theta), np.ones(2))
        Zc = np.outer(np.ones(32), [0, 1800])
        ax.plot_surface(Xc, Yc, Zc, color="#5566dd", alpha=0.12)

    for st, path, lbl in zip(_STYLES, paths, labels):
        ax.plot(path[:, 0], path[:, 1], path[:, 2],
                color=st["color"], lw=2.2, marker=st["marker"],
                ms=5, label=lbl, zorder=5)

    ax.scatter(*SOURCE[:2], TERRAIN.at(*SOURCE[:2]) + SOURCE[2],
               c="green", s=200, marker="s", zorder=10)
    ax.scatter(*DEST[:2],   TERRAIN.at(*DEST[:2])   + DEST[2],
               c="red",   s=220, marker="*", zorder=10)

    ax.set_xlabel("Easting (km)");  ax.set_ylabel("Northing (km)")
    ax.set_zlabel("Altitude AMSL (m)")
    ax.set_title("3-D Flight Paths — Terrain & Weather Avoidance",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    return fig


def plot_pareto(F: np.ndarray,
                sel_F: List[np.ndarray],
                labels: List[str]) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    pairs  = [(0, 1), (0, 2), (1, 2)]
    xlbls  = ["Mission Time (min)",      "Mission Time (min)", "Survival Risk"]
    ylbls  = ["Survival Risk (s-equiv)", "Fuel Burn (kg)",     "Fuel Burn (kg)"]
    titles = ["Time vs Survival Risk",   "Time vs Fuel",       "Risk vs Fuel"]
    x_sc   = [1/60, 1/60, 1.0]

    for ax, (a, b), xl, yl, ttl, xs in zip(axes, pairs, xlbls, ylbls, titles, x_sc):
        ax.scatter(F[:, a]*xs, F[:, b], c="#a8c7e8", s=25,
                   alpha=0.5, label="Pareto solutions")
        for sf, st, lbl in zip(sel_F, _STYLES, labels):
            ax.scatter(sf[a]*xs, sf[b], c=st["color"], s=200,
                       marker=st["marker"], edgecolors="black",
                       linewidths=0.8, zorder=6, label=lbl)
            ax.annotate(lbl, (sf[a]*xs, sf[b]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=st["color"], fontweight="bold")
        ax.set_xlabel(xl, fontsize=10); ax.set_ylabel(yl, fontsize=10)
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.28); ax.legend(fontsize=8)

    fig.suptitle(
        "Pareto Front — Weather-Avoidance Path Planning (NSGA-II)\n"
        "Trade-offs: faster ↔ safer ↔ more fuel-efficient",
        fontsize=12, fontweight="bold", y=1.03
    )
    plt.tight_layout()
    return fig


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("  Aircraft Path Planning — Weather Zone Avoidance (NSGA-II)")
    print("=" * 62)
    print(f"\n  SOURCE : ({SOURCE[0]:.0f} km, {SOURCE[1]:.0f} km, {SOURCE[2]:.0f} m AGL)")
    print(f"  DEST   : ({DEST[0]:.0f} km, {DEST[1]:.0f} km, {DEST[2]:.0f} m AGL)")
    print(f"  Direct 2-D distance : {np.linalg.norm(DEST[:2]-SOURCE[:2]):.0f} km")
    print(f"  Aircraft speed      : {SPEED*3600:.0f} km/h")
    print(f"  Weather zones       : {len(WEATHER)}  (all centred on direct route)")
    print(f"  Waypoints           : {N_WP} intermediate (3-D each)")
    print(f"\n  Altitude bands:")
    print(f"    < {SAFE_AGL:.0f} m AGL  — terrain-collision risk   → f2")
    print(f"    {SAFE_AGL:.0f}–{DETECT_AGL:.0f} m AGL — ideal low-level corridor")
    print(f"    > {DETECT_AGL:.0f} m AGL  — radar detectable         → f2")
    print(f"    Fuel-optimal AGL ≈ {FUEL_OPT_AGL:.0f} m")
    print(f"\n  Objectives:")
    print(f"    f1  Mission time   (minimise)")
    print(f"    f2  Survival risk  (weather exposure + altitude violations)")
    print(f"    f3  Fuel burn      (route length × altitude drag penalty)")

    problem = WeatherPathProblem()
    POP, NGEN = 150, 150

    algorithm = NSGA2(
        pop_size=POP,
        sampling=RouteSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    print(f"\n  Running NSGA-II  (pop={POP}, gen={NGEN}) …\n")
    np.random.seed(42)   # seed numpy global RNG used by RouteSampling
    result = minimize(problem, algorithm,
                      termination=("n_gen", NGEN),
                      seed=42, verbose=True)

    F, X = result.F, result.X
    np.savez("path_results.npz", F=F, X=X)
    print(f"\n  Saved optimisation results to path_results.npz")
    print(f"\n  Pareto-optimal solutions : {len(F)}")
    print(f"  Mission time  : {F[:,0].min()/60:.1f} – {F[:,0].max()/60:.1f} min")
    print(f"  Survival risk : {F[:,1].min():.1f} – {F[:,1].max():.1f}")
    print(f"  Fuel burn     : {F[:,2].min():.1f} – {F[:,2].max():.1f} kg")

    # Select four representative solutions for static PNG export
    weight_profiles = {
        "Fast":      np.array([0.70, 0.15, 0.15]),
        "Safe":      np.array([0.15, 0.70, 0.15]),
        "Efficient": np.array([0.15, 0.15, 0.70]),
        "Balanced":  np.array([1/3,  1/3,  1/3 ]),
    }

    sel_paths, sel_F, labels = [], [], []
    print("\n  Representative solutions:")
    print("  " + "─" * 56)

    for lbl, w in weight_profiles.items():
        idx  = _best_idx(F, w)
        wps  = X[idx].reshape(N_WP, 3)
        path = build_path(wps)
        sel_paths.append(path); sel_F.append(F[idx]); labels.append(lbl)
        agls = _path_agls(path)
        print(f"\n  [{lbl}]")
        print(f"    Time:  {F[idx,0]/60:.1f} min  |  Path: {_path_2d_length(path):.0f} km  "
              f"(+{_path_2d_length(path)-np.linalg.norm(DEST[:2]-SOURCE[:2]):.0f} km detour)")
        print(f"    Risk:  {F[idx,1]:.1f} s-equiv  |  "
              f"Fuel: {F[idx,2]:.1f} kg  |  "
              f"AGL: {agls.min():.0f}–{agls.max():.0f} m")

    # Save static PNGs
    print()
    for fname, fig in [
        ("weather_path_2d.png",       plot_2d(sel_paths, labels)),
        ("weather_path_altitude.png", plot_altitude_profile(sel_paths, labels)),
        ("weather_path_3d.png",       plot_3d(sel_paths, labels)),
        ("weather_path_pareto.png",   plot_pareto(F, sel_F, labels)),
    ]:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")

    # Launch interactive slider view
    print("\n  Launching interactive figure — drag sliders to explore trade-offs.")
    launch_interactive(F, X)
    print("\n  Done.")


if __name__ == "__main__":
    main()
