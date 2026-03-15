"""
Multi-Objective Aircraft Path Planning for Defense Scenarios
============================================================
Uses NSGA-II (pymoo) to find Pareto-optimal 3-D flight routes in a hostile
100 × 100 km operational area.

Decision variables  (N_WP × 3 real values)
-------------------------------------------
For each intermediate waypoint  i = 0 … N_WP-1:
    x_i   – X position  (km)
    y_i   – Y position  (km)
    agl_i – Altitude Above Ground Level  (m)

Objectives  (all minimised)
----------------------------
f1  Flight time           – total 3-D path length / aircraft speed (s)
f2  SAM exposure time     – seconds inside any SAM engagement envelope
f3  Radar exposure index  – altitude × range-weighted radar detection

Hard constraints  (g ≤ 0)
--------------------------
g_nfz_k      – maximum penetration depth into NFZ k across all path samples
g_ter_seg_j  – worst terrain-clearance violation on segment j
g_corridor   – maximum lateral deviation from the base→target great-circle line

Implicit behaviour
-------------------
• End-point altitude = search_agl: aircraft climbs for sensor scan on approach.
• Flying low reduces f3 (terrain masking from long-range radar).
• Warm start initialises population along/near the straight-line route.
• All inner loops are fully vectorised via NumPy for fast evaluation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.interpolate import RectBivariateSpline

from pymoo.core.problem  import Problem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2  import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.optimize   import minimize
from pymoo.decomposition.asf import ASF

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT ENTITIES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NoFlyZone:
    """Circular no-fly zone (2-D)."""
    center: np.ndarray   # (x, y) km
    radius: float        # km
    name:   str = "NFZ"


@dataclass
class SAMSite:
    """Surface-to-Air Missile site — cylindrical threat envelope."""
    position: np.ndarray   # (x, y) km
    radius:   float        # engagement radius km
    name:     str = "SAM"


@dataclass
class RadarSite:
    """Long-range ground-based early-warning radar."""
    position:       np.ndarray   # (x, y) km
    max_range:      float        # effective range km
    min_detect_alt: float        # m AMSL — below this the aircraft is masked
    name:           str = "Radar"


@dataclass
class Terrain:
    """
    Synthetic 2-D terrain: layered sinusoids + Gaussian hills.
    A bicubic spline is fitted once at construction; batch queries use `.ev()`.
    """
    x_range:    Tuple[float, float]
    y_range:    Tuple[float, float]
    resolution: int = 100

    _elevation: np.ndarray = field(default=None, repr=False)
    _interp:    object     = field(default=None, repr=False)
    _x:         np.ndarray = field(default=None, repr=False)
    _y:         np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._build()

    def _build(self) -> None:
        np.random.seed(7)
        x = np.linspace(*self.x_range, self.resolution)
        y = np.linspace(*self.y_range, self.resolution)
        X, Y = np.meshgrid(x, y)

        elev = (
              180 * np.sin(0.22 * X) * np.cos(0.18 * Y)
            + 130 * np.sin(0.55 * X + 1.0) * np.cos(0.42 * Y + 0.5)
            +  70 * np.sin(1.05 * X + 2.0) * np.cos(0.88 * Y + 1.0)
            # Prominent hills that force route decisions
            + 380 * np.exp(-((X - 35)**2 + (Y - 30)**2) / 160)   # SW hill
            + 320 * np.exp(-((X - 25)**2 + (Y - 65)**2) / 140)   # W ridge
            + 450 * np.exp(-((X - 58)**2 + (Y - 50)**2) / 200)   # Central plateau
            + 260 * np.exp(-((X - 48)**2 + (Y - 78)**2) / 130)   # N spur
            + 300 * np.exp(-((X - 75)**2 + (Y - 35)**2) / 170)   # SE hill
            + np.random.normal(0, 12, X.shape)
        )
        self._elevation = np.maximum(elev, 0.0)
        self._x, self._y = x, y
        self._interp = RectBivariateSpline(x, y, self._elevation.T)

    def elevation_at(self, x: float, y: float) -> float:
        x = float(np.clip(x, *self.x_range))
        y = float(np.clip(y, *self.y_range))
        return float(self._interp(x, y)[0, 0])

    def elevation_at_points(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return self._interp.ev(np.clip(xs, *self.x_range),
                               np.clip(ys, *self.y_range))

    def grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._x, self._y, self._elevation


@dataclass
class Aircraft:
    """Performance parameters."""
    speed:            float = 0.25    # km/s  ≈ 900 km/h
    min_agl:          float = 50.0    # m
    max_agl:          float = 1200.0  # m
    search_agl:       float = 1000.0  # m — climb for target sensor scan
    search_radius_km: float = 12.0    # km — begin climb at this range to target


# ──────────────────────────────────────────────────────────────────────────────
# BATTLE ENVIRONMENT
# ──────────────────────────────────────────────────────────────────────────────

class BattleEnvironment:
    """
    All environmental entities plus the path evaluation functions used
    by the optimiser.
    """

    X_BOUNDS = (0.0, 100.0)
    Y_BOUNDS = (0.0, 100.0)
    N_SAMP   = 20     # samples per segment for integration

    def __init__(self) -> None:
        self.aircraft = Aircraft()
        self.terrain  = Terrain(self.X_BOUNDS, self.Y_BOUNDS)

        self.base   = np.array([ 5.0,  8.0])
        self.target = np.array([92.0, 88.0])

        # ── Threats placed to force interesting trade-offs ─────────────────
        self.nfzs: List[NoFlyZone] = [
            NoFlyZone(np.array([30.0, 30.0]),  9.0, "NFZ-α"),    # SW corridor block
            NoFlyZone(np.array([55.0, 25.0]),  8.0, "NFZ-β"),    # southern route block
            NoFlyZone(np.array([48.0, 58.0]), 10.0, "NFZ-γ"),    # central block
            NoFlyZone(np.array([72.0, 50.0]),  7.5, "NFZ-δ"),    # eastern corridor
            NoFlyZone(np.array([38.0, 80.0]),  8.5, "NFZ-ε"),    # northern block
        ]

        self.sams: List[SAMSite] = [
            SAMSite(np.array([20.0, 45.0]), 12.0, "SAM-1"),   # western flank
            SAMSite(np.array([45.0, 40.0]), 14.0, "SAM-2"),   # central threat
            SAMSite(np.array([65.0, 65.0]), 13.0, "SAM-3"),   # NE approach
            SAMSite(np.array([35.0, 68.0]), 10.0, "SAM-4"),   # northern threat
            SAMSite(np.array([75.0, 28.0]), 11.0, "SAM-5"),   # SE threat
        ]

        self.radars: List[RadarSite] = [
            RadarSite(np.array([50.0, 50.0]), 70.0, 180.0, "Radar-1"),  # central
            RadarSite(np.array([80.0, 80.0]), 55.0, 220.0, "Radar-2"),  # NE (near target)
            RadarSite(np.array([15.0, 80.0]), 50.0, 160.0, "Radar-3"),  # NW
        ]

        # Precompute corridor geometry: direct base→target line
        self._corridor_dir    = self.target - self.base
        self._corridor_len_sq = float(np.dot(self._corridor_dir,
                                             self._corridor_dir))

    # ── path construction ─────────────────────────────────────────────────────

    def build_full_path(self, waypoints: np.ndarray) -> np.ndarray:
        """
        Decode (N_WP, 3) array [x_km, y_km, agl_m] → (N_WP+2, 3) AMSL path.

        Near-target blend: waypoints within search_radius_km are blended
        toward search_agl so the aircraft climbs smoothly for the sensor scan.
        """
        ac = self.aircraft

        def to_amsl(xy: np.ndarray, agl: float) -> np.ndarray:
            h = self.terrain.elevation_at(float(xy[0]), float(xy[1])) + agl
            return np.array([xy[0], xy[1], h], dtype=float)

        path = [to_amsl(self.base, ac.min_agl)]

        for wp in waypoints:
            d2t = np.linalg.norm(wp[:2] - self.target)
            if d2t < ac.search_radius_km:
                blend = 1.0 - d2t / ac.search_radius_km
                eff_agl = wp[2] + blend * (ac.search_agl - wp[2])
            else:
                eff_agl = wp[2]
            path.append(to_amsl(wp[:2], eff_agl))

        path.append(to_amsl(self.target, ac.search_agl))
        return np.array(path, dtype=float)

    # ── vectorised evaluation ─────────────────────────────────────────────────

    def evaluate_path(
        self, path: np.ndarray, waypoints: np.ndarray
    ) -> Tuple[Tuple[float, float, float], np.ndarray]:
        """
        Vectorised computation of objectives + constraints for one path.

        Returns
        -------
        objectives  : (flight_time_s, sam_time_s, radar_index)
        constraints : 1-D array  — entry > 0 means violated
        """
        ac    = self.aircraft
        S     = self.N_SAMP
        t_arr = np.linspace(0.0, 1.0, S)

        p1 = path[:-1]   # (n_seg, 3)
        p2 = path[1:]    # (n_seg, 3)
        n_seg = len(p1)

        # sample points (n_seg, S, 3)
        pts = p1[:, None, :] + t_arr[None, :, None] * (p2 - p1)[:, None, :]

        xs   = pts[:, :, 0].ravel()
        ys   = pts[:, :, 1].ravel()
        alts = pts[:, :, 2].reshape(n_seg, S)
        pts2 = pts[:, :, :2]                         # (n_seg, S, 2)

        ter_flat = self.terrain.elevation_at_points(xs, ys)
        ter      = ter_flat.reshape(n_seg, S)

        # Convert altitude difference to km before computing 3-D segment length
        # (x, y in km; altitude in m → divide by 1000 to unify units)
        dp          = p2 - p1
        dp_km       = np.column_stack([dp[:, :2], dp[:, 2] / 1000.0])
        seg3d_km    = np.linalg.norm(dp_km, axis=1)           # 3-D km
        seg2d       = np.linalg.norm(dp[:, :2], axis=1)       # 2-D km

        # ── f1: flight time ─────────────────────────────────────────────────
        flight_time = float(np.sum(seg3d_km) / ac.speed)

        # ── f2: SAM exposure time ────────────────────────────────────────────
        sam_time = 0.0
        for sam in self.sams:
            d      = np.linalg.norm(pts2 - sam.position, axis=2)   # (n_seg, S)
            inside = d < sam.radius
            frac   = inside.mean(axis=1)
            sam_time += float(np.sum(frac * seg2d / ac.speed))

        # ── f3: radar exposure ───────────────────────────────────────────────
        radar_idx = 0.0
        for radar in self.radars:
            d    = np.linalg.norm(pts2 - radar.position, axis=2)
            mask = (d < radar.max_range) & (alts > radar.min_detect_alt)
            alt_f  = np.where(mask, (alts - radar.min_detect_alt) / 2000.0, 0.0)
            rng_f  = np.where(mask, np.maximum(0.0, 1.0 - d / radar.max_range), 0.0)
            radar_idx += float(np.sum((alt_f * rng_f).mean(axis=1) * seg2d))

        objectives = (flight_time, sam_time, radar_idx)

        # ── constraints ──────────────────────────────────────────────────────
        g: List[float] = []

        # g_nfz: one per NFZ — worst penetration across all samples
        for nfz in self.nfzs:
            d = np.linalg.norm(pts2 - nfz.center, axis=2)
            g.append(float((nfz.radius - d).max()))

        # g_terrain: one per segment — worst under-clearance
        under = ac.min_agl - (alts - ter)
        for k in range(n_seg):
            g.append(float(under[k].max()))

        # g_corridor: one per waypoint — lateral deviation from base→target line
        # Allowed max deviation: 35 km
        MAX_CORRIDOR_KM = 35.0
        for wp in waypoints:
            v   = wp[:2] - self.base
            t_p = np.dot(v, self._corridor_dir) / self._corridor_len_sq
            t_p = float(np.clip(t_p, 0.0, 1.0))
            proj = self.base + t_p * self._corridor_dir
            dev  = float(np.linalg.norm(wp[:2] - proj))
            g.append(dev - MAX_CORRIDOR_KM)

        return objectives, np.array(g, dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# WARM-START SAMPLING — initialise population near the direct-route corridor
# ──────────────────────────────────────────────────────────────────────────────

class CorridorSampling(Sampling):
    """
    Place each waypoint near its position on the straight line from base to
    target, with bounded Gaussian perturbation.  This gives the optimiser a
    realistic starting region and avoids the map-edge artefact.
    """

    def _do(self, problem, n_samples, **kwargs):
        env  = problem.env
        N_WP = problem.N_WP
        xl, xu = problem.xl, problem.xu

        base   = env.base
        target = env.target
        ac     = env.aircraft

        X = np.zeros((n_samples, problem.n_var))

        for i in range(n_samples):
            for j in range(N_WP):
                t   = (j + 1) / (N_WP + 1)
                mid = base + t * (target - base)

                # Gaussian jitter within a ±20 km corridor, clipped to bounds
                dx  = np.random.normal(0, 12)
                dy  = np.random.normal(0, 12)
                agl = np.random.uniform(ac.min_agl, ac.max_agl)

                x = float(np.clip(mid[0] + dx, xl[j*3],     xu[j*3]))
                y = float(np.clip(mid[1] + dy, xl[j*3 + 1], xu[j*3 + 1]))
                X[i, j*3 : j*3 + 3] = [x, y, agl]

        return X


# ──────────────────────────────────────────────────────────────────────────────
# PYMOO PROBLEM
# ──────────────────────────────────────────────────────────────────────────────

class PathPlanningProblem(Problem):
    """
    Variable layout  (N_WP × 3):
        X[3j] = x_km,  X[3j+1] = y_km,  X[3j+2] = agl_m

    Constraints:
        n_nfz  NFZ-penetration constraints
        N_WP+1 terrain-clearance constraints (one per segment)
        N_WP   corridor constraints (one per waypoint)
    """

    N_WP = 8

    def __init__(self, env: BattleEnvironment) -> None:
        self.env = env
        ac = env.aircraft

        xl = np.tile([env.X_BOUNDS[0], env.Y_BOUNDS[0], ac.min_agl], self.N_WP)
        xu = np.tile([env.X_BOUNDS[1], env.Y_BOUNDS[1], ac.max_agl], self.N_WP)

        n_ineq = (
            len(env.nfzs)          # NFZ constraints
            + (self.N_WP + 1)      # terrain per segment
            + self.N_WP            # corridor per waypoint
        )

        super().__init__(
            n_var=self.N_WP * 3,
            n_obj=3,
            n_ieq_constr=n_ineq,
            xl=xl, xu=xu,
            type_var=float,
        )

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        n_pop = X.shape[0]
        F = np.zeros((n_pop, 3))
        G = np.zeros((n_pop, self.n_ieq_constr))

        for i in range(n_pop):
            wps  = X[i].reshape(self.N_WP, 3)
            path = self.env.build_full_path(wps)
            (ft, sam, rad), g = self.env.evaluate_path(path, wps)
            F[i] = [ft, sam, rad]
            G[i] = g

        out["F"] = F
        out["G"] = G


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISER
# ──────────────────────────────────────────────────────────────────────────────

# Distinct colour + linestyle pairs for each solution
_PATH_STYLES = [
    dict(color="#e63946", lw=2.8, ls="-",  marker="o", ms=7),    # Speed-Priority
    dict(color="#2a9d8f", lw=2.8, ls="--", marker="s", ms=6),    # SAM-Avoidance
    dict(color="#e9c46a", lw=2.8, ls="-.", marker="^", ms=7),    # Radar-Stealth
    dict(color="#457b9d", lw=2.8, ls=":",  marker="D", ms=6),    # Balanced
]


class PathVisualizer:

    def __init__(self, env: BattleEnvironment) -> None:
        self.env = env

    # ── 2-D top-down ────────────────────────────────────────────────────────

    def plot_2d(
        self,
        paths:  List[np.ndarray],
        labels: List[str],
        title:  str = "Battle Environment — 2-D",
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(13, 11))
        env = self.env
        xg, yg, elev = env.terrain.grid()
        X, Y = np.meshgrid(xg, yg)

        tcf = ax.contourf(X, Y, elev, levels=25, cmap="terrain", alpha=0.50)
        cb  = plt.colorbar(tcf, ax=ax, label="Elevation (m AMSL)", shrink=0.72)
        ax.contour(X, Y, elev, levels=12, colors="k", linewidths=0.25, alpha=0.3)

        # SAM envelopes
        for sam in env.sams:
            ax.add_patch(plt.Circle(sam.position, sam.radius,
                                    color="#e63946", alpha=0.12, zorder=2))
            ax.add_patch(plt.Circle(sam.position, sam.radius, fill=False,
                                    edgecolor="#e63946", lw=1.8, ls="--", zorder=2))
            ax.plot(*sam.position, "r^", ms=10, zorder=3)
            ax.annotate(sam.name, sam.position + [0.7, 0.9],
                        fontsize=8, color="#c1121f", fontweight="bold")

        # Radar coverage (altitude-dependent — shown as dashed outer ring)
        for radar in env.radars:
            ax.add_patch(plt.Circle(radar.position, radar.max_range,
                                    color="darkorange", alpha=0.05, zorder=2))
            ax.add_patch(plt.Circle(radar.position, radar.max_range, fill=False,
                                    edgecolor="darkorange", lw=1.4, ls=":", zorder=2))
            ax.plot(*radar.position, "D", color="darkorange", ms=10, zorder=3)
            ax.annotate(radar.name, radar.position + [0.7, 0.9],
                        fontsize=8, color="darkorange", fontweight="bold")

        # NFZs
        for nfz in env.nfzs:
            ax.add_patch(plt.Circle(nfz.center, nfz.radius,
                                    color="#222", alpha=0.40, zorder=2))
            ax.add_patch(plt.Circle(nfz.center, nfz.radius, fill=False,
                                    edgecolor="#000", lw=2.0, zorder=2))
            ax.annotate(nfz.name, nfz.center, ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")

        # Flight paths — each with unique colour + linestyle + marker
        for idx, (path, lbl) in enumerate(zip(paths, labels)):
            st = _PATH_STYLES[idx % len(_PATH_STYLES)]
            ax.plot(path[:, 0], path[:, 1],
                    color=st["color"], lw=st["lw"],
                    ls=st["ls"], marker=st["marker"], ms=st["ms"],
                    label=lbl, zorder=6,
                    markeredgecolor="white", markeredgewidth=0.8)
            # Add directional arrow at mid-point
            mid = len(path) // 2
            dx = path[mid+1, 0] - path[mid, 0]
            dy = path[mid+1, 1] - path[mid, 1]
            ax.annotate("", xy=(path[mid, 0]+dx*0.6, path[mid, 1]+dy*0.6),
                        xytext=(path[mid, 0], path[mid, 1]),
                        arrowprops=dict(arrowstyle="-|>", color=st["color"],
                                        lw=1.5), zorder=7)

        # Base & Target
        ax.plot(*env.base,   "gs", ms=16, zorder=8,
                markeredgecolor="darkgreen", markeredgewidth=1.5, label="Base")
        ax.plot(*env.target, "r*", ms=20, zorder=8,
                markeredgecolor="darkred",   markeredgewidth=1.0, label="Target")
        ax.annotate("BASE",   env.base   + [1.2, -3],
                    fontsize=10, color="darkgreen", fontweight="bold")
        ax.annotate("TARGET", env.target + [1.2, -3],
                    fontsize=10, color="darkred",   fontweight="bold")

        # Search zone ring
        ac = env.aircraft
        ax.add_patch(plt.Circle(env.target, ac.search_radius_km,
                                fill=False, edgecolor="gold",
                                lw=1.5, ls="-.", zorder=5))
        ax.annotate("Search\nzone", env.target + [ac.search_radius_km + 0.5, 0],
                    fontsize=8, color="goldenrod")

        ax.set_xlim(*env.X_BOUNDS); ax.set_ylim(*env.Y_BOUNDS)
        ax.set_xlabel("X (km)", fontsize=12); ax.set_ylabel("Y (km)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.22)

        h, l = ax.get_legend_handles_labels()
        h += [mpatches.Patch(color="#e63946", alpha=0.5, label="SAM coverage"),
              mpatches.Patch(color="darkorange", alpha=0.4, label="Radar range"),
              mpatches.Patch(color="#333", alpha=0.5, label="No-Fly Zone")]
        ax.legend(handles=h, loc="lower right", fontsize=9, ncol=2,
                  framealpha=0.85)
        plt.tight_layout()
        return fig

    # ── 3-D view ────────────────────────────────────────────────────────────

    def plot_3d(
        self,
        paths:  List[np.ndarray],
        labels: List[str],
        title:  str = "3-D Flight Paths",
    ) -> plt.Figure:
        fig = plt.figure(figsize=(15, 10))
        ax  = fig.add_subplot(111, projection="3d")
        env = self.env
        xg, yg, elev = env.terrain.grid()
        X, Y = np.meshgrid(xg, yg)
        ax.plot_surface(X, Y, elev, cmap="terrain",
                        alpha=0.38, linewidth=0, antialiased=True)

        theta = np.linspace(0, 2*np.pi, 30)
        for sam in env.sams:
            Xc = sam.position[0] + sam.radius * np.outer(np.cos(theta), [1, 1])
            Yc = sam.position[1] + sam.radius * np.outer(np.sin(theta), [1, 1])
            Zc = np.outer(np.ones(30), [0, 2200])
            ax.plot_surface(Xc, Yc, Zc, color="#e63946", alpha=0.09)

        for nfz in env.nfzs:
            Xn = nfz.center[0] + nfz.radius * np.outer(np.cos(theta), [1, 1])
            Yn = nfz.center[1] + nfz.radius * np.outer(np.sin(theta), [1, 1])
            Zn = np.outer(np.ones(30), [0, 6000])
            ax.plot_surface(Xn, Yn, Zn, color="black", alpha=0.11)

        for idx, (path, lbl) in enumerate(zip(paths, labels)):
            st = _PATH_STYLES[idx % len(_PATH_STYLES)]
            ax.plot(path[:, 0], path[:, 1], path[:, 2],
                    color=st["color"], lw=2.5,
                    ls="-",   # 3-D always solid for clarity
                    marker=st["marker"], ms=5,
                    label=lbl, zorder=5)

        bh = env.terrain.elevation_at(*env.base)
        th = env.terrain.elevation_at(*env.target)
        ax.scatter(*env.base,   bh, c="green", s=180, marker="s", zorder=10)
        ax.scatter(*env.target, th, c="red",   s=200, marker="*", zorder=10)

        ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")
        ax.set_zlabel("Altitude (m AMSL)")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        plt.tight_layout()
        return fig

    # ── Altitude profile ────────────────────────────────────────────────────

    def plot_altitude_profile(
        self,
        paths:  List[np.ndarray],
        labels: List[str],
        title:  str = "Altitude Profile",
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 6))
        env = self.env
        ac  = env.aircraft

        # Terrain profile (use first path's horizontal trace)
        if paths:
            p0    = paths[0]
            d0    = np.concatenate(([0.], np.cumsum(
                np.linalg.norm(np.diff(p0[:, :2], axis=0), axis=1))))
            ter0  = [env.terrain.elevation_at(p[0], p[1]) for p in p0]
            ax.fill_between(d0, 0, ter0, color="#8B5E3C", alpha=0.35, label="Terrain")
            ax.plot(d0, ter0, color="#5a3e28", lw=0.8)

        for idx, (path, lbl) in enumerate(zip(paths, labels)):
            st = _PATH_STYLES[idx % len(_PATH_STYLES)]
            dists = np.concatenate(([0.], np.cumsum(
                np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))))
            ax.plot(dists, path[:, 2],
                    color=st["color"], lw=st["lw"], ls=st["ls"],
                    marker=st["marker"], ms=st["ms"],
                    label=lbl, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.6)

        # Radar detection thresholds
        radar_colors = ["darkorange", "coral", "peru"]
        for i, radar in enumerate(env.radars):
            ax.axhline(radar.min_detect_alt, color=radar_colors[i],
                       ls=":", lw=1.4, alpha=0.75,
                       label=f"{radar.name} detect floor ({radar.min_detect_alt:.0f} m)")

        ax.axhline(ac.search_agl, color="gold", ls="-.", lw=1.5, alpha=0.9,
                   label=f"Search altitude ({ac.search_agl:.0f} m AGL)")

        ax.set_xlabel("Along-Track Distance (km)", fontsize=12)
        ax.set_ylabel("Altitude (m AMSL)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8.5, ncol=3, framealpha=0.85)
        ax.grid(True, alpha=0.28)
        plt.tight_layout()
        return fig

    # ── Pareto front ─────────────────────────────────────────────────────────

    def plot_pareto(
        self,
        F:          np.ndarray,
        sel_F:      List[np.ndarray],
        labels:     List[str],
    ) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        pairs  = [(0, 1), (0, 2), (1, 2)]
        xlbls  = ["Flight Time (min)", "Flight Time (min)", "SAM Exposure (min)"]
        ylbls  = ["SAM Exposure (min)", "Radar Exposure Index", "Radar Exposure Index"]
        titles = ["f1: Time  vs  f2: SAM", "f1: Time  vs  f3: Radar",
                  "f2: SAM  vs  f3: Radar"]
        sc     = [1/60, 1/60, 1.0]

        for ax, (a, b), xl, yl, ttl in zip(axes, pairs, xlbls, ylbls, titles):
            ax.scatter(F[:, a]*sc[a], F[:, b]*sc[b],
                       c="#a8c7e8", s=25, alpha=0.5, label="Pareto solutions")
            for i, (sf, lbl) in enumerate(zip(sel_F, labels)):
                st = _PATH_STYLES[i % len(_PATH_STYLES)]
                ax.scatter(sf[a]*sc[a], sf[b]*sc[b],
                           c=st["color"], s=200, zorder=6,
                           marker=st["marker"],
                           edgecolors="black", linewidths=0.8,
                           label=lbl)
                ax.annotate(lbl, (sf[a]*sc[a], sf[b]*sc[b]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=8, color=st["color"], fontweight="bold")
            ax.set_xlabel(xl, fontsize=10); ax.set_ylabel(yl, fontsize=10)
            ax.set_title(ttl, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

        fig.suptitle(
            "Pareto Front — Multi-Objective Aircraft Path Planning (NSGA-II)",
            fontsize=13, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run_optimization() -> None:
    print("=" * 65)
    print("  Multi-Objective Aircraft Path Planning  (NSGA-II / pymoo)")
    print("=" * 65)

    env     = BattleEnvironment()
    problem = PathPlanningProblem(env)

    print(f"\n  Operational area : 100 × 100 km")
    print(f"  Base  → Target   : {env.base} → {env.target} km")
    print(f"  Aircraft speed   : {env.aircraft.speed*3600:.0f} km/h")
    print(f"  NFZs / SAMs / Radars : "
          f"{len(env.nfzs)} / {len(env.sams)} / {len(env.radars)}")
    print(f"  Waypoints        : {problem.N_WP} intermediate")
    print(f"  Decision vars    : {problem.n_var}")
    print(f"  Objectives       : 3  (flight_time | sam_exposure | radar_index)")
    print(f"  Constraints      : {problem.n_ieq_constr}  "
          f"= {len(env.nfzs)} NFZ + {problem.N_WP+1} terrain + "
          f"{problem.N_WP} corridor")

    POP  = 100
    NGEN = 60

    algorithm = NSGA2(
        pop_size=POP,
        sampling=CorridorSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    print(f"\n  Running NSGA-II  (pop={POP}, gen={NGEN}) …")
    print(f"  (Increase POP/NGEN in run_optimization() for a denser Pareto front)\n")

    result = minimize(
        problem, algorithm,
        termination=("n_gen", NGEN),
        seed=42,
        verbose=True,
    )

    F = result.F
    X = result.X

    print(f"\n  Pareto-optimal solutions : {len(F)}")
    print(f"  Flight time      : {F[:,0].min()/60:.1f} – {F[:,0].max()/60:.1f} min")
    print(f"  SAM exposure     : {F[:,1].min()/60:.2f} – {F[:,1].max()/60:.2f} min")
    print(f"  Radar exposure   : {F[:,2].min():.4f} – {F[:,2].max():.4f}")

    # ── select representative solutions ──────────────────────────────────────
    weight_profiles = {
        "Speed-Priority": np.array([0.70, 0.15, 0.15]),
        "SAM-Avoidance":  np.array([0.15, 0.70, 0.15]),
        "Radar-Stealth":  np.array([0.15, 0.15, 0.70]),
        "Balanced":       np.array([1/3,  1/3,  1/3 ]),
    }

    decomp    = ASF()
    sel_paths: List[np.ndarray] = []
    sel_F:     List[np.ndarray] = []
    labels:    List[str]        = []

    print("\n  Selected representative solutions:")
    print("  " + "─" * 58)

    for lbl, w in weight_profiles.items():
        idx  = decomp.do(F, 1.0 / w).argmin()
        wps  = X[idx].reshape(problem.N_WP, 3)
        path = env.build_full_path(wps)

        sel_paths.append(path)
        sel_F.append(F[idx])
        labels.append(lbl)

        plen = float(np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))
        print(f"\n  [{lbl}]")
        print(f"    Flight time    : {F[idx,0]/60:.1f} min  |  "
              f"Path length: {plen:.1f} km")
        print(f"    SAM exposure   : {F[idx,1]/60:.2f} min  "
              f"({100*F[idx,1]/F[idx,0]:.1f}% of flight)")
        print(f"    Radar exposure : {F[idx,2]:.4f}")
        print(f"    Alt AMSL range : "
              f"{path[:,2].min():.0f} – {path[:,2].max():.0f} m")

    # ── plots ─────────────────────────────────────────────────────────────────
    viz = PathVisualizer(env)

    fig2d = viz.plot_2d(sel_paths, labels,
                        "Multi-Objective Path Planning — 2-D Overview")
    fig2d.savefig("path_planning_2d.png", dpi=150, bbox_inches="tight")
    print("\n  Saved: path_planning_2d.png")

    fig3d = viz.plot_3d(sel_paths, labels,
                        "Multi-Objective Path Planning — 3-D Paths")
    fig3d.savefig("path_planning_3d.png", dpi=150, bbox_inches="tight")
    print("  Saved: path_planning_3d.png")

    fig_alt = viz.plot_altitude_profile(
        sel_paths, labels,
        "Altitude Profile — Terrain-Following & Search Climb"
    )
    fig_alt.savefig("path_planning_altitude.png", dpi=150, bbox_inches="tight")
    print("  Saved: path_planning_altitude.png")

    fig_pf = viz.plot_pareto(F, sel_F, labels)
    fig_pf.savefig("path_planning_pareto.png", dpi=150, bbox_inches="tight")
    print("  Saved: path_planning_pareto.png")

    plt.show()
    print("\n  Done.")


if __name__ == "__main__":
    run_optimization()
