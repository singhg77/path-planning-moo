"""
MILP / LP Path Planning — Weather Zone Avoidance (Discrete Graph Solver)
=========================================================================
Same physical scenario as weather_path_planning.py but solved as a
minimum-cost network-flow LP on a discretised 3-D grid graph.

Network structure
-----------------
    Grid : NX × NY × NZ nodes  (east×north×altitude)
    Edges: forward-only (west→east), diagonal steps in Y and Z allowed
    Super-source S  →  column-0 nodes near SOURCE[1]
    Column-(NX-1) nodes near DEST[1]  →  super-sink T

LP formulation  (totally unimodular ⇒ LP relaxation gives integer 0/1 flow)
----------------------------------------------------------------------------
    Variables   x_e ∈ [0, 1]   (flow on each directed edge)
    Minimise    Σ_e  (w₁·c_time_e + w₂·c_risk_e + w₃·c_fuel_e) · x_e
    Subject to  flow conservation at every internal node (Σ_in = Σ_out)
                unit supply at S,  unit demand at T
                x_e ≥ 0

Interactive visualisation (distinct from NSGA-II viewer)
---------------------------------------------------------
    Left panel   : 2-D overhead map — graph edges coloured by risk,
                   LP-optimal path highlighted in white/gold
    Top-right    : Altitude profile of optimal path
    Bottom-left  : Per-segment cost-breakdown stacked bar
    Bottom-right : Ternary weight triangle — drag the dot to blend the
                   three objectives; the LP re-solves immediately (<0.1 s)

Run:  python milp_path_planning.py
"""

from __future__ import annotations
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.collections as mcollections
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")


# ── SHARED SCENARIO PARAMETERS (mirror weather_path_planning.py) ─────────────

SOURCE = np.array([  5.0, 40.0,  80.0])   # [x_km, y_km, agl_m]
DEST   = np.array([145.0, 40.0, 120.0])

X_BOUNDS = (0.0, 150.0)
Y_BOUNDS = (0.0,  80.0)

SPEED        = 0.083   # km/s ≈ 300 km/h
MIN_AGL      = 30.0
MAX_AGL      = 600.0
SAFE_AGL     = 61.0
DETECT_AGL   = 300.0
FUEL_OPT_AGL = 500.0
FUEL_BASE    = 1.0     # kg/km at FUEL_OPT_AGL
FUEL_LOW_K   = 0.50

W_WEATHER     = 1.0
W_TERRAIN_HIT = 0.8
W_DETECT      = 3.0
W_UNDERGROUND = 20.0

WEATHER = [
    {"center": np.array([40.0, 44.0]),  "radius": 11.0, "name": "Storm-Alpha"},
    {"center": np.array([80.0, 40.0]),  "radius": 14.0, "name": "Storm-Bravo"},
    {"center": np.array([118.0, 36.5]), "radius": 10.0, "name": "Storm-Charlie"},
]

# ── GRID DEFINITION ───────────────────────────────────────────────────────────

NX = 20       # east–west columns  (includes source & dest columns)
NY = 11       # north–south rows
NZ = 5        # altitude levels

X_GRID = np.linspace(SOURCE[0], DEST[0], NX)    # km
Y_GRID = np.linspace(8.0,  72.0, NY)            # km  (leave margin)
Z_GRID = np.array([80.0, 150.0, 300.0, 450.0, 580.0])   # m AGL


# ── TERRAIN MODEL (identical to weather_path_planning.py) ────────────────────

_TER_X = np.linspace(X_BOUNDS[0], X_BOUNDS[1], 60)
_TER_Y = np.linspace(Y_BOUNDS[0], Y_BOUNDS[1], 40)
np.random.seed(0)
_raw = (np.random.rand(40, 60) * 60
        + 30 * np.sin(np.linspace(0, 4*np.pi, 60))[None, :]
        + 20 * np.cos(np.linspace(0, 3*np.pi, 40))[:, None])
_TER_SPLINE = RectBivariateSpline(_TER_Y, _TER_X, _raw, kx=3, ky=3)


def terrain_elev(x_km: float | np.ndarray, y_km: float | np.ndarray) -> np.ndarray:
    x = np.atleast_1d(np.asarray(x_km, float))
    y = np.atleast_1d(np.asarray(y_km, float))
    return _TER_SPLINE(y, x, grid=False)


def fuel_rate(agl_m: float) -> float:
    """kg/km as a function of AGL (higher = thinner air = less drag)."""
    frac = np.clip((agl_m - MIN_AGL) / (FUEL_OPT_AGL - MIN_AGL), 0.0, 1.0)
    return FUEL_BASE * (1.0 + FUEL_LOW_K * (1.0 - frac))


# ── NODE / EDGE HELPERS ───────────────────────────────────────────────────────

def nid(ix: int, iy: int, iz: int) -> int:
    return ix * (NY * NZ) + iy * NZ + iz


N_GRID  = NX * NY * NZ
NODE_S  = N_GRID        # super-source
NODE_T  = N_GRID + 1    # super-sink
N_TOTAL = N_GRID + 2


def _seg_risk(x1, y1, agl1, x2, y2, agl2, n_samp: int = 8) -> float:
    """Compute risk cost for a segment between two 3-D points."""
    ts   = np.linspace(0, 1, n_samp)
    xs   = x1 + (x2 - x1) * ts
    ys   = y1 + (y2 - y1) * ts
    agls = agl1 + (agl2 - agl1) * ts

    dx  = (x2 - x1)
    dy  = (y2 - y1)
    dz  = (agl2 - agl1) / 1000.0   # to km
    seg_len = np.sqrt(dx**2 + dy**2 + dz**2)  # km
    seg_t   = seg_len / SPEED                  # seconds

    risk = 0.0

    # Weather penetration risk
    for storm in WEATHER:
        cx, cy, r = storm["center"][0], storm["center"][1], storm["radius"]
        dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)
        frac  = float(np.mean(dists < r))
        risk += frac * W_WEATHER * seg_t

    # Terrain collision risk
    ter = terrain_elev(xs, ys)
    amsl = ter + agls
    underground_frac = float(np.mean(agls < 0))
    low_frac         = float(np.mean((agls >= 0) & (agls < SAFE_AGL)))
    detect_excess    = np.maximum(0.0, agls - DETECT_AGL) / (MAX_AGL - DETECT_AGL)
    risk += (underground_frac * W_UNDERGROUND
             + low_frac * W_TERRAIN_HIT
             + float(detect_excess.mean()) * W_DETECT) * seg_t
    return risk


def _seg_fuel(x1, y1, agl1, x2, y2, agl2) -> float:
    mean_agl = 0.5 * (agl1 + agl2)
    horiz_km = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return horiz_km * fuel_rate(mean_agl)


def _seg_time(x1, y1, agl1, x2, y2, agl2) -> float:
    dz = (agl2 - agl1) / 1000.0
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + dz**2) / SPEED


# ── BUILD GRAPH ───────────────────────────────────────────────────────────────

def build_graph():
    """Return (edges, cost_matrix [E×3]) where cost columns = [time, risk, fuel]."""
    edges   = []
    c_time  = []
    c_risk  = []
    c_fuel  = []

    # ── internal forward edges: (ix,iy,iz) → (ix+1, iy+diy, iz+diz) ──────────
    for ix in range(NX - 1):
        for iy in range(NY):
            for iz in range(NZ):
                x1   = X_GRID[ix]; y1 = Y_GRID[iy]; a1 = Z_GRID[iz]
                for diy in (-1, 0, 1):
                    for diz in (-1, 0, 1):
                        iy2, iz2 = iy + diy, iz + diz
                        if not (0 <= iy2 < NY and 0 <= iz2 < NZ):
                            continue
                        x2 = X_GRID[ix + 1]; y2 = Y_GRID[iy2]; a2 = Z_GRID[iz2]
                        edges.append((nid(ix, iy, iz), nid(ix+1, iy2, iz2)))
                        c_time.append(_seg_time(x1, y1, a1, x2, y2, a2))
                        c_risk.append(_seg_risk(x1, y1, a1, x2, y2, a2))
                        c_fuel.append(_seg_fuel(x1, y1, a1, x2, y2, a2))

    # ── source injection edges: NODE_S → column-0 nodes ──────────────────────
    # Connect to all (iy,iz) at ix=0 — zero cost; LP will choose best entry
    sx, sy, sa = SOURCE
    for iy in range(NY):
        for iz in range(NZ):
            x2 = X_GRID[0]; y2 = Y_GRID[iy]; a2 = Z_GRID[iz]
            edges.append((NODE_S, nid(0, iy, iz)))
            # small cost to prefer nodes near actual SOURCE[1] and SOURCE[2]
            dy_pen = abs(y2 - sy) * 0.01
            da_pen = abs(a2 - sa) / 1000.0 * 0.01
            pen    = dy_pen + da_pen
            c_time.append(pen); c_risk.append(0.0); c_fuel.append(0.0)

    # ── sink collection edges: column-(NX-1) nodes → NODE_T ──────────────────
    dx_dest, dy_dest, da_dest = DEST
    for iy in range(NY):
        for iz in range(NZ):
            x1 = X_GRID[-1]; y1 = Y_GRID[iy]; a1 = Z_GRID[iz]
            edges.append((nid(NX-1, iy, iz), NODE_T))
            dy_pen = abs(y1 - dy_dest) * 0.01
            da_pen = abs(a1 - da_dest) / 1000.0 * 0.01
            pen    = dy_pen + da_pen
            c_time.append(pen); c_risk.append(0.0); c_fuel.append(0.0)

    costs = np.column_stack([c_time, c_risk, c_fuel])
    return edges, costs


# ── BUILD FLOW-CONSERVATION CONSTRAINT MATRIX ─────────────────────────────────

def build_constraints(edges: list[tuple[int,int]]):
    """
    Returns (A_eq, b_eq) for the LP:
        Σ_{in-edges of v} x_e  −  Σ_{out-edges of v} x_e = b_v
        b_S = -1 (source),  b_T = +1 (sink),  b_v = 0 (internal)
    """
    E = len(edges)
    row_idx = []
    col_idx = []
    data    = []

    for e_idx, (u, v) in enumerate(edges):
        # u is tail (outflow): coefficient -1
        row_idx.append(u); col_idx.append(e_idx); data.append(-1.0)
        # v is head (inflow):  coefficient +1
        row_idx.append(v); col_idx.append(e_idx); data.append(+1.0)

    A = csr_matrix((data, (row_idx, col_idx)), shape=(N_TOTAL, E))
    b = np.zeros(N_TOTAL)
    b[NODE_S] = -1.0   # unit supply
    b[NODE_T] = +1.0   # unit demand
    return A, b


# ── LP SOLVER ─────────────────────────────────────────────────────────────────

def solve_lp(edges, costs: np.ndarray, A, b, w: np.ndarray) -> np.ndarray:
    """
    Solve min  (w · costs.T) · x   s.t.  A x = b,  0 ≤ x ≤ 1.
    Returns flow vector x (E,).
    """
    w = np.maximum(w, 1e-9)
    w = w / w.sum()
    c = costs @ w

    res = linprog(
        c,
        A_eq=A, b_eq=b,       # sparse matrix — HiGHS handles it natively
        bounds=[(0, 1)] * len(edges),
        method="highs",
        options={"disp": False},
    )
    if res.status != 0:
        # fallback: return uniform flow (should not happen on connected graph)
        return np.ones(len(edges)) / len(edges)
    return res.x


def extract_path(edges: list, flow: np.ndarray) -> list[int]:
    """
    Follow max-flow edges from NODE_S to NODE_T to extract the optimal path
    as an ordered list of grid node ids.
    """
    # Build adjacency: for each node, sorted outgoing edges by flow
    out_edges: dict[int, list[tuple[float, int, int]]] = {}
    for e_idx, (u, v) in enumerate(edges):
        out_edges.setdefault(u, []).append((flow[e_idx], v, e_idx))

    path = []
    cur  = NODE_S
    visited = {cur}
    while cur != NODE_T:
        candidates = out_edges.get(cur, [])
        if not candidates:
            break
        _, nxt, _ = max(candidates, key=lambda t: t[0])
        if nxt in visited:
            break
        path.append(nxt)
        visited.add(nxt)
        cur = nxt
    return path


def path_to_xyz(node_path: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert list of node ids to (x_km, y_km, agl_m) arrays."""
    xs, ys, zs = [], [], []
    for nd in node_path:
        if nd in (NODE_S, NODE_T):
            continue
        ix  = nd // (NY * NZ)
        rem = nd % (NY * NZ)
        iy  = rem // NZ
        iz  = rem % NZ
        xs.append(X_GRID[ix]); ys.append(Y_GRID[iy]); zs.append(Z_GRID[iz])
    return np.array(xs), np.array(ys), np.array(zs)


# ── TERNARY TRIANGLE HELPER ───────────────────────────────────────────────────

_H = np.sqrt(3) / 2   # height of unit equilateral triangle

def weights_to_cart(w1: float, w2: float, w3: float) -> tuple[float, float]:
    """Barycentric (w1+w2+w3=1) → 2D Cartesian (equilateral triangle)."""
    return w2 + 0.5 * w3, _H * w3


def cart_to_weights(px: float, py: float) -> tuple[float, float, float]:
    """2D Cartesian → (w_time, w_risk, w_fuel), clamped to valid simplex."""
    w_fuel = np.clip(py / _H, 0, 1)
    w_risk = np.clip(px - 0.5 * w_fuel, 0, 1)
    w_time = np.clip(1.0 - w_risk - w_fuel, 0, 1)
    s = w_time + w_risk + w_fuel
    return w_time/s, w_risk/s, w_fuel/s


def point_in_tri(px: float, py: float) -> bool:
    w1, w2, w3 = cart_to_weights(px, py)
    return (w1 >= -0.02) and (w2 >= -0.02) and (w3 >= -0.02)


# ── MAIN LAUNCH ───────────────────────────────────────────────────────────────

def launch_milp_interactive():
    print("Building 3-D graph …", flush=True)
    edges, costs = build_graph()
    print(f"  {N_GRID} grid nodes,  {len(edges)} edges", flush=True)

    print("Building LP constraint matrix …", flush=True)
    A, b = build_constraints(edges)

    # Initial weights: balanced
    _w = np.array([0.33, 0.34, 0.33])

    print("Solving LP (initial weights) …", flush=True)
    flow = solve_lp(edges, costs, A, b, _w)
    path_nodes = extract_path(edges, flow)
    px, py, pz = path_to_xyz(path_nodes)
    print("  Done.  Path has", len(px), "grid waypoints.", flush=True)

    # ── Terrain grid for background ──────────────────────────────────────────
    xg = np.linspace(X_BOUNDS[0], X_BOUNDS[1], 120)
    yg = np.linspace(Y_BOUNDS[0], Y_BOUNDS[1], 80)
    XG, YG = np.meshgrid(xg, yg)
    TG = terrain_elev(XG.ravel(), YG.ravel()).reshape(XG.shape)

    # ── Pre-compute edge midpoints and risk for colouring ────────────────────
    # (only internal forward edges, not source/sink)
    n_fwd = (NX - 1) * NY * NZ * 9   # approx upper bound
    edge_risk_vals = costs[:, 1]      # risk cost column

    # For each internal forward edge record midpoint x,y
    fwd_count = 0
    for ix in range(NX - 1):
        for iy in range(NY):
            for iz in range(NZ):
                for diy in (-1, 0, 1):
                    for diz in (-1, 0, 1):
                        if 0 <= iy + diy < NY and 0 <= iz + diz < NZ:
                            fwd_count += 1

    fwd_segs   = np.zeros((fwd_count, 2, 2))  # [edge, point, (x,y)]
    fwd_risks  = np.zeros(fwd_count)
    fwd_flows  = np.zeros(fwd_count)
    fwd_eidx   = np.zeros(fwd_count, dtype=int)
    ei = 0
    for ix in range(NX - 1):
        for iy in range(NY):
            for iz in range(NZ):
                x1 = X_GRID[ix]; y1 = Y_GRID[iy]
                for diy in (-1, 0, 1):
                    for diz in (-1, 0, 1):
                        iy2, iz2 = iy + diy, iz + diz
                        if 0 <= iy2 < NY and 0 <= iz2 < NZ:
                            x2 = X_GRID[ix + 1]; y2 = Y_GRID[iy2]
                            fwd_segs[ei, 0] = [x1, y1]
                            fwd_segs[ei, 1] = [x2, y2]
                            fwd_risks[ei]   = costs[ei, 1]
                            fwd_eidx[ei]    = ei
                            ei += 1

    def update_fwd_flows(fl):
        for i in range(fwd_count):
            fwd_flows[i] = fl[fwd_eidx[i]]

    update_fwd_flows(flow)

    # ── Figure layout ─────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(17, 10), facecolor="#0d0d1a")
    fig.suptitle("MILP / LP Network-Flow Path Planner  (3-D Grid Graph)",
                 color="white", fontsize=13, y=0.98)

    gs = gridspec.GridSpec(
        3, 3,
        left=0.04, right=0.97, top=0.93, bottom=0.08,
        wspace=0.32, hspace=0.42,
        height_ratios=[2.2, 1.4, 0.18],
        width_ratios=[2.2, 1.4, 1.4],
    )

    ax_map  = fig.add_subplot(gs[0, 0])          # 2-D overhead map
    ax_alt  = fig.add_subplot(gs[0, 1:])         # altitude profile (spans 2 cols)
    ax_cost = fig.add_subplot(gs[1, 0])          # per-segment cost bar
    ax_tri  = fig.add_subplot(gs[1, 1])          # ternary triangle
    ax_info = fig.add_subplot(gs[1, 2])          # text summary

    for ax in (ax_map, ax_alt, ax_cost, ax_tri, ax_info):
        ax.set_facecolor("#12122a")

    # ── Sliders (bottom row) ──────────────────────────────────────────────────
    sl_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    for a in sl_axes:
        a.set_facecolor("#12122a")

    sl_time = Slider(sl_axes[0], "Time",  0.0, 1.0, valinit=_w[0],
                     color="#4aa8ff", track_color="#1a1a3a")
    sl_risk = Slider(sl_axes[1], "Risk",  0.0, 1.0, valinit=_w[1],
                     color="#ff6b6b", track_color="#1a1a3a")
    sl_fuel = Slider(sl_axes[2], "Fuel",  0.0, 1.0, valinit=_w[2],
                     color="#51cf66", track_color="#1a1a3a")
    for sl in (sl_time, sl_risk, sl_fuel):
        sl.label.set_color("white"); sl.valtext.set_color("white")

    # ── Draw static map elements ──────────────────────────────────────────────
    ax_map.contourf(xg, yg, TG, levels=15, cmap="terrain", alpha=0.5)
    ax_map.contour( xg, yg, TG, levels=8,  colors="gray", alpha=0.3, linewidths=0.4)

    storm_patches = []
    storm_colors  = ["#ff4500", "#ff8c00", "#ffd700"]
    for storm, col in zip(WEATHER, storm_colors):
        circ = mpatches.Circle(storm["center"], storm["radius"],
                               color=col, alpha=0.35, zorder=3)
        ax_map.add_patch(circ)
        ax_map.text(storm["center"][0], storm["center"][1] + storm["radius"] + 1.5,
                    storm["name"], ha="center", fontsize=7, color=col)
        storm_patches.append(circ)

    ax_map.plot(*SOURCE[:2], "o", color="#00ffcc", ms=10, zorder=6, label="SOURCE")
    ax_map.plot(*DEST[:2],   "s", color="#ff6600", ms=10, zorder=6, label="DEST")
    ax_map.set_xlim(X_BOUNDS); ax_map.set_ylim(Y_BOUNDS)
    ax_map.set_xlabel("East  (km)", color="white"); ax_map.set_ylabel("North  (km)", color="white")
    ax_map.set_title("2-D Map  (edge risk: blue→red)", color="white", fontsize=9)
    ax_map.tick_params(colors="white")
    ax_map.legend(loc="upper left", fontsize=7, facecolor="#1a1a3a", labelcolor="white")

    # Edge line collection (risk coloured)
    risk_max = fwd_risks.max() + 1e-9
    edge_colors_rgba = plt.cm.RdYlGn_r(fwd_risks / risk_max)
    edge_colors_rgba[:, 3] = 0.18   # mostly transparent
    lc = mcollections.LineCollection(fwd_segs, colors=edge_colors_rgba, linewidths=0.5, zorder=2)
    ax_map.add_collection(lc)

    path_line, = ax_map.plot([], [], "-", color="white", lw=2.5, zorder=7)
    flow_lc_handle = [None]  # mutable container for flow-weighted edge collection

    # ── Altitude profile ──────────────────────────────────────────────────────
    ax_alt.axhspan(MIN_AGL, SAFE_AGL,    alpha=0.15, color="#ff4444", label="< 200 ft (risky)")
    ax_alt.axhspan(SAFE_AGL, DETECT_AGL, alpha=0.10, color="#44ff88", label="Stealthy band")
    ax_alt.axhspan(DETECT_AGL, MAX_AGL,  alpha=0.10, color="#4488ff", label="> 300 m (radar)")
    ax_alt.axhline(FUEL_OPT_AGL, color="#51cf66", ls="--", lw=0.8, alpha=0.7, label="Fuel-optimal")
    alt_line, = ax_alt.plot([], [], "o-", color="#ffe066", lw=2, ms=5, zorder=5)
    ax_alt.set_xlabel("East  (km)", color="white"); ax_alt.set_ylabel("AGL  (m)", color="white")
    ax_alt.set_title("Altitude Profile", color="white", fontsize=9)
    ax_alt.set_xlim(X_BOUNDS); ax_alt.set_ylim(0, MAX_AGL + 30)
    ax_alt.tick_params(colors="white")
    ax_alt.legend(loc="upper right", fontsize=7, facecolor="#1a1a3a", labelcolor="white")

    # ── Cost breakdown bar ────────────────────────────────────────────────────
    ax_cost.set_title("Segment Cost Breakdown", color="white", fontsize=9)
    ax_cost.set_xlabel("Waypoint segment", color="white")
    ax_cost.set_ylabel("Cost (norm.)", color="white")
    ax_cost.tick_params(colors="white")
    cost_bars = [None]  # updated each redraw

    # ── Ternary triangle widget ───────────────────────────────────────────────
    ax_tri.set_xlim(-0.08, 1.08); ax_tri.set_ylim(-0.12, _H + 0.12)
    ax_tri.set_aspect("equal"); ax_tri.axis("off")
    ax_tri.set_title("Weight Triangle\n(click/drag)", color="white", fontsize=9)

    tri_verts = np.array([[0, 0], [1, 0], [0.5, _H], [0, 0]])
    ax_tri.plot(tri_verts[:, 0], tri_verts[:, 1], "w-", lw=1.5)
    ax_tri.text(-0.06, -0.07, "Time",  color="#4aa8ff", ha="center", fontsize=9, fontweight="bold")
    ax_tri.text(1.06,  -0.07, "Risk",  color="#ff6b6b", ha="center", fontsize=9, fontweight="bold")
    ax_tri.text(0.50,  _H+0.06, "Fuel", color="#51cf66", ha="center", fontsize=9, fontweight="bold")

    # Gradient fill for ternary triangle (optional visual aid)
    tx0, ty0 = weights_to_cart(1, 0, 0)
    tx1, ty1 = weights_to_cart(0, 1, 0)
    tx2, ty2 = weights_to_cart(0, 0, 1)

    tri_dot, = ax_tri.plot([], [], "o", color="white", ms=12, zorder=10)
    tri_cross, = ax_tri.plot([], [], "+", color="yellow", ms=14, mew=2, zorder=11)

    # ── Info panel ────────────────────────────────────────────────────────────
    ax_info.axis("off")
    info_text = ax_info.text(0.05, 0.95, "", transform=ax_info.transAxes,
                             va="top", ha="left", fontsize=9, color="white",
                             fontfamily="monospace")
    ax_info.set_title("Optimal Path Summary", color="white", fontsize=9)

    # ── State ─────────────────────────────────────────────────────────────────
    _st = {"w": _w.copy(), "dragging": False, "last_solve_ms": 0.0}

    def redraw(fl, w):
        nonlocal path_nodes, px, py, pz
        path_nodes = extract_path(edges, fl)
        px, py, pz = path_to_xyz(path_nodes)

        # 2-D map path
        path_line.set_data(px, py)

        # Update edge colours by flow (highlight used edges)
        update_fwd_flows(fl)
        ec2 = plt.cm.RdYlGn_r(fwd_risks / risk_max).copy()
        # Used edges (flow > 0.4) get full opacity gold; others dim
        for i in range(fwd_count):
            if fwd_flows[i] > 0.4:
                ec2[i] = [1.0, 0.85, 0.0, 0.9]
            else:
                ec2[i, 3] = 0.12
        lc.set_colors(ec2)

        # Altitude profile
        if len(px) > 0:
            alt_line.set_data(px, pz)
        else:
            alt_line.set_data([], [])

        # Cost breakdown bar
        ax_cost.cla()
        ax_cost.set_facecolor("#12122a")
        ax_cost.set_title("Segment Cost Breakdown", color="white", fontsize=9)
        ax_cost.set_xlabel("Segment #", color="white")
        ax_cost.set_ylabel("Cost (s)", color="white")
        ax_cost.tick_params(colors="white")
        if len(px) > 1:
            n_seg = len(px) - 1
            seg_t, seg_r, seg_f = [], [], []
            for k in range(n_seg):
                x1_, y1_, a1_ = px[k], py[k], pz[k]
                x2_, y2_, a2_ = px[k+1], py[k+1], pz[k+1]
                seg_t.append(_seg_time(x1_, y1_, a1_, x2_, y2_, a2_))
                seg_r.append(_seg_risk(x1_, y1_, a1_, x2_, y2_, a2_))
                seg_f.append(_seg_fuel(x1_, y1_, a1_, x2_, y2_, a2_))
            segs_t = np.array(seg_t)
            segs_r = np.array(seg_r)
            segs_f = np.array(seg_f)
            xs_b   = np.arange(n_seg)
            ax_cost.bar(xs_b, segs_t,            color="#4aa8ff", alpha=0.8, label="Time (s)")
            ax_cost.bar(xs_b, segs_r, bottom=segs_t,
                                                  color="#ff6b6b", alpha=0.8, label="Risk")
            ax_cost.bar(xs_b, segs_f, bottom=segs_t+segs_r,
                                                  color="#51cf66", alpha=0.8, label="Fuel (kg)")
            ax_cost.legend(fontsize=7, facecolor="#1a1a3a", labelcolor="white")
        else:
            ax_cost.text(0.5, 0.5, "No path found", ha="center", va="center",
                         color="red", transform=ax_cost.transAxes)

        # Info panel
        if len(px) > 1:
            tot_t = sum(_seg_time(px[k], py[k], pz[k], px[k+1], py[k+1], pz[k+1])
                        for k in range(len(px)-1))
            tot_r = sum(_seg_risk(px[k], py[k], pz[k], px[k+1], py[k+1], pz[k+1])
                        for k in range(len(px)-1))
            tot_f = sum(_seg_fuel(px[k], py[k], pz[k], px[k+1], py[k+1], pz[k+1])
                        for k in range(len(px)-1))
            mean_agl = float(pz.mean())
            txt = (
                f"Weights\n"
                f"  Time : {w[0]:.2f}\n"
                f"  Risk : {w[1]:.2f}\n"
                f"  Fuel : {w[2]:.2f}\n\n"
                f"Objectives\n"
                f"  Time : {tot_t/60:.1f} min\n"
                f"  Risk : {tot_r:.1f}\n"
                f"  Fuel : {tot_f:.1f} kg\n\n"
                f"Mean AGL: {mean_agl:.0f} m\n"
                f"Nodes   : {len(px)}\n"
                f"Solve   : {_st['last_solve_ms']:.0f} ms"
            )
        else:
            txt = "No feasible path\nfound."
        info_text.set_text(txt)

        # Ternary dot
        tx, ty = weights_to_cart(*w)
        tri_dot.set_data([tx], [ty])
        tri_cross.set_data([tx], [ty])

        fig.canvas.draw_idle()

    def do_solve(w):
        import time as _time
        t0   = _time.perf_counter()
        fl   = solve_lp(edges, costs, A, b, w)
        _st["last_solve_ms"] = (_time.perf_counter() - t0) * 1000
        return fl

    # Initial draw
    redraw(flow, _w)

    # ── Slider callbacks ──────────────────────────────────────────────────────
    def on_slider(_val):
        raw = np.array([sl_time.val, sl_risk.val, sl_fuel.val])
        s   = raw.sum()
        if s < 1e-9:
            return
        w   = raw / s
        _st["w"] = w
        fl  = do_solve(w)
        redraw(fl, w)

    sl_time.on_changed(on_slider)
    sl_risk.on_changed(on_slider)
    sl_fuel.on_changed(on_slider)

    # ── Ternary click/drag ────────────────────────────────────────────────────
    def on_press(event):
        if event.inaxes is not ax_tri:
            return
        if point_in_tri(event.xdata, event.ydata):
            _st["dragging"] = True
            w = cart_to_weights(event.xdata, event.ydata)
            _st["w"] = np.array(w)
            # sync sliders (suppress callbacks momentarily)
            sl_time.set_val(w[0]); sl_risk.set_val(w[1]); sl_fuel.set_val(w[2])

    def on_release(event):
        _st["dragging"] = False

    def on_motion(event):
        if not _st["dragging"] or event.inaxes is not ax_tri:
            return
        if event.xdata is None or event.ydata is None:
            return
        if point_in_tri(event.xdata, event.ydata):
            w = cart_to_weights(event.xdata, event.ydata)
            _st["w"] = np.array(w)
            sl_time.set_val(w[0]); sl_risk.set_val(w[1]); sl_fuel.set_val(w[2])

    fig.canvas.mpl_connect("button_press_event",   on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event",  on_motion)

    plt.show()


if __name__ == "__main__":
    launch_milp_interactive()
