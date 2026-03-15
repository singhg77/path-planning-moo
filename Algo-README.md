# Multi-Objective Aircraft Path Planning — Algorithm Reference

> **How the code works**: problem formulation, objective functions, NSGA-II mechanics,
> visualisation design, and implementation decisions — all in one place.

---

## Table of Contents

1. [Problem Overview](#1-problem-overview)
2. [Decision Variables — What the Optimizer Controls](#2-decision-variables--what-the-optimizer-controls)
3. [Path Decoding — From Numbers to a Flyable Path](#3-path-decoding--from-numbers-to-a-flyable-path)
4. [Objective Functions](#4-objective-functions)
5. [Why Classical Solvers Won't Work](#5-why-classical-solvers-wont-work)
6. [NSGA-II — The Genetic Algorithm](#6-nsga-ii--the-genetic-algorithm)
7. [Key Implementation Decisions](#7-key-implementation-decisions)
8. [Selecting One Solution — The Sliders](#8-selecting-one-solution--the-sliders)
9. [Visualisation Guide](#9-visualisation-guide)
10. [File Reference](#10-file-reference)

---

## 1. Problem Overview

An aircraft must fly from a fixed **SOURCE** (west, 3D) to a fixed **DEST** (east, 3D)
across 140 km of varied terrain. Three circular storm cells sit directly in the path.

```
SOURCE (5 km, 40 km, 80 m AGL) ──────────────────► DEST (145 km, 40 km, 120 m AGL)
                                  ☁  ☁  ☁
                               Storm  Storm  Storm
                               Alpha  Bravo  Charlie
```

The aircraft **can** fly through storms but accumulates risk proportional to time spent
inside. It can fly low (stealthy, safe from radar) or high (fuel-efficient, radar-visible).

Three **conflicting objectives** must be minimised simultaneously:

| Objective | Symbol | Meaning |
|-----------|--------|---------|
| Mission time | f₁ | Minimise total flight time |
| Survival risk | f₂ | Minimise weather exposure + altitude violations |
| Fuel burn | f₃ | Minimise total fuel consumed |

These are in direct conflict — there is no single path that is best on all three at once.
The output is a **Pareto front**: a set of 150 solutions covering the full trade-off spectrum.

---

## 2. Decision Variables — What the Optimizer Controls

Each candidate path is encoded as **8 intermediate waypoints in 3D space**, producing a
flat vector of **24 continuous real-valued variables**:

```
X = [x₁, y₁, agl₁,   x₂, y₂, agl₂,   ...   x₈, y₈, agl₈]
     └─── waypoint 1 ──┘  └─── waypoint 2 ──┘    └─── waypoint 8 ──┘
```

Variable bounds (box constraints):

```python
SOURCE = [5.0, 40.0, 80.0]    # fixed start [x_km, y_km, agl_m]
DEST   = [145.0, 40.0, 120.0] # fixed end

xl = [0 km,   0 km,  30 m AGL]   # lower bounds per waypoint
xu = [150 km, 80 km, 600 m AGL]  # upper bounds per waypoint
```

The optimizer reasons in **AGL (above ground level)**, not raw altitude. AGL is
physically meaningful regardless of terrain — 200 m AGL is always 200 m above whatever
is below, whether that is a valley or a hill. The terrain elevation is added back at
decode time.

**Why 8 waypoints?**
Fewer waypoints (e.g. 5) cannot thread precisely between storms — the path clips storm
edges when trying to avoid multiple obstacles simultaneously. Eight gives enough
flexibility to route north of Alpha, through the gap before Bravo, and south of Charlie
in a single smooth trajectory.

---

## 3. Path Decoding — From Numbers to a Flyable Path

```python
def build_path(waypoints):
    def to_amsl(xy_agl):
        terrain_elev = TERRAIN.at(xy_agl[0], xy_agl[1])   # bicubic spline lookup
        return [xy_agl[0], xy_agl[1], terrain_elev + xy_agl[2]]  # AGL → AMSL

    path = [to_amsl(SOURCE)]
    for wp in waypoints:
        path.append(to_amsl(wp))
    path.append(to_amsl(DEST))
    return np.array(path)   # shape (10, 3): SOURCE + 8 waypoints + DEST
```

The terrain model is a **synthetic elevation grid** (80×50 km, 150×100 nodes) generated
with superimposed Gaussian hills:

```python
class Terrain:
    # Built from overlapping Gaussian bumps of varying height and width
    # Interpolated with a bicubic spline (RectBivariateSpline) for smooth lookup
    def at(self, x_km, y_km) -> float: ...       # single point
    def at_points(self, xs, ys) -> ndarray: ...  # vectorised batch lookup
```

The resulting full path has 10 points (SOURCE + 8 waypoints + DEST) in AMSL coordinates.

---

## 4. Objective Functions

All three objectives are computed in a single vectorised pass through `evaluate_path()`.
Each path segment is sampled at **N_SAMP = 20 points** for numerical integration.

```
path[0] ──── path[1] ──── path[2] ── ... ── path[9]
         ↑20 pts↑   ↑20 pts↑           ↑20 pts↑
```

### f₁ — Mission Time (seconds)

```python
seg3d_km = norm(waypoint[i+1] - waypoint[i])   # true 3D distance
seg_time = seg3d_km / SPEED                     # SPEED = 0.083 km/s ≈ 300 km/h
f1 = sum(seg_time)
```

Simple physics: longer or more indirect paths take more time. Altitude changes add to
the 3D distance and therefore to mission time.

---

### f₂ — Survival Risk (second-equivalents)

Composed of four sub-penalties, each multiplied by segment time to give time-weighted
exposure:

```
f₂ = weather_risk + underground_risk + low_altitude_risk + detection_risk
```

**Weather risk** — time spent inside storm circles:
```python
for storm in WEATHER:
    d = distance(segment_sample_points, storm.center)
    wx_frac += (d < storm.radius).mean(axis=1)   # fraction of samples inside
weather_risk = sum(wx_frac * seg_time * W_WEATHER)    # W_WEATHER = 1.0
```

**Underground risk** — flying below terrain surface (catastrophic):
```python
underground_frac = (agl < 0).mean(axis=1)
underground_risk = sum(underground_frac * seg_time * W_UNDERGROUND)  # W_UNDERGROUND = 20.0
```

**Low-altitude risk** — below 61 m AGL (terrain collision):
```python
low_frac = ((agl >= 0) & (agl < SAFE_AGL)).mean(axis=1)   # SAFE_AGL = 61 m
low_risk = sum(low_frac * seg_time * W_TERRAIN_HIT)        # W_TERRAIN_HIT = 0.8
```

**Radar detection risk** — altitude-graded above 300 m AGL:
```python
# Key design: graded penalty, not binary
# 0 cost at DETECT_AGL (300 m), full W_DETECT cost at MAX_AGL (600 m)
detect_excess = max(0, (agl - DETECT_AGL)) / (MAX_AGL - DETECT_AGL)
detection_risk = sum(detect_excess.mean(axis=1) * seg_time * W_DETECT)  # W_DETECT = 3.0
```

> **Critical design choice — graded vs binary detection penalty:**
> An earlier version used a flat binary penalty (detected = W_DETECT, not detected = 0).
> With W_DETECT small, the optimizer treated "fly high through storms" as roughly
> equivalent to "fly low around storms". Making the penalty **graded and strong (W_DETECT=3.0)**
> means every extra metre above 300 m continuously increases risk — the optimizer strongly
> prefers to stay below 300 m unless fuel weight is dominant.

---

### f₃ — Fuel Burn (kg)

```python
def fuel_rate(agl):
    # Thin air at high altitude = lower drag = lower fuel burn
    # Dense air at low altitude = higher drag = higher fuel burn
    low_penalty = max(0, (FUEL_OPT_AGL - agl) / FUEL_OPT_AGL) * FUEL_LOW_K
    return FUEL_BASE + low_penalty
    # At 500 m AGL: 1.0 kg/km (minimum drag)
    # At   0 m AGL: 1.5 kg/km (50% more drag in dense air)

f3 = sum(seg2d_km * fuel_rate(mean_agl_per_segment))
```

**FUEL_OPT_AGL = 500 m** — the altitude at which drag is minimised.

This creates the **core tension** in the problem:
- Fuel wants altitude ≥ 500 m (thin air = low drag)
- Stealth wants altitude ≤ 300 m (below radar detection)
- The 200 m gap between these two optima makes f₂ and f₃ genuinely conflicting

```
AGL (m)
 600  ─── planning ceiling
 500  ─── FUEL_OPT_AGL ← minimum fuel cost here
 300  ─── DETECT_AGL   ← radar detection starts here (f₂ penalty begins)
  61  ─── SAFE_AGL     ← terrain collision risk below here
   0  ─── ground
```

---

## 5. Why Classical Solvers Won't Work

| Property | Implication |
|----------|-------------|
| **Non-convex feasible region** | Storm circles create holes in the feasible space; routing north vs south around a storm is a topology change, not a smooth perturbation — gradient methods get trapped |
| **Non-differentiable objectives** | Storm intersection uses `(distance < radius)` — a step function; terrain lookup is a spline — no analytical gradient exists |
| **Three conflicting objectives** | No single optimum exists; need the full Pareto front, not one answer |
| **24 continuous variables** | Search space is huge; exhaustive search is impossible |
| **Non-linear Pareto boundary** | The trade-off frontier between risk and fuel is curved (confirmed visually); MILP or weighted-sum linear methods cannot recover the full non-convex frontier in one solve |

---

## 6. NSGA-II — The Genetic Algorithm

NSGA-II (Non-dominated Sorting Genetic Algorithm II, Deb et al. 2002) evolves a
**population of 150 complete paths** simultaneously over 150 generations.

### 6.1 Initialisation — Warm-Start Sampling

```python
class RouteSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        for i in range(n_samples):       # 150 individuals
            for j in range(N_WP):        # 8 waypoints each
                t   = (j+1) / (N_WP+1)  # fractional position along route
                mid = SOURCE[:2] + t * (DEST[:2] - SOURCE[:2])  # point on straight line
                x   = mid[0] + np.random.normal(0, 18)  # lateral scatter ±18 km
                y   = mid[1] + np.random.normal(0, 20)
                agl = np.random.uniform(MIN_AGL, MAX_AGL)  # full altitude range: 30–600 m
```

Rather than purely random initialisation in 24-dimensional space (which produces mostly
nonsensical paths), all 150 initial paths are **seeded near the straight-line route**
with Gaussian scatter. This means Generation 0 already contains navigable paths; the
algorithm refines rather than discovering basic connectivity from scratch.

**Why sample the full AGL range (30–600 m)?**
An earlier version sampled only the stealth corridor (61–300 m). This meant the initial
population contained no high-altitude solutions, so the Pareto front had no fuel-efficient
corner — the fuel slider appeared to do nothing. Sampling the full range ensures the
initial population explores both low-level (stealthy) and high-altitude (fuel-efficient)
flight profiles.

---

### 6.2 Fitness Evaluation

```python
def _evaluate(self, X, out, *args, **kwargs):
    F = np.zeros((X.shape[0], 3))
    for i in range(X.shape[0]):              # 150 individuals per generation
        path = build_path(X[i].reshape(N_WP, 3))
        F[i] = evaluate_path(path)           # returns (time_s, risk, fuel_kg)
    out["F"] = F                             # shape: (150, 3)
```

This is the computational bottleneck: **150 evaluations × 150 generations = 22,500 total
path evaluations**. Each evaluation integrates 20 sample points across 9 segments,
performs 3 storm-circle intersection tests, and computes terrain-dependent fuel rates.
Total runtime: ~2–3 minutes on a single CPU core.

---

### 6.3 Non-Dominated Sorting — The Pareto Ranking

This is what distinguishes NSGA-II from a basic genetic algorithm. Instead of a single
fitness score, each individual gets a **Pareto rank**.

**Dominance rule:** Solution A dominates B if:
- A ≤ B on **all** three objectives, AND
- A < B on **at least one** objective

```
Example population:
             Time(min) Risk   Fuel(kg)
    Sol A:   29.3       172   181.8    ← Front 0: non-dominated
    Sol B:   28.2      1785   160.5    ← Front 0: non-dominated
    Sol C:   28.1      4751   142.0    ← Front 0: non-dominated (best time+fuel)
    Sol D:   29.5      2000   170.0    ← Front 1: B dominates D on all 3
    Sol E:   30.0      2500   175.0    ← Front 2: dominated by A, B, C, D

Front 0 = Pareto-optimal set: {A, B, C}
```

The entire population is ranked this way. By generation 46 in a typical run, all 150
solutions reach Front 0 — the full population is non-dominated.

```
Convergence trace (from run output):
Gen  1: n_nds =   4   (only 4 non-dominated solutions)
Gen 10: n_nds =  18
Gen 20: n_nds =  24
Gen 46: n_nds = 150   ← entire population becomes Pareto-optimal
Gen 150: n_nds = 150  ← front spreads and refines, no longer grows
```

---

### 6.4 Crowding Distance — Diversity Preservation

Within the same Pareto front, NSGA-II prefers solutions that are more **isolated** in
objective space. This prevents all 150 solutions clustering around one region and
ensures even coverage of the front.

```
Pareto front (Time vs Risk):

Risk ↑
4751  ●                  ← boundary: infinite crowding distance
      |    sparse region → large crowding → preferred
2000      ● ● ●          ← dense cluster → small crowding → deprioritised
1785            ●
      |
 172                ●    ← boundary: infinite crowding distance
      └────────────────── Time (min) →
             28.1  28.2  29.3  29.5
```

**In the visualisation**, crowding distance = **point size** in the pairwise scatter plots.
Larger dots represent more unique, isolated trade-off positions on the Pareto front.

---

### 6.5 Parent Selection — Binary Tournament

To fill the mating pool, NSGA-II runs **binary tournament selection** 150 times:

```
Pick 2 random individuals → compare → winner enters mating pool

Comparison rule:
  1. Lower Pareto rank wins   (front 0 beats front 1)
  2. Tie on rank → higher crowding distance wins   (diversity favoured)
```

This ensures front-0 solutions dominate reproduction while maintaining diversity
by favouring isolated (unique) solutions over densely-clustered ones.

---

### 6.6 Crossover — Simulated Binary Crossover (SBX)

```python
crossover = SBX(prob=0.9, eta=15)
```

SBX takes two parent waypoint vectors and blends them into two offspring:

```
Parent 1:  [x=45, y=52, agl=280]   ← goes north of Storm-Alpha
Parent 2:  [x=45, y=28, agl=410]   ← goes south at higher altitude

            SBX blend (eta=15)

Child 1:   [x=45, y=49, agl=305]   ← slightly north, moderate altitude
Child 2:   [x=45, y=31, agl=385]   ← slightly south, higher altitude
```

**eta=15** controls the spread of the offspring distribution:
- **High eta (15–20)**: children cluster near parents → **exploitation** of known-good regions
- **Low eta (2–5)**: children can jump far → more exploration

With eta=15 the crossover exploits rather than explores — appropriate because by
generation 30+ the population already contains good paths. The geometric intuition:
if Parent 1 avoids Storm-Alpha by going north and Parent 2 avoids Storm-Bravo by flying
high, SBX may produce a child that does both — inheriting storm-avoidance strategies
from each parent.

**prob=0.9**: 90% of parent pairs undergo crossover; 10% are passed through unchanged.

---

### 6.7 Mutation — Polynomial Mutation (PM)

```python
mutation = PM(eta=20)
```

After crossover, each variable is independently mutated with a small probability:

```
Before mutation:  agl = 280 m
After mutation:   agl = 293 m    ← small perturbation (eta=20 = tight distribution)
                  (occasionally)  agl = 450 m    ← rare large jump
```

**eta=20** gives a tight polynomial distribution — most mutations are small perturbations
(±10–30 m AGL, ±2–5 km lateral) with rare large jumps.

**Why mutate?** Crossover can only recombine genetic material that already exists in the
population. If every individual in generation 30 avoids Storm-Bravo from the south,
crossover will never discover the northern detour. Mutation occasionally kicks a waypoint
far north, potentially discovering a better topology that crossover alone could never find.

---

### 6.8 Survivor Selection — (μ + λ) Strategy

```
Parents:   150 individuals  (generation N)
Offspring: 150 individuals  (after crossover + mutation)
Combined:  300 individuals
              ↓
    Non-dominated sort → fronts 0, 1, 2, ...
    Fill new population:
      1. Add all of front 0 (if ≤ 150 individuals)
      2. If still room, add front 1, front 2, ...
      3. Last front that doesn't fully fit:
         sort by crowding distance → take highest-crowding solutions first
              ↓
    New population: 150 individuals  (generation N+1)
```

This ensures the population **never degrades** — best 150 out of 300 always survive,
with diversity preservation built into the tiebreaker.

---

### 6.9 Determinism — The Seeding Fix

```python
def _run_nsga2(pop=150, ngen=150, seed=42):
    np.random.seed(seed)          # ← explicit numpy seed (critical)
    result = minimize(
        ...,
        seed=seed,                # pymoo internal seed
        verbose=False,
    )
```

**pymoo 0.6 does NOT seed numpy's global RNG** when you pass `seed=` to `minimize()`.
Our custom `RouteSampling` class uses `np.random.normal` and `np.random.uniform` directly,
which draw from numpy's global RNG. Without the explicit `np.random.seed(seed)`, every
Re-optimize call with identical storm positions produced a different Pareto front.

With both seeds set: **same storm positions + same seed → identical Pareto front every time**.

---

### 6.10 Full Generation Loop

```
┌─────────────────────────────────────────────────────────────────┐
│  RouteSampling: 150 warm-started paths near direct route        │  Gen 0
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  evaluate_path() × 150: compute (time, risk, fuel) for each     │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  Non-dominated sort → Pareto ranks 0, 1, 2, ...                 │
│  Crowding distance within each rank                             │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  Binary tournament selection → 150 parents                      │
│  SBX crossover (prob=0.9, eta=15) → 150 offspring               │
│  PM mutation (eta=20)                                           │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  Combine 300 → sort → keep best 150  (μ + λ selection)          │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                    Repeat × 150 generations
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│  Output: 150 Pareto-optimal paths                               │
│          F shape: (150, 3)  — objective values                  │
│          X shape: (150, 24) — waypoint vectors                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Key Implementation Decisions

### 7.1 Why the Pareto Boundary is Non-Linear

The Pareto front in this problem is curved, not a straight line, for three reasons:

1. **Storm geometry**: routing around circles is inherently non-linear. The detour cost
   (extra time + fuel) grows non-linearly with storm radius and offset.

2. **Piecewise altitude physics**: the fuel penalty is linear in AGL below 500 m but flat
   above; the radar detection penalty is linear above 300 m but zero below. These
   piecewise functions make the Risk–Fuel trade-off curved.

3. **Interaction effects**: going high to save fuel simultaneously increases detection
   risk AND reduces storm exposure (because a high path might fly over a storm rather
   than around it). This coupling makes the frontier non-convex.

A weighted-sum single-objective approach with a fixed weight vector can only find points
on the **convex hull** of the Pareto front — it misses any concave regions. NSGA-II finds
the full front including concave segments.

---

### 7.2 ASF Normalisation — Why the Sliders Would Have Been Useless

```python
def _best_idx(F, w):
    w = np.maximum(w, 1e-6)
    w = w / w.sum()

    # Without this: time (~1700 s) >> fuel (~160 kg) >> risk scales differently
    # max(F_raw / w) always dominated by time regardless of slider positions
    F_min   = F.min(axis=0)
    F_range = np.maximum(F.max(axis=0) - F_min, 1e-9)
    F_norm  = (F - F_min) / F_range          # each objective scaled to [0, 1]

    return int(ASF().do(F_norm, 1.0 / w).argmin())
```

Without normalisation, mission time (~1700 s) is ~10× larger than fuel (~160 kg).
The ASF `max(F_i / w_i)` is always dominated by time regardless of slider weights —
setting fuel=0.9, risk=0.05, time=0.05 would still select based primarily on time.

Normalising to [0, 1] per objective makes all three commensurable, so the weights
directly control which trade-off region is selected.

---

### 7.3 The Risk Weight Ambiguity — Graded vs Binary Detection

The most important design decision in the objective function:

| Approach | Behaviour |
|----------|-----------|
| Binary detection: `agl > 300 m → W_DETECT penalty` | Optimizer indifferent between "fly high, skip storms" and "fly low, hit storms" when W_DETECT is small |
| **Graded detection: penalty scales linearly 0→W_DETECT between 300 m and 600 m** | Every extra metre above 300 m continuously increases cost; strong W_DETECT=3.0 makes detection much worse than weather exposure |

With the binary approach and Risk=1.0 slider, the optimizer still chose high-altitude
paths (flying at 500 m AGL through storms) because the flat detection penalty ~= storm
penalty. The graded penalty fixes this: at Risk=1.0, the optimizer firmly stays below
300 m and routes around storms.

---

## 8. Selecting One Solution — The Sliders

NSGA-II returns 150 non-dominated solutions. A single solution is selected by the
**Achievement Scalarising Function (ASF)**:

```
ASF score = max_i( F_norm[i] / w[i] )

Geometrically: the point on the Pareto front closest to the "ideal point"
               scaled by the weight vector
```

This is equivalent to finding the solution that best satisfies the preference expressed
by the weights. Unlike a simple weighted sum (which can only find convex-hull solutions),
ASF can select solutions in concave regions of the Pareto front.

### How the Three Sliders Map to Behaviour

| Slider setting | Physics | Path shape | AGL range |
|----------------|---------|------------|-----------|
| Risk=1.0, others≈0 | Minimise all exposure: storms + radar | Large detour north/south around all storms | 80–300 m (stays in stealth corridor) |
| Fuel=1.0, others≈0 | Minimise drag = fly high | Near-straight path, may enter storms | 400–600 m (above radar threshold) |
| Time=1.0, others≈0 | Minimise distance = go direct | Direct path, accepts risk | Moderate altitude |
| Equal (0.33 each) | Balanced compromise | Partial detour, mixed altitude | 80–573 m |

---

## 9. Visualisation Guide

The interactive figure (`viz_sliders.py` / `app.py`) has five panels:

### Panel 1 — Top-Down Map (2D)

Bird's-eye view of the 150×80 km airspace. Shows:
- All 150 Pareto paths as faint background lines (the full solution space)
- Active selected path in bright cyan
- 8 waypoints as cyan squares on the active path
- Storm circles (draggable in the desktop version)
- Terrain risk heatmap (yellow = high risk, dark = safe)

### Panel 2 — Altitude AGL Profile

Height above ground from takeoff to landing:
- **Red dashed line at 61 m** — terrain collision floor (f₂ penalty below)
- **Yellow dashed line at 300 m** — radar detection threshold (f₂ penalty above)
- **Green band 61–300 m** — ideal stealth corridor: safe AND undetectable
- **Cyan line** — selected path's actual AGL profile

### Panel 3 — Pairwise Pareto Scatter (3 plots)

Three 2D projections of the 3D Pareto front:

| Plot | X-axis | Y-axis | Colour |
|------|--------|--------|--------|
| Left | Time (min) | Risk | Fuel (kg) |
| Middle | Risk | Fuel (kg) | Time (min) |
| Right | Time (min) | Fuel (kg) | Risk |

- **Point size** = crowding distance (larger = more unique/isolated solution)
- **▲ red** = solution with best (lowest) mission time
- **◆ green** = solution with best (lowest) survival risk
- **■ yellow** = solution with best (lowest) fuel burn
- **★ cyan** = currently active solution (moves with sliders)
- **Cyan crosshairs** = lock to ★, show exact axis values
- **Callout box** = active solution's value on the third (colour) objective

> A point may look dominated in one 2D panel but be optimal overall — it is better
> on the third objective shown as colour. The crosshair callout makes this explicit.

### Panel 4 — Parallel Coordinates

All 3 objectives on a single plot. Each vertical axis is one objective, normalised so
0 = best value, 1 = worst value in the current Pareto front:

- **Faint gray lines** — all 150 Pareto solutions
- **Bright cyan line with glow** — active solution
- **Callout numbers on each axis** — actual values at the active solution's crossing
- **Where lines cross** — the crossing point reveals the trade-off: if Risk and Fuel
  lines cross sharply between two solutions, reducing one increases the other

### Sliders

| Slider | Controls |
|--------|----------|
| ⏱ TIME weight | Importance of short mission time |
| 🛡 RISK weight | Importance of low weather exposure + staying below radar |
| ⛽ FUEL weight | Importance of low fuel consumption |

Weights are relative — only ratios matter. The status bar shows normalised weights
`[T:xx R:xx F:xx]` alongside the active solution's full metric values.

---

## 10. File Reference

| File | Purpose |
|------|---------|
| `weather_path_planning.py` | Core module: terrain model, objective functions, NSGA-II problem definition, interactive matplotlib figure |
| `viz_sliders.py` | Desktop launcher: loads `path_results.npz`, calls `launch_interactive()` |
| `app.py` | Dash web application: full browser-based interactive viz with Plotly |
| `path_results.npz` | Pre-computed Pareto front (F: 150×3, X: 150×24) from `_run_nsga2(pop=150, ngen=150)` |
| `requirements.txt` | Python dependencies for both desktop and web deployment |
| `Procfile` | Railway/Heroku deployment: `web: gunicorn app:server` |

### Key Parameters (all in `weather_path_planning.py`)

```python
N_WP          = 8        # intermediate waypoints per path
POP           = 150      # NSGA-II population size
NGEN          = 150      # number of generations
N_SAMP        = 20       # integration samples per path segment

SAFE_AGL      = 61.0     # m  — terrain collision threshold
DETECT_AGL    = 300.0    # m  — radar detection threshold
FUEL_OPT_AGL  = 500.0    # m  — minimum drag altitude

W_WEATHER     = 1.0      # risk weight: time in storm
W_TERRAIN_HIT = 0.8      # risk weight: below 61 m AGL
W_DETECT      = 3.0      # risk weight: above 300 m AGL (graded)
W_UNDERGROUND = 20.0     # risk weight: below terrain (catastrophic)

SPEED         = 0.083    # km/s ≈ 300 km/h
FUEL_BASE     = 1.0      # kg/km at fuel-optimal altitude
FUEL_LOW_K    = 0.50     # extra drag fraction at sea level
```

---

## References

- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). **A fast and elitist
  multiobjective genetic algorithm: NSGA-II**. *IEEE Transactions on Evolutionary
  Computation*, 6(2), 182–197.
- Blank, J., & Deb, K. (2020). **pymoo: Multi-Objective Optimization in Python**.
  *IEEE Access*, 9, 89497–89509.
- Deb, K., & Jain, H. (2014). **An Evolutionary Many-Objective Optimization Algorithm
  Using Reference-Point Based Non-Dominated Sorting Approach**. *IEEE TEC*, 18(4).
