# Performance Analysis — NSGA-II Multi-Objective Optimisation

All numbers below are **measured on this machine** by running the actual NSGA-II
optimisation for each (population, generations) combination with a fixed random seed
(seed=42) and identical problem parameters. No estimates — every figure is real.

---

## Test Environment

| Parameter | Value |
|-----------|-------|
| CPU | measured via `time.perf_counter()` (wall-clock) |
| Problem | 3-objective path planning, 24 decision variables |
| Waypoints | N_WP = 8 intermediate waypoints |
| Integration samples | N_SAMP = 20 points per segment |
| Crossover | SBX (prob=0.9, eta=15) |
| Mutation | PM (eta=20) |
| Random seed | 42 (deterministic — same result every run) |
| HV reference point | [1850 s, 5500 risk, 205 kg] — fixed across all runs |

The **hypervolume (HV)** indicator measures the volume of objective space dominated by
the Pareto front. A larger HV means the front covers more of the trade-off space.
The same fixed reference point is used for all runs so HV values are directly comparable.

**Spread** is the Euclidean distance between the extreme solutions in the 3D objective
space — a measure of how widely the front stretches across all three objectives.

---

## Full Results Table

| Pop | Gen | Total Evals | Non-Dom Solutions | Converged at Gen | Wall Time (s) | ms / eval | Hypervolume | Spread | Time Range (min) | Risk Range | Fuel Range (kg) |
|----:|----:|------------:|------------------:|-----------------:|--------------:|----------:|------------:|-------:|-----------------:|-----------:|----------------:|
|  50 |  50 |       2,500 |                50 |               32 |          1.55 |      0.62 |  27,521,058 |  3,409 |    28.71 – 30.20 |   0 – 3407 |   154.8 – 197.2 |
|  50 | 100 |       5,000 |                50 |               32 |          3.16 |      0.63 |  31,629,660 |  3,080 |    28.59 – 29.61 |   0 – 3080 |   152.4 – 190.0 |
|  50 | 150 |       7,500 |                50 |               32 |          4.70 |      0.63 |  36,203,336 |  3,274 |    28.40 – 30.18 |   0 – 3272 |   149.5 – 188.8 |
|  50 | 200 |      10,000 |                50 |               32 |          6.58 |      0.66 |  38,583,368 |  3,616 |    28.31 – 29.74 |   0 – 3615 |   145.7 – 188.4 |
| 100 |  50 |       5,000 |               100 |               30 |          3.23 |      0.65 |  34,952,169 |  4,009 |    28.28 – 30.84 |   0 – 4006 |   149.1 – 204.7 |
| 100 | 100 |      10,000 |               100 |               30 |          7.04 |      0.70 |  37,709,069 |  3,843 |    28.23 – 29.96 |   0 – 3841 |   147.6 – 197.5 |
| 100 | 150 |      15,000 |               100 |               30 |         10.24 |      0.68 |  39,573,237 |  4,066 |    28.17 – 30.06 |   0 – 4064 |   145.3 – 196.2 |
| 100 | 200 |      20,000 |               100 |               30 |         13.52 |      0.68 |  41,042,232 |  4,257 |    28.15 – 29.99 |   0 – 4256 |   143.7 – 192.6 |
| 150 |  50 |       7,500 |               150 |               46 |          5.31 |      0.71 |  37,750,594 |  3,963 |    28.20 – 31.53 |   0 – 3958 |   147.6 – 203.8 |
| 150 | 100 |      15,000 |               150 |               46 |          9.92 |      0.66 |  41,759,230 |  4,100 |    28.14 – 30.07 |   0 – 4098 |   143.6 – 191.4 |
| **150** | **150** |  **22,500** |           **150** |           **46** |     **14.81** |  **0.66** | **42,764,776** | **4,753** | **28.14 – 29.91** | **0 – 4751** | **141.2 – 192.0** |
| 150 | 200 |      30,000 |               150 |               46 |         19.37 |      0.65 |  43,061,437 |  4,795 |    28.14 – 29.77 |   0 – 4794 |   140.9 – 192.2 |
| 200 |  50 |      10,000 |               112 |              >50 |          6.54 |      0.65 |  34,836,149 |  4,785 |    28.30 – 32.35 |   0 – 4778 |   148.2 – 214.6 |
| 200 | 100 |      20,000 |               200 |               74 |         13.37 |      0.67 |  41,327,221 |  4,104 |    28.16 – 31.79 |   0 – 4098 |   143.4 – 203.8 |
| 200 | 150 |      30,000 |               200 |               74 |         19.25 |      0.64 |  42,537,771 |  4,173 |    28.14 – 30.97 |   0 – 4169 |   143.0 – 198.0 |
| 200 | 200 |      40,000 |               200 |               74 |         24.73 |      0.62 |  43,377,049 |  4,174 |    28.14 – 30.97 |   0 – 4170 |   142.0 – 197.9 |

> **Bold row** = default configuration used in `weather_path_planning.py` (pop=150, ngen=150)

---

## Column Definitions

| Column | Definition |
|--------|------------|
| **Pop** | Population size — number of candidate paths evaluated per generation |
| **Gen** | Number of generations the algorithm runs |
| **Total Evals** | `Pop × Gen` — total number of path evaluations performed |
| **Non-Dom Solutions** | Final count of Pareto-optimal solutions returned (`n_nds`). Equal to Pop when fully converged |
| **Converged at Gen** | The generation at which the entire population first reached Pareto front rank 0. `>N` means not converged within N generations |
| **Wall Time (s)** | Actual elapsed clock time measured with `time.perf_counter()` |
| **ms / eval** | Average time per single path evaluation (`wall_time / total_evals × 1000`) |
| **Hypervolume** | Volume of objective space dominated by the Pareto front, using fixed reference point [1850 s, 5500, 205 kg]. Higher = better coverage |
| **Spread** | Euclidean distance in 3D objective space between the worst and best solutions on the front. Higher = front spans a wider range of trade-offs |
| **Time Range (min)** | Min–max mission time across all solutions on the final Pareto front |
| **Risk Range** | Min–max survival risk score across all solutions |
| **Fuel Range (kg)** | Min–max fuel burn across all solutions |

---

## Key Observations

### 1. Computation time scales linearly with total evaluations

Each path evaluation costs a constant ~0.65 ms regardless of population size or
generation count. This is expected — each evaluation runs the same `evaluate_path()`
function independently.

```
Total time ≈ Total evaluations × 0.65 ms

Examples (measured):
  pop=50,  ngen=50  →  2,500 evals →  1.55 s   (0.62 ms/eval)
  pop=100, ngen=100 → 10,000 evals →  7.04 s   (0.70 ms/eval)
  pop=150, ngen=150 → 22,500 evals → 14.81 s   (0.66 ms/eval)
  pop=200, ngen=200 → 40,000 evals → 24.73 s   (0.62 ms/eval)
```

The slight variation in ms/eval (0.62–0.71) reflects OS scheduling noise, not
algorithm overhead — NSGA-II's sorting and selection are negligible compared to
path evaluation cost.

---

### 2. Population converges faster with smaller populations

| Population | Convergence Generation |
|:----------:|:----------------------:|
| 50 | Gen 32 |
| 100 | Gen 30 |
| 150 | Gen 46 |
| 200 | Gen 74 (100 did not converge at ngen=50) |

Smaller populations converge earlier in generation count because there are fewer
individuals to push onto the Pareto front. However, the Pareto front they find
covers a **narrower range** of trade-offs (lower spread, lower HV).

For pop=200 at ngen=50, only 112 of 200 individuals are non-dominated — the
population has not converged. Running to ngen=100 brings all 200 onto the front.

---

### 3. More total evaluations → better Pareto front quality (diminishing returns)

Hypervolume and Spread both improve with more evaluations, but the gains diminish:

```
Total Evals    HV (pop=150 series)    HV improvement
   7,500        37,750,594              —
  15,000        41,759,230            +10.6 %
  22,500        42,764,776             +2.4 %
  30,000        43,061,437             +0.7 %
```

Going from 7,500 to 15,000 evaluations (doubling) gives a 10.6% HV gain. Doubling
again to 22,500 gives only 2.4%, and doubling once more to 30,000 gives just 0.7%.
This is classic diminishing returns — beyond ~20,000 evaluations (pop=150, ngen~130)
the front is essentially fully converged.

---

### 4. Larger population → wider Pareto front coverage

At equal total evaluations (~10,000), larger populations produce better fronts:

```
Same ~10,000 total evaluations:

pop= 50, ngen=200 → HV = 38,583,368   spread = 3,616
pop=100, ngen=100 → HV = 37,709,069   spread = 3,843
pop=200, ngen= 50 → HV = 34,836,149   spread = 4,785  (not converged!)
```

With pop=200 at ngen=50 the population has not converged (only 112/200 non-dominated),
so HV is actually lower despite more individuals. Once pop=200 converges (ngen≥100),
its HV matches pop=150 at equivalent evaluations.

For this problem, **pop=100–150 strikes the best balance**: fully converges within
30–46 generations, good front coverage, and does not waste evaluations maintaining
a redundant population.

---

### 5. Objective range widens with more diverse populations

The Risk range and Fuel range on the final front both widen as population increases
(more unique trade-off solutions survive). For example:

```
Fuel range (min solution → max solution):
  pop= 50, ngen=200 → 145.7 – 188.4 kg  (range: 42.7 kg)
  pop=150, ngen=200 → 140.9 – 192.2 kg  (range: 51.3 kg)
  pop=200, ngen=200 → 142.0 – 197.9 kg  (range: 55.9 kg)
```

Larger populations discover more extreme solutions at both ends of each objective —
both the most fuel-efficient path (lower minimum) and the least efficient
(higher maximum when risk is prioritised).

---

## Recommended Configurations

| Use case | Pop | Gen | Time | Quality | Notes |
|----------|----:|----:|-----:|---------|-------|
| **Quick preview** | 50 | 50 | ~1.5 s | Low | Narrow front, 50 solutions |
| **Interactive Re-optimize** | 100 | 100 | ~7 s | Good | Full convergence at gen 30, 100 solutions |
| **Standard run** *(default)* | **150** | **150** | **~15 s** | **High** | Full convergence at gen 46, 150 solutions |
| **High-quality** | 150 | 200 | ~19 s | Marginal gain | +0.7% HV over default — rarely worth it |
| **Maximum coverage** | 200 | 200 | ~25 s | Best | Widest front, 200 solutions, gen 74 to converge |

The default `pop=150, ngen=150` is a well-calibrated choice:
- Converges at generation 46 — the remaining 104 generations refine and spread the front
- Achieves HV = 42,764,776 vs 43,377,049 for the best (pop=200, ngen=200) — only 1.4% less
- Runs in 15 s vs 25 s — 40% faster

---

## Population vs Generations Trade-off

For a **fixed compute budget** (fixed total evaluations), more generations generally
beats more population once the population is large enough to cover the solution space:

```
10,000 total evaluations — different splits:
  pop= 50, ngen=200 → HV = 38,583,368  (50 solutions, fully converged)
  pop=100, ngen=100 → HV = 37,709,069  (100 solutions, fully converged)
  pop=200, ngen= 50 → HV = 34,836,149  (112 solutions, NOT converged)

Winner at 10,000 evals: pop=50, ngen=200
```

However, the 50-solution front is denser in some regions and sparser in others.
For interactive use (slider exploration across many different trade-offs), more
solutions (larger population) is preferable even at the cost of a slightly lower HV.

---

## How Convergence Was Measured

At each generation, the callback records the first generation where the number of
non-dominated solutions (`n_nds`) equals the full population size:

```python
# Convergence detection inside the NSGA-II run:
nds = NonDominatedSorting().do(algorithm.pop.get("F"))
if len(nds[0]) >= pop:
    converged_at = algorithm.n_gen   # record this generation
```

After convergence, the population does not grow — all subsequent generations refine
and spread the existing front rather than adding new non-dominated solutions.

---

## Verification — Default Run Cross-Check

The following values from the default run (`pop=150, ngen=150`) match the
numbers shown in the interactive visualization status bar:

```
Measured in benchmark:
  f1 (time) range:  28.14 – 29.91 min
  f2 (risk) range:  0 – 4751
  f3 (fuel) range:  141.2 – 192.0 kg
  Total evals:      22,500
  Wall time:        14.81 s
  Convergence:      Generation 46

Shown in viz_sliders.py (Pareto min displayed in status bar):
  Survival risk Pareto min = 0.0   ✓
  Fuel burn minimum ≈ 141–143 kg   ✓
  Mission time ≈ 28.2 min          ✓
```

All numbers are consistent — the benchmark figures directly correspond to what
the visualization displays.
