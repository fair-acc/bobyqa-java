# BOBYQA Java Port Design Spec

**Date:** 2026-03-16
**Status:** Revised (post-review)
**Scope:** Standalone Java 17 implementation of enhanced BOBYQA with noise handling and restart mechanisms, ported from pybobyqa (Numerical Algorithms Group)

---

## Background

The current Java BOBYQA implementation (Apache Commons Math3 `BOBYQAOptimizer`) is a faithful port of Powell's original 2009 algorithm with three configuration parameters and no noise awareness. It is inadequate for particle accelerator optimization where:

- Objective evaluations are expensive (machine time)
- Stochastic noise is present (beam jitter, measurement uncertainty)
- Multiplicative non-stochastic noise contributions exist (calibration drift)
- Escaping local minima matters (accelerator tuning landscapes are multimodal)

The Python [pybobyqa](https://github.com/numericalalgorithmsgroup/pybobyqa) library (Numerical Algorithms Group) extends the original algorithm with noise handling, restart mechanisms, geometry repair, and a rich parameter system. This spec defines a direct algorithmic port of pybobyqa to standalone Java 17.

---

## Goals

1. Standalone Java 17 library — no Apache Commons Math dependency
2. Noise-aware optimization: stochastic averaging + multiplicative noise modeling + noise-level termination
3. Restart mechanisms: soft and hard restarts with auto-detection of stagnation
4. Modular architecture mirroring pybobyqa's module structure
5. Single runtime dependency: EJML (Apache 2.0) for matrix operations
6. Clean public API decoupled from internal matrix types
7. Test suite ported from pybobyqa benchmarks for algorithmic correctness validation

---

## Architecture

### Module Structure

```
src/main/java/bobyqa/
├── Bobyqa.java              # Public entry point
├── Solver.java              # Main optimization loop
├── Controller.java          # State management, restart logic, geometry repair
├── Model.java               # Quadratic surrogate model, interpolation points
├── TrustRegion.java         # Trust region subproblem solvers (trsbox)
├── Params.java              # All configuration options with defaults (builder pattern)
├── DiagnosticInfo.java      # Per-iteration diagnostic record
├── OptimResult.java         # Return type: best point, value, exit flag
├── ExitFlag.java            # Enum of all termination conditions
├── GoalType.java            # Enum: MINIMIZE, MAXIMIZE
└── util/
    └── MathUtil.java        # EJML-based linear algebra helpers (internal only)
```

### Dependency

| Library | Version | License | Purpose |
|---------|---------|---------|---------|
| EJML | 0.43+ | Apache 2.0 | Dense matrix operations (DMatrixRMaj) |

EJML is never exposed in the public API. All public method signatures use `double[]` and `double[][]`.

---

## Public API

### Entry Point

```java
OptimResult result = Bobyqa.solve(
    x -> myObjectiveFunction(x),   // ToDoubleFunction<double[]>
    x0,                            // double[] initial point
    lower,                         // double[] lower bounds
    upper,                         // double[] upper bounds
    maxEvaluations,                // int total function evaluation budget
    GoalType.MINIMIZE,             // GoalType.MINIMIZE or GoalType.MAXIMIZE
    params                         // Params (optional)
);
```

**Minimization and maximization** are both supported via the `GoalType` enum. When `GoalType.MAXIMIZE` is specified, `Bobyqa` negates the objective function internally before passing it to `Solver` — the core algorithm always minimizes. The value returned in `result.f()` is always the **original (un-negated) objective value** at the best point found, regardless of goal type. `model.absTol` is interpreted in terms of the original objective: for maximization, the algorithm exits with `ABS_TOL_REACHED` if `f(x) > absTol`.

### Result Type

```java
result.x()           // double[]        best point found
result.f()           // double          best objective value
result.nEvals()      // int             total function evaluations used
result.exitFlag()    // ExitFlag        termination reason
result.msg()         // String          human-readable exit message
result.diagnostics() // List<DiagnosticInfo>  per-iteration log (if enabled)
```

**`DiagnosticInfo` record fields** (one record per iteration, only populated when `logging.saveDiagnosticInfo=true`):

| Field | Type | Description |
|-------|------|-------------|
| `iter` | `int` | Iteration number (0-based) |
| `iterType` | `IterType` enum | `SUCCESSFUL`, `UNSUCCESSFUL`, `SAFETY`, `GEOMETRY_REPAIR`, `RESTART_SOFT`, `RESTART_HARD` |
| `fBest` | `double` | Best objective value at end of iteration |
| `delta` | `double` | Trust region radius at end of iteration |
| `rho` | `double` | Minimum trust region radius at end of iteration |
| `ratio` | `double` | Actual-to-predicted reduction ratio (NaN for non-step iterations) |
| `nsamples` | `int` | Number of samples used for objective evaluation this iteration |
| `restartCount` | `int` | Cumulative number of restarts triggered so far |
| `poisedness` | `Double` | Poisedness constant (null if `logging.savePoisedness=false`) |
| `xk` | `double[]` | Current iterate (null if `logging.saveXk=false`) |

`logging.savePoisedness` is independent of `logging.saveDiagnosticInfo` — poisedness is always computed when `savePoisedness=true` regardless of whether the full diagnostic record is saved, since it drives geometry repair decisions internally. Setting `savePoisedness=false` saves the Lagrange polynomial computation cost at the expense of less informative diagnostics.

### Exit Flags

```java
enum ExitFlag {
    CONVERGED,               // rho reduced to rhoend
    BUDGET_EXHAUSTED,        // maxEvaluations reached
    NOISE_LEVEL_REACHED,     // all points within noise band of best
    ABS_TOL_REACHED,         // f < params.model.absTol
    SLOW_PROGRESS,           // slow iterations exceeded limit
    INPUT_ERROR              // invalid parameters
}
```

### Configuration (Builder Pattern)

```java
Params params = Params.defaults()
    .noise(n -> n
        .objfunHasNoise(true)
        .multiplicativeNoiseLevel(0.01)   // 1% relative noise
        .additiveNoiseLevel(1e-6)
        .quitOnNoiseLevel(true)
        .scaleFactorForQuit(1.0))
    .restarts(r -> r
        .useRestarts(true)
        .maxUnsuccessfulRestarts(10)
        .maxUnsuccessfulRestartsTotal(20)
        .useSoftRestarts(true)
        .autoDetect(true))
    .trRadius(tr -> tr
        .eta1(0.1).eta2(0.7)
        .gammaDec(0.98).gammaInc(2.0)
        .gammaIncOverline(4.0))
    .slow(s -> s
        .historyForSlow(5)
        .threshForSlow(1e-8))
    .logging(l -> l
        .saveDiagnosticInfo(true)
        .saveXk(false));
```

---

## Core Algorithm (`Model.java`, `TrustRegion.java`, `Controller.java`)

### Model (`Model.java`)

Owns the quadratic surrogate model:

- Stores `npt` interpolation points relative to shifting base `xbase` to avoid numerical drift. `npt` is user-configurable via `model.npt` (default: `2n+1`); valid range is `[n+2, (n+1)(n+2)/2]`. Validation is performed in `Params` at construction time — an `INPUT_ERROR` exit is returned if the range is violated. The value of `npt` determines which interpolation mode is active:
  - `npt == (n+1)(n+2)/2`: fully quadratic model
  - `n+2 ≤ npt < (n+1)(n+2)/2`: underdetermined mode with **minimum-change Hessian** update (new vs. Apache version — preserves curvature from previous iterations)
  - Note: linear models (`npt == n+1`) are outside the supported range and are not implemented in this port. The minimum supported `npt` is `n+2`.
- Maintains Z\*Zᵀ factorization (ZMAT) for the implicit Hessian — ported from the Apache Java reference implementation
- `poisednessConstant()`: explicit interpolation set quality metric computed via Lagrange polynomials, used to trigger geometry repair

### Trust Region Subproblem (`TrustRegion.java`)

- `trsbox(g, H, delta, sl, su)`: conjugate gradient solver for bound-constrained trust region subproblem, ported from the Apache Java version and enhanced with DFBOLS termination conditions (stricter convergence checks via gradient squared and curvature criteria)
- `trsboxGeometry(g, H, delta, sl, su)`: variant that maximizes absolute model value within bounds — used during geometry improvement steps

### Controller (`Controller.java`) and Solver (`Solver.java`)

`Controller` manages persistent optimization state (interpolation set, best point, radius history, restart counters) and exposes discrete operations. `Solver` owns the main iteration loop and calls into `Controller`, `Model`, and `TrustRegion`. The boundary: `Solver` decides *what* to do each iteration (trust region step, geometry repair, safety step, restart); `Controller` executes *how* (state transitions, objective evaluation, restart bookkeeping).

Key `Controller` responsibilities:

- Tracks `rho` (minimum trust region radius) and `delta` (current radius)
- Trust region radius update uses pybobyqa's `eta1/eta2/gamma` ratio-based rules, replacing the cruder heuristic in the Apache version
- `checkAndFixGeometry()`: identifies interpolation points with poor conditioning and repositions them
- `moveFurthestPoints()`: relocates distant points to improve model quality
- `evaluateObjective(x, nsamples)`: calls user objective `nsamples` times, returns average — core of noise suppression

Key `Solver` responsibilities:

- Drives the main loop: trust region step → ratio check → radius update → geometry repair → restart check
- Applies safety steps when `delta ≤ safetyStepThresh × rho`
- Evaluates all termination conditions each iteration and returns `OptimResult`

---

## Noise Handling

### Parameters

| Parameter | Default (noise on) | Description |
|-----------|-------------------|-------------|
| `noise.objfunHasNoise` | false | Activates all noise-aware defaults |
| `noise.multiplicativeNoiseLevel` | null | Relative noise magnitude (e.g. 0.01) |
| `noise.additiveNoiseLevel` | null | Absolute noise floor |
| `noise.quitOnNoiseLevel` | true | Exit when progress within noise band |
| `noise.scaleFactorForQuit` | 1.0 | Multiplier on noise tolerance |

### Multi-Sample Averaging

`Controller.evaluateObjective(x, nsamples)` evaluates the user's objective `nsamples` times at point `x` and returns the sample mean. The sample count grows dynamically when stagnation is detected. This suppresses stochastic noise (beam jitter) at the cost of additional evaluations.

### Noise-Level Termination

After each successful step, `Solver` checks whether all interpolation point values fall within the estimated noise band of the best value:

```
noise_level = additiveNoiseLevel + multiplicativeNoiseLevel × |f(xbest)|
```

If `max(f(xk) - f(xbest)) ≤ scaleFactorForQuit × noise_level` for all k, the algorithm exits with `NOISE_LEVEL_REACHED`. This prevents over-refinement past the noise floor — particularly important for expensive accelerator evaluations.

**Null noise level handling:** When `quitOnNoiseLevel=true` but both `additiveNoiseLevel` and `multiplicativeNoiseLevel` are `null` (the user has not provided estimates), `Solver` falls back to estimating the noise level from the spread of function values at identical or near-identical interpolation points across the last two iterations. If no such estimate is possible (e.g., early in the run), the noise-level termination check is skipped for that iteration. This matches pybobyqa's behaviour. If the user sets `quitOnNoiseLevel=true` but neither noise level nor a reliable spread estimate is available after `slow.historyForSlow` iterations, a warning is logged and `quitOnNoiseLevel` is silently disabled for the remainder of the run.

### Noise-Aware Trust Region Defaults

When `objfunHasNoise=true`, the following parameter defaults change to be more conservative, preventing premature radius shrinkage on noisy steps:

| Parameter | Default (clean) | Default (noisy) |
|-----------|----------------|----------------|
| `tr_radius.gamma_dec` | 0.5 | 0.98 |
| `tr_radius.alpha1` | 0.9 | 0.1 |
| `tr_radius.alpha2` | 0.95 | 0.5 |

---

## Restart Mechanisms

### Soft Restarts

- Preserves current best point `xbest` and its function value
- Resets `delta` and `rho` to `rhobegScaleAfterUnsuccessfulRestart × rhobeg`. When `objfunHasNoise=true` the default scale is `1.1` (restart with a slightly larger radius to escape noise-induced premature convergence); otherwise the default is `1.0` (restart at the original radius). The `1.1` noisy default is chosen because noisy evaluations can cause `rho` to shrink prematurely — starting the restart slightly wider avoids immediately re-entering the same stagnation.
- Repositions a subset of interpolation points around `xbest` using `soft.numGeomSteps` geometry improvement steps
- The dynamic sample count (used by `evaluateObjective`) is **reset to 1** at the start of each soft restart. The restart itself is triggered by stagnation that may have been caused by insufficient averaging — resetting allows the sample count to grow organically again from the new starting configuration.
- Cheap relative to hard restart — attempted first on stagnation detection

### Hard Restarts

- Completely reinitializes the interpolation set from current best point
- **`useOldFk` evaluation recycling:** When `hard.useOldFk=true`, `Controller` maintains an unbounded cache of all objective evaluations seen so far as a `List<double[]>` of coordinates and a parallel `List<Double>` of function values. Before calling the user objective during reinitialization, `Controller` performs a linear scan over all stored entries and returns the cached value for any stored point within `1e-12` Euclidean distance of the requested point. A `HashMap` keyed on `double[]` is deliberately avoided since Java's array identity equality would prevent coordinate-based lookup. The linear scan is acceptable because reinitialization scans occur only at restart boundaries and the cache size is bounded by `maxEvaluations`. The cache is internal and not exposed in the public API.
- `rhoend` is optionally scaled per restart via `rhoendScale`. Default is `1.0` (no scaling — same tolerance across all restarts). Set to a value less than 1.0 (e.g., 0.5) to tighten convergence with each successive restart.

### Stagnation Auto-Detection

`Controller` maintains rolling history windows (default length 30) of `delta`, gradient change magnitude, and Hessian change magnitude. Linear regression is fitted to each window per iteration. Stagnation is flagged when all three conditions hold simultaneously:

- The slope of `delta` over the window is **negative** (trust region is shrinking)
- The slope of model change magnitudes (gradient + Hessian) over the window is **positive** and exceeds `min_chg_model_slope` (0.015) — meaning the model is still evolving significantly despite the shrinking trust region, indicating the surrogate has not converged to a reliable local model
- The correlation coefficient of the model change regression exceeds `min_correl` (0.1) — ensuring the trend is statistically meaningful rather than noise

On stagnation: soft restart attempted. After `max_unsuccessful_restarts` (10) consecutive failed soft restarts: hard restart. After `max_unsuccessful_restarts_total` (20): exit.

### Slow Progress Detection

Separate from stagnation auto-detection. Tracks whether `f` improves by at least `slow.thresh_for_slow` (1e-8) over the last `slow.history_for_slow` (5) iterations. After `slow.max_slow_iters` (default 20×n) consecutive slow iterations, a restart is triggered rather than terminating — unlike the Apache version which exits unconditionally.

---

## Full Parameter Reference

### General
| Parameter | Default | Description |
|-----------|---------|-------------|
| `general.roundingErrorConstant` | 0.1 | Threshold for shifting xbase |
| `general.safetyStepThresh` | 0.5 | Safety step when delta ≤ thresh × rho |
| `general.checkObjfunForOverflow` | true | Validate for overflow on evaluation |

### Initialization
| Parameter | Default | Description |
|-----------|---------|-------------|
| `init.randomInitialDirections` | false | Random vs. coordinate directions |
| `init.randomDirectionsMakeOrthogonal` | true | Orthogonalize random directions |

### Interpolation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `interpolation.minimumChangeHessian` | true | Min-change Hessian in underdetermined case |
| `interpolation.precondition` | true | Precondition interpolation matrix |

### Trust Region Radius
| Parameter | Default | Description |
|-----------|---------|-------------|
| `trRadius.eta1` | 0.1 | Lower acceptance threshold |
| `trRadius.eta2` | 0.7 | Upper acceptance threshold |
| `trRadius.gammaDec` | 0.5/0.98* | Radius decrease factor |
| `trRadius.gammaInc` | 2.0 | Radius increase factor |
| `trRadius.gammaIncOverline` | 4.0 | Maximum radius increase factor |
| `trRadius.alpha1` | 0.9/0.1* | Lower model improvement threshold |
| `trRadius.alpha2` | 0.95/0.5* | Upper model improvement threshold |

*Noisy defaults when `noise.objfunHasNoise=true`

### Model
| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.npt` | `2n+1` | Number of interpolation points; valid range `[n+2, (n+1)(n+2)/2]`; linear mode (`n+1`) is not supported |
| `model.absTol` | -1e20 | Exit if f falls below this value |

### Noise
| Parameter | Default | Description |
|-----------|---------|-------------|
| `noise.objfunHasNoise` | false | Enable noise-aware mode |
| `noise.multiplicativeNoiseLevel` | null | Relative noise estimate |
| `noise.additiveNoiseLevel` | null | Absolute noise estimate |
| `noise.quitOnNoiseLevel` | true* | Exit within noise band |
| `noise.scaleFactorForQuit` | 1.0 | Noise tolerance multiplier |

### Restarts
| Parameter | Default | Description |
|-----------|---------|-------------|
| `restarts.useRestarts` | true* | Enable restart mechanism |
| `restarts.maxUnsuccessfulRestarts` | 10 | Soft restart limit |
| `restarts.maxUnsuccessfulRestartsTotal` | 20 | Total restart budget |
| `restarts.rhobegScaleAfterUnsuccessfulRestart` | 1.1* / 1.0 | Initial radius scaling post-restart |
| `restarts.rhoendScale` | 1.0 | Multiplier applied to `rhoend` per restart (1.0 = same tolerance across restarts; <1.0 e.g. 0.5 = tighter convergence each restart) |
| `restarts.useSoftRestarts` | true | Try soft before hard restart |
| `restarts.soft.numGeomSteps` | 3 | Geometry steps in soft restart |
| `restarts.soft.moveXk` | true | Reposition best point |
| `restarts.hard.useOldFk` | true | Recycle prior evaluations |
| `restarts.autoDetect` | true | Automatic stagnation detection |
| `restarts.autoDetect.history` | 30 | Rolling window length |
| `restarts.autoDetect.minChgModelSlope` | 0.015 | Model change slope threshold |
| `restarts.autoDetect.minCorrel` | 0.1 | Minimum regression correlation |

### Slow Progress
| Parameter | Default | Description |
|-----------|---------|-------------|
| `slow.historyForSlow` | 5 | History window for slow detection |
| `slow.threshForSlow` | 1e-8 | Minimum improvement threshold |
| `slow.maxSlowIters` | 20×n | Consecutive slow iterations before restart |

### Logging
| Parameter | Default | Description |
|-----------|---------|-------------|
| `logging.saveDiagnosticInfo` | false | Collect per-iteration diagnostics |
| `logging.savePoisedness` | true | Record poisedness metric |
| `logging.saveXk` | false | Record iteration points |
| `logging.nToPrintWholeXVector` | 6 | Dimension threshold for full vector output |

---

## Testing Strategy

### Framework

JUnit 5 (EPL 2.0 — permissive for use). No other test dependencies.

### Test Structure

```
src/test/java/bobyqa/
├── core/
│   ├── ModelTest.java                  # interpolation modes, Hessian updates, poisedness
│   ├── TrustRegionTest.java            # trsbox correctness vs. known analytical solutions
│   └── ControllerTest.java            # geometry repair, ratio computation
├── noise/
│   ├── AdditiveNoiseTest.java          # noisy quadratics with known minima
│   ├── MultiplicativeNoiseTest.java    # relative-noise problems
│   └── NoiseLevelTerminationTest.java  # exit condition at noise floor
├── restarts/
│   ├── SoftRestartTest.java            # restart triggers and recovery
│   ├── HardRestartTest.java
│   └── StagnationDetectionTest.java   # auto-detection regression logic
└── benchmarks/
    ├── RosenbrockTest.java             # smooth classic benchmark
    ├── NoisyRosenbrockTest.java        # noisy variant from pybobyqa suite
    ├── BoundedQuadraticTest.java       # bound-active problems
    └── SlowProgressTest.java           # slow convergence scenario
```

### Validation

Each benchmark compares `(exit_flag, f_best, n_evals)` against pybobyqa's output on the same problem instance. Because Python's Mersenne Twister and Java's `java.util.Random` produce different sequences from the same integer seed, cross-language seed parity is achieved via **pre-generated fixture files**: the test suite ships a set of CSV files containing the random direction sequences used by pybobyqa for each benchmark, generated once by a Python script. The Java implementation reads these fixture files during test setup to reproduce identical initialization sequences. This is a one-time setup cost and makes the validation deterministic across language runtimes.

### Thread Safety

`Bobyqa.solve()` is fully thread-safe for concurrent calls: each invocation creates its own `Solver`, `Controller`, and `Model` instances with no shared mutable state. The only shared state is parameter validation logic in `Params`, which is read-only after construction.

---

## Out of Scope

The following pybobyqa features are explicitly excluded from this port to keep scope manageable:

- **General convex constraints** (`ctrsbox` / Dykstra projection) — bounds-only for now; can be added in a follow-on iteration
- **Parallel initialization** (`init.run_in_parallel`) — sequential evaluation only
- **`trustregion` package integration** — optional accelerated subproblem solver, no Java equivalent exists yet
- **Automatic variable scaling** (`scaling.scale_based_on_bounds`) — pybobyqa optionally rescales variables to `[0,1]` based on bound widths, which helps when variables have different orders of magnitude. This is excluded here. **Implication for users:** if optimization variables differ in scale (e.g., mixing magnet currents in amperes with phase offsets in milliradians), users must manually pre-scale inputs to comparable ranges before calling `Bobyqa.solve()`. A utility method `Bobyqa.scaleToBounds(x, lower, upper)` / `Bobyqa.unscaleFromBounds(x, lower, upper)` will be provided as a convenience, but scaling is not applied automatically.
