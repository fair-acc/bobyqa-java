# bobyqa-java

[![CI](https://github.com/fair-acc/bobyqa-java/actions/workflows/ci.yml/badge.svg)](https://github.com/fair-acc/bobyqa-java/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Derivative-free, bounds-constrained optimizer in pure Java** — this is a port of [PyBOBYQA](https://github.com/numericalalgorithmsgroup/pybobyqa) with extensions for expensive, stochastic, and multiplicatively-noisy objectives encountered in real-world physical systems and with optimizations for the operational performance.

## What's new vs. PyBOBYQA

- **Warm-start** — initialise the surrogate from prior `(x, f(x))` data **or** a previous `ModelSnapshot`, skipping the `npt` interpolation evaluations. Cheap resume from a paused or external run.
- **Noise handling** — multi-sample averaging plus conservative trust-region defaults activated by `noise.objfunHasNoise=true`.
- **Restart auto-detection** — regression on rolling history windows of trust radius, gradient change, and Hessian change.
- **`GoalType.MAXIMIZE` first-class** — objective is negated internally; `result.f()` returns the original (un-negated) value.
- **Pure Java** — EJML is the sole runtime dependency. No JNI, no native libs.

## Installation

For v1.0.1, clone-and-build:

```bash
git clone https://github.com/fair-acc/bobyqa-java.git
cd bobyqa-java
mvn clean install
```

Maven Central coordinates: **arriving in v1.1** as `io.fair-acc:bobyqa-java`.

## Quick start

```java
import bobyqa.Bobyqa;
import bobyqa.GoalType;
import bobyqa.OptimResult;
import bobyqa.Params;

double[] x0    = { -1.2,  1.0 };
double[] lower = { -5.0, -5.0 };
double[] upper = {  5.0,  5.0 };

OptimResult result = Bobyqa.solve(
        x -> Math.pow(1.0 - x[0], 2) + 100.0 * Math.pow(x[1] - x[0] * x[0], 2),
        x0, lower, upper,
        2000,
        GoalType.MINIMIZE,
        Params.defaults(2));

System.out.printf("x = %s, f = %.3e, evals = %d, exit = %s%n",
        java.util.Arrays.toString(result.x()),
        result.f(),
        result.nEvals(),
        result.exitFlag());
```

## Warm-start usage

Use **prior data** when you have observations from an external run (manual measurements, parallel sweeps, etc.):

```java
import bobyqa.WarmStart;
import java.util.List;

List<double[]> priorXValues = List.of(/* sample X-values from previous runs, one double[] per sample */);
List<Double>   priorMeas = List.of(/* corresponding response values from previous runs */);
WarmStart      ws     = WarmStart.fromPriorData(priorXValues, priorMeas);

OptimResult result = Bobyqa.solve(
        objective, x0, lower, upper, maxEvals, GoalType.MINIMIZE,
        Params.defaults(n), ws);
```

Use a **model snapshot** when you are resuming a paused optimisation in the same process:

```java
ModelSnapshot modelSnapshot = previousResult.snapshot();   // captured at the end of a previous solve()
WarmStart    ws        = WarmStart.fromSnapshot(modelSnapshot);

OptimResult result = Bobyqa.solve(
        objective, x0, lower, upper, maxEvals, GoalType.MAXIMIZE,
        Params.defaults(n), ws);
```

Prior data performs a hybrid initialisation (top-k points + fresh axis-aligned fill). Snapshots perform a zero-evaluation restore.

## Documentation

The full technical design is in [docs/design/bobyqa-java-port-design.md](docs/design/bobyqa-java-port-design.md).

## Attribution

This is a Java port of **[PyBOBYQA](https://github.com/numericalalgorithmsgroup/pybobyqa)** by the Numerical Algorithms Group (NAG), itself a Python implementation of **BOBYQA** (Bound Optimization BY Quadratic Approximation) by M. J. D. Powell (2009).

Disclaimer: This software has been authored using AI, namely CLAUDE Code (version(s) 4.6 and following).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff) — GitHub renders a **"Cite this repository"** button on the repo homepage from it (also consumed by Zotero, Zenodo, and most citation managers). For manual citation, the BibTeX entries are:

```bibtex
@misc{bobyqa-java,
  author       = {Geithner, Wolfgang},
  title        = {{bobyqa-java}: a Java port of PyBOBYQA},
  year         = {2026},
  publisher    = {GSI Helmholtzzentrum für Schwerionenforschung GmbH},
  howpublished = {\url{https://github.com/fair-acc/bobyqa-java}}
}

@techreport{powell2009bobyqa,
  author      = {Powell, M.J.D.},
  title       = {The {BOBYQA} algorithm for bound constrained optimization without derivatives},
  institution = {DAMTP, University of Cambridge},
  number      = {NA2009/06},
  year        = {2009}
}
```
