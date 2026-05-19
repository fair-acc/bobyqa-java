# Changelog

All notable changes to bobyqa-java are documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

## [1.0.0] — 2026-09-14

### Added
- Initial public release: Java port of PyBOBYQA targeting expensive,
  stochastic, multiplicatively-noisy objectives. Includes:
  - Noise handling (multi-sample averaging + conservative trust-region defaults under `noise.objfunHasNoise=true`)
  - Restart auto-detection (regression on rolling history windows of delta, gradient change, Hessian change)
  - Warm-start from prior `(x, f(x))` data (`WarmStart.fromPriorData`) or a previous model snapshot (`WarmStart.fromSnapshot`) — skips or reduces the `npt` initialization evaluations
  - First-class `GoalType.MINIMIZE` / `GoalType.MAXIMIZE` (maximization is implemented by negating the objective internally; `result.f()` returns the original value)
  - Pure Java with EJML as the sole runtime dependency — no JNI, no native libs
