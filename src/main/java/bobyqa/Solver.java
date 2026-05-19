/*
 * bobyqa-java — Java port of PyBOBYQA
 * Copyright (C) 2026 GSI Helmholtzzentrum für Schwerionenforschung GmbH
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
package bobyqa;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.ToDoubleFunction;

import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;

import bobyqa.util.MathUtil;

/**
 * Main optimization loop for BOBYQA. Decides WHAT to do each iteration and delegates HOW to Controller. Loop structure
 * (mirrors pybobyqa solver.py): 1. Check termination (budget, convergence, absTol, noise) 2. Safety step if delta <=
 * safetyStepThresh * rho 3. Compute trust region step via TrustRegion.trsbox() 4. Evaluate objective at trial point 5.
 * Compute ratio (actual / predicted reduction) 6. Update model and trust radius 7. Check slow progress / stagnation →
 * restart
 */
public class Solver {

    private static final System.Logger log = System.getLogger(Solver.class.getName());

    private final ToDoubleFunction<double[]> _objective;
    private double[] _lowerLimits;
    private double[] _upperLimits;
    private final int _maxEvaluations;
    private final Params _algorithmParameters;
    private final double _rhoStart;
    private final double _rhoEnd;
    private final GoalType _goalType;
    private final WarmStart _warmStart;

    private int _totalEvals = 0;
    private final List<DiagnosticInfo> _diagnosticInfos = new ArrayList<>();
    private final double[] _hardLower;
    private final double[] _hardUpper;
    private final List<BoundExpansionEvent> _expansions = new ArrayList<>();

    public Solver(
            ToDoubleFunction<double[]> objective,
            double[] lowerLimits,
            double[] upperLimits,
            double[] hardLower,
            double[] hardUpper,
            int maxEvaluations,
            double rhoStart,
            double rhoEnd,
            Params algorithmParameters,
            GoalType goalType,
            WarmStart warmStart) {
        this._objective = objective;
        this._lowerLimits = lowerLimits.clone();
        this._upperLimits = upperLimits.clone();
        this._hardLower = hardLower;
        this._hardUpper = hardUpper;
        this._maxEvaluations = maxEvaluations;
        this._goalType = goalType;
        this._algorithmParameters = algorithmParameters;
        this._rhoStart = rhoStart;
        this._rhoEnd = rhoEnd;
        this._warmStart = warmStart;
    }

    public OptimResult optimize(double[] x0) {
        int dimension = x0.length;

        // Validate npt
        int npt = _algorithmParameters._modelParameters.npt;
        int nptLo = dimension + 2, nptHi = (dimension + 1) * (dimension + 2) / 2;
        if (npt < nptLo || npt > nptHi) return new OptimResult(x0.clone(), Double.NaN, 0,
                ExitFlag.INPUT_ERROR,
                "npt=" + npt + " out of valid range [" + nptLo + ", " + nptHi + "]",
                List.of(), List.of(), null);

        // Validate bounds
        for (int i = 0; i < dimension; i++)
            if (_lowerLimits[i] >= _upperLimits[i] || x0[i] < _lowerLimits[i] || x0[i] > _upperLimits[i])
                return new OptimResult(x0.clone(), Double.NaN, 0, ExitFlag.INPUT_ERROR,
                        "Invalid bounds or x0 outside bounds at index " + i, List.of(), List.of(), null);

        var model = new Model(dimension, npt, _rhoStart);
        var ctrl = new Controller(_objective, _algorithmParameters, _rhoStart, _rhoEnd, null);

        // Initialization
        OptimResult initResult = initialize(model, ctrl, x0, dimension, npt, _warmStart);
        if (initResult != null) return initResult;

        // Estimate noise from initialization residuals when noise mode is on and no explicit
        // noise level was provided by the caller.
        if (_algorithmParameters._noiseParameters.objfunHasNoise
                && _algorithmParameters._noiseParameters.additiveNoiseLevel == null
                && _algorithmParameters._noiseParameters.multiplicativeNoiseLevel == null) {
            Double sigmaEst = estimateNoiseFromInit(model);
            if (sigmaEst != null) ctrl.setInitialNoiseEstimate(sigmaEst);
        }

        // Track previous gradient/Hessian norms for stagnation detection
        double prevGradNorm = MathUtil.norm2(model.grad());
        double prevHessNorm = hessNorm(model);

        // Main loop
        int iter = 0;
        while (true) {
            // Budget check
            if (_totalEvals >= _maxEvaluations) return buildResult(
                    model,
                    ctrl,
                    ExitFlag.BUDGET_EXHAUSTED,
                    "Function evaluation budget exhausted.",
                    iter);

            // Convergence check
            if (ctrl.rho() <= _rhoEnd)
                return buildResult(model, ctrl, ExitFlag.CONVERGED, "Trust region radius reached rhoend.", iter);

            // Absolute tolerance check
            // Default absTol = -1e20 acts as "disabled". Only check when explicitly set.
            double fBest = model.fBest();
            if (_algorithmParameters._modelParameters.absTol > -1e20) {
                double fBestOriginal = _goalType == GoalType.MAXIMIZE ? -fBest : fBest;
                boolean absTolHit = _goalType == GoalType.MAXIMIZE ? fBestOriginal > _algorithmParameters._modelParameters.absTol // maximise: exit when f(x) > absTol
                        : fBestOriginal < _algorithmParameters._modelParameters.absTol; // minimise: exit when f(x) < absTol
                if (absTolHit) return buildResult(
                        model,
                        ctrl,
                        ExitFlag.ABS_TOL_REACHED,
                        "Objective crossed absTol threshold.",
                        iter);
            }

            IterType iterType;
            double ratio = Double.NaN;

            // Always compute the trust region step first so we can check dnorm
            double[] xopt = model.xopt();
            double[] sl = new double[dimension], su = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                sl[i] = _lowerLimits[i] - xopt[i];
                su[i] = _upperLimits[i] - xopt[i];
            }
            double[] step = TrustRegion.trsbox(model.grad(), model.hess(), ctrl.delta(), sl, su);
            double stepNorm = MathUtil.norm2(step);
            double dnorm = Math.min(stepNorm, ctrl.delta());

            if (dnorm < _algorithmParameters._generalParameters.safetyStepThresh * ctrl.rho()) {
                // Safety step: try to improve geometry before reducing rho (mirrors pybobyqa)
                double safetyDistsqThresh = 100.0 * ctrl.rho() * ctrl.rho(); // (10*rho)^2
                boolean fixedGeom = repairGeometry(model, ctrl, safetyDistsqThresh, dimension);
                if (!fixedGeom && ctrl.doneWithCurrentRho(iter)) {
                    ctrl.reduceRho();
                    ctrl.recordSuccessfulIter(iter); // mirrors pybobyqa: reset after rho reduction
                }
                iterType = IterType.SAFETY;
            } else {
                // Trust region step
                if (stepNorm < _rhoEnd * 0.01) {
                    ctrl.setDelta(ctrl.delta() * _algorithmParameters._trustRegionRadius.gammaDec);
                    iterType = IterType.UNSUCCESSFUL;
                    iter++;
                    continue;
                }

                double[] xTrial = new double[dimension];
                for (int i = 0; i < dimension; i++)
                    xTrial[i] = xopt[i] + step[i];

                double predicted = -TrustRegion.modelValue(model.grad(), model.hess(), step);
                if (predicted <= 0) {
                    ctrl.setDelta(ctrl.delta() * _algorithmParameters._trustRegionRadius.gammaDec);
                    iterType = IterType.UNSUCCESSFUL;
                } else {
                    double fTrial = callObjective(ctrl, xTrial);
                    if (fTrial == Double.MAX_VALUE) return buildResult(
                            model,
                            ctrl,
                            ExitFlag.BUDGET_EXHAUSTED,
                            "Budget exhausted during step.",
                            iter);
                    ratio = (fBest - fTrial) / predicted;
                    // Mirror pybobyqa: record success when ratio>0 or step was large enough
                    if (ratio > 0 || dnorm > ctrl.rho()) ctrl.recordSuccessfulIter(iter);

                    int kReplace = choosePointToReplace(model, step, ctrl.delta());
                    model.setPoint(kReplace, xTrial, fTrial);

                    if (fTrial < fBest) {
                        model.updateKopt();
                        iterType = IterType.SUCCESSFUL;
                    } else {
                        iterType = IterType.UNSUCCESSFUL;
                    }
                    boolean stepWasAtBoundary = TrustRegion.isAtBoundary(step, sl, su);
                    ctrl.updateTrustRadius(ratio, dnorm, stepWasAtBoundary);
                    try {
                        model.interpolateModel(_algorithmParameters._interpolationParameters.minimumChangeHessian);
                    } catch (Exception e) {
                        // Singular matrix: keep current model
                    }
                    if (_algorithmParameters._dynamicRangesParameters.enabled) {
                        tryExpandBounds(step, sl, su, model.grad(), ctrl.delta());
                    }
                    // Geometry repair after bad TR steps (ratio < eta1), mirrors pybobyqa
                    if (ratio < _algorithmParameters._trustRegionRadius.eta1 && _totalEvals < _maxEvaluations) {
                        double trDistsqThresh = Math.max(
                                4.0 * ctrl.delta() * ctrl.delta(),
                                100.0 * ctrl.rho() * ctrl.rho());
                        repairGeometry(model, ctrl, trDistsqThresh, dimension);
                    }
                }

                // Noise-level termination
                if (_algorithmParameters._noiseParameters.quitOnNoiseLevel && !Double.isNaN(ratio) && ratio >= 0)
                    if (checkNoiseLevelTermination(model, ctrl)) return buildResult(
                            model,
                            ctrl,
                            ExitFlag.NOISE_LEVEL_REACHED,
                            "All interpolation values within noise level.",
                            iter);
            }


            // Record stagnation data for auto-detection
            double gradNorm = MathUtil.norm2(model.grad());
            double hNorm = hessNorm(model);
            double gradChange = Math.abs(gradNorm - prevGradNorm);
            double hessChange = Math.abs(hNorm - prevHessNorm);
            ctrl.recordIteration(ctrl.delta(), gradChange, hessChange);
            prevGradNorm = gradNorm;
            prevHessNorm = hNorm;

            // Slow progress detection: track on all non-safety iterations (like pybobyqa for non-noise case)
            if (iterType != IterType.SAFETY) ctrl.recordFValue(model.fBest());
            if (ctrl.isSlowProgress()) {
                ctrl.incrementSlowIter();
                if (ctrl.slowIterCount() >= _algorithmParameters._slowParameters.maxSlowIters) {
                    if (_algorithmParameters._restartsParameters.useRestarts && !ctrl.restartsExhausted()) {
                        // Restart when slow progress detected and restarts enabled
                        triggerRestart(model, ctrl, false);
                        iterType = IterType.RESTART_SOFT;
                    } else {
                        // No restarts: reduce rho or exit
                        if (ctrl.rho() <= _rhoEnd) return buildResult(
                                model,
                                ctrl,
                                ExitFlag.CONVERGED,
                                "Converged (slow progress at minimum rho).",
                                iter);
                        ctrl.reduceRho();
                        ctrl.setDelta(Math.max(ctrl.delta(), ctrl.rho()));
                        ctrl.resetSlowIter();
                    }
                }
            } else {
                ctrl.resetSlowIter();
            }

            // Stagnation auto-detection → restart (mirrors pybobyqa: only active when restarts enabled)
            if (_algorithmParameters._restartsParameters.useRestarts
                    && _algorithmParameters._restartsParameters.autoDetect && ctrl.isStagnating()
                    && !ctrl.restartsExhausted()) {
                boolean hardRestart = ctrl.consecutiveRestartsExhausted();
                triggerRestart(model, ctrl, hardRestart);
                iterType = hardRestart ? IterType.RESTART_HARD : IterType.RESTART_SOFT;
            }

            // Record diagnostics
            if (_algorithmParameters._loggingParameters.saveDiagnosticInfo) {
                Double poisedness = _algorithmParameters._loggingParameters.savePoisedness ? model.poisednessConstant(ctrl.delta()) : null;
                double[] xk = _algorithmParameters._loggingParameters.saveXk ? model.xopt() : null;
                _diagnosticInfos.add(
                        new DiagnosticInfo(iter, iterType, model.fBest(), ctrl.delta(), ctrl.rho(), ratio,
                                ctrl.nsamples(), ctrl.restartCount(), poisedness, xk));
            }

            iter++;
        }
    }

    // --- Helpers ------------------------------------------------------------

    /**
     * Estimate measurement noise from initialization evaluations by fitting an OLS linear model
     * to the npt f-values and computing the RMS of the residuals. The residuals contain both
     * noise and the quadratic curvature not captured by the linear fit. If the RMS exceeds half
     * the raw f-value range, the landscape is signal-dominated and the estimate is unreliable
     * (returns null). Only called when objfunHasNoise is true.
     */
    private Double estimateNoiseFromInit(Model model) {
        int n = model.n();
        int npt = model.npt();
        int dof = npt - (n + 1);
        if (dof <= 0) return null;

        double[] xopt = model.xopt();
        double[] xbase = model.xbase();

        // Build OLS system: rows = [1, d_1, ..., d_n], response = f_k
        SimpleMatrix X = new SimpleMatrix(npt, n + 1);
        SimpleMatrix y = new SimpleMatrix(npt, 1);
        double fMin = Double.MAX_VALUE, fMax = -Double.MAX_VALUE;
        for (int k = 0; k < npt; k++) {
            X.set(k, 0, 1.0);
            double[] xrel = model.xpt(k); // relative to xbase
            for (int i = 0; i < n; i++)
                X.set(k, i + 1, xbase[i] + xrel[i] - xopt[i]);
            double fk = model.fValue(k);
            y.set(k, 0, fk);
            if (fk < fMin) fMin = fk;
            if (fk > fMax) fMax = fk;
        }

        SimpleMatrix beta = X.pseudoInverse().mult(y);
        SimpleMatrix residuals = y.minus(X.mult(beta));

        double ssq = 0;
        for (int k = 0; k < npt; k++) ssq += Math.pow(residuals.get(k, 0), 2);
        double sigmaEst = Math.sqrt(ssq / dof);

        // Guard: if residuals are dominated by quadratic curvature (signal >> noise),
        // the estimate is unreliable — don't set it.
        double fRange = fMax - fMin;
        if (fRange > 0 && sigmaEst > 0.5 * fRange) return null;

        return sigmaEst;
    }

    private OptimResult initialize(Model model, Controller ctrl, double[] x0, int n, int npt, WarmStart warmStart) {
        if (warmStart != null && warmStart.snapshot() != null && warmStart.priorX() != null)
            log.log(System.Logger.Level.WARNING,
                    "WarmStart: both snapshot and priorX provided; snapshot takes priority, priorX ignored.");

        if (warmStart != null && warmStart.snapshot() != null) {
            return initFromSnapshot(model, ctrl, x0, n, npt, warmStart.snapshot());
        } else if (warmStart != null && warmStart.priorX() != null) {
            return initFromPriorPairs(model, ctrl, x0, n, npt, warmStart.priorX(), warmStart.priorF());
        } else {
            // Cold start
            model.setXbase(x0);

            double f0 = callObjective(ctrl, x0);
            if (_totalEvals >= _maxEvaluations)
                return buildResult(model, ctrl, ExitFlag.BUDGET_EXHAUSTED, "Budget exhausted at init.", 0);
            model.setPoint(0, x0, f0);

            for (int k = 1; k < npt && _totalEvals < _maxEvaluations; k++) {
                double[] xk = axisAlignedInitPoint(x0, k, n);
                double fk = callObjective(ctrl, xk);
                model.setPoint(k, xk, fk);
            }
            if (_totalEvals >= _maxEvaluations)
                return buildResult(model, ctrl, ExitFlag.BUDGET_EXHAUSTED, "Budget exhausted at init.", 0);
            model.updateKopt();
            model.interpolateModel(_algorithmParameters._interpolationParameters.minimumChangeHessian);
            return null;
        }
    }

    private double[] axisAlignedInitPoint(double[] x0, int slot, int n) {
        double[] xk = x0.clone();
        int dir = -1;
        double step = 0;
        if (slot <= n) {
            dir = slot - 1;
            double distUp = _upperLimits[dir] - x0[dir];
            step = Math.min(_rhoStart, distUp);
            if (step == 0) step = -Math.min(_rhoStart, x0[dir] - _lowerLimits[dir]) / 2.0;
        } else if (slot <= 2 * n) {
            dir = slot - n - 1;
            double distDown = x0[dir] - _lowerLimits[dir];
            step = -Math.min(_rhoStart, distDown);
            if (step == 0) step = Math.min(_rhoStart, _upperLimits[dir] - x0[dir]) / 2.0;
        } else {
            int extraIdx = slot - 2 * n - 1;
            int d1 = extraIdx % n;
            int d2 = (d1 + 1 + extraIdx / n) % n;
            if (d2 == d1) d2 = (d1 + 1) % n;
            double s1 = Math.min(0.5 * _rhoStart, _upperLimits[d1] - x0[d1]);
            double s2 = Math.min(0.5 * _rhoStart, _upperLimits[d2] - x0[d2]);
            if (s1 == 0) s1 = -Math.min(0.5 * _rhoStart, x0[d1] - _lowerLimits[d1]);
            if (s2 == 0) s2 = -Math.min(0.5 * _rhoStart, x0[d2] - _lowerLimits[d2]);
            xk[d1] += s1;
            xk[d2] += s2;
            dir = -1;
        }
        if (dir >= 0) xk[dir] += step;
        return xk;
    }

    private OptimResult initFromPriorPairs(
            Model model, Controller ctrl, double[] x0, int n, int npt,
            List<double[]> priorX, List<Double> priorF) {

        if (priorX.isEmpty())
            return new OptimResult(x0.clone(), Double.NaN, 0, ExitFlag.INPUT_ERROR,
                    "WarmStart prior data invalid: priorX must not be empty.",
                    java.util.List.of(), java.util.List.of(), null);
        if (priorF == null)
            return new OptimResult(x0.clone(), Double.NaN, 0, ExitFlag.INPUT_ERROR,
                    "WarmStart prior data invalid: priorF must not be null when priorX is provided.",
                    java.util.List.of(), java.util.List.of(), null);
        if (priorX.size() != priorF.size())
            return new OptimResult(x0.clone(), Double.NaN, 0, ExitFlag.INPUT_ERROR,
                    "WarmStart prior data invalid: priorX and priorF must have equal length.",
                    List.of(), List.of(), null);
        for (double[] xi : priorX)
            if (xi.length != n)
                return new OptimResult(x0.clone(), Double.NaN, 0, ExitFlag.INPUT_ERROR,
                        "WarmStart prior data invalid: each priorX entry must have length n=" + n + ".",
                        List.of(), List.of(), null);

        for (Double fi : priorF)
            if (fi == null || !Double.isFinite(fi))
                return new OptimResult(x0.clone(), Double.NaN, 0, ExitFlag.INPUT_ERROR,
                        "WarmStart prior data invalid: priorF contains null or non-finite value.",
                        List.of(), List.of(), null);
        for (double[] xi : priorX)
            for (double v : xi)
                if (!Double.isFinite(v))
                    return new OptimResult(x0.clone(), Double.NaN, 0, ExitFlag.INPUT_ERROR,
                            "WarmStart prior data invalid: priorX contains non-finite coordinate.",
                            List.of(), List.of(), null);
        for (double[] xi : priorX)
            for (int i = 0; i < n; i++)
                if (xi[i] < _lowerLimits[i] || xi[i] > _upperLimits[i])
                    return new OptimResult(x0.clone(), Double.NaN, 0, ExitFlag.INPUT_ERROR,
                            "WarmStart prior data invalid: priorX contains point outside bounds at dimension " + i + ".",
                            List.of(), List.of(), null);

        Integer[] idx = new Integer[priorX.size()];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        if (_goalType == GoalType.MINIMIZE)
            Arrays.sort(idx, Comparator.comparingDouble(i -> priorF.get(i)));
        else
            Arrays.sort(idx, Comparator.comparingDouble((Integer i) -> priorF.get(i)).reversed());

        int k = Math.min(priorX.size(), npt);
        model.setXbase(x0);

        for (int j = 0; j < k; j++) {
            int src = idx[j];
            double rawF = priorF.get(src);
            double internalF = (_goalType == GoalType.MAXIMIZE) ? -rawF : rawF;
            model.setPoint(j, priorX.get(src), internalF);
        }

        for (int slot = k; slot < npt && _totalEvals < _maxEvaluations; slot++) {
            double[] xk = axisAlignedInitPoint(x0, slot, n);
            double fk = callObjective(ctrl, xk);
            model.setPoint(slot, xk, fk);
        }

        if (_totalEvals >= _maxEvaluations)
            return buildResult(model, ctrl, ExitFlag.BUDGET_EXHAUSTED, "Budget exhausted at init.", 0);

        model.updateKopt();
        model.interpolateModel(_algorithmParameters._interpolationParameters.minimumChangeHessian);
        return null;
    }

    private OptimResult initFromSnapshot(
            Model model, Controller ctrl, double[] x0, int n, int npt,
            ModelSnapshot snapshot) {

        if (snapshot.n() != n || snapshot.npt() != npt)
            return new OptimResult(x0.clone(), Double.NaN, 0, ExitFlag.INPUT_ERROR,
                    "WarmStart snapshot dimension mismatch: snapshot has n=" + snapshot.n()
                            + ", npt=" + snapshot.npt() + " but problem has n=" + n + ", npt=" + npt + ".",
                    java.util.List.of(), java.util.List.of(), null);
        if (snapshot.xbase() == null || snapshot.xbase().length != n
                || snapshot.xpt() == null || snapshot.xpt().length != npt
                || snapshot.fval() == null || snapshot.fval().length != npt
                || snapshot.grad() == null || snapshot.grad().length != n
                || snapshot.hess() == null || snapshot.hess().length != n)
            return new OptimResult(x0.clone(), Double.NaN, 0, ExitFlag.INPUT_ERROR,
                    "WarmStart snapshot array structure is invalid (null or wrong-length arrays).",
                    java.util.List.of(), java.util.List.of(), null);
        for (int i = 0; i < n; i++)
            if (snapshot.hess()[i] == null || snapshot.hess()[i].length != n)
                return new OptimResult(x0.clone(), Double.NaN, 0, ExitFlag.INPUT_ERROR,
                        "WarmStart snapshot hess row " + i + " is null or has wrong length.",
                        java.util.List.of(), java.util.List.of(), null);
        for (int k = 0; k < npt; k++)
            if (snapshot.xpt()[k] == null || snapshot.xpt()[k].length != n)
                return new OptimResult(x0.clone(), Double.NaN, 0, ExitFlag.INPUT_ERROR,
                        "WarmStart snapshot xpt row " + k + " is null or has wrong length.",
                        java.util.List.of(), java.util.List.of(), null);

        // xbase comes from the snapshot, not from x0
        model.setXbase(snapshot.xbase());

        for (int k = 0; k < npt; k++) {
            // snapshot.xpt()[k] is relative to snapshot.xbase() — setPoint() expects absolute coords
            double[] xAbs = new double[n];
            for (int i = 0; i < n; i++) xAbs[i] = snapshot.xbase()[i] + snapshot.xpt()[k][i];
            model.setPoint(k, xAbs, snapshot.fval()[k]);
        }

        model.setGrad(snapshot.grad());

        var hMat = new DMatrixRMaj(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                hMat.set(i, j, snapshot.hess()[i][j]);
        model.setHess(hMat);

        model.updateKopt();
        // Skip interpolateModel() — surrogate is already fitted
        return null;
    }

    private double callObjective(Controller ctrl, double[] x) {
        if (_totalEvals >= _maxEvaluations) return Double.MAX_VALUE;
        try {
            double f = ctrl.evaluateObjective(x, ctrl.nsamples());
            _totalEvals += ctrl.nsamples();
            return f;
        } catch (HardBoundViolationException e) {
            _totalEvals += ctrl.nsamples();
            if (e.direction == Direction.UPPER) {
                _hardUpper[e.paramIndex] = e.violatingValue;
                _upperLimits[e.paramIndex] = Math.min(_upperLimits[e.paramIndex], e.violatingValue);
                log.log(System.Logger.Level.INFO,
                        "Hard upper bound discovered for parameter {0} at {1} via framework exception; "
                                + "soft upper bound clipped to {2}.",
                        e.paramIndex, e.violatingValue, _upperLimits[e.paramIndex]);
            } else {
                _hardLower[e.paramIndex] = e.violatingValue;
                _lowerLimits[e.paramIndex] = Math.max(_lowerLimits[e.paramIndex], e.violatingValue);
                log.log(System.Logger.Level.INFO,
                        "Hard lower bound discovered for parameter {0} at {1} via framework exception; "
                                + "soft lower bound clipped to {2}.",
                        e.paramIndex, e.violatingValue, _lowerLimits[e.paramIndex]);
            }
            return Double.MAX_VALUE;
        }
    }

    private boolean checkNoiseLevelTermination(Model model, Controller ctrl) {
        double fBest = model.fBest();
        Double noiseLevel = computeNoiseLevel(ctrl, fBest);
        if (noiseLevel == null) return false;
        double threshold = _algorithmParameters._noiseParameters.scaleFactorForQuit * noiseLevel;
        for (int k = 0; k < model.npt(); k++)
            if (model.fValue(k) - fBest > threshold) return false;
        return true;
    }

    private Double computeNoiseLevel(Controller ctrl, double fBest) {
        Double mult = _algorithmParameters._noiseParameters.multiplicativeNoiseLevel;
        Double add = _algorithmParameters._noiseParameters.additiveNoiseLevel;
        if (mult != null || add != null)
            return (add != null ? add : 0.0) + (mult != null ? mult * Math.abs(fBest) : 0.0);
        return ctrl.estimateNoiseFallback(fBest);
    }

    /**
     * Choose the interpolation point to replace using the Lagrange polynomial criterion (pybobyqa
     * choose_point_to_replace): maximize |lambda_k(xopt+step)| weighted by (dist/delta)^2.
     * Falls back to worst-f-value if all Lagrange computations fail.
     */
    private int choosePointToReplace(Model model, double[] step, double delta) {
        int n = model.n();
        double delsq = delta * delta;
        double[] g = new double[n];
        DMatrixRMaj H = new DMatrixRMaj(n, n);
        double[] xoptRel = model.xpt(model.kopt());
        int knew = -1;
        double scaden = 0;
        for (int k = 0; k < model.npt(); k++) {
            if (k == model.kopt()) continue;
            if (!model.lagrangePolynomial(k, g, H)) continue;
            double den = TrustRegion.modelValue(g, H, step); // lambda_k(xopt+step), c=0 for k!=kopt
            double[] xk = model.xpt(k);
            double distsq = 0;
            for (int i = 0; i < n; i++) {
                double d = xk[i] - xoptRel[i];
                distsq += d * d;
            }
            double weight = Math.max(1.0, (distsq / delsq) * (distsq / delsq));
            if (weight * Math.abs(den) > scaden) {
                scaden = weight * Math.abs(den);
                knew = k;
            }
        }
        if (knew >= 0) return knew;
        // Fallback: worst f-value
        int worst = -1;
        double worstF = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < model.npt(); k++) {
            if (k == model.kopt()) continue;
            if (model.fValue(k) > worstF) { worstF = model.fValue(k); worst = k; }
        }
        return worst;
    }

    /**
     * Attempt to repair interpolation set geometry by repositioning the point furthest from xopt
     * (if it exceeds distsqThresh) via a Lagrange-polynomial geometry step.
     * Returns true if a geometry step was taken (eval consumed), false if geometry was already acceptable.
     */
    private boolean repairGeometry(Model model, Controller ctrl, double distsqThresh, int n) {
        double[] xoptRel = model.xpt(model.kopt());
        int worst = -1;
        double maxDistsq = 0;
        for (int k = 0; k < model.npt(); k++) {
            if (k == model.kopt()) continue;
            double[] xk = model.xpt(k);
            double distsq = 0;
            for (int i = 0; i < n; i++) {
                double d = xk[i] - xoptRel[i];
                distsq += d * d;
            }
            if (distsq > maxDistsq) { maxDistsq = distsq; worst = k; }
        }
        if (worst < 0 || maxDistsq <= distsqThresh) return false;

        double dist = Math.sqrt(maxDistsq);
        double adelt = Math.max(Math.min(0.1 * dist, ctrl.delta()), ctrl.rho());

        double[] g = new double[n];
        DMatrixRMaj H = new DMatrixRMaj(n, n);
        if (!model.lagrangePolynomial(worst, g, H)) return false;

        double[] xopt = model.xopt();
        double[] sl = new double[n], su = new double[n];
        for (int i = 0; i < n; i++) {
            sl[i] = _lowerLimits[i] - xopt[i];
            su[i] = _upperLimits[i] - xopt[i];
        }
        double[] geomStep = TrustRegion.trsboxGeometry(g, H, adelt, sl, su);
        double[] xnew = new double[n];
        for (int i = 0; i < n; i++) xnew[i] = xopt[i] + geomStep[i];

        double fNew = callObjective(ctrl, xnew);
        if (fNew == Double.MAX_VALUE) return true; // budget exhausted

        model.setPoint(worst, xnew, fNew);
        if (fNew < model.fBest()) model.updateKopt();
        try {
            model.interpolateModel(_algorithmParameters._interpolationParameters.minimumChangeHessian);
        } catch (Exception e) {
            // Singular: keep current model
        }
        return true;
    }

    private void triggerRestart(Model model, Controller ctrl, boolean hard) {
        double fBefore = model.fBest();
        double[] xbest = model.xopt();
        if (hard) {
            ctrl.hardRestart(model, xbest);
            reinitializeModel(model, ctrl, xbest, _rhoStart);
        } else {
            ctrl.softRestart(model, xbest);
            // Rebuild with post-restart rho as perturbation distance (= scale * rhobeg after soft reset)
            reinitializeModel(model, ctrl, xbest, ctrl.rho());
        }
        if (_totalEvals < _maxEvaluations && model.fBest() < fBefore) {
            ctrl.recordSuccessfulRestart();
        }
    }

    /**
     * Rebuild all npt interpolation points around xbest with the given perturbation step.
     * Mirrors initialize() but centred on xbest instead of x0.
     * Called after soft/hard restart once Controller has repositioned model.xbase.
     * All npt slots (0 through npt-1) are overwritten, so no stale data remains.
     */
    private void reinitializeModel(Model model, Controller ctrl,
                                    double[] xbest, double perturbStep) {
        int n = xbest.length;
        int npt = _algorithmParameters._modelParameters.npt;

        double f0 = callObjective(ctrl, xbest);
        if (f0 == Double.MAX_VALUE) return; // budget exhausted
        model.setPoint(0, xbest, f0);

        for (int k = 1; k < npt && _totalEvals < _maxEvaluations; k++) {
            double[] xk = xbest.clone();
            int dir = -1;
            double step;
            if (k <= n) {
                dir = k - 1;
                step = perturbStep;
                if (xbest[dir] + step > _upperLimits[dir]) step = -step;
                xk[dir] += step;
            } else if (k <= 2 * n) {
                dir = k - n - 1;
                step = -perturbStep;
                if (xbest[dir] + step < _lowerLimits[dir]) step = perturbStep;
                xk[dir] += step;
            } else {
                int extraIdx = k - 2 * n - 1;
                int d1 = extraIdx % n;
                int d2 = (d1 + 1 + extraIdx / n) % n;
                if (d2 == d1) d2 = (d1 + 1) % n;
                double s1 = 0.5 * perturbStep;
                double s2 = 0.5 * perturbStep;
                if (xbest[d1] + s1 > _upperLimits[d1]) s1 = -s1;
                if (xbest[d2] + s2 > _upperLimits[d2]) s2 = -s2;
                xk[d1] += s1;
                xk[d2] += s2;
            }
            double fk = callObjective(ctrl, xk);
            model.setPoint(k, xk, fk);
        }

        // If budget was exhausted mid-loop, don't interpolate on partial data
        if (_totalEvals >= _maxEvaluations) return;

        model.updateKopt();
        try {
            model.interpolateModel(_algorithmParameters._interpolationParameters.minimumChangeHessian);
        } catch (Exception e) {
            // Singular interpolation: keep current model state
        }
    }

    private void tryExpandBounds(double[] step, double[] sl, double[] su, double[] grad, double delta) {
        double thresh = _algorithmParameters._dynamicRangesParameters.gradientThreshold;
        double stepSize = _algorithmParameters._dynamicRangesParameters.expansionStep != null
                ? _algorithmParameters._dynamicRangesParameters.expansionStep : delta;

        for (int i = 0; i < _lowerLimits.length; i++) {
            // Upper bound hit and gradient points further upward → expand upper soft bound
            if (Math.abs(step[i] - su[i]) < 1e-12 && grad[i] < -thresh) {
                double oldBound = _upperLimits[i];
                double newBound = Math.min(oldBound + stepSize, _hardUpper[i]);
                if (newBound > oldBound + 1e-15) {
                    _upperLimits[i] = newBound;
                    _expansions.add(new BoundExpansionEvent(i, Direction.UPPER, oldBound, newBound));
                    log.log(System.Logger.Level.INFO,
                            "Dynamic range expansion: parameter {0} upper soft bound {1} → {2} (hard limit {3}).",
                            i, oldBound, newBound, _hardUpper[i]);
                }
            }
            // Lower bound hit and gradient points further downward → expand lower soft bound
            if (Math.abs(step[i] - sl[i]) < 1e-12 && grad[i] > thresh) {
                double oldBound = _lowerLimits[i];
                double newBound = Math.max(oldBound - stepSize, _hardLower[i]);
                if (newBound < oldBound - 1e-15) {
                    _lowerLimits[i] = newBound;
                    _expansions.add(new BoundExpansionEvent(i, Direction.LOWER, oldBound, newBound));
                    log.log(System.Logger.Level.INFO,
                            "Dynamic range expansion: parameter {0} lower soft bound {1} → {2} (hard limit {3}).",
                            i, oldBound, newBound, _hardLower[i]);
                }
            }
        }
    }

    private static double hessNorm(Model model) {
        var h = model.hess();
        double s = 0;
        for (int i = 0; i < model.n(); i++)
            for (int j = 0; j < model.n(); j++)
                s += h.get(i, j) * h.get(i, j);
        return Math.sqrt(s);
    }

    private ModelSnapshot buildSnapshot(Model model) {
        int n = model.n(), npt = model.npt();
        double[][] xptCopy = new double[npt][];
        for (int k = 0; k < npt; k++) xptCopy[k] = model.xpt(k);
        double[] fvalCopy = new double[npt];
        for (int k = 0; k < npt; k++) fvalCopy[k] = model.fValue(k);
        var h = model.hess();
        double[][] hessCopy = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                hessCopy[i][j] = h.get(i, j);
        return new ModelSnapshot(n, npt, model.xbase(), xptCopy, fvalCopy, model.grad(), hessCopy);
    }

    private OptimResult buildResult(Model model, Controller ctrl, ExitFlag flag, String msg, int iter) {
        return new OptimResult(model.xopt(), model.fBest(), _totalEvals, flag, msg,
                List.copyOf(_diagnosticInfos), List.copyOf(_expansions), buildSnapshot(model));
    }

}
