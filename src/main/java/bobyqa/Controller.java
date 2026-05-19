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

import bobyqa.util.MathUtil;
import java.util.ArrayList;
import java.util.List;
import java.util.function.ToDoubleFunction;

/**
 * Manages persistent optimization state and exposes discrete operations.
 * Solver decides WHAT to do; Controller executes HOW.
 */
public class Controller {

    private final ToDoubleFunction<double[]> objective;
    private final Params params;
    private final double rhobeg;
    private final double rhoend;

    // Trust region radii
    private double delta;
    private double rho;

    // Restart tracking
    private int restartCount = 0;
    private int consecutiveUnsuccessfulRestarts = 0;
    private int totalUnsuccessfulRestarts = 0;
    private int nsamples = 1; // dynamic sample count, reset on soft restart

    // Evaluation cache for useOldFk (parallel lists, linear scan)
    private final List<double[]> cachedX = new ArrayList<>();
    private final List<Double>   cachedF = new ArrayList<>();
    private static final double CACHE_TOL = 1e-12;

    // Rolling history for stagnation auto-detection
    private final List<Double> histDelta      = new ArrayList<>();
    private final List<Double> histGradChange = new ArrayList<>();
    private final List<Double> histHessChange = new ArrayList<>();

    // Slow progress tracking
    private final List<Double> fHistory = new ArrayList<>();
    private int slowIterCount = 0;

    // Gate for rho reduction: pybobyqa only reduces rho after 3+ iters since last success
    private int lastSuccessfulIter = -3; // start so doneWithCurrentRho is true at iter 0

    // Noise fallback estimation
    private Double estimatedNoiseFallback = null;
    private int itersSinceNoiseEstimate = 0;
    private boolean noiseFallbackDisabled = false;

    public Controller(ToDoubleFunction<double[]> objective,
                      Params params, double rhobeg, double rhoend,
                      Double initialNoiseEstimate) {
        this.objective = objective;
        this.params    = params;
        this.rhobeg    = rhobeg;
        this.rhoend    = rhoend;
        this.delta     = rhobeg;
        this.rho       = rhobeg;
        this.estimatedNoiseFallback = initialNoiseEstimate;
    }

    // --- Objective evaluation -----------------------------------------------

    /**
     * Evaluate objective nsamples times and return the mean.
     * Checks cache first if useOldFk is enabled.
     * Throws ArithmeticException if result is infinite and checkObjfunForOverflow is set.
     */
    public double evaluateObjective(double[] x, int nSamples) {
        // Cache lookup
        if (params._restartsParameters.hardUseOldFk) {
            Double cached = cacheLookup(x);
            if (cached != null) return cached;
        }
        double sum = 0;
        for (int i = 0; i < nSamples; i++) {
            double f = objective.applyAsDouble(x);
            if (params._generalParameters.checkObjfunForOverflow && !Double.isFinite(f))
                throw new ArithmeticException("Objective returned non-finite value: " + f);
            sum += f;
        }
        double avg = sum / nSamples;
        cacheEvaluation(x, avg);
        return avg;
    }

    public void cacheEvaluation(double[] x, double f) {
        cachedX.add(x.clone());
        cachedF.add(f);
    }

    private Double cacheLookup(double[] x) {
        for (int i = 0; i < cachedX.size(); i++)
            if (MathUtil.distance(cachedX.get(i), x) <= CACHE_TOL)
                return cachedF.get(i);
        return null;
    }

    // --- Trust region radius update -----------------------------------------

    /**
     * Update delta based on ratio of actual to predicted reduction.
     * Uses pybobyqa's eta1/eta2/gamma rules.
     * dnorm = min(||step||, delta) — the effective step length.
     */
    public void updateTrustRadius(double ratio, double dnorm, boolean stepWasAtBoundary) {
        var tr = params._trustRegionRadius;
        if (ratio >= tr.eta2) {
            // Very successful step: grow delta, tracking actual step size
            delta = Math.min(Math.max(tr.gammaInc * delta, tr.gammaIncOverline * dnorm), 1e10);
        } else if (ratio >= tr.eta1) {
            // Acceptable step: keep delta at least as large as the actual step
            delta = Math.max(tr.gammaDec * delta, dnorm);
        } else if (stepWasAtBoundary) {
            // Boundary-truncated step is not a true model minimizer — apply conservative reduction
            // (same rule as acceptable step: keep delta at least as large as the actual step)
            delta = Math.max(tr.gammaDec * delta, dnorm);
        } else {
            // Bad interior step: shrink delta aggressively
            delta = Math.min(tr.gammaDec * delta, dnorm);
        }
        // Cap: if delta is very close to rho, snap to rho to keep them aligned
        if (delta <= 1.5 * rho) delta = rho;
    }

    /** Reduce rho toward rhoend following pybobyqa's schedule. */
    public void reduceRho() {
        var tr = params._trustRegionRadius;
        double ratio = rho / rhoend;
        double oldRho = rho;
        double newRho;
        if (ratio <= 16.0)
            newRho = rhoend;
        else if (ratio <= 250.0)
            newRho = Math.sqrt(ratio) * rhoend; // = sqrt(rho * rhoend), geometric mean
        else
            newRho = tr.alpha1 * rho; // alpha1 = 0.9
        // delta: reset to just below old rho, at least new rho (pybobyqa: max(alpha2*rho_old, new_rho))
        delta = Math.max(tr.alpha2 * oldRho, newRho);
        rho = newRho;
    }

    // --- Stagnation detection -----------------------------------------------

    /** Record per-iteration values for rolling window regression. */
    public void recordIteration(double currentDelta, double gradChange, double hessChange) {
        histDelta.add(currentDelta);
        histGradChange.add(gradChange);
        histHessChange.add(hessChange);
        int maxH = params._restartsParameters.autoDetectHistory;
        if (histDelta.size() > maxH) {
            histDelta.remove(0); histGradChange.remove(0); histHessChange.remove(0);
        }
    }

    /** Returns true if stagnation is detected via regression. */
    public boolean isStagnating() {
        if (!params._restartsParameters.autoDetect) return false;
        int h = params._restartsParameters.autoDetectHistory;
        if (histDelta.size() < h) return false;
        double[] xs = new double[h];
        for (int i = 0; i < h; i++) xs[i] = i;
        double[] deltaArr      = histDelta.stream().mapToDouble(Double::doubleValue).toArray();
        double[] modelChangeArr = new double[h];
        for (int i = 0; i < h; i++)
            modelChangeArr[i] = histGradChange.get(i) + histHessChange.get(i);

        double[] deltaReg = bobyqa.util.MathUtil.linreg(xs, deltaArr);
        double[] modelReg = bobyqa.util.MathUtil.linreg(xs, modelChangeArr);

        // MathUtil.linreg returns [slope, pearsonCorrelation] — index 1 is correlation, not intercept
        boolean deltaShrinking     = deltaReg[0] < 0;
        boolean modelStillChanging = modelReg[0] > params._restartsParameters.autoDetectMinChgModelSlope;
        boolean correlSignificant  = modelReg[1] > params._restartsParameters.autoDetectMinCorrel; // [1] = r

        return deltaShrinking && modelStillChanging && correlSignificant;
    }

    // --- Slow progress detection --------------------------------------------

    public void recordFValue(double f) {
        fHistory.add(f);
        int h = params._slowParameters.historyForSlow;
        if (fHistory.size() > h) fHistory.remove(0);
    }

    public boolean isSlowProgress() {
        int h = params._slowParameters.historyForSlow;
        if (fHistory.size() < h) return false;
        double best  = fHistory.stream().mapToDouble(Double::doubleValue).min().orElse(0);
        double worst = fHistory.stream().mapToDouble(Double::doubleValue).max().orElse(0);
        return (worst - best) < params._slowParameters.threshForSlow;
    }

    public void incrementSlowIter() { slowIterCount++; }
    public void resetSlowIter()     { slowIterCount = 0; }
    public int  slowIterCount()     { return slowIterCount; }

    public void setInitialNoiseEstimate(double sigma) { estimatedNoiseFallback = sigma; }

    // --- Rho reduction gate (mirrors pybobyqa done_with_current_rho) --------

    /** Record that this iteration was successful (ratio > 0 or step > rho). */
    public void recordSuccessfulIter(int iter) { lastSuccessfulIter = iter; }

    /** True if 3+ iters have passed since last success — safe to reduce rho. */
    public boolean doneWithCurrentRho(int iter) { return iter > lastSuccessfulIter + 2; }

    // --- Noise estimation fallback ------------------------------------------

    /** Estimate noise from spread of near-duplicate evaluations in cache. */
    public Double estimateNoiseFallback(double fBest) {
        if (noiseFallbackDisabled) return null;
        // Find any two cached points within 1e-6 of each other
        for (int i = 0; i < cachedX.size(); i++)
            for (int j = i + 1; j < cachedX.size(); j++)
                if (MathUtil.distance(cachedX.get(i), cachedX.get(j)) < 1e-6) {
                    double rawEstimate = Math.abs(cachedF.get(i) - cachedF.get(j)) / 2.0;
                    return Math.abs(fBest) > 1e-15 ? rawEstimate * Math.abs(fBest) : rawEstimate;
                }
        itersSinceNoiseEstimate++;
        if (itersSinceNoiseEstimate > params._slowParameters.historyForSlow) {
            noiseFallbackDisabled = true;
            System.err.println("[BOBYQA] Warning: quitOnNoiseLevel disabled — " +
                               "no reliable noise estimate available.");
        }
        return null;
    }

    // --- Restart operations -------------------------------------------------

    public void softRestart(Model model, double[] xbest) {
        double scale = params._restartsParameters.rhobegScaleAfterUnsuccessfulRestart;
        rho   = scale * rhobeg;
        delta = rho;
        nsamples = 1; // reset sample count
        restartCount++;
        consecutiveUnsuccessfulRestarts++;
        totalUnsuccessfulRestarts++;
        histDelta.clear(); histGradChange.clear(); histHessChange.clear();
        slowIterCount = 0;
        model.setXbase(xbest);           // reposition model origin at best point
        model.invalidateFactorisation();  // mark factorisation stale
    }

    public void hardRestart(Model model, double[] xbest) {
        rho   = rhobeg;
        delta = rhobeg;
        nsamples = 1;
        restartCount++;
        consecutiveUnsuccessfulRestarts = 0;
        totalUnsuccessfulRestarts++;
        histDelta.clear(); histGradChange.clear(); histHessChange.clear();
        slowIterCount = 0;
        model.setXbase(xbest);           // reposition model origin at best point
        model.invalidateFactorisation();  // mark factorisation stale
    }

    public void recordSuccessfulRestart() { consecutiveUnsuccessfulRestarts = 0; }

    // --- Geometry repair ----------------------------------------------------

    /**
     * Check poisedness and reposition the worst interpolation point if needed.
     * Returns true if a geometry step was taken.
     */
    public boolean checkAndFixGeometry(Model model, TrustRegion tr,
                                       double[] lower, double[] upper) {
        double poisedness = model.poisednessConstant(delta);
        if (poisedness < 100.0) return false; // geometry is acceptable
        // Find the interpolation point with largest Lagrange polynomial value
        // and reposition it via a geometry step
        int worst = findWorstPoint(model);
        double[] sl = new double[model.n()], su = new double[model.n()];
        double[] xopt = model.xopt();
        for (int i = 0; i < model.n(); i++) {
            sl[i] = lower[i] - xopt[i];
            su[i] = upper[i] - xopt[i];
        }
        double[] step = TrustRegion.trsboxGeometry(model.grad(), model.hess(), delta, sl, su);
        double[] xnew = new double[model.n()];
        for (int i = 0; i < model.n(); i++) xnew[i] = xopt[i] + step[i];
        double fNew = evaluateObjective(xnew, nsamples);
        model.setPoint(worst, xnew, fNew);
        model.updateKopt();
        return true;
    }

    private int findWorstPoint(Model model) {
        // Return the index of the interpolation point furthest from xopt
        int worst = 0;
        double maxDist = 0;
        double[] xopt = model.xopt();
        for (int k = 0; k < model.npt(); k++) {
            if (k == model.kopt()) continue;
            double d = MathUtil.distance(model.xopt(), absolutePoint(model, k));
            if (d > maxDist) { maxDist = d; worst = k; }
        }
        return worst;
    }

    private double[] absolutePoint(Model model, int k) {
        double[] xbase = model.xbase();
        double[] rel   = model.xpt(k);
        double[] abs   = new double[model.n()];
        for (int i = 0; i < model.n(); i++) abs[i] = xbase[i] + rel[i];
        return abs;
    }

    // --- Accessors / mutators -----------------------------------------------

    public double delta()       { return delta; }
    public double rho()         { return rho; }
    public int    restartCount(){ return restartCount; }
    public int    nsamples()    { return nsamples; }
    public void   setDelta(double v) { delta = v; }
    public void   setRho(double v)   { rho = v; }
    public void   incrementNsamples() { nsamples++; }

    public boolean restartsExhausted() {
        return totalUnsuccessfulRestarts >= params._restartsParameters.maxUnsuccessfulRestartsTotal;
    }
    public boolean consecutiveRestartsExhausted() {
        return consecutiveUnsuccessfulRestarts >= params._restartsParameters.maxUnsuccessfulRestarts;
    }
}
