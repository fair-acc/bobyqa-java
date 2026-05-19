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

import org.ejml.data.DMatrixRMaj;

/**
 * Bound-constrained trust region subproblem solver (trsbox).
 * <p>
 * Solves: min g'd + 0.5 d'Hd  subject to  ||d|| &lt;= delta,  sl &lt;= d &lt;= su
 * <p>
 * Uses a truncated conjugate gradient method following Powell (2009) / DFBOLS.
 * Variables that hit their bounds are fixed and the CG iteration continues
 * in the subspace of free variables.
 */
public final class TrustRegion {

    private TrustRegion() {}

    /**
     * Solve the bound-constrained trust region subproblem.
     *
     * @param g     gradient vector (length n)
     * @param H     Hessian matrix (n x n, symmetric)
     * @param delta trust region radius (&gt; 0)
     * @param sl    lower bounds relative to current point (sl = lower - xopt)
     * @param su    upper bounds relative to current point (su = upper - xopt)
     * @return step d that approximately minimizes g'd + 0.5 d'Hd within constraints
     */
    public static double[] trsbox(double[] g, DMatrixRMaj H,
                                  double delta, double[] sl, double[] su) {
        final int n = g.length;
        double[] d = new double[n];   // current step
        double[] s = new double[n];   // search direction
        double[] hs = new double[n];  // H * s
        double[] r = new double[n];   // residual (negative gradient of quadratic at d)
        boolean[] fixed = new boolean[n]; // true if variable is fixed at a bound

        // Clamp initial point to bounds (d starts at 0, should be in [sl, su])
        // and set up initial search direction s = -g (steepest descent)
        for (int i = 0; i < n; i++) {
            // Clamp d to bounds
            d[i] = Math.max(sl[i], Math.min(su[i], 0.0));
        }

        // Compute residual r = -(g + H*d). Since d may be nonzero after clamping,
        // we need to compute the gradient of the quadratic at d.
        // gradient at d = g + H*d
        for (int i = 0; i < n; i++) {
            double hd_i = 0;
            for (int j = 0; j < n; j++) {
                hd_i += H.get(i, j) * d[j];
            }
            r[i] = -(g[i] + hd_i);
        }

        // Fix variables at bounds: if at lower bound and gradient pushes further down,
        // or at upper bound and gradient pushes further up, fix them.
        for (int i = 0; i < n; i++) {
            if (d[i] <= sl[i] + 1e-15 && r[i] < 0) {
                fixed[i] = true;
                r[i] = 0;
            } else if (d[i] >= su[i] - 1e-15 && r[i] > 0) {
                fixed[i] = true;
                r[i] = 0;
            }
        }

        // Initialize search direction
        for (int i = 0; i < n; i++) {
            s[i] = fixed[i] ? 0.0 : r[i];
        }

        double rr = dot(r, r); // ||r||^2

        // If gradient is zero, nothing to do
        if (rr < 1e-30) {
            return d;
        }

        int maxIter = 2 * n * n + 10 * n; // generous iteration limit

        for (int iter = 0; iter < maxIter; iter++) {
            // Compute H * s
            for (int i = 0; i < n; i++) {
                hs[i] = 0;
                for (int j = 0; j < n; j++) {
                    hs[i] += H.get(i, j) * s[j];
                }
            }

            double sHs = dot(s, hs);
            double ss = dot(s, s);
            double ds = dot(d, s);
            double dd = dot(d, d);

            if (ss < 1e-30) break;

            // Compute maximum step alpha along s before hitting trust region boundary
            // ||d + alpha*s||^2 = delta^2
            // dd + 2*alpha*ds + alpha^2*ss = delta^2
            double alphaTR = Double.MAX_VALUE;
            double disc = ds * ds - ss * (dd - delta * delta);
            if (disc > 0) {
                alphaTR = (-ds + Math.sqrt(disc)) / ss;
                if (alphaTR < 0) alphaTR = 0;
            } else if (dd >= delta * delta) {
                // Already at or beyond trust region
                break;
            }

            // CG step length
            double alphaCG = (sHs > 0) ? rr / sHs : alphaTR;

            // Compute maximum alpha before hitting any bound
            double alphaBound = alphaTR;
            int boundIdx = -1;
            boolean hitLower = false;
            for (int i = 0; i < n; i++) {
                if (fixed[i] || Math.abs(s[i]) < 1e-30) continue;
                double alphaLo = (sl[i] - d[i]) / s[i];
                double alphaUp = (su[i] - d[i]) / s[i];
                double alphaB;
                boolean isLower;
                if (s[i] > 0) {
                    alphaB = alphaUp;
                    isLower = false;
                } else {
                    alphaB = alphaLo;
                    isLower = true;
                }
                if (alphaB < alphaBound && alphaB > 0) {
                    alphaBound = alphaB;
                    boundIdx = i;
                    hitLower = isLower;
                }
            }

            // Choose the minimum step length
            double alpha = Math.min(alphaCG, Math.min(alphaTR, alphaBound));
            alpha = Math.max(alpha, 0);

            // Take the step
            for (int i = 0; i < n; i++) {
                d[i] += alpha * s[i];
                // Clamp for safety
                d[i] = Math.max(sl[i], Math.min(su[i], d[i]));
            }

            // Check which constraint was hit
            if (alpha >= alphaTR - 1e-15 && alphaTR <= alphaCG && alphaTR <= alphaBound) {
                // Hit trust region boundary - we're done
                break;
            }

            if (alpha >= alphaBound - 1e-15 && boundIdx >= 0 && alphaBound <= alphaCG) {
                // Hit a bound - fix variable and restart CG in subspace
                fixed[boundIdx] = true;
                if (hitLower) {
                    d[boundIdx] = sl[boundIdx];
                } else {
                    d[boundIdx] = su[boundIdx];
                }

                // Recompute residual r = -(g + H*d) for free variables
                for (int i = 0; i < n; i++) {
                    if (fixed[i]) {
                        r[i] = 0;
                        continue;
                    }
                    double hd_i = 0;
                    for (int j = 0; j < n; j++) {
                        hd_i += H.get(i, j) * d[j];
                    }
                    r[i] = -(g[i] + hd_i);
                    // If at bound and residual pushes past it, fix
                    if (d[i] <= sl[i] + 1e-15 && r[i] < 0) {
                        fixed[i] = true;
                        r[i] = 0;
                    } else if (d[i] >= su[i] - 1e-15 && r[i] > 0) {
                        fixed[i] = true;
                        r[i] = 0;
                    }
                }

                rr = dot(r, r);
                if (rr < 1e-30) break;

                // Restart CG: s = r
                for (int i = 0; i < n; i++) {
                    s[i] = fixed[i] ? 0.0 : r[i];
                }
                continue;
            }

            // Normal CG step completed (alphaCG was smallest)
            // Update residual: r = r - alpha * H*s
            double rrOld = rr;
            for (int i = 0; i < n; i++) {
                if (fixed[i]) {
                    r[i] = 0;
                } else {
                    r[i] -= alpha * hs[i];
                }
            }
            rr = dot(r, r);

            // Check convergence
            if (rr < 1e-30) break;

            // Update search direction: s = r + beta * s
            double beta = rr / rrOld;
            for (int i = 0; i < n; i++) {
                if (fixed[i]) {
                    s[i] = 0;
                } else {
                    s[i] = r[i] + beta * s[i];
                    // Project s: if d is at bound and s would push past it, zero out
                    if (d[i] <= sl[i] + 1e-15 && s[i] < 0) {
                        s[i] = 0;
                    } else if (d[i] >= su[i] - 1e-15 && s[i] > 0) {
                        s[i] = 0;
                    }
                }
            }
        }

        return d;
    }

    /**
     * Find the geometry-improving step that maximizes |m(d)| within the trust region
     * and bounds. Solves both min m(d) and min -m(d), returning whichever gives
     * larger absolute model value.
     *
     * @param g     gradient vector
     * @param H     Hessian matrix (full, symmetric)
     * @param delta trust region radius
     * @param sl    lower bounds relative to current point
     * @param su    upper bounds relative to current point
     * @return step d that maximizes |g'd + 0.5 d'Hd|
     */
    public static double[] trsboxGeometry(double[] g, DMatrixRMaj H,
                                          double delta, double[] sl, double[] su) {
        int n = g.length;
        double[] dMin = trsbox(g, H, delta, sl, su);

        // Negate FULL H matrix and g for maximization
        double[] gNeg = new double[n];
        DMatrixRMaj hNeg = new DMatrixRMaj(n, n);
        for (int i = 0; i < n; i++) {
            gNeg[i] = -g[i];
            for (int j = 0; j < n; j++) {
                hNeg.set(i, j, -H.get(i, j));
            }
        }
        double[] dMax = trsbox(gNeg, hNeg, delta, sl, su);

        double mMin = modelValue(g, H, dMin);
        double mMax = modelValue(g, H, dMax);
        return Math.abs(mMax) >= Math.abs(mMin) ? dMax : dMin;
    }

    /**
     * Compute the quadratic model value m(d) = g'd + 0.5 d'Hd.
     */
    static double modelValue(double[] g, DMatrixRMaj H, double[] d) {
        int n = g.length;
        double v = 0;
        for (int i = 0; i < n; i++) {
            v += g[i] * d[i];
            for (int j = 0; j < n; j++) {
                v += 0.5 * H.get(i, j) * d[i] * d[j];
            }
        }
        return v;
    }

    /**
     * Returns true if any component of step is within 1e-12 of its bound,
     * indicating the CG solver was truncated by a variable bound.
     */
    public static boolean isAtBoundary(double[] step, double[] sl, double[] su) {
        for (int i = 0; i < step.length; i++) {
            if (Math.abs(step[i] - sl[i]) < 1e-12 || Math.abs(step[i] - su[i]) < 1e-12)
                return true;
        }
        return false;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }
}
