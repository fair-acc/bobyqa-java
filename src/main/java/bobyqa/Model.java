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

import java.util.Arrays;

import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;

/**
 * Quadratic surrogate model for BOBYQA. Maintains npt interpolation points relative to xbase. Stores the implicit
 * Hessian via ZMAT (Z*Z^T factorization). Variable names match Powell's Fortran and the Apache Commons Math port to
 * ease cross-referencing during porting: xpt[k] = interpolation point k relative to xbase (n-vector) fval[k] = f(xbase
 * + xpt[k]) bmat = explicit part of the Lagrange basis matrix zmat = Z factor such that implicit Hessian ~ Z*Z^T kopt =
 * index of current best interpolation point
 */
public class Model {

    private final int n;
    private final int npt;

    // Interpolation points relative to xbase: xpt[k] is a double[n]
    private final double[][] xpt;
    private final double[] fval;
    private double[] xbase;
    private int kopt;

    // ZMAT: npt × (npt - n - 1) matrix for implicit Hessian
    // BMAT: (npt + n) × n matrix for explicit Lagrange basis
    final DMatrixRMaj zmat;
    final DMatrixRMaj bmat;

    // Explicit gradient and Hessian of the quadratic model (at xopt)
    private final double[] grad;
    private final DMatrixRMaj hess; // n×n

    private boolean factorisationCurrent = false;
    private final double rhoStart;

    public Model(int n, int npt, double rhoStart) {
        this.n = n;
        this.npt = npt;
        this.xpt = new double[npt][n];
        this.fval = new double[npt];
        Arrays.fill(this.fval, Double.POSITIVE_INFINITY);
        this.xbase = new double[n];
        this.kopt = 0;
        int nptMinusNMinus1 = npt - n - 1;
        this.zmat = new DMatrixRMaj(npt, Math.max(1, nptMinusNMinus1));
        this.bmat = new DMatrixRMaj(npt + n, n);
        this.grad = new double[n];
        this.hess = new DMatrixRMaj(n, n);
        this.rhoStart = rhoStart;
        // Seed ZMAT(0,0) with 1/rhoStart^2 following Powell's initial factorization scaling.
        // Note: interpolateModel() currently rebuilds grad/hess from scratch via EJML and does
        // not read zmat/bmat, so this seeding has no effect on current solves. It is retained
        // for correctness when incremental factorization updates are added in the future.
        if (nptMinusNMinus1 > 0) {
            zmat.set(0, 0, 1.0 / (rhoStart * rhoStart));
        }
    }

    // --- Point management ---------------------------------------------------

    public void setPoint(int k, double[] xAbs, double fv) {
        for (int i = 0; i < n; i++)
            xpt[k][i] = xAbs[i] - xbase[i];
        fval[k] = fv;
        factorisationCurrent = false;
    }

    public void setFValue(int k, double fv) {
        fval[k] = fv;
    }

    public void setXbase(double[] newXbase) {
        // Shift all stored relative points
        for (int k = 0; k < npt; k++)
            for (int i = 0; i < n; i++)
                xpt[k][i] += xbase[i] - newXbase[i];
        xbase = newXbase.clone();
        factorisationCurrent = false;
    }

    public void updateKopt() {
        kopt = 0;
        for (int k = 1; k < npt; k++)
            if (fval[k] < fval[kopt]) kopt = k;
    }

    public void swapPoints(int i, int j) {
        double[] tmp = xpt[i];
        xpt[i] = xpt[j];
        xpt[j] = tmp;
        double tf = fval[i];
        fval[i] = fval[j];
        fval[j] = tf;
        if (kopt == i)
            kopt = j;
        else if (kopt == j) kopt = i;
        factorisationCurrent = false;
    }

    // --- Accessors ----------------------------------------------------------

    public int n() {
        return n;
    }

    public int npt() {
        return npt;
    }

    public int kopt() {
        return kopt;
    }

    public double[] xopt() {
        double[] r = new double[n];
        for (int i = 0; i < n; i++)
            r[i] = xbase[i] + xpt[kopt][i];
        return r;
    }

    public double[] xbase() {
        return xbase.clone();
    }

    public double[] xpt(int k) {
        return xpt[k].clone();
    }

    public double fValue(int k) {
        return fval[k];
    }

    public double fBest() {
        return fval[kopt];
    }

    public double[] grad() {
        return grad.clone();
    }

    public DMatrixRMaj hess() {
        return hess.copy();
    }

    public void setGrad(double[] g) {
        System.arraycopy(g, 0, grad, 0, n);
    }

    public void setHess(DMatrixRMaj h) {
        hess.setTo(h);
    }

    public void invalidateFactorisation() {
        factorisationCurrent = false;
    }

    /**
     * Returns a single ZMAT entry. Package-private to avoid exposing EJML types in the public API.
     */
    public double zmatGet(int row, int col) {
        return zmat.get(row, col);
    }

    // --- Stubs for Task 6 ---------------------------------------------------

    /**
     * Fit quadratic model to current interpolation points. Populates grad[] and hess[][] by solving the interpolation
     * system.
     * 
     * @param minimumChangeHessian if true, use min-change Hessian for underdetermined case
     */
    public void interpolateModel(boolean minimumChangeHessian) {
        int fullyQuadraticNpt = (n + 1) * (n + 2) / 2;

        // Build displacement vectors from xopt for each point
        double[][] disp = new double[npt][n];
        for (int k = 0; k < npt; k++)
            for (int i = 0; i < n; i++)
                disp[k][i] = xpt[k][i] - xpt[kopt][i];

        // Map row index (0..npt-2) to point index (skipping kopt)
        int[] rowToK = new int[npt - 1];
        int r = 0;
        for (int k = 0; k < npt; k++)
            if (k != kopt) rowToK[r++] = k;

        int p = npt - 1; // number of interpolation constraints (excluding kopt)

        if (npt >= fullyQuadraticNpt) {
            // Fully determined or overdetermined case: standard [g, H] parameterization
            int nQuad = n * (n + 1) / 2;
            int nParams = n + nQuad;
            SimpleMatrix W = new SimpleMatrix(p, nParams);
            SimpleMatrix b = new SimpleMatrix(p, 1);
            for (int i = 0; i < p; i++) {
                int k = rowToK[i];
                for (int j = 0; j < n; j++)
                    W.set(i, j, disp[k][j]);
                int col = n;
                for (int a = 0; a < n; a++)
                    for (int bb = a; bb < n; bb++) {
                        W.set(i, col, (a == bb) ? 0.5 * disp[k][a] * disp[k][bb] : disp[k][a] * disp[k][bb]);
                        col++;
                    }
                b.set(i, 0, fval[k] - fval[kopt]);
            }
            SimpleMatrix coeffs = (p == nParams) ? W.solve(b) : W.pseudoInverse().mult(b);
            for (int j = 0; j < n; j++)
                grad[j] = coeffs.get(j, 0);
            int col = n;
            for (int a = 0; a < n; a++)
                for (int bb = a; bb < n; bb++) {
                    double v = coeffs.get(col++, 0);
                    hess.set(a, bb, v);
                    hess.set(bb, a, v);
                }
        } else {
            // Underdetermined case: Powell's implicit Hessian H = H_old + Σ c_k * d_k * d_k^T
            // System: [Omega, Y^T; Y, 0] * [c; g] = [rhs_c; 0]
            // where Omega[i,j] = 0.5*(d_i·d_j)^2, Y[:,i] = d_i
            // This matches pybobyqa's build_interpolation_matrix (underdetermined case).
            int sysSize = p + n;
            SimpleMatrix A = new SimpleMatrix(sysSize, sysSize);
            SimpleMatrix bvec = new SimpleMatrix(sysSize, 1);

            // Omega block (p×p): A[i,j] = 0.5*(d_i · d_j)^2
            for (int i = 0; i < p; i++) {
                int ki = rowToK[i];
                for (int j = 0; j < p; j++) {
                    int kj = rowToK[j];
                    double dot = 0;
                    for (int dim = 0; dim < n; dim++) dot += disp[ki][dim] * disp[kj][dim];
                    A.set(i, j, 0.5 * dot * dot);
                }
            }

            // Y^T block (p×n): A[i, p+j] = d_i[j]
            // Y block   (n×p): A[p+i, j] = d_j[i]
            for (int i = 0; i < p; i++) {
                int ki = rowToK[i];
                for (int j = 0; j < n; j++) {
                    A.set(i, p + j, disp[ki][j]);
                    A.set(p + j, i, disp[ki][j]);
                }
            }
            // Lower-right (n×n) block stays 0

            // RHS for interpolation rows: f_k - f_kopt, adjusted for min-change Hessian
            for (int i = 0; i < p; i++) {
                int k = rowToK[i];
                double val = fval[k] - fval[kopt];
                if (minimumChangeHessian) {
                    // Subtract old Hessian contribution: 0.5 * d^T H_old d
                    double hContrib = 0;
                    for (int a = 0; a < n; a++)
                        for (int bb = 0; bb < n; bb++)
                            hContrib += hess.get(a, bb) * disp[k][a] * disp[k][bb];
                    val -= 0.5 * hContrib;
                }
                bvec.set(i, 0, val);
            }
            // RHS for gradient rows: 0 (already initialized)

            SimpleMatrix coeffs = A.pseudoInverse().mult(bvec);

            // Extract gradient (last n entries)
            for (int j = 0; j < n; j++)
                grad[j] = coeffs.get(p + j, 0);

            // Update Hessian: H = H_old (if min-change) + Σ c_k * d_k * d_k^T
            if (!minimumChangeHessian) {
                for (int a = 0; a < n; a++)
                    for (int bb = 0; bb < n; bb++)
                        hess.set(a, bb, 0);
            }
            for (int i = 0; i < p; i++) {
                double ck = coeffs.get(i, 0);
                int k = rowToK[i];
                for (int a = 0; a < n; a++)
                    for (int bb = 0; bb < n; bb++)
                        hess.set(a, bb, hess.get(a, bb) + ck * disp[k][a] * disp[k][bb]);
            }
        }
    }

    /**
     * Compute the gradient g and Hessian H of the k-th Lagrange basis polynomial, centered at xopt.
     * For k != kopt the constant term is zero: lambda_k(xopt) = 0.
     * The polynomial satisfies lambda_k(x_j) = delta_{k,j} for all interpolation points j.
     * Fills gOut[0..n-1] and hOut[n×n] in-place.
     * Returns false if the interpolation matrix is singular (caller should fall back).
     */
    boolean lagrangePolynomial(int k, double[] gOut, DMatrixRMaj hOut) {
        int nQuad = n * (n + 1) / 2;
        int nParams = n + nQuad;
        int fullyQuadraticNpt = (n + 1) * (n + 2) / 2;

        double[][] disp = new double[npt][n];
        for (int j = 0; j < npt; j++)
            for (int i = 0; i < n; i++)
                disp[j][i] = xpt[j][i] - xpt[kopt][i];

        int rows = npt - 1;
        SimpleMatrix W = new SimpleMatrix(rows, nParams);
        SimpleMatrix b = new SimpleMatrix(rows, 1);
        int row = 0;
        for (int j = 0; j < npt; j++) {
            if (j == kopt) continue;
            for (int i = 0; i < n; i++)
                W.set(row, i, disp[j][i]);
            int col = n;
            for (int i = 0; i < n; i++)
                for (int jj = i; jj < n; jj++) {
                    W.set(row, col, (i == jj) ? 0.5 * disp[j][i] * disp[j][jj] : disp[j][i] * disp[j][jj]);
                    col++;
                }
            b.set(row, 0, j == k ? 1.0 : 0.0);
            row++;
        }

        try {
            SimpleMatrix coeffs;
            if (npt >= fullyQuadraticNpt) {
                coeffs = W.solve(b);
            } else {
                // Min-norm via SVD pseudoinverse (numerically stable)
                coeffs = W.pseudoInverse().mult(b);
            }
            for (int i = 0; i < n; i++)
                gOut[i] = coeffs.get(i, 0);
            int col = n;
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++) {
                    double val = coeffs.get(col, 0);
                    hOut.set(i, j, val);
                    hOut.set(j, i, val);
                    col++;
                }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Compute the poisedness constant of the interpolation set within radius delta. Returns max over all k of
     * max_{||d||<=delta} |lambda_k(xopt + d)|. Simplified approach: evaluate Lagrange polynomials at candidate points
     * {xopt +/- delta*e_i} and return the maximum absolute value.
     */
    public double poisednessConstant(double delta) {
        // Build the interpolation matrix W (same as in interpolateModel)
        int nLinear = n;
        int nQuad = n * (n + 1) / 2;
        int nParams = nLinear + nQuad;

        double[][] disp = new double[npt][n];
        for (int k = 0; k < npt; k++)
            for (int i = 0; i < n; i++)
                disp[k][i] = xpt[k][i] - xpt[kopt][i];

        // Build full interpolation matrix including kopt row
        // Each row: [linear terms | quadratic terms | 1 for constant]
        // Actually we use: row 0..npt-1, columns: constant(1) + linear(n) + quadratic(n*(n+1)/2)
        int nAll = 1 + nParams; // constant + linear + quadratic
        SimpleMatrix W = new SimpleMatrix(npt, nAll);
        for (int k = 0; k < npt; k++) {
            W.set(k, 0, 1.0); // constant term
            for (int i = 0; i < n; i++)
                W.set(k, 1 + i, disp[k][i]);
            int col = 1 + n;
            for (int i = 0; i < n; i++) {
                for (int j = i; j < n; j++) {
                    double val = (i == j) ? 0.5 * disp[k][i] * disp[k][j] : disp[k][i] * disp[k][j];
                    W.set(k, col, val);
                    col++;
                }
            }
        }

        // If npt < nAll, we need to handle underdetermined basis.
        // For poisedness, we compute Lagrange polynomials via: L = W^{-T} or pseudoinverse
        // Lambda_k(x) = e_k^T * W^{-1} * phi(x)
        // So we need W^{-1} (or pseudoinverse if not square)

        SimpleMatrix Winv;
        if (npt == nAll) {
            Winv = W.invert();
        } else if (npt < nAll) {
            // Underdetermined: use right pseudoinverse W^T (W W^T)^{-1}
            SimpleMatrix Wt = W.transpose();
            Winv = Wt.mult((W.mult(Wt)).invert());
        } else {
            // Overdetermined: use left pseudoinverse (W^T W)^{-1} W^T
            SimpleMatrix Wt = W.transpose();
            Winv = (Wt.mult(W)).invert().mult(Wt);
        }

        // Generate candidate points: xopt +/- delta*e_i (2n candidates)
        // Displacement from xopt is +/- delta*e_i
        double maxLambda = 0.0;
        for (int i = 0; i < n; i++) {
            for (int sign = -1; sign <= 1; sign += 2) {
                // Build phi vector for candidate point
                double[] d = new double[n];
                d[i] = sign * delta;
                double[] phi = new double[nAll];
                phi[0] = 1.0; // constant
                for (int ii = 0; ii < n; ii++)
                    phi[1 + ii] = d[ii];
                int col = 1 + n;
                for (int ii = 0; ii < n; ii++) {
                    for (int jj = ii; jj < n; jj++) {
                        phi[col] = (ii == jj) ? 0.5 * d[ii] * d[jj] : d[ii] * d[jj];
                        col++;
                    }
                }

                // Compute Lagrange values: lambda_k = (Winv * phi)[k]
                // Winv is nAll x npt, but we want npt x 1 result
                // Actually Winv here is nAll x npt when underdetermined, npt x nAll when square/over
                // Let's compute lambda = Winv^T * phi if Winv is nAll x npt
                // No - let's think again.
                // W is npt x nAll. W * coeffs = e_k gives Lagrange poly k.
                // coeffs_k = W^{-1} * e_k (column k of W^{-1})
                // Lambda_k(x) = phi(x)^T * coeffs_k = phi(x)^T * W^{-1} * e_k
                // = (W^{-T} * phi(x))[k]

                SimpleMatrix phiVec = new SimpleMatrix(nAll, 1);
                for (int p = 0; p < nAll; p++)
                    phiVec.set(p, 0, phi[p]);

                SimpleMatrix lambdas = Winv.transpose().mult(phiVec);
                // lambdas is npt x 1 when Winv is nAll x npt... no.
                // Winv is computed differently based on case.
                // Let me reconsider.

                // For square case: W is npt x nAll (npt==nAll). Winv is nAll x npt.
                // Winv^T is npt x nAll. Winv^T * phi is npt x 1. Good.
                // For underdetermined (npt < nAll): Winv is nAll x npt.
                // Winv^T is npt x nAll. Winv^T * phi is npt x 1. Good.

                for (int k = 0; k < npt; k++) {
                    double lam = lambdas.get(k, 0);
                    maxLambda = Math.max(maxLambda, Math.abs(lam));
                }
            }
        }

        return maxLambda;
    }

}
