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
package bobyqa.util;

public final class MathUtil {

    private MathUtil() {}

    public static double norm2(double[] v) {
        double s = 0;
        for (double x : v) s += x * x;
        return Math.sqrt(s);
    }

    public static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    public static double distance(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) { double d = a[i] - b[i]; s += d * d; }
        return Math.sqrt(s);
    }

    /**
     * Simple linear regression of y on x.
     * Returns double[]{slope, pearsonCorrelation}.
     */
    public static double[] linreg(double[] x, double[] y) {
        int n = x.length;
        double sx = 0, sy = 0;
        for (int i = 0; i < n; i++) { sx += x[i]; sy += y[i]; }
        double mx = sx / n, my = sy / n;
        double sxx = 0, syy = 0, sxy = 0;
        for (int i = 0; i < n; i++) {
            double dx = x[i] - mx, dy = y[i] - my;
            sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
        }
        double slope = sxx == 0 ? 0 : sxy / sxx;
        double correl = (sxx == 0 || syy == 0) ? 0 : sxy / Math.sqrt(sxx * syy);
        return new double[]{slope, correl};
    }

    /** Clamp x to [lo, hi]. */
    public static double clamp(double x, double lo, double hi) {
        return Math.max(lo, Math.min(hi, x));
    }

    /** Copy of arr with element i replaced by v. */
    public static double[] with(double[] arr, int i, double v) {
        double[] copy = arr.clone();
        copy[i] = v;
        return copy;
    }
}
