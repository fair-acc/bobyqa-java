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
package bobyqa.core;

import bobyqa.Model;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ModelTest {

    @Test void constructionSetsNpt() {
        // n=2, npt=5 (2n+1)
        var m = new Model(2, 5, 1.0);
        assertEquals(5, m.npt());
        assertEquals(2, m.n());
    }

    @Test void addPointAndGetValue() {
        var m = new Model(2, 5, 1.0);
        m.setPoint(0, new double[]{0.0, 0.0}, 42.0);
        assertEquals(42.0, m.fValue(0), 1e-12);
    }

    @Test void koptTracksMinimum() {
        var m = new Model(2, 5, 1.0);
        m.setPoint(0, new double[]{0.0, 0.0}, 10.0);
        m.setPoint(1, new double[]{1.0, 0.0}, 5.0);
        m.setPoint(2, new double[]{0.0, 1.0}, 8.0);
        m.updateKopt();
        assertEquals(1, m.kopt());
        assertArrayEquals(new double[]{1.0, 0.0}, m.xopt(), 1e-12);
    }

    @Test void fBestReturnsMinimum() {
        var m = new Model(2, 5, 1.0);
        m.setPoint(0, new double[]{0,0}, 3.0);
        m.setPoint(1, new double[]{1,0}, 1.0);
        m.updateKopt();
        assertEquals(1.0, m.fBest(), 1e-12);
    }

    @Test void xbaseShiftingKeepsRelativePoints() {
        var m = new Model(2, 5, 1.0);
        m.setPoint(0, new double[]{2.0, 3.0}, 0.0);
        m.setXbase(new double[]{2.0, 3.0});
        // relative coords of point 0 should be {0,0}
        assertArrayEquals(new double[]{0.0, 0.0}, m.xpt(0), 1e-12);
    }

    @Test void interpolateModelFullyQuadratic() {
        // 1D fully quadratic: n=1, npt=3=(n+1)(n+2)/2
        // Points at x={-1,0,1} relative to xbase=0
        // f values: f(-1)=1, f(0)=0, f(1)=1  -> minimum at x=0
        var m = new Model(1, 3, 1.0);
        m.setPoint(0, new double[]{-1.0}, 1.0);
        m.setPoint(1, new double[]{ 0.0}, 0.0);
        m.setPoint(2, new double[]{ 1.0}, 1.0);
        m.updateKopt();
        m.interpolateModel(true); // minimumChangeHessian=true
        // gradient at xopt (index 1) should be ~0
        assertEquals(0.0, m.grad()[0], 1e-8);
        // Hessian should be ~2 (second derivative of x^2)
        assertEquals(2.0, m.hess().get(0,0), 1e-8);
    }

    @Test void poisednessIsPositive() {
        var m = new Model(2, 5, 1.0);
        m.setPoint(0, new double[]{0,0}, 0.0);
        m.setPoint(1, new double[]{1,0}, 1.0);
        m.setPoint(2, new double[]{0,1}, 1.0);
        m.setPoint(3, new double[]{-1,0}, 1.0);
        m.setPoint(4, new double[]{0,-1}, 1.0);
        m.updateKopt();
        m.interpolateModel(true);
        double p = m.poisednessConstant(1.0);
        assertTrue(p > 0, "Poisedness must be positive");
    }

    @Test void modelConstructorSeedsZmatWithRhoStartScaling() {
        // Architectural note: ZMAT(0,0) is seeded here for future incremental-update code paths;
        // current interpolateModel() rebuilds matrices from scratch so this value is not yet read.
        // n=2, npt=5 → ZMAT is 5×2 (npt-n-1=2 columns)
        var model = new Model(2, 5, 0.5);
        // ZMAT(0,0) should be 1/(0.5^2) = 4.0
        assertEquals(4.0, model.zmatGet(0, 0), 1e-10,
            "ZMAT(0,0) must be seeded with 1/rhoStart^2 at construction");
    }
}
