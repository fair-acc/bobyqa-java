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

import bobyqa.TrustRegion;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TrustRegionTest {

    private static DMatrixRMaj identity(int n) {
        DMatrixRMaj m = new DMatrixRMaj(n, n);
        CommonOps_DDRM.setIdentity(m);
        return m;
    }

    /** Unconstrained minimum of 0.5*d^T*H*d + g^T*d is -H^{-1}*g.
     *  For H=I, g=[1,0], unconstrained min is d=[-1,0].
     *  If delta=2 and bounds allow, trsbox should return ~[-1,0]. */
    @Test void unconstrainedStepIdentityHessian() {
        int n = 2;
        double[] g = {1.0, 0.0};
        DMatrixRMaj H = identity(n);
        double delta = 2.0;
        double[] sl = {-10, -10}, su = {10, 10};
        double[] step = TrustRegion.trsbox(g, H, delta, sl, su);
        assertEquals(-1.0, step[0], 1e-6);
        assertEquals( 0.0, step[1], 1e-6);
    }

    /** Step must satisfy ||d|| <= delta. */
    @Test void stepRespectsTrustRadius() {
        int n = 3;
        double[] g = {10, 10, 10};
        DMatrixRMaj H = identity(n);
        double delta = 0.5;
        double[] sl = {-1,-1,-1}, su = {1,1,1};
        double[] step = TrustRegion.trsbox(g, H, delta, sl, su);
        double norm = 0;
        for (double d : step) norm += d*d;
        assertTrue(Math.sqrt(norm) <= delta + 1e-8, "Step must lie within trust region");
    }

    /** Step must respect variable bounds. */
    @Test void stepRespectsBounds() {
        int n = 2;
        double[] g = {-5.0, 0.0}; // wants to go in +x direction
        DMatrixRMaj H = identity(n);
        double delta = 2.0;
        double[] sl = {-0.3, -1}, su = {0.3, 1}; // tight bound on x
        double[] step = TrustRegion.trsbox(g, H, delta, sl, su);
        assertTrue(step[0] <= 0.3 + 1e-10, "Step must not exceed upper bound");
        assertTrue(step[0] >= -0.3 - 1e-10, "Step must not violate lower bound");
    }

    /** trsboxGeometry maximizes |m(d)| and result should differ from trsbox. */
    @Test void geometryStepMaximizesAbsModelValue() {
        int n = 2;
        double[] g = {1.0, 0.0};
        DMatrixRMaj H = identity(n);
        double delta = 1.0;
        double[] sl = {-2,-2}, su = {2,2};
        double[] geomStep = TrustRegion.trsboxGeometry(g, H, delta, sl, su);
        // geometry step should maximize |m(d)|; result must stay within trust region + bounds
        double norm = 0;
        for (double d : geomStep) norm += d*d;
        assertTrue(Math.sqrt(norm) <= delta + 1e-8, "Geometry step must not exceed trust radius");
        // For this unconstrained case with identity H the geometry step should reach the boundary
        assertEquals(delta, Math.sqrt(norm), 1e-6);
    }

    @Test void isAtBoundaryReturnsTrueWhenStepHitsBound() {
        double[] step = {-1.0, 0.3};
        double[] sl   = {-1.0, -2.0};
        double[] su   = { 2.0,  2.0};
        assertTrue(TrustRegion.isAtBoundary(step, sl, su),
            "Step at lower bound for dim 0 should be detected");
    }

    @Test void isAtBoundaryReturnsFalseForInteriorStep() {
        double[] step = {0.5, 0.3};
        double[] sl   = {-1.0, -2.0};
        double[] su   = { 2.0,  2.0};
        assertFalse(TrustRegion.isAtBoundary(step, sl, su));
    }
}
