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

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SolverSmokeTest {

    /** Simple 2D quadratic: f(x,y)=(x-1)^2+(y-2)^2, minimum at (1,2). */
    @Test void minimize2DQuadratic() {
        var result = Bobyqa.solve(
            x -> Math.pow(x[0] - 1, 2) + Math.pow(x[1] - 2, 2),
            new double[]{0.0, 0.0},
            new double[]{-5.0, -5.0},
            new double[]{ 5.0,  5.0},
            500, GoalType.MINIMIZE, Params.defaults(2));
        assertArrayEquals(new double[]{1.0, 2.0}, result.x(), 1e-4);
        assertEquals(0.0, result.f(), 1e-8);
        assertEquals(ExitFlag.CONVERGED, result.exitFlag());
    }

    /** Maximize f(x,y) = -(x^2+y^2): maximum at (0,0). */
    @Test void maximize2DQuadratic() {
        var result = Bobyqa.solve(
            x -> -(x[0]*x[0] + x[1]*x[1]),
            new double[]{2.0, 2.0},
            new double[]{-5.0, -5.0},
            new double[]{ 5.0,  5.0},
            500, GoalType.MAXIMIZE, Params.defaults(2));
        assertArrayEquals(new double[]{0.0, 0.0}, result.x(), 1e-4);
        assertEquals(0.0, result.f(), 1e-6);
    }

    /** Budget exhaustion exits cleanly. */
    @Test void budgetExhausted() {
        var result = Bobyqa.solve(
            x -> Math.pow(x[0] - 3, 2),
            new double[]{0.0},
            new double[]{-10.0},
            new double[]{ 10.0},
            3, GoalType.MINIMIZE, Params.defaults(1)); // 3 evals = exact init cost for n=1,npt=3
        assertEquals(ExitFlag.BUDGET_EXHAUSTED, result.exitFlag());
    }

    /** INPUT_ERROR on invalid npt. */
    @Test void inputErrorOnBadNpt() {
        var params = Params.defaults(2).model(m -> m.npt(2)); // below n+2=4
        var result = Bobyqa.solve(x -> 0.0, new double[]{0,0},
                                  new double[]{-1,-1}, new double[]{1,1},
                                  100, GoalType.MINIMIZE, params);
        assertEquals(ExitFlag.INPUT_ERROR, result.exitFlag());
    }

    /** scaleToBounds / unscaleFromBounds roundtrip. */
    @Test void scalingRoundtrip() {
        double[] x = {3.0, -1.0};
        double[] lo = {0.0, -2.0}, hi = {6.0, 2.0};
        double[] scaled   = Bobyqa.scaleToBounds(x, lo, hi);
        double[] unscaled = Bobyqa.unscaleFromBounds(scaled, lo, hi);
        assertArrayEquals(x, unscaled, 1e-12);
    }
}
