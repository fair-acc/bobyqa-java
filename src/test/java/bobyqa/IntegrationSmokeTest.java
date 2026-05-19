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

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * End-to-end smoke test: minimise the Rosenbrock banana function defined
 * inline. Proves the optimizer converges from a standard start without
 * depending on any external test-function library.
 */
class IntegrationSmokeTest {

    /** f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2 — global minimum 0 at (1, 1). */
    private static double rosenbrock(double[] x) {
        final double a = 1.0 - x[0];
        final double b = x[1] - x[0] * x[0];
        return a * a + 100.0 * b * b;
    }

    @Test
    void minimisesRosenbrockFromStandardStart() {
        final double[] x0    = { -1.2,  1.0 };
        final double[] lower = { -5.0, -5.0 };
        final double[] upper = {  5.0,  5.0 };
        final int      maxEvals = 2000;

        OptimResult result = Bobyqa.solve(
                IntegrationSmokeTest::rosenbrock,
                x0, lower, upper, maxEvals,
                GoalType.MINIMIZE,
                Params.defaults(2));

        assertTrue(Math.abs(result.x()[0] - 1.0) < 1e-5,
                "x[0] should be ≈ 1.0, was " + result.x()[0]);
        assertTrue(Math.abs(result.x()[1] - 1.0) < 1e-5,
                "x[1] should be ≈ 1.0, was " + result.x()[1]);
        assertTrue(result.f() < 1e-10,
                "f should be ≈ 0, was " + result.f());
        assertTrue(result.nEvals() <= maxEvals,
                "should respect budget, used " + result.nEvals());
    }
}
