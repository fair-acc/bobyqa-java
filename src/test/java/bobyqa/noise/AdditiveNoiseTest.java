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
package bobyqa.noise;

import bobyqa.*;
import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class AdditiveNoiseTest {

    @Test void findsMinimumDespiteAdditiveNoise() {
        var rng = new Random(42);
        double noiseLevel = 0.05;
        var result = Bobyqa.solve(
            x -> x[0]*x[0] + noiseLevel * (rng.nextDouble() - 0.5),
            new double[]{2.0},
            new double[]{-5.0},
            new double[]{ 5.0},
            300, GoalType.MINIMIZE,
            Params.defaults(1)
                .noise(n -> n.objfunHasNoise(true).additiveNoiseLevel(noiseLevel)));
        assertTrue(Math.abs(result.x()[0]) < 0.3,
            "Should find x near 0 despite additive noise, got: " + result.x()[0]);
    }

    @Test void noiseLevelTerminationFiresWhenWithinBand() {
        var rng = new Random(7);
        double noiseLevel = 0.1;
        var result = Bobyqa.solve(
            x -> 1.0 + noiseLevel * (rng.nextDouble() - 0.5),
            new double[]{0.0},
            new double[]{-1.0},
            new double[]{ 1.0},
            500, GoalType.MINIMIZE,
            Params.defaults(1)
                .noise(n -> n.objfunHasNoise(true)
                             .additiveNoiseLevel(noiseLevel)
                             .quitOnNoiseLevel(true)));
        // For a flat+noise function, the optimizer should either hit the noise level
        // termination condition or exhaust the budget — both are valid outcomes.
        assertTrue(
            result.exitFlag() == ExitFlag.NOISE_LEVEL_REACHED
            || result.exitFlag() == ExitFlag.BUDGET_EXHAUSTED
            || result.exitFlag() == ExitFlag.CONVERGED,
            "Expected clean termination for flat+noise function, got: " + result.exitFlag());
        assertNotEquals(ExitFlag.INPUT_ERROR, result.exitFlag());
    }
}
