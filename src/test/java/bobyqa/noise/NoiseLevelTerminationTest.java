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
import static org.junit.jupiter.api.Assertions.*;

class NoiseLevelTerminationTest {

    @Test void noiseTerminationDisabledWhenBothLevelsNullAndNoEstimate() {
        var result = Bobyqa.solve(
            x -> x[0]*x[0],
            new double[]{1.0},
            new double[]{-3.0},
            new double[]{ 3.0},
            200, GoalType.MINIMIZE,
            Params.defaults(1).noise(n -> n.objfunHasNoise(true).quitOnNoiseLevel(true)));
        assertNotEquals(ExitFlag.INPUT_ERROR, result.exitFlag());
        assertTrue(result.f() < 0.01, "Should still find approximate minimum");
    }

    @Test void bothNoiseLevelsUsedInFormula() {
        double addNoise = 0.01, multNoise = 0.001;
        double fBest = 10.0;
        var result = Bobyqa.solve(
            x -> fBest + 0.001 * x[0],
            new double[]{0.0},
            new double[]{-1.0},
            new double[]{ 1.0},
            200, GoalType.MINIMIZE,
            Params.defaults(1).noise(n -> n
                .objfunHasNoise(true)
                .additiveNoiseLevel(addNoise)
                .multiplicativeNoiseLevel(multNoise)
                .quitOnNoiseLevel(true)
                .scaleFactorForQuit(10.0)));
        assertNotEquals(ExitFlag.INPUT_ERROR, result.exitFlag());
    }
}
