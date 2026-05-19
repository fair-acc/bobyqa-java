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

class MultiplicativeNoiseTest {

    @Test void findsMinimumDespiteMultiplicativeNoise() {
        var rng = new Random(123);
        double noiseLevel = 0.02;
        var result = Bobyqa.solve(
            x -> {
                double f = Math.pow(x[0] - 3.0, 2);
                return f * (1.0 + noiseLevel * (rng.nextDouble() - 0.5));
            },
            new double[]{0.0},
            new double[]{-5.0},
            new double[]{ 8.0},
            300, GoalType.MINIMIZE,
            Params.defaults(1)
                .noise(n -> n.objfunHasNoise(true).multiplicativeNoiseLevel(noiseLevel)));
        assertEquals(3.0, result.x()[0], 0.2,
            "Should find x near 3 despite multiplicative noise");
    }
}
