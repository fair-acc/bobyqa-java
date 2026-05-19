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
package bobyqa.restarts;

import bobyqa.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class HardRestartTest {

    @Test void hardRestartRecyclesEvaluationsFromCache() {
        var params = Params.defaults(1);
        int[] callCount = {0};
        var ctrl = new Controller(x -> { callCount[0]++; return 7.0; },
                                  params, 1.0, 1e-6, null);
        ctrl.cacheEvaluation(new double[]{0.5}, 3.14);
        int callsBefore = callCount[0];
        double v = ctrl.evaluateObjective(new double[]{0.5 + 5e-13}, 1);
        assertEquals(3.14, v, 1e-12, "Should return cached value without calling objective");
        assertEquals(callsBefore, callCount[0], "Objective must not be called on cache hit");
    }

    @Test void hardRestartResetsConsecutiveCount() {
        var params = Params.defaults(1);
        var ctrl = new Controller(x -> 0.0, params, 1.0, 1e-6, null);
        var model = new Model(1, 3, 1.0);
        model.setPoint(0, new double[]{0}, 0.0);
        model.updateKopt();
        ctrl.softRestart(model, model.xopt());
        ctrl.hardRestart(model, model.xopt());
        assertFalse(ctrl.consecutiveRestartsExhausted(),
            "Consecutive restart counter should reset after hard restart");
    }

    @Test void hardRestartRelocatesModelXbaseToXbest() {
        var params = Params.defaults(1);
        var ctrl = new Controller(x -> 0.0, params, 1.0, 1e-6, null);
        var model = new Model(1, 3, 1.0);
        double[] xbest = {3.0};
        model.setPoint(0, xbest, 0.0);
        model.updateKopt();
        ctrl.hardRestart(model, xbest);
        assertArrayEquals(xbest, model.xbase(), 1e-10,
            "hardRestart must move model.xbase to xbest");
    }
}
