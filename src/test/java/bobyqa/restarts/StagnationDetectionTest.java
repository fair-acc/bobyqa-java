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

class StagnationDetectionTest {

    @Test void detectsStagnationWhenDeltaShrinksAndModelChanges() {
        var params = Params.defaults(1)
            .restarts(r -> r.autoDetect(true).maxUnsuccessfulRestarts(10));
        var ctrl = new Controller(x -> 0.0, params, 1.0, 1e-6, null);
        int h = params._restartsParameters.autoDetectHistory;
        for (int i = 0; i < h; i++) {
            double delta = 0.5 - i * 0.01;
            double modelChg = 0.1 + i * 0.05;  // slope=0.05 > autoDetectMinChgModelSlope=0.015
            ctrl.recordIteration(delta, modelChg / 2, modelChg / 2);
        }
        assertTrue(ctrl.isStagnating(), "Should detect stagnation");
    }

    @Test void noStagnationWhenDeltaGrowing() {
        var params = Params.defaults(1).restarts(r -> r.autoDetect(true));
        var ctrl = new Controller(x -> 0.0, params, 1.0, 1e-6, null);
        int h = params._restartsParameters.autoDetectHistory;
        for (int i = 0; i < h; i++) {
            double delta = 0.1 + i * 0.02;
            ctrl.recordIteration(delta, 0.05, 0.05);
        }
        assertFalse(ctrl.isStagnating(), "Should not detect stagnation when delta grows");
    }

    @Test void slowProgressTriggerWithHistory() {
        var params = Params.defaults(1)
            .slow(s -> s.historyForSlow(5).threshForSlow(1e-8));
        var ctrl = new Controller(x -> 0.0, params, 1.0, 1e-6, null);
        for (int i = 0; i < 5; i++) ctrl.recordFValue(1.0);
        assertTrue(ctrl.isSlowProgress(), "Should detect slow progress on flat history");
    }
}
