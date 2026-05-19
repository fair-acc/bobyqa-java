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

class SoftRestartTest {

    @Test void softRestartResetsNsamplesToOne() {
        var params = Params.defaults(1);
        var ctrl = new Controller(x -> 0.0, params, 1.0, 1e-6, null);
        ctrl.incrementNsamples();
        ctrl.incrementNsamples();
        var model = new Model(1, 3, 1.0);
        model.setPoint(0, new double[]{0}, 0.0);
        model.updateKopt();
        ctrl.softRestart(model, model.xopt());
        assertEquals(1, ctrl.nsamples(), "nsamples must reset to 1 after soft restart");
    }

    @Test void softRestartResetsRadii() {
        var params = Params.defaults(1);
        var ctrl = new Controller(x -> 0.0, params, 1.0, 1e-6, null);
        ctrl.setDelta(0.001); ctrl.setRho(0.001);
        var model = new Model(1, 3, 1.0);
        model.setPoint(0, new double[]{0}, 0.0);
        model.updateKopt();
        ctrl.softRestart(model, model.xopt());
        assertEquals(1.0, ctrl.rho(), 1e-10, "rho should reset to scale*rhobeg after soft restart");
    }

    @Test void softRestartRelocatesModelXbaseToXbest() {
        var params = Params.defaults(1);
        var ctrl = new Controller(x -> 0.0, params, 1.0, 1e-6, null);
        var model = new Model(1, 3, 1.0);
        double[] xbest = {5.0};
        model.setPoint(0, xbest, 0.0);
        model.updateKopt();
        ctrl.softRestart(model, xbest);
        // xbase should have been moved to xbest
        assertArrayEquals(xbest, model.xbase(), 1e-10,
            "softRestart must move model.xbase to xbest");
    }

    @Test void solverConvergesAfterSoftRestartWithModelRebuild() {
        // Run optimizer on 1D quadratic with restarts enabled and very tight slow-progress window
        // so a soft restart is triggered early, forcing reinitializeModel to be called
        var params = Params.defaults(1)
            .restarts(r -> r.useRestarts(true))
            .slow(s -> s.historyForSlow(3).threshForSlow(1e-10).maxSlowIters(3));

        double[] lb = {-10.0}, ub = {10.0};
        var result = Bobyqa.solve(x -> (x[0] - 2.0) * (x[0] - 2.0),
            new double[]{0.0}, lb, ub, 300, GoalType.MINIMIZE, params);
        // Optimizer should converge to x=2, f=0 regardless of restart
        assertEquals(2.0, result.x()[0], 0.1, "Should converge to minimum after restart");
        assertTrue(result.f() < 0.01, "Objective should be near zero after restart");
    }
}
