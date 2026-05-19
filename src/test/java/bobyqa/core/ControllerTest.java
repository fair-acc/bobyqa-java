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

import bobyqa.*;
import org.junit.jupiter.api.Test;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

class ControllerTest {

    @Test void evaluateObjectiveAveragesNSamples() {
        var counter = new AtomicInteger(0);
        // objective returns counter value each call: 1, 2, 3
        var ctrl = new Controller(x -> (double) counter.incrementAndGet(),
                                  Params.defaults(1), 1.0, 1e-4, null);
        double avg = ctrl.evaluateObjective(new double[]{0.0}, 3);
        assertEquals(2.0, avg, 1e-12); // (1+2+3)/3
        assertEquals(3, counter.get());
    }

    @Test void evaluateObjectiveChecksOverflow() {
        // Explicitly enable overflow check so test is robust to default changes
        var params = Params.defaults(1).general(g -> new Params.GeneralParams(
            g.roundingErrorConstant, g.safetyStepThresh, true));
        var ctrl = new Controller(x -> Double.POSITIVE_INFINITY, params, 1.0, 1e-4, null);
        assertThrows(ArithmeticException.class,
            () -> ctrl.evaluateObjective(new double[]{0}, 1));
    }

    @Test void radiusUpdateSuccessfulStep() {
        var params = Params.defaults(2);
        var ctrl = new Controller(x -> 0.0, params, 1.0, 1e-6, null);
        ctrl.setDelta(0.5);
        ctrl.setRho(0.1);
        // ratio > eta2 (0.7) → increase delta; dnorm = 0.5 (step at trust region boundary)
        ctrl.updateTrustRadius(0.9, 0.5, false);
        assertTrue(ctrl.delta() > 0.5, "Delta should increase on good step");
    }

    @Test void radiusUpdateUnsuccessfulStep() {
        var params = Params.defaults(2);
        var ctrl = new Controller(x -> 0.0, params, 1.0, 1e-6, null);
        ctrl.setDelta(0.5);
        ctrl.setRho(0.1);
        // ratio < eta1 (0.1), interior step (not at boundary): min(gammaDec*delta, dnorm)
        // gammaDec=0.5, delta=0.5 → 0.25; dnorm=0.1 → min(0.25,0.1)=0.1
        ctrl.updateTrustRadius(0.05, 0.1, false);
        assertEquals(0.1, ctrl.delta(), 1e-10, "Interior bad step should use min(gammaDec*delta, dnorm)");
    }

    @Test void oldFkCacheHitsOnNearPoint() {
        var ctrl = new Controller(x -> 99.0,
                                  Params.defaults(1), 1.0, 1e-4, null);
        ctrl.cacheEvaluation(new double[]{1.0}, 42.0);
        // same point → should return cached value, not call objective
        double v = ctrl.evaluateObjective(new double[]{1.0 + 1e-13}, 1);
        assertEquals(42.0, v, 1e-12);
    }

    @Test void estimateNoiseFallbackScalesByFBest() {
        // Two near-duplicate x's with differing f values → raw estimate = |Δf|/2
        var ctrl = new Controller(x -> 0.0, Params.defaults(1), 1.0, 1e-6, null);
        ctrl.cacheEvaluation(new double[]{0.0}, 100.0);
        ctrl.cacheEvaluation(new double[]{1e-7}, 102.0); // distance < 1e-6, Δf = 2
        // raw = 1.0; fBest = 50.0 → scaled = 1.0 * 50.0 = 50.0
        Double estimate = ctrl.estimateNoiseFallback(50.0);
        assertNotNull(estimate);
        assertEquals(50.0, estimate, 1e-10);
    }

    @Test void estimateNoiseFallbackZeroFBestIsUnscaled() {
        var ctrl = new Controller(x -> 0.0, Params.defaults(1), 1.0, 1e-6, null);
        ctrl.cacheEvaluation(new double[]{0.0}, 0.4);
        ctrl.cacheEvaluation(new double[]{1e-7}, 0.6); // Δf = 0.2, raw = 0.1
        // fBest ≈ 0 → return raw estimate unchanged
        Double estimate = ctrl.estimateNoiseFallback(0.0);
        assertNotNull(estimate);
        assertEquals(0.1, estimate, 1e-10);
    }

    @Test void radiusUpdateBoundaryStepUsesConservativeReduction() {
        var params = Params.defaults(2);
        var ctrl = new Controller(x -> 0.0, params, 1.0, 1e-6, null);
        ctrl.setDelta(0.5);
        ctrl.setRho(0.05);
        // ratio < eta1 (0.1) → bad step, but stepWasAtBoundary=true
        // gammaDec = 0.5, delta = 0.5 → gammaDec*delta = 0.25, dnorm = 0.4
        // conservative (boundary): max(0.25, 0.4) = 0.4
        // aggressive (interior):   min(0.25, 0.4) = 0.25
        ctrl.updateTrustRadius(0.02, 0.4, true);
        assertEquals(0.4, ctrl.delta(), 1e-10,
            "Boundary step should use conservative max(gammaDec*delta, dnorm)");
    }
}
