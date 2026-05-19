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
package bobyqa.dynamicranges;

import bobyqa.Bobyqa;
import bobyqa.BoundExpansionEvent;
import bobyqa.Direction;
import bobyqa.GoalType;
import bobyqa.HardBoundViolationException;
import bobyqa.Params;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class DynamicRangesTest {

    @Test
    void hardBoundViolationException_storesFields() {
        var ex = new HardBoundViolationException(2, 3.5, Direction.UPPER);
        assertEquals(2, ex.paramIndex);
        assertEquals(3.5, ex.violatingValue, 1e-15);
        assertEquals(Direction.UPPER, ex.direction);
        assertInstanceOf(RuntimeException.class, ex);
    }

    @Test
    void hardBoundViolationException_lowerDirection() {
        var ex = new HardBoundViolationException(0, -4.1, Direction.LOWER);
        assertEquals(Direction.LOWER, ex.direction);
        assertEquals(-4.1, ex.violatingValue, 1e-15);
    }

    @Test
    void boundExpansionEvent_storesFields() {
        var ev = new BoundExpansionEvent(1, Direction.LOWER, -1.0, -1.5);
        assertEquals(1, ev.paramIndex());
        assertEquals(Direction.LOWER, ev.direction());
        assertEquals(-1.0, ev.originalSoftBound(), 1e-15);
        assertEquals(-1.5, ev.newSoftBound(), 1e-15);
    }

    @Test
    void dynamicRangesParams_defaults() {
        var p = Params.defaults(1)._dynamicRangesParameters;
        assertFalse(p.enabled);
        assertNull(p.expansionStep);
        assertEquals(1e-8, p.gradientThreshold, 1e-15);
    }

    @Test
    void dynamicRangesParams_builder() {
        var p = Params.defaults(1)._dynamicRangesParameters
                .enabled(true)
                .expansionStep(0.2)
                .gradientThreshold(1e-6);
        assertTrue(p.enabled);
        assertEquals(0.2, p.expansionStep, 1e-15);
        assertEquals(1e-6, p.gradientThreshold, 1e-15);
    }

    @Test
    void params_dynamicRangesBuilder_wiredThroughParams() {
        int n = 2;
        var params = Params.defaults(n)
                .dynamicRanges(p -> p.enabled(true).gradientThreshold(1e-5));
        assertTrue(params._dynamicRangesParameters.enabled);
        assertEquals(1e-5, params._dynamicRangesParameters.gradientThreshold, 1e-15);
        // other param groups unaffected
        assertEquals(2 * n + 1, params._modelParameters.npt);
    }

    @Test
    void optimResult_hasExpansionsField_emptyByDefault() {
        var result = Bobyqa.solve(
                x -> x[0] * x[0],
                new double[]{0.5},
                new double[]{0.0}, new double[]{1.0},
                100);
        assertNotNull(result.expansions());
        assertTrue(result.expansions().isEmpty());
    }

    @Test
    void hardBoundViolation_caughtBySolver_resultValid() {
        // Objective: min at x=0; throws HardBoundViolationException if x > 0.6.
        // Soft bounds [0.0, 1.5] — solver may propose x > 0.6 during exploration.
        // Hard bounds null → ±∞; solver discovers the hard upper bound at 0.6 via exception.
        var objective = (java.util.function.ToDoubleFunction<double[]>) x -> {
            if (x[0] > 0.6)
                throw new HardBoundViolationException(0, x[0], Direction.UPPER);
            return x[0] * x[0];
        };
        var params = Params.defaults(1).init(p -> p.rhoBeg(0.1));
        var result = Bobyqa.solve(objective,
                new double[]{0.3},
                new double[]{0.0}, new double[]{1.5},
                null, null,
                200, GoalType.MINIMIZE, params);
        assertNotNull(result);
        assertTrue(result.x()[0] <= 0.6 + 1e-9,
                "x=" + result.x()[0] + " exceeded discovered hard bound 0.6");
    }

    @Test
    void softBound_expandedWhenOptimumBeyondInitialRange() {
        // f(x) = (x - 2.0)^2, minimum at x=2.0 which is beyond the initial soft upper bound of 1.0.
        // Hard bounds [−1, 3] allow expansion up to 3.0.
        var params = Params.defaults(1)
                .init(p -> p.rhoBeg(0.2).rhoEnd(1e-6))
                .dynamicRanges(p -> p.enabled(true).gradientThreshold(1e-6));
        var result = Bobyqa.solve(
                x -> (x[0] - 2.0) * (x[0] - 2.0),
                new double[]{0.5},
                new double[]{-0.5}, new double[]{1.0},
                new Double[]{-1.0}, new Double[]{3.0},
                500, GoalType.MINIMIZE, params);
        assertFalse(result.expansions().isEmpty(), "Expected at least one bound expansion");
        assertTrue(result.expansions().stream()
                .anyMatch(e -> e.direction() == Direction.UPPER && e.paramIndex() == 0),
                "Expected upper bound expansion for parameter 0");
        assertTrue(result.x()[0] > 1.0,
                "Expected optimizer to move past initial soft bound; got x=" + result.x()[0]);
    }

    @Test
    void softBound_notExpandedWhenDynamicRangesDisabled() {
        // Same objective but dynamic ranges disabled — no expansion.
        var params = Params.defaults(1).init(p -> p.rhoBeg(0.2).rhoEnd(1e-6));
        var result = Bobyqa.solve(
                x -> (x[0] - 2.0) * (x[0] - 2.0),
                new double[]{0.5},
                new double[]{-0.5}, new double[]{1.0},
                new Double[]{-1.0}, new Double[]{3.0},
                200, GoalType.MINIMIZE, params);
        assertTrue(result.expansions().isEmpty(), "Expected no expansions when feature disabled");
        assertTrue(result.x()[0] <= 1.0 + 1e-9,
                "Expected optimizer to stay within soft bound; got x=" + result.x()[0]);
    }

    @Test
    void softBound_notExpandedBeyondHardBound() {
        // f(x) = (x - 5.0)^2; hard upper bound at 2.0.
        // Expansion should stop at the hard limit, never exceed it.
        var params = Params.defaults(1)
                .init(p -> p.rhoBeg(0.2).rhoEnd(1e-6))
                .dynamicRanges(p -> p.enabled(true).gradientThreshold(1e-6));
        var result = Bobyqa.solve(
                x -> (x[0] - 5.0) * (x[0] - 5.0),
                new double[]{0.5},
                new double[]{-0.5}, new double[]{1.0},
                new Double[]{-1.0}, new Double[]{2.0},
                500, GoalType.MINIMIZE, params);
        for (BoundExpansionEvent ev : result.expansions()) {
            if (ev.direction() == Direction.UPPER) {
                assertTrue(ev.newSoftBound() <= 2.0 + 1e-12,
                        "Soft bound expanded past hard limit: " + ev.newSoftBound());
            }
        }
        assertTrue(result.x()[0] <= 2.0 + 1e-9,
                "x=" + result.x()[0] + " exceeded hard bound 2.0");
    }
}
