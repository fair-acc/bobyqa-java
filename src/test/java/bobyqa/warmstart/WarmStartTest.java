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
package bobyqa.warmstart;

import bobyqa.*;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.ToDoubleFunction;

import static org.junit.jupiter.api.Assertions.*;

class WarmStartTest {

    @Test
    void modelSnapshot_recordFields() {
        double[] xbase = {1.0, 2.0};
        double[][] xpt  = {{0.1, 0.0}, {0.0, 0.2}, {-0.1, 0.0}, {0.0, -0.2}};
        double[] fval  = {1.1, 1.2, 1.3, 1.4};
        double[] grad  = {0.5, -0.3};
        double[][] hess = {{2.0, 0.1}, {0.1, 3.0}};

        var snap = new ModelSnapshot(2, 4, xbase, xpt, fval, grad, hess);

        assertEquals(2, snap.n());
        assertEquals(4, snap.npt());
        assertArrayEquals(xbase, snap.xbase());
        assertArrayEquals(fval,  snap.fval());
        assertArrayEquals(grad,  snap.grad());
        assertEquals(xpt[0][0], snap.xpt()[0][0]);
        assertEquals(hess[1][1], snap.hess()[1][1]);
    }

    @Test
    void warmStart_fromPriorData() {
        List<double[]> xs = List.of(new double[]{1.0}, new double[]{2.0});
        List<Double>   fs = List.of(3.0, 4.0);

        WarmStart ws = WarmStart.fromPriorData(xs, fs);

        assertSame(xs, ws.priorX());
        assertSame(fs, ws.priorF());
        assertNull(ws.snapshot());
    }

    @Test
    void warmStart_fromSnapshot() {
        var snap = new ModelSnapshot(1, 3,
                new double[]{0.0}, new double[][]{{0.1}, {-0.1}, {0.0}},
                new double[]{1.0, 1.1, 1.2}, new double[]{0.5}, new double[][]{{2.0}});

        WarmStart ws = WarmStart.fromSnapshot(snap);

        assertNull(ws.priorX());
        assertNull(ws.priorF());
        assertSame(snap, ws.snapshot());
    }

    @Test
    void warmStart_of_bothProvided() {
        List<double[]> xs = List.of(new double[]{1.0});
        List<Double>   fs = List.of(0.5);
        var snap = new ModelSnapshot(1, 3,
                new double[]{0.0}, new double[][]{{0.1}, {-0.1}, {0.0}},
                new double[]{1.0, 1.1, 1.2}, new double[]{0.5}, new double[][]{{2.0}});

        WarmStart ws = WarmStart.of(xs, fs, snap);

        assertSame(xs, ws.priorX());
        assertSame(fs, ws.priorF());
        assertSame(snap, ws.snapshot());
    }

    @Test
    void warmStart_fromPriorData_reducesInitEvals() {
        ToDoubleFunction<double[]> f = x -> (x[0] - 3.0) * (x[0] - 3.0);
        double[] x0 = {0.5};
        double[] lo  = {0.0}, hi = {6.0};
        // npt=3 is the minimum for n=1 (n+2=3); equals the cold-start init budget
        Params params = Params.defaults(1).model(m -> m.npt(3));

        // 5 prior pairs — top 3 by f are near the global minimum at x=3
        WarmStart ws = WarmStart.fromPriorData(
                List.of(new double[]{2.5}, new double[]{3.0}, new double[]{3.5},
                        new double[]{1.0}, new double[]{5.0}),
                List.of(0.25, 0.0, 0.25, 4.0, 4.0));

        int budget = 3; // = npt: cold exhausts budget entirely on init
        OptimResult cold = Bobyqa.solve(f, x0, lo, hi, budget, GoalType.MINIMIZE, params);
        OptimResult warm = Bobyqa.solve(f, x0, lo, hi, budget, GoalType.MINIMIZE, params, ws);

        // Cold: all 3 evals used for init, best f near x0=0.5 (f≈5.76)
        // Warm: 0 init evals (5 prior pairs ≥ npt=3), 3 optimizer steps from x≈3 (f≈0)
        assertTrue(warm.f() < cold.f(),
                "Warm start from prior data must achieve better f than cold start with same budget. "
                        + "cold.f()=" + cold.f() + ", warm.f()=" + warm.f());
        assertNotNull(warm.snapshot(), "warm result must carry a snapshot");
        assertTrue(warm.nEvals() <= budget,
                "Warm start must not exceed budget. warm.nEvals()=" + warm.nEvals());
    }

    @Test
    void warmStart_fromSnapshot_zeroInitEvals() {
        ToDoubleFunction<double[]> f = x -> (x[0] - 3.0) * (x[0] - 3.0);
        double[] x0 = {0.5};
        double[] lo  = {0.0}, hi = {6.0};
        Params params = Params.defaults(1).model(m -> m.npt(3));

        // Full cold run to convergence — captures snapshot
        OptimResult cold = Bobyqa.solve(f, x0, lo, hi, 200, GoalType.MINIMIZE, params);
        assertNotNull(cold.snapshot(), "snapshot must be non-null on successful convergence");

        // Tiny budget: 3 evals (= npt). Cold exhausts on init. Warm skips init entirely.
        int tinyBudget = 3;
        OptimResult coldTiny = Bobyqa.solve(f, x0, lo, hi, tinyBudget, GoalType.MINIMIZE, params);
        OptimResult warmTiny = Bobyqa.solve(f, x0, lo, hi, tinyBudget, GoalType.MINIMIZE, params,
                WarmStart.fromSnapshot(cold.snapshot()));

        assertTrue(warmTiny.f() < coldTiny.f(),
                "Warm start from snapshot must achieve better f than cold start with same tiny budget. "
                        + "coldTiny.f()=" + coldTiny.f() + ", warmTiny.f()=" + warmTiny.f());
        assertNotNull(warmTiny.snapshot(), "warm result must carry a snapshot");
        assertTrue(warmTiny.nEvals() <= tinyBudget,
                "Warm start must not exceed budget. warmTiny.nEvals()=" + warmTiny.nEvals());
    }

    @Test
    void warmStart_fromPriorData_maximize() {
        // Maximize -(x-3)^2; maximum at x=3 where f=0, worse away from x=3
        ToDoubleFunction<double[]> f = x -> -(x[0] - 3.0) * (x[0] - 3.0);
        double[] x0 = {0.5};
        double[] lo  = {0.0}, hi = {6.0};
        Params params = Params.defaults(1).model(m -> m.npt(3));

        // Prior pairs near the maximum at x=3 (highest raw f = 0)
        WarmStart ws = WarmStart.fromPriorData(
                List.of(new double[]{2.5}, new double[]{3.0}, new double[]{3.5},
                        new double[]{1.0}, new double[]{5.0}),
                List.of(-0.25, 0.0, -0.25, -4.0, -4.0));

        int budget = 3;
        OptimResult cold = Bobyqa.solve(f, x0, lo, hi, budget, GoalType.MAXIMIZE, params);
        OptimResult warm = Bobyqa.solve(f, x0, lo, hi, budget, GoalType.MAXIMIZE, params, ws);

        // Cold: all 3 evals used for init near x0=0.5 where f≈-5.76
        // Warm: 0 init evals, 3 optimizer steps near x=3 where f≈0
        assertTrue(warm.f() > cold.f(),
                "Warm start with prior data (MAXIMIZE) must achieve better f than cold start. "
                        + "cold.f()=" + cold.f() + ", warm.f()=" + warm.f());
        assertNotNull(warm.snapshot(), "warm result must carry a snapshot");
    }
}
