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

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class FoundationTypesTest {

    @Test void exitFlagValues() {
        assertNotNull(ExitFlag.CONVERGED);
        assertNotNull(ExitFlag.BUDGET_EXHAUSTED);
        assertNotNull(ExitFlag.NOISE_LEVEL_REACHED);
        assertNotNull(ExitFlag.ABS_TOL_REACHED);
        assertNotNull(ExitFlag.SLOW_PROGRESS);
        assertNotNull(ExitFlag.INPUT_ERROR);
    }

    @Test void goalTypeValues() {
        assertEquals(2, GoalType.values().length);
        assertNotNull(GoalType.MINIMIZE);
        assertNotNull(GoalType.MAXIMIZE);
    }

    @Test void iterTypeValues() {
        assertNotNull(IterType.SUCCESSFUL);
        assertNotNull(IterType.UNSUCCESSFUL);
        assertNotNull(IterType.SAFETY);
        assertNotNull(IterType.GEOMETRY_REPAIR);
        assertNotNull(IterType.RESTART_SOFT);
        assertNotNull(IterType.RESTART_HARD);
    }

    @Test void diagnosticInfoRecord() {
        var d = new DiagnosticInfo(0, IterType.SUCCESSFUL, 1.5, 0.1, 0.01,
                                   0.8, 1, 0, null, null);
        assertEquals(0, d.iter());
        assertEquals(IterType.SUCCESSFUL, d.iterType());
        assertEquals(1.5, d.fBest());
        assertEquals(0.1, d.delta());
        assertEquals(0.01, d.rho());
        assertEquals(0.8, d.ratio());
        assertEquals(1, d.nsamples());
        assertEquals(0, d.restartCount());
        assertNull(d.poisedness());
        assertNull(d.xk());
    }

    @Test void optimResultFields() {
        var r = new OptimResult(new double[]{1.0}, -3.0, 42,
                                ExitFlag.CONVERGED, "Converged.", java.util.List.of(), java.util.List.of(), null);
        assertArrayEquals(new double[]{1.0}, r.x());
        assertEquals(-3.0, r.f());
        assertEquals(42, r.nEvals());
        assertEquals(ExitFlag.CONVERGED, r.exitFlag());
        assertEquals("Converged.", r.msg());
        assertTrue(r.diagnostics().isEmpty());
    }
}
