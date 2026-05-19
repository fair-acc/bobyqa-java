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

class ParamsTest {

    @Test void defaultsAreClean() {
        var p = Params.defaults(5); // n=5
        assertEquals(11, p._modelParameters.npt);  // 2*5+1
        assertEquals(-1e20, p._modelParameters.absTol);
        assertFalse(p._noiseParameters.objfunHasNoise);
        assertNull(p._noiseParameters.multiplicativeNoiseLevel);
        assertNull(p._noiseParameters.additiveNoiseLevel);
        assertFalse(p._noiseParameters.quitOnNoiseLevel);
        assertFalse(p._restartsParameters.useRestarts);
        assertEquals(0.5, p._trustRegionRadius.gammaDec);
        assertEquals(0.1, p._trustRegionRadius.alpha1);  // smooth: fast rho reduction
        assertEquals(0.5, p._trustRegionRadius.alpha2);
        assertEquals(0.1, p._generalParameters.roundingErrorConstant);
        assertEquals(0.5, p._generalParameters.safetyStepThresh);
    }

    @Test void noisyDefaultsActivated() {
        var p = Params.defaults(5).noise(n -> n.objfunHasNoise(true));
        assertTrue(p._noiseParameters.objfunHasNoise);
        assertTrue(p._noiseParameters.quitOnNoiseLevel);
        assertTrue(p._restartsParameters.useRestarts);
        assertEquals(0.98, p._trustRegionRadius.gammaDec);
        assertEquals(0.9, p._trustRegionRadius.alpha1);  // noisy: slow rho reduction
        assertEquals(0.95, p._trustRegionRadius.alpha2);
        assertEquals(1.1, p._restartsParameters.rhobegScaleAfterUnsuccessfulRestart);
    }

    @Test void builderOverridesDefaults() {
        var p = Params.defaults(3)
            .trRadius(tr -> tr.eta1(0.2).eta2(0.8).gammaDec(0.6))
            .slow(s -> s.historyForSlow(10).threshForSlow(1e-6));
        assertEquals(0.2, p._trustRegionRadius.eta1);
        assertEquals(0.8, p._trustRegionRadius.eta2);
        assertEquals(0.6, p._trustRegionRadius.gammaDec);
        assertEquals(10, p._slowParameters.historyForSlow);
        assertEquals(1e-6, p._slowParameters.threshForSlow);
    }
}
