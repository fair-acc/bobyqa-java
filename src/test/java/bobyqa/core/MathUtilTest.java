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

import bobyqa.util.MathUtil;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MathUtilTest {

    @Test void norm2() {
        assertEquals(5.0, MathUtil.norm2(new double[]{3, 4}), 1e-12);
    }

    @Test void dot() {
        assertEquals(14.0, MathUtil.dot(new double[]{1,2,3}, new double[]{1,2,3}), 1e-12);
        assertEquals(11.0, MathUtil.dot(new double[]{1,2,3}, new double[]{1,2,2}), 1e-12); // 1+4+6=11
        assertEquals( 0.0, MathUtil.dot(new double[]{1,0}, new double[]{0,1}), 1e-12);
    }

    @Test void linregSlopeAndCorrel() {
        // y = 2x + 1 → slope=2, correl=1.0
        double[] x = {0,1,2,3,4};
        double[] y = {1,3,5,7,9};
        double[] res = MathUtil.linreg(x, y);
        assertEquals(2.0, res[0], 1e-10); // slope
        assertEquals(1.0, res[1], 1e-10); // correlation
    }

    @Test void euclideanDistance() {
        assertEquals(Math.sqrt(2), MathUtil.distance(new double[]{0,0}, new double[]{1,1}), 1e-12);
    }
}
