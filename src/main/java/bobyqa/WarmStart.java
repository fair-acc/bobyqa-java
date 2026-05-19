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

import java.util.List;

public record WarmStart(
    List<double[]> priorX,
    List<Double>   priorF,
    ModelSnapshot  snapshot
) {
    public WarmStart {
        if ((priorX == null) != (priorF == null))
            throw new IllegalArgumentException(
                    "WarmStart: priorX and priorF must both be non-null or both be null.");
    }

    public static WarmStart fromPriorData(List<double[]> x, List<Double> f) {
        return new WarmStart(x, f, null);
    }

    public static WarmStart fromSnapshot(ModelSnapshot s) {
        return new WarmStart(null, null, s);
    }

    public static WarmStart of(List<double[]> x, List<Double> f, ModelSnapshot s) {
        return new WarmStart(x, f, s);
    }
}
