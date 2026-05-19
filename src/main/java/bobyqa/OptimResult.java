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

public record OptimResult(
    double[] x,
    double f,
    int nEvals,
    ExitFlag exitFlag,
    String msg,
    List<DiagnosticInfo> diagnostics,
    List<BoundExpansionEvent> expansions,
    ModelSnapshot snapshot
) {}
