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

public class HardBoundViolationException extends RuntimeException {

    public final int paramIndex;
    public final double violatingValue;
    public final Direction direction;

    public HardBoundViolationException(int paramIndex, double violatingValue, Direction direction) {
        super("Hard bound violated for parameter " + paramIndex
                + " in direction " + direction + " at value " + violatingValue);
        this.paramIndex = paramIndex;
        this.violatingValue = violatingValue;
        this.direction = direction;
    }
}
