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

import java.util.function.ToDoubleFunction;

public final class Bobyqa {

    private Bobyqa() {}

    /**
     * Solver method with hard bounds from the device database. Hard bound elements may be
     * null (no recorded device limit); null is treated as ±∞ and the limit may be
     * discovered at runtime if the objective throws {@link HardBoundViolationException}.
     *
     * @param softLower scan-range lower limits (expandable)
     * @param softUpper scan-range upper limits (expandable)
     * @param hardLower database hard lower limits; null array or null elements treated as -∞
     * @param hardUpper database hard upper limits; null array or null elements treated as +∞
     */
    public static OptimResult solve(
            ToDoubleFunction<double[]> objective,
            double[] startValues,
            double[] softLower,
            double[] softUpper,
            Double[] hardLower,
            Double[] hardUpper,
            int maxEvaluations,
            GoalType goalType,
            Params algorithmParameters,
            WarmStart warmStart) {

        int n = startValues.length;

        double[] effectiveHardLower = new double[n];
        double[] effectiveHardUpper = new double[n];
        for (int i = 0; i < n; i++) {
            effectiveHardLower[i] = (hardLower != null && hardLower[i] != null)
                    ? hardLower[i] : Double.NEGATIVE_INFINITY;
            effectiveHardUpper[i] = (hardUpper != null && hardUpper[i] != null)
                    ? hardUpper[i] : Double.POSITIVE_INFINITY;
        }

        ToDoubleFunction<double[]> internalObj =
                goalType == GoalType.MAXIMIZE ? x -> -objective.applyAsDouble(x) : objective;

        Params resolvedParams = algorithmParameters != null ? algorithmParameters : Params.defaults(n);

        Double customRhoBeg = resolvedParams._initparameters.rhoBeg;
        double rhoStart;
        if (customRhoBeg != null) {
            rhoStart = customRhoBeg;
        } else {
            double maxAbsX0 = 0.0;
            for (double v : startValues) maxAbsX0 = Math.max(maxAbsX0, Math.abs(v));
            rhoStart = 0.1 * Math.max(maxAbsX0, 1.0);
        }
        Double customRhoEnd = resolvedParams._initparameters.rhoEnd;
        double rhoEnd = customRhoEnd != null ? customRhoEnd : 1e-8;

        var solver = new Solver(internalObj, softLower, softUpper,
                effectiveHardLower, effectiveHardUpper,
                maxEvaluations, rhoStart, rhoEnd, resolvedParams, goalType, warmStart);
        OptimResult result = solver.optimize(startValues);

        if (goalType == GoalType.MAXIMIZE)
            result = new OptimResult(result.x(), -result.f(), result.nEvals(),
                    result.exitFlag(), result.msg(), result.diagnostics(), result.expansions(),
                    result.snapshot());

        return result;
    }

    /**
     * Convenience overload without WarmStart — delegates to the full overload with {@code warmStart=null}.
     */
    public static OptimResult solve(
            ToDoubleFunction<double[]> objective,
            double[] startValues,
            double[] softLower,
            double[] softUpper,
            Double[] hardLower,
            Double[] hardUpper,
            int maxEvaluations,
            GoalType goalType,
            Params algorithmParameters) {
        return solve(objective, startValues, softLower, softUpper,
                hardLower, hardUpper, maxEvaluations, goalType, algorithmParameters, null);
    }

    /**
     * Solver method with configurable algorithm parameters and warm-start
     *
     * @param objective
     * @param startValues
     * @param lowerLimits
     * @param upperLimits
     * @param maxEvaluations
     * @param goalType
     * @param algorithmParameters
     * @param warmStart
     * @return
     */
    public static OptimResult solve(
            ToDoubleFunction<double[]> objective,
            double[] startValues,
            double[] lowerLimits,
            double[] upperLimits,
            int maxEvaluations,
            GoalType goalType,
            Params algorithmParameters,
            WarmStart warmStart) {
        // Hard bounds equal soft bounds: no expansion possible via this overload.
        int n = startValues.length;
        Double[] hardLower = new Double[n];
        Double[] hardUpper = new Double[n];
        for (int i = 0; i < n; i++) {
            hardLower[i] = lowerLimits[i];
            hardUpper[i] = upperLimits[i];
        }
        return solve(objective, startValues, lowerLimits, upperLimits,
                hardLower, hardUpper, maxEvaluations, goalType, algorithmParameters, warmStart);
    }

    /**
     * Solver method with configurable algorithm parameters
     *
     * @param objective
     * @param startValues
     * @param lowerLimits
     * @param upperLimits
     * @param maxEvaluations
     * @param goalType
     * @param algorithmParameters
     * @return
     */
    public static OptimResult solve(
            ToDoubleFunction<double[]> objective,
            double[] startValues,
            double[] lowerLimits,
            double[] upperLimits,
            int maxEvaluations,
            GoalType goalType,
            Params algorithmParameters) {
        return solve(objective, startValues, lowerLimits, upperLimits,
                maxEvaluations, goalType, algorithmParameters, null);
    }

    /**
     * Solver method with default algorithm parameters seeking minimum
     * 
     * @param objective
     * @param startValues
     * @param lowerLimits
     * @param upperLimits
     * @param maxEvaluations
     * @return
     */
    public static OptimResult solve(
            ToDoubleFunction<double[]> objective,
            double[] startValues,
            double[] lowerLimits,
            double[] upperLimits,
            int maxEvaluations) {
        return solve(
                objective,
                startValues,
                lowerLimits,
                upperLimits,
                maxEvaluations,
                GoalType.MINIMIZE,
                Params.defaults(startValues.length));
    }

    /**
     * Method normalizing external values from [-Inf, Inf] to [0, 1] from bounds for each dimension
     * 
     * @param externalValues
     * @param lowerLimits
     * @param upperLimits
     * @return
     */
    public static double[] scaleToBounds(double[] externalValues, double[] lowerLimits, double[] upperLimits) {
        double[] normalizedValues = new double[externalValues.length];
        for (int i = 0; i < externalValues.length; i++)
            normalizedValues[i] = (externalValues[i] - lowerLimits[i]) / (upperLimits[i] - lowerLimits[i]);
        return normalizedValues;
    }

    /**
     * Method rescaling internal values from [0, 1] to exernal values [-Inf, Inf] from bounds for each dimension
     *
     * @param normalizedValues
     * @param lowerLimits
     * @param upperLimits
     * @return
     */
    public static double[] unscaleFromBounds(double[] normalizedValues, double[] lowerLimits, double[] upperLimits) {
        double[] rescaledValues = new double[normalizedValues.length];
        for (int i = 0; i < normalizedValues.length; i++)
            rescaledValues[i] = lowerLimits[i] + normalizedValues[i] * (upperLimits[i] - lowerLimits[i]);
        return rescaledValues;
    }

}
