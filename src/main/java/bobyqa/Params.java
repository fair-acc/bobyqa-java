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

import java.util.function.UnaryOperator;

public final class Params {

    public final GeneralParams _generalParameters;
    public final InitParams _initparameters;
    public final InterpolationParams _interpolationParameters;
    public final TrRadiusParams _trustRegionRadius;
    public final ModelParams _modelParameters;
    public final NoiseParams _noiseParameters;
    public final RestartParams _restartsParameters;
    public final SlowParams _slowParameters;
    public final LoggingParams _loggingParameters;
    public final DynamicRangesParams _dynamicRangesParameters;

    private Params(
            GeneralParams general,
            InitParams init,
            InterpolationParams interpolation,
            TrRadiusParams trRadius,
            ModelParams model,
            NoiseParams noise,
            RestartParams restarts,
            SlowParams slow,
            LoggingParams logging,
            DynamicRangesParams dynamicRanges) {
        this._generalParameters = general;
        this._initparameters = init;
        this._interpolationParameters = interpolation;
        this._trustRegionRadius = trRadius;
        this._modelParameters = model;
        this._noiseParameters = noise;
        this._restartsParameters = restarts;
        this._slowParameters = slow;
        this._loggingParameters = logging;
        this._dynamicRangesParameters = dynamicRanges;
    }

    /** Create default params for a problem of dimension n. */
    public static Params defaults(int n) {
        return new Params(GeneralParams.defaults(), InitParams.defaults(), InterpolationParams.defaults(),
                TrRadiusParams.cleanDefaults(), ModelParams.defaults(n), NoiseParams.cleanDefaults(),
                RestartParams.cleanDefaults(n), SlowParams.defaults(n), LoggingParams.defaults(),
                DynamicRangesParams.defaults());
    }

    public Params general(UnaryOperator<GeneralParams> f) {
        return copy(
                f.apply(_generalParameters),
                _initparameters,
                _interpolationParameters,
                _trustRegionRadius,
                _modelParameters,
                _noiseParameters,
                _restartsParameters,
                _slowParameters,
                _loggingParameters,
                _dynamicRangesParameters);
    }

    public Params init(UnaryOperator<InitParams> f) {
        return copy(
                _generalParameters,
                f.apply(_initparameters),
                _interpolationParameters,
                _trustRegionRadius,
                _modelParameters,
                _noiseParameters,
                _restartsParameters,
                _slowParameters,
                _loggingParameters,
                _dynamicRangesParameters);
    }

    public Params interpolation(UnaryOperator<InterpolationParams> f) {
        return copy(
                _generalParameters,
                _initparameters,
                f.apply(_interpolationParameters),
                _trustRegionRadius,
                _modelParameters,
                _noiseParameters,
                _restartsParameters,
                _slowParameters,
                _loggingParameters,
                _dynamicRangesParameters);
    }

    public Params trRadius(UnaryOperator<TrRadiusParams> f) {
        return copy(
                _generalParameters,
                _initparameters,
                _interpolationParameters,
                f.apply(_trustRegionRadius),
                _modelParameters,
                _noiseParameters,
                _restartsParameters,
                _slowParameters,
                _loggingParameters,
                _dynamicRangesParameters);
    }

    public Params model(UnaryOperator<ModelParams> f) {
        return copy(
                _generalParameters,
                _initparameters,
                _interpolationParameters,
                _trustRegionRadius,
                f.apply(_modelParameters),
                _noiseParameters,
                _restartsParameters,
                _slowParameters,
                _loggingParameters,
                _dynamicRangesParameters);
    }

    public Params restarts(UnaryOperator<RestartParams> f) {
        return copy(
                _generalParameters,
                _initparameters,
                _interpolationParameters,
                _trustRegionRadius,
                _modelParameters,
                _noiseParameters,
                f.apply(_restartsParameters),
                _slowParameters,
                _loggingParameters,
                _dynamicRangesParameters);
    }

    public Params slow(UnaryOperator<SlowParams> f) {
        return copy(
                _generalParameters,
                _initparameters,
                _interpolationParameters,
                _trustRegionRadius,
                _modelParameters,
                _noiseParameters,
                _restartsParameters,
                f.apply(_slowParameters),
                _loggingParameters,
                _dynamicRangesParameters);
    }

    public Params logging(UnaryOperator<LoggingParams> f) {
        return copy(
                _generalParameters,
                _initparameters,
                _interpolationParameters,
                _trustRegionRadius,
                _modelParameters,
                _noiseParameters,
                _restartsParameters,
                _slowParameters,
                f.apply(_loggingParameters),
                _dynamicRangesParameters);
    }

    public Params dynamicRanges(UnaryOperator<DynamicRangesParams> f) {
        return copy(_generalParameters, _initparameters, _interpolationParameters,
                _trustRegionRadius, _modelParameters, _noiseParameters, _restartsParameters,
                _slowParameters, _loggingParameters, f.apply(_dynamicRangesParameters));
    }

    /** Activating noise also flips trust-region and restart defaults. */
    public Params noise(UnaryOperator<NoiseParams> f) {
        NoiseParams n = f.apply(_noiseParameters);
        TrRadiusParams tr = n.objfunHasNoise ? TrRadiusParams.noisyDefaults() : _trustRegionRadius;
        RestartParams r = n.objfunHasNoise ? RestartParams.noisyDefaults() : _restartsParameters;
        return copy(
                _generalParameters,
                _initparameters,
                _interpolationParameters,
                tr,
                _modelParameters,
                n,
                r,
                _slowParameters,
                _loggingParameters,
                _dynamicRangesParameters);
    }

    private static Params copy(
            GeneralParams g,
            InitParams i,
            InterpolationParams ip,
            TrRadiusParams tr,
            ModelParams mo,
            NoiseParams no,
            RestartParams re,
            SlowParams sl,
            LoggingParams lo,
            DynamicRangesParams dr) {
        return new Params(g, i, ip, tr, mo, no, re, sl, lo, dr);
    }

    // ---- Nested param groups ------------------------------------------------

    public static final class GeneralParams {

        public final double roundingErrorConstant;
        public final double safetyStepThresh;
        public final boolean checkObjfunForOverflow;

        public GeneralParams(double roundingErrorConstant, double safetyStepThresh, boolean checkObjfunForOverflow) {
            this.roundingErrorConstant = roundingErrorConstant;
            this.safetyStepThresh = safetyStepThresh;
            this.checkObjfunForOverflow = checkObjfunForOverflow;
        }

        static GeneralParams defaults() {
            return new GeneralParams(0.1, 0.5, true);
        }

    }

    public static final class InitParams {

        public final boolean randomInitialDirections;
        public final boolean randomDirectionsMakeOrthogonal;
        /** Override for the initial trust-region radius. null → use the default formula
         *  (0.1 × max(|x₀|, 1.0) for physical inputs, 0.1 for [0,1]-normalised inputs).
         *  For noisy beam objectives with [0,1]-normalised inputs, values of 0.3–0.5
         *  give wider initialization probes and a stronger gradient signal. */
        public final Double rhoBeg;
        /** Override for the convergence trust-region radius. null → 1e-8.
         *  For noisy objectives the default 1e-8 causes the optimizer to over-refine
         *  in the noise floor. Set to 1e-3 or 1e-4 to stop earlier and avoid the
         *  dense cluster near x₀ that appears when δ collapses toward rhoEnd. */
        public final Double rhoEnd;

        public InitParams(boolean randomInitialDirections, boolean randomDirectionsMakeOrthogonal,
                          Double rhoBeg, Double rhoEnd) {
            this.randomInitialDirections = randomInitialDirections;
            this.randomDirectionsMakeOrthogonal = randomDirectionsMakeOrthogonal;
            this.rhoBeg = rhoBeg;
            this.rhoEnd = rhoEnd;
        }

        static InitParams defaults() {
            return new InitParams(false, true, null, null);
        }

        public InitParams randomInitialDirections(boolean v) {
            return new InitParams(v, randomDirectionsMakeOrthogonal, rhoBeg, rhoEnd);
        }

        public InitParams randomDirectionsMakeOrthogonal(boolean v) {
            return new InitParams(randomInitialDirections, v, rhoBeg, rhoEnd);
        }

        public InitParams rhoBeg(double v) {
            return new InitParams(randomInitialDirections, randomDirectionsMakeOrthogonal, v, rhoEnd);
        }

        public InitParams rhoEnd(double v) {
            return new InitParams(randomInitialDirections, randomDirectionsMakeOrthogonal, rhoBeg, v);
        }

    }

    public static final class InterpolationParams {

        public final boolean minimumChangeHessian;
        public final boolean precondition;

        public InterpolationParams(boolean minimumChangeHessian, boolean precondition) {
            this.minimumChangeHessian = minimumChangeHessian;
            this.precondition = precondition;
        }

        static InterpolationParams defaults() {
            return new InterpolationParams(true, true);
        }

        public InterpolationParams minimumChangeHessian(boolean v) {
            return new InterpolationParams(v, precondition);
        }

        public InterpolationParams precondition(boolean v) {
            return new InterpolationParams(minimumChangeHessian, v);
        }

    }

    public static final class TrRadiusParams {

        public final double eta1, eta2, gammaDec, gammaInc, gammaIncOverline, alpha1, alpha2;

        public TrRadiusParams(
                double eta1,
                double eta2,
                double gammaDec,
                double gammaInc,
                double gammaIncOverline,
                double alpha1,
                double alpha2) {
            this.eta1 = eta1;
            this.eta2 = eta2;
            this.gammaDec = gammaDec;
            this.gammaInc = gammaInc;
            this.gammaIncOverline = gammaIncOverline;
            this.alpha1 = alpha1;
            this.alpha2 = alpha2;
        }

        static TrRadiusParams cleanDefaults() {
            // alpha1=0.1, alpha2=0.5: fast rho reduction (matches pybobyqa smooth defaults)
            return new TrRadiusParams(0.1, 0.7, 0.5, 2.0, 4.0, 0.1, 0.5);
        }

        static TrRadiusParams noisyDefaults() {
            // alpha1=0.9, alpha2=0.95: slow rho reduction (matches pybobyqa noisy defaults)
            return new TrRadiusParams(0.1, 0.7, 0.98, 2.0, 4.0, 0.9, 0.95);
        }

        public TrRadiusParams eta1(double v) {
            return new TrRadiusParams(v, eta2, gammaDec, gammaInc, gammaIncOverline, alpha1, alpha2);
        }

        public TrRadiusParams eta2(double v) {
            return new TrRadiusParams(eta1, v, gammaDec, gammaInc, gammaIncOverline, alpha1, alpha2);
        }

        public TrRadiusParams gammaDec(double v) {
            return new TrRadiusParams(eta1, eta2, v, gammaInc, gammaIncOverline, alpha1, alpha2);
        }

        public TrRadiusParams gammaInc(double v) {
            return new TrRadiusParams(eta1, eta2, gammaDec, v, gammaIncOverline, alpha1, alpha2);
        }

        public TrRadiusParams gammaIncOverline(double v) {
            return new TrRadiusParams(eta1, eta2, gammaDec, gammaInc, v, alpha1, alpha2);
        }

        public TrRadiusParams alpha1(double v) {
            return new TrRadiusParams(eta1, eta2, gammaDec, gammaInc, gammaIncOverline, v, alpha2);
        }

        public TrRadiusParams alpha2(double v) {
            return new TrRadiusParams(eta1, eta2, gammaDec, gammaInc, gammaIncOverline, alpha1, v);
        }

    }

    public static final class ModelParams {

        public final int npt;
        public final double absTol;

        public ModelParams(int npt, double absTol) {
            this.npt = npt;
            this.absTol = absTol;
        }

        static ModelParams defaults(int n) {
            return new ModelParams(2 * n + 1, -1e20);
        }

        public ModelParams npt(int v) {
            return new ModelParams(v, absTol);
        }

        public ModelParams absTol(double v) {
            return new ModelParams(npt, v);
        }

        /** Call after setting npt; throws if out of range for dimension n. */
        public ModelParams validate(int n) {
            int lo = n + 2, hi = (n + 1) * (n + 2) / 2;
            if (npt < lo || npt > hi) throw new IllegalArgumentException(
                    "npt=" + npt + " out of valid range [" + lo + ", " + hi + "] for n=" + n);
            return this;
        }

    }

    public static final class NoiseParams {

        public final boolean objfunHasNoise;
        public final Double multiplicativeNoiseLevel;
        public final Double additiveNoiseLevel;
        public final boolean quitOnNoiseLevel;
        public final double scaleFactorForQuit;

        public NoiseParams(
                boolean objfunHasNoise,
                Double multiplicativeNoiseLevel,
                Double additiveNoiseLevel,
                boolean quitOnNoiseLevel,
                double scaleFactorForQuit) {
            this.objfunHasNoise = objfunHasNoise;
            this.multiplicativeNoiseLevel = multiplicativeNoiseLevel;
            this.additiveNoiseLevel = additiveNoiseLevel;
            this.quitOnNoiseLevel = quitOnNoiseLevel;
            this.scaleFactorForQuit = scaleFactorForQuit;
        }

        static NoiseParams cleanDefaults() {
            return new NoiseParams(false, null, null, false, 1.0);
        }

        public NoiseParams objfunHasNoise(boolean v) {
            return new NoiseParams(v, multiplicativeNoiseLevel, additiveNoiseLevel, v || quitOnNoiseLevel,
                    scaleFactorForQuit);
        }

        public NoiseParams multiplicativeNoiseLevel(double v) {
            return new NoiseParams(objfunHasNoise, v, additiveNoiseLevel, quitOnNoiseLevel, scaleFactorForQuit);
        }

        public NoiseParams additiveNoiseLevel(double v) {
            return new NoiseParams(objfunHasNoise, multiplicativeNoiseLevel, v, quitOnNoiseLevel, scaleFactorForQuit);
        }

        public NoiseParams quitOnNoiseLevel(boolean v) {
            return new NoiseParams(objfunHasNoise, multiplicativeNoiseLevel, additiveNoiseLevel, v, scaleFactorForQuit);
        }

        public NoiseParams scaleFactorForQuit(double v) {
            return new NoiseParams(objfunHasNoise, multiplicativeNoiseLevel, additiveNoiseLevel, quitOnNoiseLevel, v);
        }

    }

    public static final class RestartParams {

        public final boolean useRestarts;
        public final int maxUnsuccessfulRestarts;
        public final int maxUnsuccessfulRestartsTotal;
        public final double rhobegScaleAfterUnsuccessfulRestart;
        public final double rhoendScale;
        public final boolean useSoftRestarts;
        public final int softNumGeomSteps;
        public final boolean softMoveXk;
        public final boolean hardUseOldFk;
        public final boolean autoDetect;
        public final int autoDetectHistory;
        public final double autoDetectMinChgModelSlope;
        public final double autoDetectMinCorrel;

        public RestartParams(
                boolean useRestarts,
                int maxUnsuccessfulRestarts,
                int maxUnsuccessfulRestartsTotal,
                double rhobegScale,
                double rhoendScale,
                boolean useSoftRestarts,
                int softNumGeomSteps,
                boolean softMoveXk,
                boolean hardUseOldFk,
                boolean autoDetect,
                int autoDetectHistory,
                double autoDetectMinChgModelSlope,
                double autoDetectMinCorrel) {
            this.useRestarts = useRestarts;
            this.maxUnsuccessfulRestarts = maxUnsuccessfulRestarts;
            this.maxUnsuccessfulRestartsTotal = maxUnsuccessfulRestartsTotal;
            this.rhobegScaleAfterUnsuccessfulRestart = rhobegScale;
            this.rhoendScale = rhoendScale;
            this.useSoftRestarts = useSoftRestarts;
            this.softNumGeomSteps = softNumGeomSteps;
            this.softMoveXk = softMoveXk;
            this.hardUseOldFk = hardUseOldFk;
            this.autoDetect = autoDetect;
            this.autoDetectHistory = autoDetectHistory;
            this.autoDetectMinChgModelSlope = autoDetectMinChgModelSlope;
            this.autoDetectMinCorrel = autoDetectMinCorrel;
        }

        static RestartParams cleanDefaults(int n) {
            return new RestartParams(false, 10, 20, 1.0, 1.0, true, 3, true, true, true, 30, 0.015, 0.1);
        }

        static RestartParams noisyDefaults() {
            return new RestartParams(true, 10, 20, 1.1, 1.0, true, 3, true, true, true, 30, 0.015, 0.1);
        }

        public RestartParams useRestarts(boolean v) {
            return new RestartParams(v, maxUnsuccessfulRestarts, maxUnsuccessfulRestartsTotal,
                    rhobegScaleAfterUnsuccessfulRestart, rhoendScale, useSoftRestarts, softNumGeomSteps, softMoveXk,
                    hardUseOldFk, autoDetect, autoDetectHistory, autoDetectMinChgModelSlope, autoDetectMinCorrel);
        }

        public RestartParams maxUnsuccessfulRestarts(int v) {
            return new RestartParams(useRestarts, v, maxUnsuccessfulRestartsTotal, rhobegScaleAfterUnsuccessfulRestart,
                    rhoendScale, useSoftRestarts, softNumGeomSteps, softMoveXk, hardUseOldFk, autoDetect,
                    autoDetectHistory, autoDetectMinChgModelSlope, autoDetectMinCorrel);
        }

        public RestartParams maxUnsuccessfulRestartsTotal(int v) {
            return new RestartParams(useRestarts, maxUnsuccessfulRestarts, v, rhobegScaleAfterUnsuccessfulRestart,
                    rhoendScale, useSoftRestarts, softNumGeomSteps, softMoveXk, hardUseOldFk, autoDetect,
                    autoDetectHistory, autoDetectMinChgModelSlope, autoDetectMinCorrel);
        }

        public RestartParams useSoftRestarts(boolean v) {
            return new RestartParams(useRestarts, maxUnsuccessfulRestarts, maxUnsuccessfulRestartsTotal,
                    rhobegScaleAfterUnsuccessfulRestart, rhoendScale, v, softNumGeomSteps, softMoveXk, hardUseOldFk,
                    autoDetect, autoDetectHistory, autoDetectMinChgModelSlope, autoDetectMinCorrel);
        }

        public RestartParams autoDetect(boolean v) {
            return new RestartParams(useRestarts, maxUnsuccessfulRestarts, maxUnsuccessfulRestartsTotal,
                    rhobegScaleAfterUnsuccessfulRestart, rhoendScale, useSoftRestarts, softNumGeomSteps, softMoveXk,
                    hardUseOldFk, v, autoDetectHistory, autoDetectMinChgModelSlope, autoDetectMinCorrel);
        }

    }

    public static final class SlowParams {

        public final int historyForSlow;
        public final double threshForSlow;
        public final int maxSlowIters;

        public SlowParams(int historyForSlow, double threshForSlow, int maxSlowIters) {
            this.historyForSlow = historyForSlow;
            this.threshForSlow = threshForSlow;
            this.maxSlowIters = maxSlowIters;
        }

        static SlowParams defaults(int n) {
            return new SlowParams(5, 1e-8, 20 * n);
        }

        public SlowParams historyForSlow(int v) {
            return new SlowParams(v, threshForSlow, maxSlowIters);
        }

        public SlowParams threshForSlow(double v) {
            return new SlowParams(historyForSlow, v, maxSlowIters);
        }

        public SlowParams maxSlowIters(int v) {
            return new SlowParams(historyForSlow, threshForSlow, v);
        }

    }

    public static final class LoggingParams {

        public final boolean saveDiagnosticInfo;
        public final boolean savePoisedness;
        public final boolean saveXk;
        public final int nToPrintWholeXVector;

        public LoggingParams(
                boolean saveDiagnosticInfo,
                boolean savePoisedness,
                boolean saveXk,
                int nToPrintWholeXVector) {
            this.saveDiagnosticInfo = saveDiagnosticInfo;
            this.savePoisedness = savePoisedness;
            this.saveXk = saveXk;
            this.nToPrintWholeXVector = nToPrintWholeXVector;
        }

        public static LoggingParams defaults() {
            return new LoggingParams(false, true, false, 6);
        }

        public LoggingParams saveDiagnosticInfo(boolean v) {
            return new LoggingParams(v, savePoisedness, saveXk, nToPrintWholeXVector);
        }

        public LoggingParams savePoisedness(boolean v) {
            return new LoggingParams(saveDiagnosticInfo, v, saveXk, nToPrintWholeXVector);
        }

        public LoggingParams saveXk(boolean v) {
            return new LoggingParams(saveDiagnosticInfo, savePoisedness, v, nToPrintWholeXVector);
        }

    }

    public static final class DynamicRangesParams {

        public final boolean enabled;
        /** How far to widen the soft bound per expansion event. null = use current trust radius delta. */
        public final Double expansionStep;
        /** Minimum outward surrogate-gradient magnitude to trigger expansion. */
        public final double gradientThreshold;

        public DynamicRangesParams(boolean enabled, Double expansionStep, double gradientThreshold) {
            this.enabled = enabled;
            this.expansionStep = expansionStep;
            this.gradientThreshold = gradientThreshold;
        }

        static DynamicRangesParams defaults() {
            return new DynamicRangesParams(false, null, 1e-8);
        }

        public DynamicRangesParams enabled(boolean v) {
            return new DynamicRangesParams(v, expansionStep, gradientThreshold);
        }

        public DynamicRangesParams expansionStep(double v) {
            return new DynamicRangesParams(enabled, v, gradientThreshold);
        }

        public DynamicRangesParams gradientThreshold(double v) {
            return new DynamicRangesParams(enabled, expansionStep, v);
        }

    }

}
