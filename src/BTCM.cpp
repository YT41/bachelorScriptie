#include "BTCM.hpp"

#include "Matrix.hpp"
#include "MemArena.hpp"
#include "MiscMath.hpp"
#include "NeuralNetwork.hpp"
#include "TrajectorySim.hpp"
#include "Random.hpp"
#include "ReactionParser.hpp"
#include "SRN.hpp"
#include <cmath>
#include <cstdio>
#include <math.h>
#include <cstddef>
#include <cstdint>
#include <stdio.h>
#include <ctime>


/*============================ helper functions ============================*/

static inline double GetInitialConditionProbability(const SRN* srn, IntMatrix n)
{
    uint32_t M = SRNGetSpeciesCount(srn);
    for(uint32_t i = 0; i < M; i++)
    {
        if(GetValueIntMatrix(n, i, 0) != (int32_t)(srn->species[i].initialCount))
            return 0.0;
    }
    return 1.0;
}

/*return probability of having picked this sample in initial distribution (its always 1)*/
static inline double GetInitialConditionSample(const SRN* srn, IntMatrix s)
{
    uint32_t M = SRNGetSpeciesCount(srn);
    for(uint32_t i = 0; i < M; i++)
        SetValueIntMatrix(s, (srn->species[i].initialCount), i, 0);
    return 1.0;
}

static inline void AppendValueVector(Matrix matrix, double val, size_t* size)
{
    SetValueMatrix(matrix, val, *size, 0);
    (*size)++;
}

/*prepare token i: x_i^{(0)} := (i, t, n_{<i})*/
static inline void PrepareInputToken(BTCM* m, uint32_t i, IntMatrix n, double t)
{
    uint32_t K = SRNGetReactionCount((m->srn));

    size_t tokenSize = 0;

    /*prepare token i: x_i^{(0)} := (i, t, \eta_i, n_{<i})*/
    SetMatrix((m->tokenCache), 0.0); /*padding is just set to 0*/

    AppendValueVector((m->tokenCache), (double)i, &tokenSize);
    AppendValueVector((m->tokenCache), t, &tokenSize);
    for(uint32_t k = 0; k < K; k++)
    {
        AppendValueVector((m->tokenCache), (double)GetValueIntMatrix((m->srn->reactantMatrix), i, k), &tokenSize);
        AppendValueVector((m->tokenCache), (double)GetValueIntMatrix((m->srn->productMatrix), i, k), &tokenSize);
        AppendValueVector((m->tokenCache), (m->srn->reactionRates[k]), &tokenSize);
    }
    for(uint32_t j = 0; j < i; j++)
        AppendValueVector((m->tokenCache), (double)GetValueIntMatrix(n, j, 0), &tokenSize);
}

static inline double GetTargetProbability(BTCM* m, IntMatrix sample, IntMatrix previousState, double t, double deltaT)
{
    /*we know that we can only enter the sample state from connected states, all other propensities are 0*/
    double enteringSampleStateProbability = 0.0;
    for(uint32_t k = 0; k < SRNGetReactionCount((m->srn)); k++)
    {
        double possiblePreviousStatePropensity = GetPreviousConnectedState((m->srn), sample, previousState, k);

        if(possiblePreviousStatePropensity != 0.0)
            enteringSampleStateProbability += (possiblePreviousStatePropensity * BTCMPredict(m, previousState, t));
    }
    double tSampleProbability = BTCMPredict(m, sample, t);
    /*note that this is just one step of explicit euler with the approximation of CME*/
    return (tSampleProbability + (deltaT * (enteringSampleStateProbability - (GetEscapeRate((m->srn), sample) * tSampleProbability))));
}


/*============================ public functions ============================*/

BTCM* BTCMCreate(SRN* srn, uint32_t* neuronsPerHiddenLayer, uint32_t hiddenLayerCount, double learningRate)
{
    uint32_t M = SRNGetSpeciesCount(srn);
    uint32_t K = SRNGetReactionCount(srn);

    uint32_t inputTokenDim = 1 + 1 + (K * 3) + (M - 1);

    MemArena arena = CreateMemArena(sizeof(BTCM) + GetMatrixAllocSize(inputTokenDim, 1));

    BTCM* ret = (BTCM*)MemArenaAlloc(&arena, sizeof(BTCM));

    ret->tokenCache = CreateMatrix(&arena, inputTokenDim, 1, NULL);

    /*specify neurons per layer*/
    uint32_t neuronsPerLayer[hiddenLayerCount + 2];
    neuronsPerLayer[0] = inputTokenDim;
    for(uint32_t i = 1; i <= hiddenLayerCount; i++)
        neuronsPerLayer[i] = neuronsPerHiddenLayer[i - 1];
    neuronsPerLayer[hiddenLayerCount + 1] = SRNGetMaxSpeciesCount(srn);

    /*specify activation functions for layers*/
    ActivationFnID activationFnPerLayer[hiddenLayerCount + 1];
    for(uint32_t i = 0; i < hiddenLayerCount; i++)
        activationFnPerLayer[i] = TANH;
    activationFnPerLayer[hiddenLayerCount] = SOFTMAX;

    ret->arena = arena;
    ret->nn = NNCreate(neuronsPerLayer, activationFnPerLayer, hiddenLayerCount, learningRate);
    ret->srn = srn;

    return ret;
}

void BTCMDelete(BTCM* m)
{
    NNDelete((m->nn));
    DeleteMemArena(&(m->arena));
}

size_t BTCMGetParamCount(const BTCM* m)
{
    return NNGetParamCount((m->nn));
}

BTCM* BTCMCopy(const BTCM* m)
{
    uint32_t hiddenLayerCount = (m->nn->hiddenLayerCount);
    uint32_t neuronsPerHiddenLayer[hiddenLayerCount];
    for(uint32_t i = 0; i < hiddenLayerCount; i++)
        neuronsPerHiddenLayer[i] = (m->nn->layerVectors[i+1].rowCount);

    BTCM* ret = BTCMCreate((m->srn), neuronsPerHiddenLayer, hiddenLayerCount, (m->nn->learningRate));

    BTCMCopyParameters(ret, m);

    return ret;
}

void BTCMCopyParameters(BTCM* dest, const BTCM* src)
{
    NNCopyParameters((dest->nn), (src->nn));
}


/*doing it this way, it takes a sample and returns the probability of having taken this sample at the same time*/
double BTCMTakeSample(BTCM* m, IntMatrix s, double t, Matrix desiredNudgesMLPOutput)
{
    if(t == 0.0)
        return GetInitialConditionSample((m->srn), s);

    SetMatrix(desiredNudgesMLPOutput, 0.0);

    uint32_t M = SRNGetSpeciesCount((m->srn));
    double conditionalProbabilityProduct = 1.0;
    for(uint32_t i = 0; i < M; i++)
    {
        PrepareInputToken(m, i, s, t);
        Matrix out = NNPredict((m->nn), (m->tokenCache));

        /*simulate a count for species i using generated conditional probabilities*/
        uint32_t sim = PickUintWithChances(out.data, out.rowCount);
        SetValueIntMatrix(s, sim, i, 0);

        double conditionalProbability = GetValueMatrix(out, GetValueIntMatrix(s, i, 0), 0);

        /*increment desired nudges*/
        SetValueMatrix(desiredNudgesMLPOutput, (GetValueMatrix(desiredNudgesMLPOutput, sim, 0) + (1.0 / conditionalProbability)), sim, 0);

        conditionalProbabilityProduct *= conditionalProbability;
    }

    return conditionalProbabilityProduct;
}

double BTCMTakeSampleNoGradient(BTCM* m, IntMatrix s, double t)
{
    if(t == 0.0)
        return GetInitialConditionSample((m->srn), s);

    uint32_t M = SRNGetSpeciesCount((m->srn));
    double conditionalProbabilityProduct = 1.0;
    for(uint32_t i = 0; i < M; i++)
    {
        PrepareInputToken(m, i, s, t);
        Matrix out = NNPredict((m->nn), (m->tokenCache));

        /*simulate a count for species i using generated conditional probabilities*/
        uint32_t sim = PickUintWithChances(out.data, out.rowCount);
        SetValueIntMatrix(s, sim, i, 0);

        conditionalProbabilityProduct *= GetValueMatrix(out, GetValueIntMatrix(s, i, 0), 0); 
    }

    return conditionalProbabilityProduct;
}

double BTCMPredict(BTCM* m, IntMatrix n, double t)
{
    /*in this case we know the probabilities exactly*/
    if(t == 0.0)
        return GetInitialConditionProbability((m->srn), n);

    uint32_t M = SRNGetSpeciesCount((m->srn));
    double conditionalProbabilityProduct = 1.0;
    for(uint32_t i = 0; i < M; i++)
    {
        PrepareInputToken(m, i, n, t);
        Matrix out = NNPredict((m->nn), (m->tokenCache));
        conditionalProbabilityProduct *= GetValueMatrix(out, GetValueIntMatrix(n, i, 0), 0);
    }

    return conditionalProbabilityProduct;
}

/*TODO: naive way of calculating full distribution, change if it ever takes too long*/
void BTCMGetFullProbabilityDistribution(BTCM* m, Tensor probabilities, double t)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(GetIntMatrixAllocSize(M, 1));

    IntMatrix n = CreateBlankIntMatrix(&arena, M, 1);
    SetIntMatrix(n, 0);
    do
    {
        SetValueTensor(probabilities, BTCMPredict(m, n, t), n);
        IncrementTensorIndex(probabilities, n);
    }
    while(!IntMatrixIsZero(n));

    DeleteMemArena(&arena);
}

static inline void GetTotalGradient(uint32_t B, double rewardBaseline, const double* rewards, Matrix* desiredNudgesMLPOutput, Matrix totalDesiredNudgesMLPOutput)
{
    for(uint32_t b = 0; b < B; b++)
    {
        if(isfinite(rewards[b]) && isfinite(rewardBaseline))
        {
            MatrixScaleSelf(desiredNudgesMLPOutput[b], (rewards[b] - rewardBaseline));
            MatrixAddSelf(totalDesiredNudgesMLPOutput, desiredNudgesMLPOutput[b]);
        }
        else if(isfinite(rewards[b]))
        {
            MatrixScaleSelf(desiredNudgesMLPOutput[b], rewards[b]);
            MatrixAddSelf(totalDesiredNudgesMLPOutput, desiredNudgesMLPOutput[b]);
        }
    }
}

void BTCMTrain(BTCM* m, double T, double deltaT, double p, uint32_t B, uint32_t Q, uint64_t epochs, const char* fileName)
{
    FILE* lossFile = fopen(fileName, "w");

    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(
        (GetIntMatrixAllocSize(M, 1) * 2) + 
        (GetMatrixAllocSize(NNGetOutputDimension((m->nn)), 1) * (1 + B)) + 
        (B * sizeof(Matrix))
    );

    BTCM* targetModelCopy = BTCMCopy(m);

    IntMatrix sample = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix previousState = CreateBlankIntMatrix(&arena, M, 1);

    /*for gradient descent*/
    Matrix totalDesiredNudgesMLPOutput = CreateMatrix(&arena, NNGetOutputDimension((m->nn)), 1, NULL);
    Matrix* desiredNudgesMLPOutput = (Matrix*)MemArenaAlloc(&arena, (B * sizeof(Matrix)));
    for(uint32_t b = 0; b < B; b++)
        desiredNudgesMLPOutput[b] = CreateMatrix(&arena, NNGetOutputDimension((m->nn)), 1, NULL);

    for(uint64_t e = 0; e < epochs; e++)
    {
        double t = (BernoulliDistributionSim(p)) ? 0.0 : (StandardClosedUniformSim() * (((double)e / (double)epochs) * (T - deltaT)));
        //double t = (BernoulliDistributionSim(p)) ? 0.0 : (StandardClosedUniformSim() * (T - deltaT));

        /*update parameters every Q epochs*/
        if(((e + 1) % Q) == 0) { BTCMCopyParameters(targetModelCopy, m); }

        double loss = 0.0; /*KL-divergence a.k.a. cross-entropy*/

        SetMatrix(totalDesiredNudgesMLPOutput, 0.0);

        double rewards[B];
        double rewardBaseline = 0.0;
        for(uint32_t b = 0; b < B; b++)
        {
            double sampleProbability = BTCMTakeSample(m, sample, (t + deltaT), desiredNudgesMLPOutput[b]);
            double targetProbability = GetTargetProbability(targetModelCopy, sample, previousState, t, deltaT);

            rewards[b] = (log(sampleProbability) - log(targetProbability));
            rewardBaseline += rewards[b];

            loss += rewards[b];
        }
        loss /= (double)B;
        rewardBaseline /= (double)B;

        GetTotalGradient(B, rewardBaseline, rewards, desiredNudgesMLPOutput, totalDesiredNudgesMLPOutput);
        MatrixScaleSelf(totalDesiredNudgesMLPOutput, (1.0 / (double)B));

        //PrintMatrix(totalDesiredNudgesMLPOutput);

        NNSetLastLayer((m->nn), totalDesiredNudgesMLPOutput);
        NNBackPropagation((m->nn));

        /*log the loss to file*/
        if((e % 1000) == 0) { printf("loss of epoch %lu: %f\n", e, loss); }
        if(((e % 10) == 0) && isfinite(loss)) { fprintf(lossFile, "%lu %f\n", e, loss); }


        /*TODO: debug*/
        //if(e == 10000) { m->nn->learningRate = 1.0; }
    }

    BTCMDelete(targetModelCopy);
    DeleteMemArena(&arena);
    fclose(lossFile);
}


/*==================== statistics ====================*/

static inline void PrintStateProbability(BTCM* m, IntMatrix n, double t)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    printf("P(");
    for(uint32_t i = 0; i < (M - 1); i++)
        printf("n_%u = %d, ", i, GetValueIntMatrix(n, i, 0));
    printf("n_%u = %d | t = %.3f) = %f\n", (M - 1), GetValueIntMatrix(n, (M - 1), 0), t, BTCMPredict(m, n, t));
}

void BTCMGetPerSpeciesMean(BTCM* m, Matrix mean, double t, size_t sampleCount)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena((GetIntMatrixAllocSize(M, 1) * 2));

    IntMatrix sample = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix sampleSum = CreateBlankIntMatrix(&arena, M, 1);
    SetIntMatrix(sampleSum, 0);

    for(uint32_t n = 0; n < sampleCount; n++)
    {
        BTCMTakeSampleNoGradient(m, sample, t);
        IntMatrixAddSelf(sampleSum, sample);
    }

    for(uint32_t i = 0; i < M; i++)
        SetValueMatrix(mean, ((double)GetValueIntMatrix(sampleSum, i, 0) / (double)sampleCount), i, 0);

    DeleteMemArena(&arena);
}

void BTCMGetPerSpeciesStandardDeviation(BTCM* m, Matrix std, double t, size_t sampleCount)
{    
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(GetMatrixAllocSize(M, 1) + GetIntMatrixAllocSize(M, 1));

    Matrix mean = CreateMatrix(&arena, M, 1, NULL);
    BTCMGetPerSpeciesMean(m, mean, t, sampleCount);

    IntMatrix sample = CreateBlankIntMatrix(&arena, M, 1);

    SetMatrix(std, 0.0);
    for(uint32_t n = 0; n < sampleCount; n++)
    {
        BTCMTakeSampleNoGradient(m, sample, t);

        for(uint32_t i = 0; i < M; i++)
            MatrixAddValue(std, pow((double)GetValueIntMatrix(sample, i, 0) - GetValueMatrix(mean, i, 0), 2.0), i, 0);
    }

    MatrixScaleSelf(std, (1.0 / (double)sampleCount));
    MatrixTransformSelf(std, sqrt);

    DeleteMemArena(&arena);
}

void BTCMLogPerSpeciesMean(BTCM* m, double T, double tStep, size_t sampleCount, FILE* logFile)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(GetMatrixAllocSize(M, 1));

    size_t segmentCount = ((size_t)(T / tStep) + 1);
    Matrix mean = CreateMatrix(&arena, M, 1, NULL);
    for(size_t i = 0; i < segmentCount; i++)
    {
        BTCMGetPerSpeciesMean(m, mean, ((double)i * tStep), sampleCount);
        LogDataPoint(((double)i * tStep), mean, logFile);
    }

    DeleteMemArena(&arena);
}

void BTCMLogPerSpeciesStandardDeviation(BTCM* m, double T, double tStep, size_t sampleCount, FILE* logFile)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(GetMatrixAllocSize(M, 1));

    size_t segmentCount = ((size_t)(T / tStep) + 1);
    Matrix std = CreateMatrix(&arena, M, 1, NULL);
    for(size_t i = 0; i < segmentCount; i++)
    {
        BTCMGetPerSpeciesStandardDeviation(m, std, ((double)i * tStep), sampleCount);
        LogDataPoint(((double)i * tStep), std, logFile);
    }

    DeleteMemArena(&arena);
}

void BTCMPrintFullProbabilityDistribution(BTCM* m, double t)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(GetIntMatrixAllocSize(M, 1));

    IntMatrix n = CreateBlankIntMatrix(&arena, M, 1);
    SetIntMatrix(n, 0);
    do
    {
        PrintStateProbability(m, n, t);
        IncrementStateInStateSpace((m->srn), n);
    }
    while(!IntMatrixIsZero(n));

    DeleteMemArena(&arena);
}

void BTCMGetFullProbabilityDistribution(BTCM* m, double t, Tensor dest)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(GetIntMatrixAllocSize(M, 1));

    IntMatrix n = CreateBlankIntMatrix(&arena, M, 1);
    SetIntMatrix(n, 0);
    do
    {
        SetValueTensor(dest, BTCMPredict(m, n, t), n);
        IncrementStateInStateSpace((m->srn), n);
    }
    while(!IntMatrixIsZero(n));

    DeleteMemArena(&arena);
}

void BTCMLogFullProbabilityDistribution(BTCM* m, double T, double tStep, size_t sampleCount, FILE* logFile)
{
    size_t segmentCount = ((size_t)(T / tStep) + 1);

    MemArena arena = CreateMemArena(SRNGetStateSpaceTensorAllocSize((m->srn)));
    
    Tensor stateSpaceProbabilities = SRNCreateStateSpaceTensor(&arena, (m->srn));
    for(size_t i = 0; i < segmentCount; i++)
    {
        BTCMGetFullProbabilityDistribution(m, ((double)i * tStep), stateSpaceProbabilities);
        LogFullDistribution(stateSpaceProbabilities, ((double)i * tStep), logFile);
    }

    DeleteMemArena(&arena);
}


/*==================== experiments ====================*/

void BTCMSingleTimeStepExperiment(void)
{
    SetSeedRandU48(time(NULL));

    SRN* srn = ParseSRN("res/birthDeathModel.txt");
    uint32_t M = SRNGetSpeciesCount(srn);

    uint32_t hiddenLayerNeuronCount[] = {16 };
    BTCM* m = BTCMCreate(srn, hiddenLayerNeuronCount, (sizeof(hiddenLayerNeuronCount) / sizeof(uint32_t)), 0.5);

    PrintIntMatrix((srn->stoichiometricMatrix));
    printf("Param count: %lu\n", BTCMGetParamCount(m));

    MemArena arena = CreateMemArena(
        (GetMatrixAllocSize(M, 1) * 2) + 
        (SRNGetStateSpaceTensorAllocSize(srn) * 2)
    );

    double T = 1.0;
    BTCMTrain(m, T, T, 1.0, 500, 1, 50000, "res/BTCMSingleTimeStepExperiment/Loss.data");

    BTCMPrintFullProbabilityDistribution(m, T);

    size_t sampleCount = 10000;
    Matrix mean = CreateMatrix(&arena, M, 1, NULL);
    Matrix std = CreateMatrix(&arena, M, 1, NULL);

    BTCMGetPerSpeciesMean(m, mean, T, sampleCount);
    BTCMGetPerSpeciesStandardDeviation(m, std, T, sampleCount);

    printf("BTCM mean: ");
    PrintMatrix(mean);
    printf("BTCM standard deviation: ");
    PrintMatrix(std);

    GillespieSRNTrajectorySimGetPerSpeciesMean(srn, mean, T, sampleCount);
    GillespieSRNTrajectorySimGetPerSpeciesStandardDeviation(srn, std, T, sampleCount);

    printf("Gillespie trajectory simulation mean: ");
    PrintMatrix(mean);
    printf("Gillespie trajectory simulation standard deviation: ");
    PrintMatrix(std);

    Tensor modelStateSpaceProbabilities = SRNCreateStateSpaceTensor(&arena, srn);
    Tensor marginalStateSpaceProbabilities = SRNCreateStateSpaceTensor(&arena, srn);
    BTCMGetFullProbabilityDistribution(m, modelStateSpaceProbabilities, T);
    GillespieSRNTrajectorySimGetFullDistribution(srn, marginalStateSpaceProbabilities, T, sampleCount);

    FILE* BTCMFullDistributionFile = fopen("res/BTCMSingleTimeStepExperiment/BTCMFullDistribution.data", "w");
    FILE* TrajectorySimFullDistributionFile = fopen("res/BTCMSingleTimeStepExperiment/TrajectorySimFullDistribution.data", "w");

    LogFullDistribution(modelStateSpaceProbabilities, T, BTCMFullDistributionFile);
    LogFullDistribution(marginalStateSpaceProbabilities, T, TrajectorySimFullDistributionFile);

    printf("Hellinger Distance at time %.2f: %f\n", T, HellingerDistance(modelStateSpaceProbabilities, marginalStateSpaceProbabilities));

    fclose(TrajectorySimFullDistributionFile);
    fclose(BTCMFullDistributionFile);
    DeleteMemArena(&arena);
    BTCMDelete(m);
    DeleteSRN(srn);
}

void BTCMGlobalTimeExperiment(void)
{
    SetSeedRandU48(time(NULL));

    SRN* srn = ParseSRN("res/birthDeathModel.txt");

    uint32_t hiddenLayerNeuronCount[] = {16 };
    BTCM* m = BTCMCreate(srn, hiddenLayerNeuronCount, (sizeof(hiddenLayerNeuronCount) / sizeof(uint32_t)), 1.0);

    PrintIntMatrix((srn->stoichiometricMatrix));
    printf("Param count: %lu\n", BTCMGetParamCount(m));

    double deltaT = 0.01;
    double T = 100.0;
    BTCMTrain(m, T, deltaT, 0.1, 500, 1, 500000, "res/BTCMGlobalTimeExperiment/Loss.data");

    size_t sampleCount = 10000;
    double sampleStep = (deltaT * 10.0);

    FILE* meanFile = fopen("res/BTCMGlobalTimeExperiment/mean.data", "w");
    GillespieSRNTrajectorySimLogPerSpeciesMean(srn, T, sampleStep, sampleCount, meanFile);
    fputs("\n\n", meanFile);
    BTCMLogPerSpeciesMean(m, T, sampleStep, sampleCount, meanFile);
    fclose(meanFile);

    FILE* stdFile = fopen("res/BTCMGlobalTimeExperiment/std.data", "w");
    GillespieSRNTrajectorySimLogPerSpeciesStandardDeviation(srn, T, sampleStep, sampleCount, stdFile);
    fputs("\n\n", stdFile);
    BTCMLogPerSpeciesStandardDeviation(m, T, sampleStep, sampleCount, stdFile);
    fclose(stdFile);

    FILE* fullDistributionFile = fopen("res/BTCMGlobalTimeExperiment/fullDistribution.data", "w");
    GillespieSRNTrajectorySimLogFullDistribution((m->srn), T, sampleStep, sampleCount, fullDistributionFile);
    fputs("\n", fullDistributionFile);
    BTCMLogFullProbabilityDistribution(m, T, sampleStep, sampleCount, fullDistributionFile);
    fclose(fullDistributionFile);

    MemArena arena = CreateMemArena((SRNGetStateSpaceTensorAllocSize(srn) * 2));

    FILE* hellingerDistanceFile = fopen("res/BTCMGlobalTimeExperiment/hellingerDistance.data", "w");
    Tensor BTCMProbabilityDistribution = SRNCreateStateSpaceTensor(&arena, srn);
    Tensor GillespieSRNTrajectorySimDistribution = SRNCreateStateSpaceTensor(&arena, srn);
    for(double t = 0.0; t <= T; t += sampleStep)
    {
        BTCMGetFullProbabilityDistribution(m, BTCMProbabilityDistribution, T);
        GillespieSRNTrajectorySimGetFullDistribution(srn, GillespieSRNTrajectorySimDistribution, T, sampleCount);
        fprintf(hellingerDistanceFile, "%f %f\n", t, HellingerDistance(BTCMProbabilityDistribution, GillespieSRNTrajectorySimDistribution));
    }
    fclose(hellingerDistanceFile);

    DeleteMemArena(&arena);
    BTCMDelete(m);
    DeleteSRN(srn);
}

void BTCMSignallingCascadeExperiment(uint32_t M)
{
    SetSeedRandU48(time(NULL));

    SRN* srn = SRNCreateSignallingCascade(M);

    uint32_t hiddenLayerNeuronCount[] = {64, 32 };
    BTCM* m = BTCMCreate(srn, hiddenLayerNeuronCount, (sizeof(hiddenLayerNeuronCount) / sizeof(uint32_t)), 0.02);

    PrintIntMatrix((srn->stoichiometricMatrix));
    printf("Param count: %lu\n", BTCMGetParamCount(m));

    double deltaT = 0.01;
    double T = 10.0;
    BTCMTrain(m, T, deltaT, 0.1, 500, 1, 1000000, "res/BTCMSignallingCascadeExperiment/Loss.data");

    size_t sampleCount = 10000;
    double sampleStep = (deltaT * 10.0);

    FILE* meanFile = fopen("res/BTCMSignallingCascadeExperiment/mean.data", "w");
    GillespieSRNTrajectorySimLogPerSpeciesMean(srn, T, sampleStep, sampleCount, meanFile);
    fputs("\n\n", meanFile);
    BTCMLogPerSpeciesMean(m, T, sampleStep, sampleCount, meanFile);
    fclose(meanFile);

    FILE* stdFile = fopen("res/BTCMSignallingCascadeExperiment/std.data", "w");
    GillespieSRNTrajectorySimLogPerSpeciesStandardDeviation(srn, T, sampleStep, sampleCount, stdFile);
    fputs("\n\n", stdFile);
    BTCMLogPerSpeciesStandardDeviation(m, T, sampleStep, sampleCount, stdFile);
    fclose(stdFile);

    // printf("Hellinger Distance at time %.2f: %f\n", T, HellingerDistance(modelStateSpaceProbabilities, marginalStateSpaceProbabilities));

    BTCMDelete(m);
    DeleteSRN(srn);
}