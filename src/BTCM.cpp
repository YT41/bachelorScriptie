#include "BTCM.hpp"

#include "Matrix.hpp"
#include "MemArena.hpp"
#include "MiscMath.hpp"
#include "NeuralNetwork.hpp"
#include "TrajectorySim.hpp"
#include "Random.hpp"
#include "ReactionParser.hpp"
#include "SRN.hpp"
#include "generalModelFunctions.hpp"
#include <cmath>
#include <cstdio>
#include <math.h>
#include <cstddef>
#include <cstdint>
#include <stdio.h>
#include <ctime>


/*============================ helper functions ============================*/

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

BTCM* BTCMCreate(SRN* srn, uint32_t* neuronsPerHiddenLayer, uint32_t hiddenLayerCount, uint32_t timeEmbeddingDim, double learningRate)
{
    uint32_t M = SRNGetSpeciesCount(srn);
    uint32_t K = SRNGetReactionCount(srn);

    uint32_t inputTokenDim = 1 + timeEmbeddingDim + (K * 3) + (M - 1);

    MemArena arena = CreateMemArena(sizeof(BTCM) + GetMatrixAllocSize(inputTokenDim, M));

    BTCM* ret = (BTCM*)MemArenaAlloc(&arena, sizeof(BTCM));

    ret->batchCache = CreateMatrix(&arena, inputTokenDim, M, NULL);

    /*specify neurons per layer for MLP*/
    uint32_t neuronsPerLayer[hiddenLayerCount + 2];
    neuronsPerLayer[0] = inputTokenDim;
    for(uint32_t i = 1; i <= hiddenLayerCount; i++) { neuronsPerLayer[i] = neuronsPerHiddenLayer[i - 1]; }
    neuronsPerLayer[hiddenLayerCount + 1] = SRNGetMaxSpeciesCount(srn);

    /*specify activation function per layer for MLP*/
    ActivationFnID activationFnPerLayer[hiddenLayerCount + 1];
    for(uint32_t i = 0; i < hiddenLayerCount; i++) { activationFnPerLayer[i] = TANH; }
    activationFnPerLayer[hiddenLayerCount] = SOFTMAX;

    /*specify neurons and activation functions per layer for learnable time embedding*/
    uint32_t neuronsPerLayerTimeEmbedding[2] = { 2, timeEmbeddingDim }; /*input is (t, log(t + eps))*/
    ActivationFnID activationFnPerLayerTimeEmbedding[1] = { IDENTITY };

    ret->arena = arena;
    ret->MLP = NNCreate(neuronsPerLayer, activationFnPerLayer, hiddenLayerCount, M, learningRate);
    ret->timeEmbedding = NNCreate(neuronsPerLayerTimeEmbedding, activationFnPerLayerTimeEmbedding, 0, 1, learningRate);

    ret->srn = srn;

    return ret;
}

void BTCMDelete(BTCM* m)
{
    NNDelete((m->timeEmbedding));
    NNDelete((m->MLP));
    DeleteMemArena(&(m->arena));
}

size_t BTCMGetParamCount(const BTCM* m)
{
    return (NNGetParamCount((m->MLP)) + NNGetParamCount((m->timeEmbedding)));
}

BTCM* BTCMCopy(const BTCM* m)
{
    uint32_t hiddenLayerCount = (m->MLP->hiddenLayerCount);
    uint32_t neuronsPerHiddenLayer[hiddenLayerCount];
    for(uint32_t i = 0; i < hiddenLayerCount; i++)
        neuronsPerHiddenLayer[i] = (m->MLP->layerVectors[i+1].rowCount);
    uint32_t timeEmbeddingDim = NNGetOutputDimension((m->timeEmbedding));

    BTCM* ret = BTCMCreate((m->srn), neuronsPerHiddenLayer, hiddenLayerCount, timeEmbeddingDim, (m->MLP->learningRate));

    BTCMCopyParameters(ret, m);

    return ret;
}

void BTCMCopyParameters(BTCM* dest, const BTCM* src)
{
    NNCopyParameters((dest->MLP), (src->MLP));
    NNCopyParameters((dest->timeEmbedding), (src->timeEmbedding));
}


/*doing it this way, it takes a sample and returns the probability of having taken this sample at the same time*/
double BTCMTakeSample(BTCM* m, IntMatrix s, double t, Matrix desiredNudgesMLPOutput)
{
    if(t == 0.0)
        return GetInitialConditionSample((m->srn), s);

    SetMatrix(desiredNudgesMLPOutput, 0.0);

    Matrix embeddedTime = GetEmbeddedTime((m->timeEmbedding), t);

    uint32_t M = SRNGetSpeciesCount((m->srn));
    double conditionalProbabilityProduct = 1.0;
    for(uint32_t i = 0; i < M; i++)
    {
        GetSingleInputToken((m->srn), (m->batchCache), i, s, embeddedTime);
        Matrix out = NNPredictSingleDataPoint((m->MLP), i, (m->batchCache));

        /*simulate a count for species i using generated conditional probabilities*/
        uint32_t sim = PickUintWithChances(out.data, out.rowCount);
        SetValueIntMatrix(s, sim, i, 0);

        double conditionalProbability = GetValueMatrix(out, GetValueIntMatrix(s, i, 0), 0);

        /*set desired nudges*/
        MatrixSubSelf(GetColumnVectorMatrix(desiredNudgesMLPOutput, i), out);
        MatrixAddValue(desiredNudgesMLPOutput, 1.0, sim, i);

        conditionalProbabilityProduct *= conditionalProbability;
    }

    return conditionalProbabilityProduct;
}

double BTCMTakeSampleNoGradient(BTCM* m, IntMatrix s, double t)
{
    if(t == 0.0)
        return GetInitialConditionSample((m->srn), s);

    Matrix embeddedTime = GetEmbeddedTime((m->timeEmbedding), t);

    uint32_t M = SRNGetSpeciesCount((m->srn));
    double conditionalProbabilityProduct = 1.0;
    for(uint32_t i = 0; i < M; i++)
    {
        GetSingleInputToken((m->srn), (m->batchCache), i, s, embeddedTime);
        Matrix out = NNPredictSingleDataPoint((m->MLP), i, (m->batchCache));

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

    Matrix embeddedTime = GetEmbeddedTime((m->timeEmbedding), t);
    GetInputTokens((m->srn), (m->batchCache), n, embeddedTime);
    Matrix out = NNPredict((m->MLP), (m->batchCache));

    /*TODO: make general function for calculating conditionalProbabilityProduct*/
    uint32_t M = SRNGetSpeciesCount((m->srn));
    double conditionalProbabilityProduct = 1.0;
    for(uint32_t i = 0; i < M; i++)
        conditionalProbabilityProduct *= GetValueMatrix(out, GetValueIntMatrix(n, i, 0), i);

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
        else /*TODO: test for clipping infinite values that we can allow, instead of throwing it away*/
            MatrixAddSelf(totalDesiredNudgesMLPOutput, desiredNudgesMLPOutput[b]);
    }
    MatrixScaleSelf(totalDesiredNudgesMLPOutput, (1.0 / (double)B));
}

static inline void ClampTargetProbability(double* probability)
{
    static uint64_t totalSampledStateCounter = 0;
    static uint64_t negativeProbabilityCounter = 0;

    if((*probability) < 0.0)
    {
        negativeProbabilityCounter++;

        double negativeProbabilityPercentage = (((double)negativeProbabilityCounter / (double)totalSampledStateCounter) * 100.0);
        if(((totalSampledStateCounter % 1000) == 0) && (negativeProbabilityPercentage > 1.0))
            printf("So far %.2f%% of target probabilities have been negative. Consider lowering deltaT\n", negativeProbabilityPercentage);
    }

    const double eps = 1e-10;
    (*probability) = MAX((*probability), eps);

    totalSampledStateCounter++;
}

static inline double GenerateTrainTime(uint64_t epoch, uint64_t totalEpochs, double T, double deltaT, double p)
{
    if(BernoulliDistributionSim(p)) /*to learn initial condition*/
        return 0.0;

    /*This version picks t uniformly from 0 to (T - deltaT)*/
    //return (StandardClosedUniformSim() * (T - deltaT));

    /*This version picks t uniformly from a linearly growing interval with max size (T - deltaT) and min size (T - deltaT)/4*/
    return (StandardClosedUniformSim() * Lerp(((T - deltaT) / 4.0), (T - deltaT), ((double)epoch / (double)totalEpochs)));

    /*This version picks t uniformly from a linearly growing interval with max size (T - deltaT)*/
    //return (StandardClosedUniformSim() * Lerp(0.0, (T - deltaT), ((double)epoch / (double)totalEpochs)));

    /*This version picks t uniformly from a linearly growing interval with max size (T - deltaT) reached after completing half of all epochs*/
    //return (StandardClosedUniformSim() * MIN(((1.0 * (double)epoch) / (double)totalEpochs) * (T - deltaT), (T - deltaT)));

    /*this version makes it more likely to generate smaller t*/
    // const double eps = 1e-10;
    // double logt = UniformSim(log(eps), log(T - deltaT), false , false);
    // return exp(logt);

    /*this version does make the interval linearly grow with the number of epochs, it also makes it more likely to generate smaller t*/
    // const double eps = 1e-10;
    // double logt = UniformSim(log(eps), log(((double)epoch / (double)totalEpochs) * (T - deltaT)), false , false);
    // return exp(logt);

    /*this version makes the interval linearly grow for the first quarter of epochs, it also makes it more likely to generate smaller t*/
    // double intervalEnd = MIN(((4.0 * (double)epoch) / (double)totalEpochs) * (T - deltaT), (T - deltaT));
    // const double eps = 1e-10;
    // double logt = UniformSim(log(eps), log(((double)epoch / (double)totalEpochs) * (T - deltaT)), false , false);
    // return exp(logt);
}

static inline void BTCMBackPropagation(BTCM* m, Matrix totalDesiredNudgesMLPOutput)
{
    NNSetLastLayer((m->MLP), totalDesiredNudgesMLPOutput);
    NNBackPropagation((m->MLP));

    /*set the right gradient with respects to the last layer for time embedding, this is simply the average gradient for the psi(t) input of the MLP*/
    uint32_t M = NNGetBatchSize((m->MLP));
    Matrix timeEmbeddingOut = NNGetLastLayer((m->timeEmbedding));
    SetMatrix(timeEmbeddingOut, 0.0);
    for(uint32_t i = 0; i < M; i++)
        MatrixAddSelf(timeEmbeddingOut, GetSubColumnVectorMatrix(NNGetFirstLayer((m->MLP)), i, 1, 32));
    MatrixScaleSelf(timeEmbeddingOut, (1.0 / (double)M));

    NNBackPropagation((m->timeEmbedding));
}

void BTCMTrain(BTCM* m, double T, double deltaT, double p, uint32_t B, uint32_t Q, uint64_t epochs, const char* logFile)
{
    FILE* lossFile = fopen(logFile, "w");

    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(
        (GetIntMatrixAllocSize(M, 1) * 2) + 
        (GetMatrixAllocSize(NNGetOutputDimension((m->MLP)), M) * (1 + B)) + 
        (B * sizeof(Matrix))
    );

    BTCM* targetModelCopy = BTCMCopy(m);

    IntMatrix sample = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix previousState = CreateBlankIntMatrix(&arena, M, 1);

    /*for gradient descent*/
    Matrix totalDesiredNudgesMLPOutput = CreateMatrix(&arena, NNGetOutputDimension((m->MLP)), M, NULL);
    Matrix* desiredNudgesMLPOutput = (Matrix*)MemArenaAlloc(&arena, (B * sizeof(Matrix)));
    for(uint32_t b = 0; b < B; b++)
        desiredNudgesMLPOutput[b] = CreateMatrix(&arena, NNGetOutputDimension((m->MLP)), M, NULL);
    for(uint64_t e = 0; e < epochs; e++)
    {
        double t = GenerateTrainTime(e, epochs, T, deltaT, p);

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

            /*prevents NaN value for loss, it is a sign that deltaT is too big*/
            ClampTargetProbability(&targetProbability);

            rewards[b] = (log(sampleProbability) - log(targetProbability));
            rewardBaseline += rewards[b];

            loss += rewards[b];
        }
        loss /= (double)B;
        rewardBaseline /= (double)B;

        GetTotalGradient(B, rewardBaseline, rewards, desiredNudgesMLPOutput, totalDesiredNudgesMLPOutput);
        BTCMBackPropagation(m, totalDesiredNudgesMLPOutput);

        /*log the loss to file*/
        if((e % 1000) == 0) { printf("loss of epoch %lu: %f\n", e, loss); }
        if(((e % 10) == 0) && isfinite(loss)) { fprintf(lossFile, "%lu %f\n", e, loss); }
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

// void BTCMGetProbabilityDistribution(BTCM* m, double t, uint32_t* speciesIndices, uint32_t speciesCount, Tensor dest)
// {
//     uint32_t M = SRNGetSpeciesCount((m->srn));

//     MemArena arena = CreateMemArena(GetIntMatrixAllocSize(M, 1));

//     IntMatrix n = CreateBlankIntMatrix(&arena, M, 1);
//     SetIntMatrix(n, 0);
//     do
//     {
//         SetValueTensor(dest, BTCMPredict(m, n, t), n);
//         IncrementStateInStateSpace((m->srn), n);
//     }
//     while(!IntMatrixIsZero(n));

//     DeleteMemArena(&arena);
// }

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

/*get part of the Probability Distribution, only for the species in speciesIndices*/
// void BTCMLogProbabilityDistribution(BTCM* m, double T, double tStep, size_t sampleCount, uint32_t* speciesIndices, uint32_t speciesCount, FILE* logFile)
// {
//     size_t segmentCount = ((size_t)(T / tStep) + 1);

//     MemArena arena = CreateMemArena(SRNGetStateSpaceTensorAllocSize((m->srn)));
    
//     Tensor stateSpaceProbabilities = SRNCreateStateSpaceTensor(&arena, (m->srn));
//     for(size_t i = 0; i < segmentCount; i++)
//     {
//         BTCMGetFullProbabilityDistribution(m, ((double)i * tStep), stateSpaceProbabilities);
//         LogFullDistribution(stateSpaceProbabilities, ((double)i * tStep), logFile);
//     }

//     DeleteMemArena(&arena);
// }


/*==================== experiments ====================*/

void BTCMSingleTimeStepExperiment(void)
{
    SetSeedRandU48(time(NULL));

    SRN* srn = ParseSRN("res/birthDeathModel.txt");
    uint32_t M = SRNGetSpeciesCount(srn);

    double T = 1.0;
    uint32_t hiddenLayerNeuronCount[] = {16 };
    BTCM* m = BTCMCreate(srn, hiddenLayerNeuronCount, (sizeof(hiddenLayerNeuronCount) / sizeof(uint32_t)), 32, 0.5);

    PrintIntMatrix((srn->stoichiometricMatrix));
    printf("Param count: %lu\n", BTCMGetParamCount(m));

    MemArena arena = CreateMemArena(
        (GetMatrixAllocSize(M, 1) * 2) + 
        (SRNGetStateSpaceTensorAllocSize(srn) * 2)
    );

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

    double deltaT = 0.01;
    double T = 100.0;

    uint32_t hiddenLayerNeuronCount[] = {16 };
    BTCM* m = BTCMCreate(srn, hiddenLayerNeuronCount, (sizeof(hiddenLayerNeuronCount) / sizeof(uint32_t)), 32, 1.0);

    PrintIntMatrix((srn->stoichiometricMatrix));
    printf("Param count: %lu\n", BTCMGetParamCount(m));

    BTCMTrain(m, T, deltaT, 0.1, 500, 1, 500000, "res/BTCMGlobalTimeExperiment/Loss.data");


    /*quick test for whether the time embedding is working correctly*/
    double testTimes[6] = {0.001, 0.01, 0.1, 1.0, 5.0, 10.0 };
    for(uint32_t i = 0; i < 6; i++)
    {
        PrintMatrix(GetEmbeddedTime((m->timeEmbedding), testTimes[i]));
        printf("\n\n");
    }


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

void BTCMGeneExpressionExperiment(void)
{
    SetSeedRandU48(time(NULL));

    SRN* srn = ParseSRN("res/GeneExpressionModel.txt");

    double deltaT = 0.1;
    double T = 3600.0;
    uint32_t hiddenLayerNeuronCount[] = {64 };
    BTCM* m = BTCMCreate(srn, hiddenLayerNeuronCount, (sizeof(hiddenLayerNeuronCount) / sizeof(uint32_t)), 32, 0.005);

    PrintIntMatrix((srn->stoichiometricMatrix));
    printf("Param count: %lu\n", BTCMGetParamCount(m));

    BTCMTrain(m, T, deltaT, 0.1, 2000, 1, 100000, "res/BTCMGeneExpressionModelExperiment/Loss.data");

    size_t sampleCount = 10000;
    double sampleStep = (deltaT * 100.0);

    FILE* meanFile = fopen("res/BTCMGeneExpressionModelExperiment/mean.data", "w");
    GillespieSRNTrajectorySimLogPerSpeciesMean(srn, T, sampleStep, sampleCount, meanFile);
    fputs("\n\n", meanFile);
    BTCMLogPerSpeciesMean(m, T, sampleStep, sampleCount, meanFile);
    fclose(meanFile);

    FILE* stdFile = fopen("res/BTCMGeneExpressionModelExperiment/std.data", "w");
    GillespieSRNTrajectorySimLogPerSpeciesStandardDeviation(srn, T, sampleStep, sampleCount, stdFile);
    fputs("\n\n", stdFile);
    BTCMLogPerSpeciesStandardDeviation(m, T, sampleStep, sampleCount, stdFile);
    fclose(stdFile);

    // FILE* fullDistributionFile = fopen("res/BTCMGeneExpressionModelExperiment/fullDistribution.data", "w");
    // GillespieSRNTrajectorySimLogFullDistribution((m->srn), 0.5, 0.1, sampleCount, fullDistributionFile);
    // fputs("\n", fullDistributionFile);
    // BTCMLogFullProbabilityDistribution(m, 0.5, 0.1, sampleCount, fullDistributionFile);
    // fclose(fullDistributionFile);

    MemArena arena = CreateMemArena((SRNGetStateSpaceTensorAllocSize(srn) * 2));

    FILE* hellingerDistanceFile = fopen("res/BTCMGeneExpressionModelExperiment/hellingerDistance.data", "w");
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

    double deltaT = 0.01;
    double T = 10.0;
    uint32_t hiddenLayerNeuronCount[] = {128 };
    BTCM* m = BTCMCreate(srn, hiddenLayerNeuronCount, (sizeof(hiddenLayerNeuronCount) / sizeof(uint32_t)), 32, 0.015);

    PrintIntMatrix((srn->stoichiometricMatrix));
    printf("Param count: %lu\n", BTCMGetParamCount(m));

    BTCMTrain(m, T, deltaT, 0.1, 1000, 1, 200000, "res/BTCMSignallingCascadeExperiment/Loss.data");

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

    
    if(M <= 5) /*for larger M, getting the full distribution hellinger distance would become too computationally expensive*/
    {
        MemArena arena = CreateMemArena((SRNGetStateSpaceTensorAllocSize(srn) * 2));

        FILE* hellingerDistanceFile = fopen("res/BTCMSignallingCascadeExperiment/hellingerDistance.data", "w");
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
    }

    BTCMDelete(m);
    DeleteSRN(srn);
}