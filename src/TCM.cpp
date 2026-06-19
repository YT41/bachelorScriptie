#include "TCM.hpp"

#include "Attention.hpp"
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

static inline double GetTargetProbability(TCM* m, IntMatrix sample, IntMatrix previousState, double t, double deltaT)
{
    /*we know that we can only enter the sample state from connected states, all other propensities are 0*/
    double enteringSampleStateProbability = 0.0;
    for(uint32_t k = 0; k < SRNGetReactionCount((m->srn)); k++)
    {
        double possiblePreviousStatePropensity = GetPreviousConnectedState((m->srn), sample, previousState, k);

        if(possiblePreviousStatePropensity != 0.0)
            enteringSampleStateProbability += (possiblePreviousStatePropensity * TCMPredict(m, previousState, t, 0.0));
    }
    double tSampleProbability = TCMPredict(m, sample, t, 0.0);
    /*note that this is just one step of explicit euler with the approximation of CME*/
    return (tSampleProbability + (deltaT * (enteringSampleStateProbability - (GetEscapeRate((m->srn), sample) * tSampleProbability))));
}

static inline bool AttentionEnabled(const TCM* m) { return ((m->a) != NULL); }


/*============================ public functions ============================*/

TCM* TCMCreate(SRN* srn, uint32_t* neuronsPerHiddenLayer, uint32_t hiddenLayerCount, uint32_t timeEmbeddingDim, uint32_t attentionDim)
{
    uint32_t M = SRNGetSpeciesCount(srn);
    uint32_t K = SRNGetReactionCount(srn);

    uint32_t inputTokenDim = 1 + timeEmbeddingDim + (K * 3) + (M - 1);

    MemArena arena = CreateMemArena(sizeof(TCM) + GetMatrixAllocSize(inputTokenDim, M));

    TCM* ret = (TCM*)MemArenaAlloc(&arena, sizeof(TCM));

    ret->batchCache = CreateMatrix(&arena, inputTokenDim, M, NULL);

    /*specify neurons per layer for MLP*/
    uint32_t neuronsPerLayer[hiddenLayerCount + 2];
    neuronsPerLayer[0] = ((attentionDim == 0) ? inputTokenDim : attentionDim);
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
    ret->MLP = NNCreate(neuronsPerLayer, activationFnPerLayer, hiddenLayerCount, M);
    ret->timeEmbedding = NNCreate(neuronsPerLayerTimeEmbedding, activationFnPerLayerTimeEmbedding, 0, 1);
    ret->a = ((attentionDim == 0) ? NULL : AMCreate(M, inputTokenDim, attentionDim));

    ret->srn = srn;

    return ret;
}

void TCMDelete(TCM* m)
{
    if(AttentionEnabled(m)) { AMDelete((m->a)); }
    NNDelete((m->timeEmbedding));
    NNDelete((m->MLP));
    DeleteMemArena(&(m->arena));
}

size_t TCMGetParamCount(const TCM* m)
{
    return (
        NNGetParamCount((m->MLP)) + 
        NNGetParamCount((m->timeEmbedding)) +
        (AttentionEnabled(m) ? AMGetParamCount((m->a)) : 0)
    );
}

TCM* TCMCopy(const TCM* m)
{
    uint32_t hiddenLayerCount = (m->MLP->hiddenLayerCount);
    uint32_t neuronsPerHiddenLayer[hiddenLayerCount];
    for(uint32_t i = 0; i < hiddenLayerCount; i++)
        neuronsPerHiddenLayer[i] = (m->MLP->layerVectors[i+1].rowCount);
    uint32_t timeEmbeddingDim = NNGetOutputDimension((m->timeEmbedding));
    uint32_t attentionDim = (AttentionEnabled(m) ? AMGetOutputDimension((m->a)) : 0);

    TCM* ret = TCMCreate((m->srn), neuronsPerHiddenLayer, hiddenLayerCount, timeEmbeddingDim, attentionDim);

    TCMCopyParameters(ret, m);

    return ret;
}

void TCMCopyParameters(TCM* dest, const TCM* src)
{
    NNCopyParameters((dest->MLP), (src->MLP));
    if(AttentionEnabled(src)) { AMCopyParameters((dest->a), (src->a)); }
    NNCopyParameters((dest->timeEmbedding), (src->timeEmbedding));
}

static inline Matrix GetAttentionToMLPOutSingleToken(TCM* m, uint32_t i, double dropoutProbability)
{
    if(AttentionEnabled(m))
    { 
        Matrix attOut = AMGetSingleMaskedAttention((m->a), i, (m->batchCache));
        return NNPredictSingleDataPoint((m->MLP), i, attOut, dropoutProbability);
    }
    else
        return NNPredictSingleDataPoint((m->MLP), i, GetColumnVectorMatrix((m->batchCache), i), dropoutProbability);
}

static inline Matrix GetAttentionToMLPOut(TCM* m, double dropoutProbability)
{
    if(AttentionEnabled(m))
    { 
        Matrix attOut = AMGetMaskedAttention((m->a), (m->batchCache));
        return NNPredict((m->MLP), attOut, dropoutProbability);
    }
    else
        return NNPredict((m->MLP), (m->batchCache), dropoutProbability);

}

double TCMTakeSample(TCM* m, IntMatrix s, double t, Matrix gradient, double dropoutProbability)
{
    double probability = TCMTakeSampleNoGradient(m, s, t, dropoutProbability);
    Matrix out = NNGetLastLayer((m->MLP));

    SetMatrix(gradient, 0.0);
    MatrixSubSelf(gradient, out);
    for(uint32_t i = 0; i < SRNGetSpeciesCount((m->srn)); i++)
        MatrixAddValue(gradient, 1.0, GetValueIntMatrix(s, i, 0), i);

    return probability;
}

/*doing it this way, it takes a sample and returns the probability of having taken this sample at the same time*/
double TCMTakeSampleNoGradient(TCM* m, IntMatrix s, double t, double dropoutProbability)
{
    if(t == 0.0)
        return GetInitialConditionSample((m->srn), s);

    Matrix embeddedTime = GetEmbeddedTime((m->timeEmbedding), t, dropoutProbability);

    uint32_t M = SRNGetSpeciesCount((m->srn));
    double conditionalProbabilityProduct = 1.0;
    for(uint32_t i = 0; i < M; i++)
    {
        GetSingleInputToken((m->srn), (m->batchCache), i, s, embeddedTime);
        Matrix out = GetAttentionToMLPOutSingleToken(m, i, dropoutProbability);

        /*simulate a count for species i using generated conditional probabilities*/
        uint32_t sim = PickUintWithChances(out.data, out.rowCount);
        SetValueIntMatrix(s, sim, i, 0);

        conditionalProbabilityProduct *= GetValueMatrix(out, GetValueIntMatrix(s, i, 0), 0);
    }

    return conditionalProbabilityProduct;
}

/*assumes n is already known*/
double TCMPredict(TCM* m, IntMatrix n, double t, double dropoutProbability)
{
    /*in this case we know the probabilities exactly*/
    if(t == 0.0)
        return GetInitialConditionProbability((m->srn), n);

    Matrix embeddedTime = GetEmbeddedTime((m->timeEmbedding), t, dropoutProbability);
    GetInputTokens((m->srn), (m->batchCache), n, embeddedTime);
    Matrix out = GetAttentionToMLPOut(m, dropoutProbability);

    uint32_t M = SRNGetSpeciesCount((m->srn));
    double conditionalProbabilityProduct = 1.0;
    for(uint32_t i = 0; i < M; i++)
        conditionalProbabilityProduct *= GetValueMatrix(out, GetValueIntMatrix(n, i, 0), i);

    return conditionalProbabilityProduct;
}

/*naive way of calculating full distribution, should be good enough for any application though*/
void TCMGetFullProbabilityDistribution(TCM* m, Tensor probabilities, double t)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(GetIntMatrixAllocSize(M, 1));

    IntMatrix n = CreateBlankIntMatrix(&arena, M, 1);
    SetIntMatrix(n, 0);
    do
    {
        SetValueTensor(probabilities, TCMPredict(m, n, t, 0.0), n);
        IncrementTensorIndex(probabilities, n);
    }
    while(!IntMatrixIsZero(n));

    DeleteMemArena(&arena);
}

static inline void GetTotalGradient(uint32_t B, double rewardBaseline, const double* rewards, Matrix* desiredNudgesMLPOutput, Matrix totalDesiredNudgesMLPOutput)
{
    for(uint32_t b = 0; b < B; b++)
    {
        MatrixScaleSelf(desiredNudgesMLPOutput[b], (rewards[b] - rewardBaseline));
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
    //return (StandardClosedUniformSim() * Lerp(((T - deltaT) / 4.0), (T - deltaT), ((double)epoch / (double)totalEpochs)));

    /*This version picks t uniformly from a linearly growing interval with max size (T - deltaT)*/
    return (StandardClosedUniformSim() * Lerp(0.0, (T - deltaT), ((double)(epoch + 1) / (double)totalEpochs)));

    /*This version picks t uniformly from a linearly growing interval with max size (T - deltaT) reached after completing half of all epochs*/
    //return (StandardClosedUniformSim() * MIN(((2.0 * (double)epoch) / (double)totalEpochs) * (T - deltaT), (T - deltaT)));

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

static inline void TCMBackwardPass(TCM* m, Matrix desiredNudgesMLPOutput)
{
    NNSetLastLayer((m->MLP), desiredNudgesMLPOutput);
    NNBackwardPass((m->MLP));

    /*set the right gradient with respects to the last layer for time embedding (and attention mechanism if enabled)*/
    uint32_t M = NNGetBatchSize((m->MLP));
    Matrix timeEmbeddingOut = NNGetLastLayer((m->timeEmbedding));
    SetMatrix(timeEmbeddingOut, 0.0);
    if(AttentionEnabled(m))
    {
        CopyMatrixData((m->a->O), NNGetFirstLayer((m->MLP)));
        AMBackwardPass((m->a));

        for(uint32_t i = 0; i < M; i++)
            MatrixAddSelf(timeEmbeddingOut, GetSubColumnVectorMatrix((m->a->X), i, 1, (timeEmbeddingOut.rowCount)));
    }
    else
    {
        for(uint32_t i = 0; i < M; i++)
            MatrixAddSelf(timeEmbeddingOut, GetSubColumnVectorMatrix(NNGetFirstLayer((m->MLP)), i, 1, (timeEmbeddingOut.rowCount)));
    }
    MatrixScaleSelf(timeEmbeddingOut, (1.0 / (double)M));

    NNBackwardPass((m->timeEmbedding));
}

static inline void TCMGradientDescent(TCM* m, double learningRate)
{
    NNGradientDescent((m->MLP), learningRate);
    if(AttentionEnabled(m)) { AMGradientDescent((m->a), learningRate); }
    NNGradientDescent((m->timeEmbedding), learningRate);
}

void TCMTrain(TCM* m, double T, double deltaT, double p, uint32_t B, uint32_t Q, uint64_t epochs, double learningRate, double dropoutProbability, const char* logFile)
{
    FILE* lossFile = fopen(logFile, "w");

    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(
        (GetIntMatrixAllocSize(M, 1) * 2) + 
        GetMatrixAllocSize(NNGetOutputDimension((m->MLP)), M)
    );

    TCM* targetModelCopy = TCMCopy(m);

    IntMatrix sample = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix previousState = CreateBlankIntMatrix(&arena, M, 1);

    Matrix desiredNudgesMLPOutput = CreateMatrix(&arena, NNGetOutputDimension((m->MLP)), M, NULL);

    const double alpha = 0.95;
    double rewardBaseline = 0.0;
    for(uint64_t e = 0; e < epochs; e++)
    {
        double t = GenerateTrainTime(e, epochs, T, deltaT, p);

        /*update parameters every Q epochs*/
        if(((e + 1) % Q) == 0) { TCMCopyParameters(targetModelCopy, m); }

        double loss = 0.0; /*KL-divergence a.k.a. cross-entropy*/
        for(uint32_t b = 0; b < B; b++)
        {
            double sampleProbability = TCMTakeSample(m, sample, (t + deltaT), desiredNudgesMLPOutput, dropoutProbability);
            double targetProbability = GetTargetProbability(targetModelCopy, sample, previousState, t, deltaT);

            ClampTargetProbability(&targetProbability); /*prevents NaN value for loss, it is a sign that deltaT is too big*/

            double reward = (log(sampleProbability) - log(targetProbability));

            MatrixScaleSelf(desiredNudgesMLPOutput, (reward - rewardBaseline));
            TCMBackwardPass(m, desiredNudgesMLPOutput);

            loss += reward;
        }
        loss /= (double)B;

        rewardBaseline = (alpha * rewardBaseline) + ((1.0 - alpha) * loss); /*EMA baseline for next epoch*/

        TCMGradientDescent(m, (learningRate / (double)B));

        /*log the loss to file*/
        if((e % 1000) == 0)
        { 
            if(AttentionEnabled(m)) { printf("\n"); PrintMatrix(AMGetA((m->a))); printf("\n"); }
            printf("loss of epoch %lu: %f\n", e, loss); 
        }
        if((e % 10) == 0) { fprintf(lossFile, "%lu %f\n", e, loss); }
    }

    TCMDelete(targetModelCopy);
    DeleteMemArena(&arena);
    fclose(lossFile);
}


/*==================== statistics ====================*/

static inline void PrintStateProbability(TCM* m, IntMatrix n, double t)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    printf("P(");
    for(uint32_t i = 0; i < (M - 1); i++)
        printf("n_%u = %d, ", i, GetValueIntMatrix(n, i, 0));
    printf("n_%u = %d | t = %.3f) = %f\n", (M - 1), GetValueIntMatrix(n, (M - 1), 0), t, TCMPredict(m, n, t, 0.0));
}

void TCMGetPerSpeciesMean(TCM* m, Matrix mean, double t, size_t sampleCount)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena((GetIntMatrixAllocSize(M, 1) * 2));

    IntMatrix sample = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix sampleSum = CreateBlankIntMatrix(&arena, M, 1);
    SetIntMatrix(sampleSum, 0);

    for(uint32_t n = 0; n < sampleCount; n++)
    {
        TCMTakeSampleNoGradient(m, sample, t, 0.0);
        IntMatrixAddSelf(sampleSum, sample);
    }

    for(uint32_t i = 0; i < M; i++)
        SetValueMatrix(mean, ((double)GetValueIntMatrix(sampleSum, i, 0) / (double)sampleCount), i, 0);

    DeleteMemArena(&arena);
}

void TCMGetPerSpeciesStandardDeviation(TCM* m, Matrix std, double t, size_t sampleCount)
{    
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(GetMatrixAllocSize(M, 1) + GetIntMatrixAllocSize(M, 1));

    Matrix mean = CreateMatrix(&arena, M, 1, NULL);
    TCMGetPerSpeciesMean(m, mean, t, sampleCount);

    IntMatrix sample = CreateBlankIntMatrix(&arena, M, 1);

    SetMatrix(std, 0.0);
    for(uint32_t n = 0; n < sampleCount; n++)
    {
        TCMTakeSampleNoGradient(m, sample, t, 0.0);

        for(uint32_t i = 0; i < M; i++)
            MatrixAddValue(std, pow((double)GetValueIntMatrix(sample, i, 0) - GetValueMatrix(mean, i, 0), 2.0), i, 0);
    }

    MatrixScaleSelf(std, (1.0 / (double)sampleCount));
    MatrixTransformSelf(std, sqrt);

    DeleteMemArena(&arena);
}

static inline void TCMLogPerSpeciesMean(TCM* m, double T, double tStep, size_t sampleCount, FILE* logFile)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(GetMatrixAllocSize(M, 1));

    size_t segmentCount = ((size_t)(T / tStep) + 1);
    Matrix mean = CreateMatrix(&arena, M, 1, NULL);
    for(size_t i = 0; i < segmentCount; i++)
    {
        TCMGetPerSpeciesMean(m, mean, ((double)i * tStep), sampleCount);
        LogDataPoint(((double)i * tStep), mean, logFile);
    }

    DeleteMemArena(&arena);
}

static inline void TCMLogPerSpeciesStandardDeviation(TCM* m, double T, double tStep, size_t sampleCount, FILE* logFile)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(GetMatrixAllocSize(M, 1));

    size_t segmentCount = ((size_t)(T / tStep) + 1);
    Matrix std = CreateMatrix(&arena, M, 1, NULL);
    for(size_t i = 0; i < segmentCount; i++)
    {
        TCMGetPerSpeciesStandardDeviation(m, std, ((double)i * tStep), sampleCount);
        LogDataPoint(((double)i * tStep), std, logFile);
    }

    DeleteMemArena(&arena);
}

void TCMPrintFullProbabilityDistribution(TCM* m, double t)
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

static inline void TCMLogFullProbabilityDistribution(TCM* m, double T, double tStep, size_t sampleCount, FILE* logFile)
{
    size_t segmentCount = ((size_t)(T / tStep) + 1);

    MemArena arena = CreateMemArena(SRNGetStateSpaceTensorAllocSize((m->srn)));
    
    Tensor stateSpaceProbabilities = SRNCreateStateSpaceTensor(&arena, (m->srn));
    for(size_t i = 0; i < segmentCount; i++)
    {
        TCMGetFullProbabilityDistribution(m, stateSpaceProbabilities, ((double)i * tStep));
        LogFullDistribution(stateSpaceProbabilities, ((double)i * tStep), logFile);
    }

    DeleteMemArena(&arena);
}

/*get part of the Probability Distribution, only for the species in speciesIndices (not finished! but might be handy for some future experiments)*/
// void BTCMLogProbabilityDistribution(BTCM* m, double T, double tStep, size_t sampleCount, uint32_t* speciesIndices, uint32_t speciesCount, FILE* logFile)
// {
//     size_t segmentCount = ((size_t)(T / tStep) + 1);

//     MemArena arena = CreateMemArena(SRNGetStateSpaceTensorAllocSize((m->srn)));
    
//     Tensor stateSpaceProbabilities = SRNCreateStateSpaceTensor(&arena, (m->srn));
//     for(size_t i = 0; i < segmentCount; i++)
//     {
//         TCMGetFullProbabilityDistribution(m, ((double)i * tStep), stateSpaceProbabilities);
//         LogFullDistribution(stateSpaceProbabilities, ((double)i * tStep), logFile);
//     }

//     DeleteMemArena(&arena);
// }

/*==================== experiment helper functions ====================*/

void TCMLogMeanComparisonOverTime(TCM* m, double T, double sampleStep, size_t sampleCount, const char* fileName)
{
    FILE* meanFile = fopen(fileName, "w");
    GillespieSRNTrajectorySimLogPerSpeciesMean((m->srn), T, sampleStep, sampleCount, meanFile);
    fputs("\n\n", meanFile);
    TCMLogPerSpeciesMean(m, T, sampleStep, sampleCount, meanFile);
    fclose(meanFile);
}

void TCMLogStdComparisonOverTime(TCM* m, double T, double sampleStep, size_t sampleCount, const char* fileName)
{
    FILE* stdFile = fopen(fileName, "w");
    GillespieSRNTrajectorySimLogPerSpeciesStandardDeviation((m->srn), T, sampleStep, sampleCount, stdFile);
    fputs("\n\n", stdFile);
    TCMLogPerSpeciesStandardDeviation(m, T, sampleStep, sampleCount, stdFile);
    fclose(stdFile);
}

void TCMLogFullDistributionComparisonOverTime(TCM* m, double T, double sampleStep, size_t sampleCount, const char* fileName)
{
    FILE* fullDistributionFile = fopen(fileName, "w");
    GillespieSRNTrajectorySimLogFullDistribution((m->srn), T, sampleStep, sampleCount, fullDistributionFile);
    fputs("\n", fullDistributionFile);
    TCMLogFullProbabilityDistribution(m, T, sampleStep, sampleCount, fullDistributionFile);
    fclose(fullDistributionFile);
}

void TCMLogHellingerDistanceToGillespieOverTime(TCM* m, double T, double sampleStep, size_t sampleCount, const char* fileName)
{
    MemArena arena = CreateMemArena((SRNGetStateSpaceTensorAllocSize((m->srn)) * 2));

    FILE* hellingerDistanceFile = fopen(fileName, "w");
    Tensor TCMProbabilityDistribution = SRNCreateStateSpaceTensor(&arena, (m->srn));
    Tensor GillespieSRNTrajectorySimDistribution = SRNCreateStateSpaceTensor(&arena, (m->srn));
    for(double t = 0.0; t <= T; t += sampleStep)
    {
        TCMGetFullProbabilityDistribution(m, TCMProbabilityDistribution, T);
        GillespieSRNTrajectorySimGetFullDistribution((m->srn), GillespieSRNTrajectorySimDistribution, T, sampleCount);
        fprintf(hellingerDistanceFile, "%f %f\n", t, HellingerDistance(TCMProbabilityDistribution, GillespieSRNTrajectorySimDistribution));
    }
    fclose(hellingerDistanceFile);
    DeleteMemArena(&arena);
}


/*==================== experiments ====================*/

void TCMSingleTimeStepExperiment(void)
{
    SetSeedRandU48(time(NULL));

    SRN* srn = ParseSRN("res/birthDeathModel.txt");
    uint32_t M = SRNGetSpeciesCount(srn);

    double T = 1.0;
    uint32_t hiddenLayerNeuronCount[] = {16 };
    TCM* m = TCMCreate(srn, hiddenLayerNeuronCount, (sizeof(hiddenLayerNeuronCount) / sizeof(uint32_t)), 1, 0);

    PrintIntMatrix((srn->stoichiometricMatrix));
    printf("Param count: %lu\n", TCMGetParamCount(m));

    MemArena arena = CreateMemArena(
        (GetMatrixAllocSize(M, 1) * 2) + 
        (SRNGetStateSpaceTensorAllocSize(srn) * 2)
    );

    TCMTrain(m, T, T, 1.0, 1500, 1, 50000, 0.5, 0.0, "res/TCMSingleTimeStepExperiment/Loss.data");

    TCMPrintFullProbabilityDistribution(m, T);

    size_t sampleCount = 10000;
    Matrix mean = CreateMatrix(&arena, M, 1, NULL);
    Matrix std = CreateMatrix(&arena, M, 1, NULL);

    TCMGetPerSpeciesMean(m, mean, T, sampleCount);
    TCMGetPerSpeciesStandardDeviation(m, std, T, sampleCount);

    printf("TCM mean: ");
    PrintMatrix(mean);
    printf("TCM standard deviation: ");
    PrintMatrix(std);

    GillespieSRNTrajectorySimGetPerSpeciesMean(srn, mean, T, sampleCount);
    GillespieSRNTrajectorySimGetPerSpeciesStandardDeviation(srn, std, T, sampleCount);

    printf("Gillespie trajectory simulation mean: ");
    PrintMatrix(mean);
    printf("Gillespie trajectory simulation standard deviation: ");
    PrintMatrix(std);

    Tensor modelStateSpaceProbabilities = SRNCreateStateSpaceTensor(&arena, srn);
    Tensor marginalStateSpaceProbabilities = SRNCreateStateSpaceTensor(&arena, srn);
    TCMGetFullProbabilityDistribution(m, modelStateSpaceProbabilities, T);
    GillespieSRNTrajectorySimGetFullDistribution(srn, marginalStateSpaceProbabilities, T, sampleCount);

    FILE* TCMFullDistributionFile = fopen("res/TCMSingleTimeStepExperiment/TCMFullDistribution.data", "w");
    FILE* TrajectorySimFullDistributionFile = fopen("res/TCMSingleTimeStepExperiment/TrajectorySimFullDistribution.data", "w");

    LogFullDistribution(modelStateSpaceProbabilities, T, TCMFullDistributionFile);
    LogFullDistribution(marginalStateSpaceProbabilities, T, TrajectorySimFullDistributionFile);

    printf("Hellinger Distance at time %.2f: %f\n", T, HellingerDistance(modelStateSpaceProbabilities, marginalStateSpaceProbabilities));

    fclose(TrajectorySimFullDistributionFile);
    fclose(TCMFullDistributionFile);
    DeleteMemArena(&arena);
    TCMDelete(m);
    DeleteSRN(srn);
}

void TCMGlobalTimeExperiment(void)
{
    SetSeedRandU48(time(NULL));

    SRN* srn = ParseSRN("res/birthDeathModel.txt");

    double deltaT = 0.01;
    double T = 100.0;

    uint32_t hiddenLayerNeuronCount[] = { 16 };
    TCM* m = TCMCreate(srn, hiddenLayerNeuronCount, (sizeof(hiddenLayerNeuronCount) / sizeof(uint32_t)), 1, 0);

    PrintIntMatrix((srn->stoichiometricMatrix));
    printf("Param count: %lu\n", TCMGetParamCount(m));

    clock_t startTime = clock();
    TCMTrain(m, T, deltaT, 0.01, 1000, 1, 500000, 1.0, 0.0, "res/TCMGlobalTimeExperiment/Loss.data");
    clock_t endTime = clock();

    double trainTimeHours = ((double)(endTime - startTime) / (double)(CLOCKS_PER_SEC * 60 * 60));
    printf("Training (CPU) time in hours: %f\n", trainTimeHours);

    double testTimes[] = { 0.001, 0.01, 0.1, 1.0, 5.0, 10.0 };
    for(uint32_t i = 0; i < 6; i++)
    {
        PrintMatrix(GetEmbeddedTime((m->timeEmbedding), testTimes[i], 0.0));
        printf("\n");
    }

    size_t sampleCount = 10000;
    double sampleStep = (deltaT * 10.0);

    TCMLogMeanComparisonOverTime(m, T, sampleStep, sampleCount, "res/TCMGlobalTimeExperiment/mean.data");
    TCMLogStdComparisonOverTime(m, T, sampleStep, sampleCount, "res/TCMGlobalTimeExperiment/std.data");
    TCMLogFullDistributionComparisonOverTime(m, T, sampleStep, sampleCount, "res/TCMGlobalTimeExperiment/fullDistribution.data");
    TCMLogHellingerDistanceToGillespieOverTime(m, T, sampleStep, sampleCount, "res/TCMGlobalTimeExperiment/hellingerDistance.data");

    TCMDelete(m);
    DeleteSRN(srn);
}

void TCMGeneExpressionExperiment(void)
{
    SetSeedRandU48(time(NULL));

    SRN* srn = ParseSRN("res/GeneExpressionModel.txt");

    double deltaT = 0.1;
    double T = 3600.0;
    uint32_t hiddenLayerNeuronCount[] = { 64 };
    TCM* m = TCMCreate(srn, hiddenLayerNeuronCount, (sizeof(hiddenLayerNeuronCount) / sizeof(uint32_t)), 32, 32);

    PrintIntMatrix((srn->stoichiometricMatrix));
    printf("Param count: %lu\n", TCMGetParamCount(m));

    clock_t startTime = clock();
    TCMTrain(m, T, deltaT, 0.01, 1000, 1, 300000, 0.002, 0.0, "res/TCMGeneExpressionModelExperiment/Loss.data");
    clock_t endTime = clock();

    double trainTimeHours = ((double)(endTime - startTime) / (double)(CLOCKS_PER_SEC * 60 * 60));
    printf("Training (CPU) time in hours: %f\n", trainTimeHours);

    size_t sampleCount = 10000;
    double sampleStep = (deltaT * 100.0);

    TCMLogMeanComparisonOverTime(m, T, sampleStep, sampleCount, "res/TCMGeneExpressionModelExperiment/mean.data");
    TCMLogStdComparisonOverTime(m, T, sampleStep, sampleCount, "res/TCMGeneExpressionModelExperiment/std.data");
    TCMLogFullDistributionComparisonOverTime(m, T, sampleStep, sampleCount, "res/TCMGeneExpressionModelExperiment/fullDistribution.data");
    TCMLogHellingerDistanceToGillespieOverTime(m, T, sampleStep, sampleCount, "res/TCMGeneExpressionModelExperiment/hellingerDistance.data");

    TCMDelete(m);
    DeleteSRN(srn);
}

void TCMSignalingCascadeExperiment(uint32_t M)
{
    SetSeedRandU48(time(NULL));

    SRN* srn = SRNCreateSignalingCascade(M, 20, 0);

    double deltaT = 0.01;
    double T = 10.0;
    uint32_t hiddenLayerNeuronCount[] = { 64 };
    TCM* m = TCMCreate(srn, hiddenLayerNeuronCount, (sizeof(hiddenLayerNeuronCount) / sizeof(uint32_t)), 32, 0);

    PrintIntMatrix((srn->stoichiometricMatrix));
    printf("Param count: %lu\n", TCMGetParamCount(m));

    clock_t startTime = clock();
    TCMTrain(m, T, deltaT, 0.01, 1000, 1, 100000, 0.01, 0.0, "res/TCMSignalingCascadeExperiment/Loss.data");
    clock_t endTime = clock();

    double trainTimeHours = ((double)(endTime - startTime) / (double)(CLOCKS_PER_SEC * 60 * 60));
    printf("Training (CPU) time in hours: %f\n", trainTimeHours);

    size_t sampleCount = 10000;
    double sampleStep = (deltaT * 10.0);

    TCMLogMeanComparisonOverTime(m, T, sampleStep, sampleCount, "res/TCMSignalingCascadeExperiment/mean.data");
    TCMLogStdComparisonOverTime(m, T, sampleStep, sampleCount, "res/TCMSignalingCascadeExperiment/std.data");

    if(M <= 4) /*for larger M, getting the full distribution hellinger distance would become too computationally expensive*/
    {
        TCMLogFullDistributionComparisonOverTime(m, T, sampleStep, sampleCount, "res/TCMSignalingCascadeExperiment/fullDistribution.data");
        TCMLogHellingerDistanceToGillespieOverTime(m, T, sampleStep, sampleCount, "res/TCMSignalingCascadeExperiment/hellingerDistance.data");
    }

    TCMDelete(m);
    DeleteSRN(srn);
}

void TCMTimeEmbeddingExperiment()
{
    SetSeedRandU48(time(NULL));

    SRN* srn = SRNCreateSignalingCascade(2, 15, 0);

    double deltaT = 0.01;
    double T = 1.0;
    uint32_t hiddenLayerNeuronCount[] = { 32 };
    TCM* m = TCMCreate(srn, hiddenLayerNeuronCount, (sizeof(hiddenLayerNeuronCount) / sizeof(uint32_t)), 32, 0);

    PrintIntMatrix((srn->stoichiometricMatrix));
    printf("Param count: %lu\n", TCMGetParamCount(m));

    clock_t startTime = clock();
    TCMTrain(m, T, deltaT, 0.01, 1000, 1, 100000, 0.01, 0.0, "res/TCMTimeEmbeddingExperiment/Loss.data");
    clock_t endTime = clock();

    double trainTimeHours = ((double)(endTime - startTime) / (double)(CLOCKS_PER_SEC * 60 * 60));
    printf("Training (CPU) time in hours: %f\n", trainTimeHours);

    size_t sampleCount = 10000;
    double sampleStep = deltaT;

    TCMLogMeanComparisonOverTime(m, T, sampleStep, sampleCount, "res/TCMTimeEmbeddingExperiment/mean.data");
    TCMLogStdComparisonOverTime(m, T, sampleStep, sampleCount, "res/TCMTimeEmbeddingExperiment/std.data");
    TCMLogFullDistributionComparisonOverTime(m, T, sampleStep, sampleCount, "res/TCMTimeEmbeddingExperiment/fullDistribution.data");
    TCMLogHellingerDistanceToGillespieOverTime(m, T, sampleStep, sampleCount, "res/TCMTimeEmbeddingExperiment/hellingerDistance.data");

    TCMDelete(m);
    DeleteSRN(srn);
}