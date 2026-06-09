#include "TrajectorySim.hpp"

#include <cstddef>
#include <cstdint>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "Matrix.hpp"
#include "MiscMath.hpp"
#include "MemArena.hpp"
#include "Random.hpp"
#include "SRN.hpp"


static inline void saveSpeciesDataPoint(FILE* saveFilePointer, double time, IntMatrix currentState, uint32_t M)
{
    fprintf(saveFilePointer, "%f", time);
    for(uint32_t j = 0; j < M; j++)
        fprintf(saveFilePointer, " %u", GetValueIntMatrix(currentState, j, 0));
    fputs("\n", saveFilePointer);
}

/*return exponentially distributed deltaT and the reaction index of the reaction that occured at time deltaT*/
static inline void SimReaction(double* deltaT, uint32_t* activeReactionIndex, const SRN* srn, IntMatrix n)
{
    uint32_t K = SRNGetReactionCount(srn);
    double propensities[K];
    GetReactionPropensities(srn, n, propensities);

    double propensitySum = propensities[0];
    for(uint32_t k = 1; k < K; k++)
        propensitySum += propensities[k];

    double r1 = StandardUniformSim(true, false);
    double r2 = StandardOpenUniformSim();

    *deltaT = (log(1.0 / r1) / propensitySum);

    *activeReactionIndex = 0;
    double propensityComparisonSum = propensities[0];
    while(propensityComparisonSum <= (r2 * propensitySum))
    {
        (*activeReactionIndex)++;
        propensityComparisonSum += propensities[(*activeReactionIndex)];
    }
}


void NaiveSRNTrajectorySim(double deltaT, uint64_t timeStepCount, uint32_t epochs, const SRN* srn, const char* saveFileName)
{
    FILE* saveFilePointer = fopen(saveFileName, "w");

    uint32_t M = SRNGetSpeciesCount(srn);
    uint32_t K = SRNGetReactionCount(srn);

    MemArena arena = CreateMemArena(GetIntMatrixAllocSize(M, 1) * 2);

    IntMatrix stoichiometricColumn = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix currentState = CreateBlankIntMatrix(&arena, M, 1);
    double propensities[K];
    for(uint32_t e = 0; e < epochs; e++)
    {
        for(uint32_t i = 0; i < M; i++)
            SetValueIntMatrix(currentState, (srn->species[i].initialCount), i, 0);

        for(uint64_t i = 0; i < timeStepCount; i++)
        {
            saveSpeciesDataPoint(saveFilePointer, (deltaT * (double)i), currentState, M);

            GetReactionPropensities(srn, currentState, propensities);
            for(uint32_t k = 0; k < K; k++)
            {   
                if(BernoulliDistributionSim(propensities[k] * deltaT))
                {
                    GetColumnVectorIntMatrix((srn->stoichiometricMatrix), stoichiometricColumn, k);
                    IntMatrixAddSelf(currentState, stoichiometricColumn);
                }
            }
        }
        fputs("\n\n", saveFilePointer);
    }

    DeleteMemArena(&arena);
    fclose(saveFilePointer);
}

void GillespieSRNTrajectorySim(double time, uint32_t epochs, const SRN* srn, const char* saveFileName)
{
    FILE* saveFilePointer = fopen(saveFileName, "w");

    uint32_t M = SRNGetSpeciesCount(srn);

    MemArena arena = CreateMemArena(GetIntMatrixAllocSize(M, 1) * 2);

    IntMatrix stoichiometricColumn = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix currentState = CreateBlankIntMatrix(&arena, M, 1);
    for(uint32_t e = 0; e < epochs; e++)
    {
        for(uint32_t i = 0; i < M; i++)
            SetValueIntMatrix(currentState, (srn->species[i].initialCount), i, 0);

        double currentTime = 0.0;
        while(currentTime < time)
        {
            saveSpeciesDataPoint(saveFilePointer, currentTime, currentState, M);

            double deltaT;
            uint32_t activeReactionIndex;
            SimReaction(&deltaT, &activeReactionIndex, srn, currentState);
            
            GetColumnVectorIntMatrix((srn->stoichiometricMatrix), stoichiometricColumn, activeReactionIndex);
            IntMatrixAddSelf(currentState, stoichiometricColumn);

            currentTime += deltaT;
        }
        fputs("\n\n", saveFilePointer);
    }

    DeleteMemArena(&arena);
    fclose(saveFilePointer);
}

static inline void GillespieSRNTrajectorySimTakeSample(const SRN* srn, IntMatrix n, IntMatrix stoichiometricColumn, double t)
{
    SetInitialState(srn, n);
    SetIntMatrix(stoichiometricColumn, 0);
    double currentTime = 0.0;
    do
    {
        IntMatrixAddSelf(n, stoichiometricColumn);

        if(!IsValidState(srn, n)) /*This should almost never happen*/
        {
            printf("Invalid state detected; state has been clipped to closest valid state. Consider increasing N. The invalid state:\n");
            PrintIntMatrix(n);
            ClipToValidState(srn, n);
        }

        double deltaT;
        uint32_t activeReactionIndex;
        SimReaction(&deltaT, &activeReactionIndex, srn, n);

        GetColumnVectorIntMatrix((srn->stoichiometricMatrix), stoichiometricColumn, activeReactionIndex);

        currentTime += deltaT;
    }
    while(currentTime < t);
}

static inline void GillespieSRNTrajectorySimGetPerSpeciesMean(const SRN* srn, Matrix* means, double T, double tStep, size_t sampleCount)
{
    uint32_t M = SRNGetSpeciesCount(srn);
    size_t segmentCount = ((size_t)(T / tStep) + 1);

    MemArena arena = CreateMemArena((GetIntMatrixAllocSize(M, 1) * 2));

    IntMatrix stoichiometricColumn = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix currentState = CreateBlankIntMatrix(&arena, M, 1);

    for(size_t i = 0; i < segmentCount; i++)
        SetMatrix(means[i], 0.0);

    for(uint32_t n = 0; n < sampleCount; n++)
    {
        SetInitialState(srn, currentState);
        SetIntMatrix(stoichiometricColumn, 0);
        double currentTime = 0.0;
        for(size_t i = 0; i < segmentCount; i++)
        {
            while(currentTime < ((double)i * tStep))
            {
                IntMatrixAddSelf(currentState, stoichiometricColumn);

                double deltaT;
                uint32_t activeReactionIndex;
                SimReaction(&deltaT, &activeReactionIndex, srn, currentState);

                GetColumnVectorIntMatrix((srn->stoichiometricMatrix), stoichiometricColumn, activeReactionIndex);

                currentTime += deltaT;
            }

            for(uint32_t j = 0; j < M; j++)
                MatrixAddValue(means[i], (double)GetValueIntMatrix(currentState, j, 0), j, 0);
        }
    }

    for(size_t i = 0; i < segmentCount; i++)
        MatrixScaleSelf(means[i], (1.0 / (double)sampleCount));

    DeleteMemArena(&arena);
}

static inline void GillespieSRNTrajectorySimGetProbabilityDistributions(const SRN* srn, Tensor* stateSpaceProbabilities, double T, double tStep, size_t sampleCount)
{
    uint32_t M = SRNGetSpeciesCount(srn);
    size_t segmentCount = ((size_t)(T / tStep) + 1);

    MemArena arena = CreateMemArena((GetIntMatrixAllocSize(M, 1) * 2));

    IntMatrix stoichiometricColumn = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix currentState = CreateBlankIntMatrix(&arena, M, 1);

    for(size_t i = 0; i < segmentCount; i++)
        SetTensor(stateSpaceProbabilities[i], 0.0);

    for(uint32_t n = 0; n < sampleCount; n++)
    {
        SetInitialState(srn, currentState);
        SetIntMatrix(stoichiometricColumn, 0);
        double currentTime = 0.0;
        for(size_t i = 0; i < segmentCount; i++)
        {
            while(currentTime < ((double)i * tStep))
            {
                IntMatrixAddSelf(currentState, stoichiometricColumn);

                double deltaT;
                uint32_t activeReactionIndex;
                SimReaction(&deltaT, &activeReactionIndex, srn, currentState);

                GetColumnVectorIntMatrix((srn->stoichiometricMatrix), stoichiometricColumn, activeReactionIndex);

                currentTime += deltaT;
            }
            TensorAddValue(stateSpaceProbabilities[i], 1.0, currentState);
        }
    }
    for(size_t i = 0; i < segmentCount; i++)
        TensorScaleSelf(stateSpaceProbabilities[i], (1.0 / (double)sampleCount));

    DeleteMemArena(&arena);
}


/*==================== statistics  ====================*/

void GillespieSRNTrajectorySimGetFullDistribution(const SRN* srn, Tensor probabilities, double t, size_t sampleCount)
{
    uint32_t M = SRNGetSpeciesCount(srn);
    
    MemArena arena = CreateMemArena((GetIntMatrixAllocSize(M, 1) * 2));

    IntMatrix stoichiometricColumn = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix currentState = CreateBlankIntMatrix(&arena, M, 1);
    SetTensor(probabilities, 0.0);
    for(size_t n = 0; n < sampleCount; n++)
    {
        GillespieSRNTrajectorySimTakeSample(srn, currentState, stoichiometricColumn, t);
        
        TensorAddValue(probabilities, 1.0, currentState);
    }
    TensorScaleSelf(probabilities, (1.0 / (double)sampleCount));

    DeleteMemArena(&arena);
}

void GillespieSRNTrajectorySimGetPerSpeciesMean(const SRN* srn, Matrix mean, double t, size_t sampleCount)
{
    uint32_t M = SRNGetSpeciesCount(srn);

    MemArena arena = CreateMemArena((GetIntMatrixAllocSize(M, 1) * 2));

    IntMatrix stoichiometricColumn = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix currentState = CreateBlankIntMatrix(&arena, M, 1);
    SetMatrix(mean, 0.0);
    for(uint32_t n = 0; n < sampleCount; n++)
    {
        GillespieSRNTrajectorySimTakeSample(srn, currentState, stoichiometricColumn, t);
        
        for(uint32_t i = 0; i < M; i++)
            MatrixAddValue(mean, (double)GetValueIntMatrix(currentState, i, 0), i, 0);
    }
    MatrixScaleSelf(mean, (1.0 / (double)sampleCount));

    DeleteMemArena(&arena);
}

void GillespieSRNTrajectorySimGetPerSpeciesStandardDeviation(const SRN* srn, Matrix std, double t, size_t sampleCount)
{
    uint32_t M = SRNGetSpeciesCount(srn);

    MemArena arena = CreateMemArena((GetIntMatrixAllocSize(M, 1) * 2) + GetMatrixAllocSize(M, 1));

    Matrix mean = CreateMatrix(&arena, M, 1, NULL);
    GillespieSRNTrajectorySimGetPerSpeciesMean(srn, mean, t, sampleCount);

    IntMatrix stoichiometricColumn = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix sample = CreateBlankIntMatrix(&arena, M, 1);
    
    SetMatrix(std, 0.0);
    for(uint32_t n = 0; n < sampleCount; n++)
    {
        GillespieSRNTrajectorySimTakeSample(srn, sample, stoichiometricColumn, t);
        
        for(uint32_t i = 0; i < M; i++)
            MatrixAddValue(std, pow((double)GetValueIntMatrix(sample, i, 0) - GetValueMatrix(mean, i, 0), 2.0), i, 0);
    }

    MatrixScaleSelf(std, (1.0 / (double)sampleCount));
    MatrixTransformSelf(std, sqrt);

    DeleteMemArena(&arena);
}

void GillespieSRNTrajectorySimLogFullDistribution(const SRN* srn, double T, double tStep, size_t sampleCount, FILE* logFile)
{
    size_t segmentCount = ((size_t)(T / tStep) + 1);

    MemArena arena = CreateMemArena((sizeof(Tensor) * segmentCount) + (SRNGetStateSpaceTensorAllocSize(srn) * segmentCount));

    Tensor* stateSpaceProbabilities = (Tensor*)MemArenaAlloc(&arena, (sizeof(Tensor) * segmentCount));
    for(size_t i = 0; i < segmentCount; i++)
        stateSpaceProbabilities[i] = SRNCreateStateSpaceTensor(&arena, srn);

    GillespieSRNTrajectorySimGetProbabilityDistributions(srn, stateSpaceProbabilities, T, tStep, sampleCount);

    for(size_t i = 0; i < segmentCount; i++)
        LogFullDistribution(stateSpaceProbabilities[i], ((double)i * tStep), logFile);

    DeleteMemArena(&arena);
}

void GillespieSRNTrajectorySimLogPerSpeciesMean(const SRN* srn, double T, double tStep, size_t sampleCount, FILE* logFile)
{
    uint32_t M = SRNGetSpeciesCount(srn);
    size_t segmentCount = ((size_t)(T / tStep) + 1);

    MemArena arena = CreateMemArena((sizeof(Matrix) * segmentCount) + (GetMatrixAllocSize(M, 1) * segmentCount));

    Matrix* means = (Matrix*)MemArenaAlloc(&arena, (sizeof(Matrix) * segmentCount));
    for(size_t i = 0; i < segmentCount; i++)
        means[i] = CreateMatrix(&arena, M, 1, NULL);
    GillespieSRNTrajectorySimGetPerSpeciesMean(srn, means, T, tStep, sampleCount);

    for(size_t i = 0; i < segmentCount; i++)
        LogDataPoint(((double)i * tStep), means[i], logFile);

    DeleteMemArena(&arena);
}

void GillespieSRNTrajectorySimLogPerSpeciesStandardDeviation(const SRN* srn, double T, double tStep, size_t sampleCount, FILE* logFile)
{
    uint32_t M = SRNGetSpeciesCount(srn);
    size_t segmentCount = ((size_t)(T / tStep) + 1);

    MemArena arena = CreateMemArena(
        ((sizeof(Matrix) * segmentCount) * 2) + 
        ((GetMatrixAllocSize(M, 1) * segmentCount) * 2) +
        (GetIntMatrixAllocSize(M, 1) * 2)
    );

    Matrix* means = (Matrix*)MemArenaAlloc(&arena, (sizeof(Matrix) * segmentCount));
    for(size_t i = 0; i < segmentCount; i++)
        means[i] = CreateMatrix(&arena, M, 1, NULL);
    GillespieSRNTrajectorySimGetPerSpeciesMean(srn, means, T, tStep, sampleCount);

    Matrix* stds = (Matrix*)MemArenaAlloc(&arena, (sizeof(Matrix) * segmentCount));
    for(size_t i = 0; i < segmentCount; i++)
        stds[i] = CreateMatrix(&arena, M, 1, NULL);

    IntMatrix stoichiometricColumn = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix currentState = CreateBlankIntMatrix(&arena, M, 1);


    for(size_t i = 0; i < segmentCount; i++)
        SetMatrix(stds[i], 0.0);

    for(uint32_t n = 0; n < sampleCount; n++)
    {
        SetInitialState(srn, currentState);
        SetIntMatrix(stoichiometricColumn, 0);
        double currentTime = 0.0;
        for(size_t i = 0; i < segmentCount; i++)
        {
            while(currentTime < ((double)i * tStep))
            {
                IntMatrixAddSelf(currentState, stoichiometricColumn);

                double deltaT;
                uint32_t activeReactionIndex;
                SimReaction(&deltaT, &activeReactionIndex, srn, currentState);

                GetColumnVectorIntMatrix((srn->stoichiometricMatrix), stoichiometricColumn, activeReactionIndex);

                currentTime += deltaT;
            }

            for(uint32_t j = 0; j < M; j++)
                MatrixAddValue(stds[i], pow((double)GetValueIntMatrix(currentState, j, 0) - GetValueMatrix(means[i], j, 0), 2.0), j, 0);
        }
    }

    for(size_t i = 0; i < segmentCount; i++)
    {
        MatrixScaleSelf(stds[i], (1.0 / (double)sampleCount));
        MatrixTransformSelf(stds[i], sqrt);
        LogDataPoint(((double)i * tStep), stds[i], logFile);
    }

    DeleteMemArena(&arena);
}