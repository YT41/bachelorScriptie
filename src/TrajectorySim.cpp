#include "TrajectorySim.hpp"

#include <cstdint>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "Matrix.hpp"
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
static inline void SimReaction(double* deltaT, uint32_t* activeReactionIndex, const SRN* srn, IntMatrix n, uint32_t K)
{
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


/*TODO: refactored but not tested*/
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

/*TODO: refactored but not tested*/
void GillespieSRNTrajectorySim(double time, uint32_t epochs, const SRN* srn, const char* saveFileName)
{
    FILE* saveFilePointer = fopen(saveFileName, "w");

    uint32_t M = SRNGetSpeciesCount(srn);
    uint32_t K = SRNGetReactionCount(srn);

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
            SimReaction(&deltaT, &activeReactionIndex, srn, currentState, K);
            
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
    uint32_t K = SRNGetReactionCount(srn);

    SetInitialState(srn, n);
    SetIntMatrix(stoichiometricColumn, 0);
    double currentTime = 0.0;
    do
    {
        IntMatrixAddSelf(n, stoichiometricColumn);

        double deltaT;
        uint32_t activeReactionIndex;
        SimReaction(&deltaT, &activeReactionIndex, srn, n, K);

        GetColumnVectorIntMatrix((srn->stoichiometricMatrix), stoichiometricColumn, activeReactionIndex);

        currentTime += deltaT;
    }
    while(currentTime < t);
}

void GetFullMarginalDistributionGillespieSRNTrajectorySim(const SRN* srn, Tensor probabilities, double t, uint32_t epochs)
{
    uint32_t M = SRNGetSpeciesCount(srn);
    
    MemArena arena = CreateMemArena((GetIntMatrixAllocSize(M, 1) * 2));

    IntMatrix stoichiometricColumn = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix currentState = CreateBlankIntMatrix(&arena, M, 1);
    SetTensor(probabilities, 0.0);
    for(uint32_t e = 0; e < epochs; e++)
    {
        GillespieSRNTrajectorySimTakeSample(srn, currentState, stoichiometricColumn, t);
        
        TensorAddValue(probabilities, 1.0, currentState);
    }
    TensorScaleSelf(probabilities, (1.0 / (double)epochs));

    DeleteMemArena(&arena);
}

void GillespieSRNTrajectorySimGetPerSpeciesMean(const SRN* srn, Matrix mean, double t)
{
    uint32_t M = SRNGetSpeciesCount(srn);

    MemArena arena = CreateMemArena((GetIntMatrixAllocSize(M, 1) * 2));

    IntMatrix stoichiometricColumn = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix currentState = CreateBlankIntMatrix(&arena, M, 1);
    SetMatrix(mean, 0.0);

    const uint32_t sampleCount = 1000;
    for(uint32_t n = 0; n < sampleCount; n++)
    {
        GillespieSRNTrajectorySimTakeSample(srn, currentState, stoichiometricColumn, t);
        
        for(uint32_t i = 0; i < M; i++)
            MatrixAddValue(mean, (double)GetValueIntMatrix(currentState, i, 0), i, 0);
    }
    MatrixScaleSelf(mean, (1.0 / (double)sampleCount));

    DeleteMemArena(&arena);
}

void GillespieSRNTrajectorySimGetPerSpeciesStandardDeviation(const SRN* srn, Matrix std, double t)
{
    uint32_t M = SRNGetSpeciesCount(srn);

    MemArena arena = CreateMemArena((GetIntMatrixAllocSize(M, 1) * 2) + GetMatrixAllocSize(M, 1));

    Matrix mean = CreateMatrix(&arena, M, 1, NULL);
    GillespieSRNTrajectorySimGetPerSpeciesMean(srn, mean, t);

    IntMatrix stoichiometricColumn = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix sample = CreateBlankIntMatrix(&arena, M, 1);
    
    SetMatrix(std, 0.0);

    const uint32_t sampleCount = 1000;
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