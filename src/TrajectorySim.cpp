#include "TrajectorySim.hpp"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "Matrix.hpp"
#include "MemArena.hpp"
#include "Random.hpp"
#include "SRN.hpp"


static void saveSpeciesDataPoint(FILE* saveFilePointer, double time, IntMatrix currentState, uint32_t M)
{
    fprintf(saveFilePointer, "%f", time);
    for(uint32_t j = 0; j < M; j++)
        fprintf(saveFilePointer, " %u", GetValueIntMatrix(currentState, j, 0));
    fputs("\n", saveFilePointer);
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
    double propensities[K];
    for(uint32_t e = 0; e < epochs; e++)
    {
        for(uint32_t i = 0; i < M; i++)
            SetValueIntMatrix(currentState, (srn->species[i].initialCount), i, 0);

        double currentTime = 0.0;
        while(currentTime < time)
        {
            saveSpeciesDataPoint(saveFilePointer, currentTime, currentState, M);

            GetReactionPropensities(srn, currentState, propensities);
            double propensitySum = propensities[0];
            for(uint32_t k = 1; k < K; k++)
                propensitySum += propensities[k];

            double r1 = StandardUniformSim(true, false);
            double r2 = StandardOpenUniformSim();

            double deltaT = (log(1.0 / r1) / propensitySum);

            uint32_t activeReactionIndex = 0;
            double propensityComparisonSum = propensities[0];
            while(propensityComparisonSum <= (r2 * propensitySum))
            {
                activeReactionIndex++;
                propensityComparisonSum += propensities[activeReactionIndex];
            }
            
            GetColumnVectorIntMatrix((srn->stoichiometricMatrix), stoichiometricColumn, activeReactionIndex);
            IntMatrixAddSelf(currentState, stoichiometricColumn);

            currentTime += deltaT;
        }
        fputs("\n\n", saveFilePointer);
    }

    DeleteMemArena(&arena);
    fclose(saveFilePointer);
}