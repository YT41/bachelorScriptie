#include "TrajectorySim.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "Matrix.h"
#include "Random.h"
#include "SRN.h"


static double GetPropensity(const uint64_t* speciesCounts, const int32_t* reactantColumn, uint32_t speciesCount, double reactionRate)
{
    double product = 1.0;
    for(uint32_t j = 0; j < speciesCount; j++)
    {
        if((reactantColumn[j] > 0))
        {
            if(speciesCounts[j] < reactantColumn[j])
                return 0.0;
            else
                product *= pow((double)(speciesCounts[j]), (double)(reactantColumn[j]));
        }
    }
    return (product * reactionRate);
}

static void GetReactionPropensities(double* propensities, const uint64_t* speciesCounts, const SRN* srn)
{
    uint32_t speciesCount = SRNGetSpeciesCount(srn);
    int32_t reactantColumn[speciesCount];
    for(uint32_t k = 0; k < SRNGetReactionCount(srn); k++)
    {
        GetColumnMatrix((srn->reactantMatrix), reactantColumn, k);
        propensities[k] = GetPropensity(speciesCounts, reactantColumn, speciesCount, (srn->reactionRates[k]));
    }
}

static void saveSpeciesDataPoint(FILE* saveFilePointer, double time, const uint64_t* speciesCounts, uint32_t speciesCount)
{
    fprintf(saveFilePointer, "%f", time);
    for(uint32_t j = 0; j < speciesCount; j++)
        fprintf(saveFilePointer, " %lu", speciesCounts[j]);
    fputs("\n", saveFilePointer);
}


void NaiveSRNTrajectorySim(double deltaT, uint64_t timeStepCount, uint32_t epochs, const SRN* srn, const char* saveFileName)
{
    FILE* saveFilePointer = fopen(saveFileName, "w");

    uint32_t speciesCount = SRNGetSpeciesCount(srn);
    uint32_t reactionCount = SRNGetReactionCount(srn);

    int32_t stoichiometricColumn[speciesCount];
    uint64_t currentSpeciesCounts[speciesCount];
    double propensities[reactionCount];
    for(uint32_t e = 0; e < epochs; e++)
    {
        for(uint32_t i = 0; i < speciesCount; i++)
            currentSpeciesCounts[i] = (srn->species[i].initialCount);

        for(uint64_t i = 0; i < timeStepCount; i++)
        {
            saveSpeciesDataPoint(saveFilePointer, (deltaT * (double)i), currentSpeciesCounts, speciesCount);

            GetReactionPropensities(propensities, currentSpeciesCounts, srn);
            for(uint32_t k = 0; k < reactionCount; k++)
            {   
                if(BernoulliDistributionSim(propensities[k] * deltaT))
                {
                    GetColumnMatrix((srn->stoichiometricMatrix), stoichiometricColumn, k);

                    for(uint32_t j = 0; j < speciesCount; j++)
                        currentSpeciesCounts[j] += stoichiometricColumn[j];
                }
            }
        }
        fputs("\n\n", saveFilePointer);
    }
    fclose(saveFilePointer);
}

void GillespieSRNTrajectorySim(double time, uint32_t epochs, const SRN* srn, const char* saveFileName)
{
    FILE* saveFilePointer = fopen(saveFileName, "w");

    uint32_t speciesCount = SRNGetSpeciesCount(srn);
    uint32_t reactionCount = SRNGetReactionCount(srn);

    int32_t stoichiometricColumn[speciesCount];
    uint64_t currentSpeciesCounts[speciesCount];
    double propensities[reactionCount];
    for(uint32_t e = 0; e < epochs; e++)
    {
        for(uint32_t i = 0; i < speciesCount; i++)
            currentSpeciesCounts[i] = (srn->species[i].initialCount);

        double currentTime = 0.0;
        while(currentTime < time)
        {
            saveSpeciesDataPoint(saveFilePointer, currentTime, currentSpeciesCounts, speciesCount);

            GetReactionPropensities(propensities, currentSpeciesCounts, srn);
            double propensitySum = propensities[0];
            for(uint32_t k = 1; k < reactionCount; k++)
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
            
            GetColumnMatrix((srn->stoichiometricMatrix), stoichiometricColumn, activeReactionIndex);
            for(uint32_t j = 0; j < speciesCount; j++)
                currentSpeciesCounts[j] += stoichiometricColumn[j];

            currentTime += deltaT;
        }
        fputs("\n\n", saveFilePointer);
    }
    fclose(saveFilePointer);
}