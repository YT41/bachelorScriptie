#include "SRN.hpp"
#include "Matrix.hpp"

#include <cmath>
#include <cstdint>
#include <stdint.h>
#include <string.h>


Reaction* CreateReaction(double reactionRate, uint32_t speciesCount, int32_t* reactantColumnVals, int32_t* productColumnVals)
{
    MemArena arena = CreateMemArena( 
        sizeof(Reaction) +
        (sizeof(int32_t) * speciesCount) +
        (sizeof(int32_t) * speciesCount)
    );
    Reaction* reaction = (Reaction*)MemArenaAlloc(&arena, sizeof(Reaction));

    reaction->arena = arena;

    reaction->reactantColumn = (int32_t*)MemArenaAlloc(&(reaction->arena), (sizeof(int32_t) * speciesCount));
    memmove((reaction->reactantColumn), reactantColumnVals, (sizeof(int32_t) * speciesCount));

    reaction->productColumn = (int32_t*)MemArenaAlloc(&(reaction->arena), (sizeof(int32_t) * speciesCount));
    memmove((reaction->productColumn), productColumnVals, (sizeof(int32_t) * speciesCount));

    reaction->speciesCount = speciesCount;
    reaction->reactionRate = reactionRate;

    return reaction;
}

void DeleteReaction(Reaction* reaction)
{
    DeleteMemArena(&(reaction->arena));
}


SRN* CreateSRN(uint32_t reactionCount, Reaction** reactions, Species* species)
{
    if(reactionCount == 0)
        return NULL;

    uint32_t speciesCount = (reactions[0]->speciesCount);
    
    MemArena arena = CreateMemArena( 
        sizeof(SRN) +
        GetIntMatrixAllocSize(speciesCount, reactionCount) + 
        GetIntMatrixAllocSize(speciesCount, reactionCount) + 
        GetIntMatrixAllocSize(speciesCount, reactionCount) + 
        (sizeof(double) * reactionCount) + 
        (sizeof(Species) * speciesCount)
    );
    SRN* srn = (SRN*)MemArenaAlloc(&arena, sizeof(SRN));
    srn->arena = arena;

    srn->reactantMatrix = CreateBlankIntMatrix(&(srn->arena), speciesCount, reactionCount);
    srn->productMatrix = CreateBlankIntMatrix(&(srn->arena), speciesCount, reactionCount);
    srn->stoichiometricMatrix = CreateBlankIntMatrix(&(srn->arena), speciesCount, reactionCount);

    srn->reactionRates = (double*)MemArenaAlloc(&(srn->arena), (sizeof(double) * reactionCount));
    srn->species = (Species*)MemArenaAlloc(&(srn->arena), (sizeof(Species) * speciesCount));

    for(uint32_t i = 0; i < reactionCount; i++)
    {
        SetColumnIntMatrix((srn->reactantMatrix), (reactions[i]->reactantColumn), i);
        SetColumnIntMatrix((srn->productMatrix), (reactions[i]->productColumn), i);

        for(uint32_t j = 0; j < speciesCount; j++)
            SetValueIntMatrix((srn->stoichiometricMatrix), ((reactions[i]->productColumn[j]) - (reactions[i]->reactantColumn[j])), j, i);

        srn->reactionRates[i] = reactions[i]->reactionRate;
    }

    memmove((void*)(srn->species), (void*)species, (sizeof(Species) * speciesCount));

    return srn;
}

void DeleteSRN(SRN* srn)
{
    DeleteMemArena(&(srn->arena));
}


uint32_t SRNGetMaxSpeciesCount(const SRN* srn)
{ 
    uint32_t maxCountAllSpecies = 0;
    for(uint32_t i = 0; i < SRNGetSpeciesCount(srn); i++)
    {
        if((srn->species[i].maxCount) > maxCountAllSpecies)
            maxCountAllSpecies = (srn->species[i].maxCount);
    }
    return maxCountAllSpecies; 
}

double GetPropensity(const SRN* srn, IntMatrix n, uint32_t reactionIndex)
{
    double product = 1.0;
    for(uint32_t i = 0; i < SRNGetSpeciesCount(srn); i++)
    {
        int32_t reactantCount = GetValueIntMatrix((srn->reactantMatrix), i, reactionIndex);
        if((reactantCount > 0))
        {
            int32_t countJ = GetValueIntMatrix(n, i, 0);
            if(countJ < reactantCount)
                return 0.0;
            else
                product *= pow((double)countJ, (double)reactantCount);
        }
    }
    return (product * (srn->reactionRates[reactionIndex]));
}

void GetReactionPropensities(const SRN* srn, IntMatrix n, double* propensities)
{
    for(uint32_t k = 0; k < SRNGetReactionCount(srn); k++)
        propensities[k] = GetPropensity(srn, n, k);
}

double GetEscapeRate(const SRN* srn, IntMatrix n)
{
    double sum = 0.0;
    for(uint32_t k = 0; k < SRNGetReactionCount(srn); k++)
        sum += GetPropensity(srn, n, k); 
    return sum;
}

double GetPreviousConnectedState(const SRN* srn, IntMatrix currentState, IntMatrix previousState, uint32_t reactionIndex)
{
    for(uint32_t i = 0; i < SRNGetSpeciesCount(srn); i++)
    {
        SetValueIntMatrix(previousState, (GetValueIntMatrix(currentState, i, 0) - GetValueIntMatrix((srn->stoichiometricMatrix), i, reactionIndex)), i, 0);
        int32_t ni = GetValueIntMatrix(previousState, i, 0);
        if((ni < 0) || (ni >= (int32_t)(srn->species[i].maxCount)))
            return 0.0; /*not a valid previous state*/
    }

    return GetPropensity(srn, previousState, reactionIndex);
}