#include "SRN.hpp"

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
        SetColumnMatrix((srn->reactantMatrix), (reactions[i]->reactantColumn), i);
        SetColumnMatrix((srn->productMatrix), (reactions[i]->productColumn), i);

        for(uint32_t j = 0; j < speciesCount; j++)
            SetValueMatrix((srn->stoichiometricMatrix), ((reactions[i]->productColumn[j]) - (reactions[i]->reactantColumn[j])), j, i);

        srn->reactionRates[i] = reactions[i]->reactionRate;
    }

    memmove((void*)(srn->species), (void*)species, (sizeof(Species) * speciesCount));

    return srn;
}

void DeleteSRN(SRN* srn)
{
    DeleteMemArena(&(srn->arena));
}

uint32_t SRNGetReactionCount(const SRN* srn)
{
    return (srn->stoichiometricMatrix.ColumnCount);
}

uint32_t SRNGetSpeciesCount(const SRN* srn)
{
    return (srn->stoichiometricMatrix.RowCount);
}