#pragma once

#include <stdint.h>

#include "MemArena.h"
#include "Matrix.h"


typedef struct Species
{
    uint64_t initialCount;
    char name[32];
} Species;

typedef struct Reaction
{
    MemArena arena;

    int32_t* reactantColumn; /*the column in the reactant matrix belonging to this reaction*/
    int32_t* productColumn; /*the column in the product matrix belonging to this reaction*/
    uint32_t speciesCount;
    
    double reactionRate;
} Reaction;

typedef struct SRN
{
    MemArena arena;

    IntMatrixNxM reactantMatrix; /*speciesCount x reactionCount dimensional matrix*/
    IntMatrixNxM productMatrix; /*speciesCount x reactionCount dimensional matrix*/
    IntMatrixNxM stoichiometricMatrix; /*speciesCount x reactionCount dimensional matrix*/
    double* reactionRates;

    Species* species;
} SRN;


Reaction* CreateReaction(double reactionRate, uint32_t speciesCount, int32_t* reactantColumnVals, int32_t* productColumnVals);
void DeleteReaction(Reaction* reaction);

SRN* CreateSRN(uint32_t reactionCount, Reaction** reactions, Species* species);
void DeleteSRN(SRN* srn);

uint32_t SRNGetReactionCount(const SRN* srn);
uint32_t SRNGetSpeciesCount(const SRN* srn);