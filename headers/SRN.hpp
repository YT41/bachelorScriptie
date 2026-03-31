#pragma once

#include <cstdint>
#include <stdint.h>

#include "MemArena.hpp"
#include "Matrix.hpp"


typedef struct Species
{
    char name[32];
    uint32_t initialCount;
    uint32_t maxCount;
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

static inline uint32_t SRNGetReactionCount(const SRN* srn) { return (srn->stoichiometricMatrix.columnCount); }
static inline uint32_t SRNGetSpeciesCount(const SRN* srn) { return (srn->stoichiometricMatrix.rowCount); }

