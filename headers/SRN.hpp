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

    IntMatrix reactantMatrix; /*speciesCount x reactionCount dimensional matrix*/
    IntMatrix productMatrix; /*speciesCount x reactionCount dimensional matrix*/
    IntMatrix stoichiometricMatrix; /*speciesCount x reactionCount dimensional matrix*/
    double* reactionRates;

    Species* species;
} SRN;


Reaction* CreateReaction(double reactionRate, uint32_t speciesCount, int32_t* reactantColumnVals, int32_t* productColumnVals);
void DeleteReaction(Reaction* reaction);

SRN* CreateSRN(uint32_t reactionCount, Reaction** reactions, Species* species);
void DeleteSRN(SRN* srn);

static inline uint32_t SRNGetReactionCount(const SRN* srn) { return (srn->stoichiometricMatrix.columnCount); }
static inline uint32_t SRNGetSpeciesCount(const SRN* srn) { return (srn->stoichiometricMatrix.rowCount); }
uint32_t SRNGetMaxSpeciesCount(const SRN* srn);

Tensor SRNCreateStateSpaceTensor(MemArena* arena, const SRN* srn);
void IncrementStateInStateSpace(const SRN* srn, IntMatrix n); /*iterates n through the state space*/

double GetPropensity(const SRN* srn, IntMatrix n, uint32_t reactionIndex);
void GetReactionPropensities(const SRN* srn, IntMatrix n, double* propensities);
double GetEscapeRate(const SRN* srn, IntMatrix n);

/*returns propensity of previous state with respects to reaction, previous state is set into previousState*/
double GetPreviousConnectedState(const SRN* srn, IntMatrix currentState, IntMatrix previousState, uint32_t reactionIndex);

void SetInitialState(const SRN* srn, IntMatrix n);