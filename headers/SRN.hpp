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


/*Creates an instance of a reaction, see section 2.1 of the thesis report

INPUTS:
double reactionRate: reaction rate of reaction
uint32_t speciesCount: amount of species, M
int32_t* reactantColumnVals: C-style array of the reactant vector belonging to reaction, indexed by species index
int32_t* productColumnVals: C-style array of the product vector belonging to reaction, indexed by species index

RETURNS:
Heap-allocated pointer to Reaction structure, be sure to delete it with DeleteReaction whenever you're not using it anymore to prevent memory leaks.*/
Reaction* CreateReaction(double reactionRate, uint32_t speciesCount, int32_t* reactantColumnVals, int32_t* productColumnVals);
void DeleteReaction(Reaction* reaction);


/*Creates an instance of a stochastic reaction network, see section 2 of the thesis report

INPUTS:
uint32_t reactionCount: the amount of reaction in the SRN, K
Reaction** reactions: C-style array of pointers to Reaction structures, these will all be added to the SRN
Species* species: C-style array of Species structures, these species will all be added to the SRN, note that these need to be indexed the same as they are indexed in the Reaction structures

RETURNS:
Heap-allocated pointer to SRN structure, be sure to delete it with DeleteSRN whenever you're not using it anymore to prevent memory leaks.*/
SRN* CreateSRN(uint32_t reactionCount, Reaction** reactions, Species* species);
void DeleteSRN(SRN* srn);


/*Creates an instance of a linear signaling cascade stochastic reaction network, see section 4.3 of the thesis report

INPUTS:
uint32_t M: amount of species in the linear signaling cascade
uint32_t N: truncation count for all species, set higher the lower M is, 20 seems to be enough for M = 1
uint32_t initialCount: initial species count of all species (initial probability distribution is thus the delta distribution)

RETURNS:
Heap-allocated pointer to SRN structure, be sure to delete it with DeleteSRN whenever you're not using it anymore to prevent memory leaks.*/
SRN* SRNCreateSignalingCascade(uint32_t M, uint32_t N, uint32_t initialCount);

static inline uint32_t SRNGetReactionCount(const SRN* srn) { return (srn->stoichiometricMatrix.columnCount); }
static inline uint32_t SRNGetSpeciesCount(const SRN* srn) { return (srn->stoichiometricMatrix.rowCount); }
uint32_t SRNGetMaxSpeciesCount(const SRN* srn); /*internally, all species have their own truncation counts, say N_i, this returns max{N_1, ..., N_M}*/

size_t SRNGetStateSpaceSize(const SRN* srn); /*returns truncated state space size of srn*/
size_t SRNGetStateSpaceTensorAllocSize(const SRN* srn); /*returns memory allocation size for a state space tensor, for state space of srn*/


/*Creates a tensor, allocated on arena. It has an axis for every species, each axis has the states {0, ..., N_i-1}, where N_i is the truncation count for that species

INPUTS:
MemArena* arena: pointer to memory arena structure to allocate the tensor on
const SRN* srn: pointer to stochastic reaction network structure to get state space tensor for

RETURNS:
state space tensor of srn*/
Tensor SRNCreateStateSpaceTensor(MemArena* arena, const SRN* srn);


/*Iterates n through the state space, kind of works like adding 1 to a number in arabic numerals, where each numeral represents a species count
if we call this function 3 times on the state (9, 0), where the truncation count is 10, then we get:
(9, 0) --> (0, 1) --> (1, 1) --> (2, 1)

INPUTS:
const SRN* srn: pointer to stochastic reaction network structure to increment n in
IntMatrix n: species count vector to increment state for, so this is changed when calling this function*/
void IncrementStateInStateSpace(const SRN* srn, IntMatrix n);


bool IsValidState(const SRN* srn, IntMatrix n); /*just checks whether n is a valid state in the state space of srn*/
void ClipToValidState(const SRN* srn, IntMatrix n); /*does nothing whenever the state is already valid, otherwise clips the state n to the closest valid state*/

double GetInitialConditionProbability(const SRN* srn, IntMatrix n); /*returns the probability of being in the state n at time t = 0, n is not changed*/
double GetInitialConditionSample(const SRN* srn, IntMatrix s); /*a sample is taken from P(0), s is set to this sample and it returns the probability of being in the state s at time t = 0*/


/*function to get propensity k (reactionIndex) evaluated in n, see section 2.2 in the thesis report.
Where we assumed stochastic mass-action.

INPUTS:
const SRN* srn: pointer to stochastic reaction network structure to get propensities from
IntMatrix n: species count vector to evaluate propensity in, isn't changed after function call, assumes that the state is valid otherwise behavior is undefined
uint32_t reactionIndex: index of reaction to get propensity for (usually denoted k in the thesis report)

RETURNS:
propensity k (reactionIndex) evaluated in n*/
double GetPropensity(const SRN* srn, IntMatrix n, uint32_t reactionIndex);
void GetReactionPropensities(const SRN* srn, IntMatrix n, double* propensities); /*just repeatedly calls GetPropensity for every reaction, puts the propensities in the C-Style array "propensities"*/


/*Gets the escape rate away from the state n, this is just the sum of all the propensities evaluated in n, see equation 2.13 in thesis report

INPUTS:
const SRN* srn: pointer to stochastic reaction network structure to get propensities from
IntMatrix n: species count vector to evaluate escape rate in, isn't changed after function call

RETURNS:
escape rate away from the state n*/
double GetEscapeRate(const SRN* srn, IntMatrix n);


/*Gets the previous state with respects to reaction k (reactionIndex), which is: (currentState - \nu_k), where \nu_k is the stoichiometric vector associated with reaction k

INPUTS:
const SRN* srn: pointer to stochastic reaction network structure, for context
IntMatrix currentState: species count vector to go back to previous state from, isn't changed after function call
IntMatrix previousState: is set to (currentState - \nu_k), where \nu_k is the stoichiometric vector associated with reaction k
uint32_t reactionIndex: index of reaction to go back to previous state with (usually denoted k in the thesis report)

RETURNS:
propensity of previous state with respects to reaction, is 0 whenever there is no valid previous state*/
double GetPreviousConnectedState(const SRN* srn, IntMatrix currentState, IntMatrix previousState, uint32_t reactionIndex);


void SetInitialState(const SRN* srn, IntMatrix n); /*sets initial distribution at t = 0 to be the delta distribution with probability 1 on the state n*/