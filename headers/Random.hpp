#pragma once

#include "Matrix.hpp"
#include <stdint.h>
#include <stdbool.h>

#define UINT48_MAX  0x0000FFFFFFFFFFFFULL

#define MEMOI_COUNT     67 /*this will be the maximum for n and k, otherwise n choose n/2 wont fit in 64-bit uint*/

#define MIN(a, b)       (((a) < (b)) ? (a) : (b))
#define MAX(a, b)       (((a) > (b)) ? (a) : (b))


/*======================= PRNG's =======================*/

void SetSeedRandU48(uint64_t seed);

uint64_t RandU48(void); /*has a period of 2^48 - 1*/


/*======================= Distributions =======================*/

static inline double StandardClosedUniformSim(void) /*uniform chance variable realisation within [0, 1]*/
{ 
    return ((double)RandU48() / (double)UINT48_MAX); 
}

static inline double StandardOpenUniformSim(void) /*uniform chance variable realisation within (0, 1)*/
{ 
    const double epsilon = (1.0 / (double)(UINT48_MAX + 2));
    return (epsilon + (StandardClosedUniformSim() * (1.0 - (2.0 * epsilon))));
}

/*Standard uniform distribution, can be of types: (0, 1), [0, 1], [0, 1) and (0, 1]*/
static inline double StandardUniformSim(bool lowerBoundOpen, bool upperBoundOpen)
{
    uint32_t openEndsCount = ((uint8_t)lowerBoundOpen + (uint8_t)upperBoundOpen);
    double epsilon = (1.0 / (double)(UINT48_MAX + openEndsCount)); /*discrete making of continuous interval*/

    return (((lowerBoundOpen) ? epsilon : 0.0) + (StandardClosedUniformSim() * (1.0 - ((double)openEndsCount * epsilon))));
}

static inline double UniformSim(double a, double b, bool lowerBoundOpen, bool upperBoundOpen)
{
    return (a + (StandardUniformSim(lowerBoundOpen, upperBoundOpen) * (b - a)));
}

static inline uint8_t BernoulliDistributionSim(double p)
{
    return ((StandardUniformSim(false, true) < p) ? 1 : 0);
}

static inline double SelectRandomDouble(double x, double y) { return BernoulliDistributionSim(0.5) ? x : y; }
static inline uint64_t SelectRandomUint64(uint64_t x, uint64_t y) { return BernoulliDistributionSim(0.5) ? x : y; }

/*expects at least (count - 1) probabilities, as the last one is just (1 minus the sum of the others)*/
static inline uint32_t PickUintWithChances(const double* probabilities, uint32_t count)
{
    double standardSim = StandardClosedUniformSim();
    double probabilitySum = 0.0;
    for(uint32_t i = 0; i < (count - 1); i++)
    {
        probabilitySum += probabilities[i];
        if(probabilitySum >= standardSim)
            return i;
    }
    return (count - 1);
}


uint64_t NChooseK(uint32_t n, uint32_t k);

uint32_t BinomialDistributionSim(uint32_t n, double p);


/*======================= Distances functions =======================*/

double HellingerDistance(Tensor P, Tensor Q);