#include "Random.h"

#include <math.h>


/*======================= PRNG's =======================*/

static uint64_t randU48Seed = 0;

void SetSeedRandU48(uint64_t seed)
{
    randU48Seed = seed;
    randU48Seed &= UINT48_MAX;
}

uint64_t RandU48(void)
{
	randU48Seed ^= (randU48Seed << 13);
    randU48Seed &= UINT48_MAX;
	randU48Seed ^= (randU48Seed >> 7);
    randU48Seed &= UINT48_MAX;
	randU48Seed ^= (randU48Seed << 17);
    randU48Seed &= UINT48_MAX;

	return randU48Seed;
}


/*======================= Distributions =======================*/

static uint64_t RecNChooseK(uint32_t n, uint32_t k)
{
    static uint64_t memoisationLUT[MEMOI_COUNT - 2][MEMOI_COUNT / 2] = { 0 };

    k = MIN(k, (n - k)); /*use symmetry*/

    if(k == 0) /*base cases*/
        return 1;

    if(memoisationLUT[n-2][k-1] == 0)
        memoisationLUT[n-2][k-1] = (RecNChooseK((n - 1), k) + RecNChooseK((n - 1), (k - 1)));
    return memoisationLUT[n-2][k-1];
}

uint64_t NChooseK(uint32_t n, uint32_t k)
{
    if((k > n) || (n >= MEMOI_COUNT)) /*not defined or too big to fit in 64 bit unsigned int*/
        return 0;
    return RecNChooseK(n, k);
}

static uint32_t BinomialDistributionSimHelper(uint32_t n, double p)
{
    if((p <= 0.0) || (p > 1.0) || (n >= MEMOI_COUNT)) /*not defined or always 0 in case of p = 0*/
        return 0;
    else if(p == 1.0)
        return n;

    double notp = (1.0 - p);
    double notpPown = pow(notp, (double)n);

    double pPowk = 1.0;
    double notpPowk = 1.0;

    double uniformRealisation = StandardOpenUniformSim();

    double binomialProbabilitySum = 0.0;
    for(uint32_t k = 0; k <= n; k++)
    {
        binomialProbabilitySum += ((double)NChooseK(n, k) * pPowk * (notpPown / notpPowk)); /*(n choose k) p^k(1 - p)^{n - k}*/
        if(binomialProbabilitySum > uniformRealisation)
            return k;

        pPowk *= p;
        notpPowk *= notp;
    }
    return n;
}

uint32_t BinomialDistributionSim(uint32_t n, double p)
{
    uint32_t divisionCount = ((n / (MEMOI_COUNT - 1)) + (uint32_t)((n % (MEMOI_COUNT - 1)) != 0));

    uint32_t resultSum = 0;
    for(uint32_t i = 0; i < divisionCount; i++)
        resultSum += BinomialDistributionSimHelper(((n / divisionCount) + (uint32_t)((n % divisionCount) > i)), p);
    return resultSum;
}