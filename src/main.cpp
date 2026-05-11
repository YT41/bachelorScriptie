#include <cstdint>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "Matrix.hpp"
#include "MemArena.hpp"
#include "Random.hpp"
#include "ReactionParser.hpp"
#include "SRN.hpp"
#include "BTCM.hpp"
#include "TrajectorySim.hpp"


int main(int argc, char** argv)
{
    SetSeedRandU48(time(NULL));

    //TestSimpleSinNN();

    SRN* srn = ParseSRN("res/birthDeathModel.txt");

    PrintIntMatrix((srn->stoichiometricMatrix));

    uint32_t hiddenLayerNeuronCount[] = { 32 };
    BTCM* m = BTCMCreate(srn, hiddenLayerNeuronCount, (sizeof(hiddenLayerNeuronCount) / sizeof(uint32_t)), 0.01);

    MemArena arena = CreateMemArena(100000000);


    // IntMatrixNxM n = CreateBlankIntMatrix(&arena, M, 1);
    // SetValueIntMatrix(n, 10, 0, 0);
    // double probability = BTCMPredict(m, n, 0.0);

    // printf("predicted probability: %.2f\n", probability);

    double testTime = 0.01;

    BTCMTrain(m, 10.0, testTime, 0.5, 20, 1, 5000000);

    Tensor modelStateSpaceProbabilities = SRNCreateStateSpaceTensor(&arena, srn);
    BTCMGetFullProbabilityDistribution(m, modelStateSpaceProbabilities, testTime);

    uint32_t M = SRNGetSpeciesCount((m->srn));
    Matrix mean = CreateMatrix(&arena, M, 1, NULL);
    Matrix std = CreateMatrix(&arena, M, 1, NULL);

    BTCMGetPerSpeciesMean(m, mean, testTime);
    BTCMGetPerSpeciesStandardDeviation(m, std, testTime);

    PrintMatrix(mean);
    PrintMatrix(std);

    GillespieSRNTrajectorySimGetPerSpeciesMean(srn, mean, testTime);
    GillespieSRNTrajectorySimGetPerSpeciesStandardDeviation(srn, std, testTime);

    PrintMatrix(mean);
    PrintMatrix(std);

    Tensor marginalStateSpaceProbabilities = SRNCreateStateSpaceTensor(&arena, srn);
    GetFullMarginalDistributionGillespieSRNTrajectorySim(srn, marginalStateSpaceProbabilities, testTime, 10000000);

    printf("Hellinger Distance: %f\n", HellingerDistance(modelStateSpaceProbabilities, marginalStateSpaceProbabilities));

    // IntMatrix s = CreateBlankIntMatrix(&arena, M, 1);
    // double probability = BTCMTakeSample(m, s, 0.01);

    // printf("Sample count: %u\nprobability of taking this sample: %.2f\n\n", s.data[0], probability);


    DeleteMemArena(&arena);
    BTCMDelete(m);
    DeleteSRN(srn);

    return 0;
}