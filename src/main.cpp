#include "TCM.hpp"
#include "Attention.hpp"
#include "Matrix.hpp"
#include "MemArena.hpp"
#include "Random.hpp"
#include "SRN.hpp"
#include "TrajectorySim.hpp"
#include "ReactionParser.hpp"
#include <cstdint>
#include <cstdio>
#include <ctime>


static inline void TCMExperimentTemplate()
{
    SetSeedRandU48(time(NULL));

    SRN* srn = ParseSRN("res/nameOfYourSRN.txt");

    double deltaT = 0.01;
    double T = 100.0;
    double p = 0.01;
    uint32_t B = 1000;
    uint32_t Q = 1;
    uint64_t epochs = 500000;
    double learningRate = 0.01;
    double dropoutProbability = 0.0;
    const char* pathToYourLossDataFile = "res/yourExperimentFolder/Loss.data";
    uint32_t hiddenLayerNeuronCount[] = { 16 };
    uint32_t timeEmbeddingDim = 32;
    uint32_t attentionDim = 0; /*0 means attention is disabled*/

    TCM* m = TCMCreate(srn, hiddenLayerNeuronCount, (sizeof(hiddenLayerNeuronCount) / sizeof(uint32_t)), timeEmbeddingDim, attentionDim);

    PrintIntMatrix((srn->stoichiometricMatrix));
    printf("Param count: %lu\n", TCMGetParamCount(m));

    clock_t startTime = clock();
    TCMTrain(m, T, deltaT, p, B, Q, epochs, learningRate, dropoutProbability, pathToYourLossDataFile);
    clock_t endTime = clock();

    double trainTimeHours = ((double)(endTime - startTime) / (double)(CLOCKS_PER_SEC * 60 * 60));
    printf("Training (CPU) time in hours: %f\n", trainTimeHours);

    size_t sampleCount = 10000;
    double sampleStep = (deltaT * 10.0);

    /*uncomment one of these to obtain more data on your experiment after training*/
    // TCMLogMeanComparisonOverTime(m, T, sampleStep, sampleCount, "res/BTCMGlobalTimeExperiment/mean.data");
    // TCMLogStdComparisonOverTime(m, T, sampleStep, sampleCount, "res/BTCMGlobalTimeExperiment/std.data");
    // TCMLogFullDistributionComparisonOverTime(m, T, sampleStep, sampleCount, "res/BTCMGlobalTimeExperiment/fullDistribution.data");
    // TCMLogHellingerDistanceToGillespieOverTime(m, T, sampleStep, sampleCount, "res/BTCMGlobalTimeExperiment/hellingerDistance.data");

    TCMDelete(m);
    DeleteSRN(srn);
}

int main(int argc, char** argv)
{
    /*uncomment one of these to run the corresponding experiment*/
    //TCMSingleTimeStepExperiment();
    TCMGlobalTimeExperiment();
    //TCMGeneExpressionExperiment();
    //TCMSignalingCascadeExperiment(2);
    //TCMTimeEmbeddingExperiment();
    //TCMExperimentTemplate()

    return 0;
}