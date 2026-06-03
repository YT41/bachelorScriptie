#pragma once

#include <cstdint>
#include <cstdio>
#include <stdint.h>

#include "Matrix.hpp"
#include "MemArena.hpp"
#include "NeuralNetwork.hpp"
#include "SRN.hpp"


/*first model: MLP using KL-divergence, basic time-conditioned model*/
typedef struct BTCM
{
    MemArena arena;

    Matrix batchCache;
    NeuralNetwork* MLP;
    NeuralNetwork* timeEmbedding;
    
    SRN* srn;
} BTCM;


BTCM* BTCMCreate(SRN* srn, uint32_t* neuronsPerHiddenLayer, uint32_t hiddenLayerCount, uint32_t timeEmbeddingDim, double learningRate);
void BTCMDelete(BTCM* m);

size_t BTCMGetParamCount(const BTCM* m);
BTCM* BTCMCopy(const BTCM* m);
void BTCMCopyParameters(BTCM* dest, const BTCM* src);

/*returns the probability of getting that sample, the sample is returned in s, the gradient of KL-loss with respects to last layer is incremented in desiredNudgesMLPOutput*/
double BTCMTakeSample(BTCM* m, IntMatrix s, double t, Matrix desiredNudgesMLPOutput);
double BTCMTakeSampleNoGradient(BTCM* m, IntMatrix s, double t);
double BTCMPredict(BTCM* m, IntMatrix n, double t); /*directly gives P(n, t), so this is NOT a full-grid evaluation*/
void BTCMGetFullProbabilityDistribution(BTCM* m, Tensor probabilities, double t);
void BTCMTrain(BTCM* m, double T, double deltaT, double p, uint32_t B, uint32_t Q, uint64_t epochs, const char* logFile); /*every Q epochs parameters of target are updated*/


/*==================== statistics ====================*/

void BTCMGetPerSpeciesMean(BTCM* m, Matrix mean, double t, size_t sampleCount);
void BTCMGetPerSpeciesStandardDeviation(BTCM* m, Matrix std, double t, size_t sampleCount);

void BTCMLogPerSpeciesMean(BTCM* m, double T, double tStep, size_t sampleCount, FILE* logFile);
void BTCMLogPerSpeciesStandardDeviation(BTCM* m, double T, double tStep, size_t sampleCount, FILE* logFile);

void BTCMPrintFullProbabilityDistribution(BTCM* m, double t);
void BTCMGetFullProbabilityDistribution(BTCM* m, double t, Tensor dest);

void BTCMLogFullProbabilityDistribution(BTCM* m, double T, double tStep, size_t sampleCount, FILE* logFile);


/*==================== experiments ====================*/

void BTCMSingleTimeStepExperiment(void);
void BTCMGlobalTimeExperiment(void);
void BTCMGeneExpressionExperiment(void);
void BTCMSignallingCascadeExperiment(uint32_t M);