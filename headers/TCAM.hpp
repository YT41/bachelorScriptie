#pragma once

#include <cstdint>
#include <cstdio>
#include <stdint.h>

#include "Attention.hpp"
#include "Matrix.hpp"
#include "MemArena.hpp"
#include "NeuralNetwork.hpp"
#include "SRN.hpp"


/*second model: masked attention mechanism --> MLP with KL-divergence loss, time-conditioned attention model*/
typedef struct TCAM
{
    MemArena arena;

    Matrix tokenCache;
    Matrix X;

    AttentionMechanism* a;
    NeuralNetwork* nn;
    
    SRN* srn;
} TCAM;


TCAM* TCAMCreate(SRN* srn, uint32_t attentionDim, uint32_t* neuronsPerHiddenLayer, uint32_t hiddenLayerCount, double learningRate);
void TCAMDelete(TCAM* m);

size_t TCAMGetParamCount(const TCAM* m);
TCAM* TCAMCopy(const TCAM* m);
void TCAMCopyParameters(TCAM* dest, const TCAM* src);

/*returns the probability of getting that sample, the sample is returned in s, the gradient of KL-loss with respects to last layer is incremented in desiredNudgesMLPOutput*/
double TCAMTakeSample(TCAM* m, IntMatrix s, double t, Matrix desiredNudgesMLPOutput);
double TCAMTakeSampleNoGradient(TCAM* m, IntMatrix s, double t);
double TCAMPredict(TCAM* m, IntMatrix n, double t); /*directly gives P(n, t), so this is NOT a full-grid evaluation*/
void TCAMGetFullProbabilityDistribution(TCAM* m, Tensor probabilities, double t);
void TCAMTrain(TCAM* m, double T, double deltaT, double p, uint32_t B, uint32_t Q, uint64_t epochs, const char* fileName); /*every Q epochs parameters of target are updated*/


/*==================== statistics ====================*/

void TCAMGetPerSpeciesMean(TCAM* m, Matrix mean, double t, size_t sampleCount);
void TCAMGetPerSpeciesStandardDeviation(TCAM* m, Matrix std, double t, size_t sampleCount);

void TCAMLogPerSpeciesMean(TCAM* m, double T, double tStep, size_t sampleCount, FILE* logFile);
void TCAMLogPerSpeciesStandardDeviation(TCAM* m, double T, double tStep, size_t sampleCount, FILE* logFile);

void TCAMPrintFullProbabilityDistribution(TCAM* m, double t);
void TCAMGetFullProbabilityDistribution(TCAM* m, double t, Tensor dest);

void TCAMLogFullProbabilityDistribution(TCAM* m, double T, double tStep, size_t sampleCount, FILE* logFile);


/*==================== experiments ====================*/

void TCAMBirthDeathExperiment(void);
void TCAMGeneExpressionExperiment(void);
void TCAMSignallingCascadeExperiment(uint32_t M);