#pragma once

#include <cstdint>
#include <cstdio>
#include <stdint.h>

#include "Attention.hpp"
#include "Matrix.hpp"
#include "MemArena.hpp"
#include "NeuralNetwork.hpp"
#include "SRN.hpp"


/*time-conditioned model, MLP using KL-divergence, with time embedding and attention mechanism
pipeline: time embedding --> attention mechanism --> MLP*/
typedef struct TCM
{
    MemArena arena;

    Matrix batchCache;
    NeuralNetwork* MLP;
    NeuralNetwork* timeEmbedding;
    AttentionMechanism* a;
    
    SRN* srn;
} TCM;


TCM* TCMCreate(SRN* srn, uint32_t* neuronsPerHiddenLayer, uint32_t hiddenLayerCount, uint32_t timeEmbeddingDim, uint32_t attentionDim);
void TCMDelete(TCM* m);

size_t TCMGetParamCount(const TCM* m);
TCM* TCMCopy(const TCM* m);
void TCMCopyParameters(TCM* dest, const TCM* src);

/*returns the probability of getting that sample, the sample is returned in s, the gradient of KL-loss with respects to last layer is incremented in desiredNudgesMLPOutput*/
double TCMTakeSample(TCM* m, IntMatrix s, double t, Matrix desiredNudgesMLPOutput, double dropoutProbability);
double TCMTakeSampleNoGradient(TCM* m, IntMatrix s, double t, double dropoutProbability);
double TCMPredict(TCM* m, IntMatrix n, double t, double dropoutProbability); /*directly gives P(n, t), so assumes n is known*/
void TCMGetFullProbabilityDistribution(TCM* m, Tensor probabilities, double t);
void TCMTrain(TCM* m, double T, double deltaT, double p, uint32_t B, uint32_t Q, uint64_t epochs, double learningRate, double dropoutProbability, const char* logFile); /*every Q epochs parameters of target are updated*/


/*==================== statistics ====================*/

void TCMGetPerSpeciesMean(TCM* m, Matrix mean, double t, size_t sampleCount);
void TCMGetPerSpeciesStandardDeviation(TCM* m, Matrix std, double t, size_t sampleCount);

void TCMLogPerSpeciesMean(TCM* m, double T, double tStep, size_t sampleCount, FILE* logFile);
void TCMLogPerSpeciesStandardDeviation(TCM* m, double T, double tStep, size_t sampleCount, FILE* logFile);

void TCMPrintFullProbabilityDistribution(TCM* m, double t);
void TCMGetFullProbabilityDistribution(TCM* m, double t, Tensor dest);

void TCMLogFullProbabilityDistribution(TCM* m, double T, double tStep, size_t sampleCount, FILE* logFile);


/*==================== experiments ====================*/

void TCMSingleTimeStepExperiment(void);
void TCMGlobalTimeExperiment(void);
void TCMGeneExpressionExperiment(void);
void TCMSignallingCascadeExperiment(uint32_t M);