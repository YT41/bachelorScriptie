#pragma once

#include <stdint.h>

#include "Matrix.hpp"
#include "MemArena.hpp"
#include "NeuralNetwork.hpp"
#include "SRN.hpp"


/*first model: MLP using KL-divergence, basic time-conditioned model*/
typedef struct BTCM
{
    MemArena arena;

    Matrix tokenCache;
    NeuralNetwork* nn;
    
    SRN* srn;
} BTCM;


BTCM* BTCMCreate(SRN* srn, uint32_t* neuronsPerHiddenLayer, uint32_t hiddenLayerCount, double learningRate);
void BTCMDelete(BTCM* m);

/*returns the probability of getting that sample, the sample is returned in s, the gradient of KL-loss with respects to last layer is incremented in desiredNudgesMLPOutput*/
double BTCMTakeSample(BTCM* m, IntMatrix s, double t, Matrix desiredNudgesMLPOutput);
double BTCMPredict(BTCM* m, IntMatrix n, double t); /*directly gives P(n, t), so this is NOT a full-grid evaluation*/
void BTCMTrain(BTCM* m, double T, double deltaT, uint32_t B, uint64_t epochs);