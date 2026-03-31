#pragma once

#include <cstdint>
#include <stdint.h>

#include "Matrix.hpp"
#include "MemArena.hpp"
#include "NeuralNetwork.hpp"
#include "SRN.hpp"


/*first model: path signature ---> NN*/
typedef struct SigNN
{
    MemArena arena;

    NeuralNetwork* nn;
    SRN* srn;
    Matrix signatureInput;

    uint32_t truncationOrder;
    uint32_t sigLen;
} SigNN;


SigNN* CreateSigNN(SRN* srn, uint32_t* neuronsPerHiddenLayer, ActivationFnID* activationFnPerLayer, uint32_t hiddenLayerCount, double learningRate, uint32_t truncationOrder);
void DeleteSigNN(SigNN* sigNN);

double SigNNPredict(SigNN* sigNN, uint32_t* n, double t);
void SigNNTrain(SigNN* sigNN, double T, uint64_t epochs);