#pragma once

#include <stdint.h>

#include "Matrix.hpp"
#include "MemArena.hpp"

#define MAX_HIDDEN_LAYER_COUNT              4

typedef enum ActivationFnID 
{
    IDENTITY = 0,
    SIGMOID = 1,
    TANH = 2,
    RELU = 3
} ActivationFnID;


typedef struct NeuralNetwork
{
    MemArena arena;

    /*a_0, ..., a_{l+1}*/
    Matrix layerVectors[MAX_HIDDEN_LAYER_COUNT + 2]; /*from input at index 0, to output layer*/

    /*saved on forward pass, used in backpropagation algorithm*/
    /*(f')_0, ..., (f')_{l+1}*/
    Matrix activationFunctionDerivativeCache[MAX_HIDDEN_LAYER_COUNT + 2]; /*from input at index 0, to output layer*/

    /*parameters*/
    /*W_0, ..., W_l*/
    Matrix weightMatrices[MAX_HIDDEN_LAYER_COUNT + 1]; /*from before first hidden layer at index 0, to before output layer*/
    /*b_0, ..., b_l*/
    Matrix biasVectors[MAX_HIDDEN_LAYER_COUNT + 1]; /*from first hidden layer at index 0, to output layer*/

    double learningRate; //this determines how fast the backpropagation strides towards the local minimum, in a gradient descent sense
    ActivationFnID activationFnPerLayer[MAX_HIDDEN_LAYER_COUNT + 1];
    uint32_t hiddenLayerCount;
} NeuralNetwork;


NeuralNetwork* NNCreate(uint32_t* neuronsPerLayer, ActivationFnID* activationFnPerLayer, uint32_t hiddenLayerCount, double learningRate);
void NNDelete(NeuralNetwork* network);

// void NNSaveToFile(const NeuralNetwork* network, const char* fileName);
// void NNLoadFromFile(NeuralNetwork* network, const char* fileName);

//double NNLoss(const NeuralNetwork* network, const unsigned int correctOutputIndex);

void NNPredict(NeuralNetwork* network, Matrix x);
double NNTrain(NeuralNetwork* network, Matrix x, Matrix y);