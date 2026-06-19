#pragma once

#include <cstdint>
#include <stdint.h>

#include "Matrix.hpp"
#include "MemArena.hpp"

#define MAX_HIDDEN_LAYER_COUNT              4

typedef enum ActivationFnID 
{
    IDENTITY = 0,
    SIGMOID = 1,
    TANH = 2,
    RELU = 3,
    SOFTMAX = 4
} ActivationFnID;


typedef struct NeuralNetwork
{
    MemArena arena;

    /*a_0, ..., a_{l+1}*/
    Matrix layerVectors[MAX_HIDDEN_LAYER_COUNT + 2]; /*from input at index 0, to output layer*/

    /*saved on forward pass, used in backpropagation algorithm*/
    /*(f')_0, ..., (f')_{l+1}*/
    Matrix activationFunctionDerivativeCache[MAX_HIDDEN_LAYER_COUNT + 2]; /*from input at index 0, to output layer*/

    /*================ parameters ================*/
    /*W_0, ..., W_l*/
    Matrix weightMatrices[MAX_HIDDEN_LAYER_COUNT + 1]; /*from before first hidden layer at index 0, to before output layer*/
    /*b_0, ..., b_l*/
    Matrix biasVectors[MAX_HIDDEN_LAYER_COUNT + 1]; /*from first hidden layer at index 0, to output layer*/

    /*================ gradient caches ================*/

    Matrix weightMatrixGradientCaches[MAX_HIDDEN_LAYER_COUNT + 1]; /*from before first hidden layer at index 0, to before output layer*/
    Matrix biasVectorGradientCaches[MAX_HIDDEN_LAYER_COUNT + 1]; /*from first hidden layer at index 0, to output layer*/

    ActivationFnID activationFnPerLayer[MAX_HIDDEN_LAYER_COUNT + 1];
    uint32_t hiddenLayerCount;
} NeuralNetwork;


/*Creates an instance of a neural network

INPUTS:
uint32_t* neuronsPerLayer: C-Style array specifying the amount of neurons to be in every layer in order from input to output layer
ActivationFnID* activationFnPerLayer: C-Style array specifying the activation function to be between every layer in order from input to output layer
uint32_t hiddenLayerCount: amount of hidden layers
uint32_t batchSize: amount of input tokens to be in a single batch

RETURNS:
Heap-allocated pointer to NeuralNetwork structure, be sure to delete it with NNDelete whenever you're not using it anymore to prevent memory leaks.*/
NeuralNetwork* NNCreate(uint32_t* neuronsPerLayer, ActivationFnID* activationFnPerLayer, uint32_t hiddenLayerCount, uint32_t batchSize);
void NNDelete(NeuralNetwork* network);

size_t NNGetParamCount(const NeuralNetwork* network);
void NNCopyParameters(NeuralNetwork* dest, const NeuralNetwork* src);

void NNSetLastLayer(NeuralNetwork* network, Matrix Y); /*can be used to set cost gradient with respects to last layer*/

void NNBackwardPass(NeuralNetwork* network); /*performs backward pass, adding the gradients to gradient cache matrices. These are then used with NNGradientDescent*/
void NNGradientDescent(NeuralNetwork* network, double learningRate); /*performs gradient descent, using stored gradients from NNBackwardPass runs*/

/*for these 3 functions dropout probability only applies to neurons within the MLP hidden layers (not that important, just set it to 0 if you dont know what this is)*/
Matrix NNPredict(NeuralNetwork* network, Matrix X, double dropoutProbability); /*processes whole batch X through MLP, expects X to be shape inputDim x batchSize*/
Matrix NNPredictNoCopy(NeuralNetwork* network, double dropoutProbability); /*like NNPredict, but assumes input of NN is already set to X instead*/
Matrix NNPredictSingleDataPoint(NeuralNetwork* network, uint32_t i, Matrix x, double dropoutProbability); /*like NNPredict, but for single token x_i of batch, specified in x*/

static inline uint32_t NNGetOutputDimension(const NeuralNetwork* network) { return (network->layerVectors[(network->hiddenLayerCount) + 1].rowCount); };
static inline uint32_t NNGetBatchSize(const NeuralNetwork* network) { return (network->layerVectors[0].columnCount); };
static inline Matrix NNGetLastLayer(const NeuralNetwork* network) { return (network->layerVectors[(network->hiddenLayerCount) + 1]); };
static inline Matrix NNGetFirstLayer(const NeuralNetwork* network) { return (network->layerVectors[0]); };


/*========================== NN testing ==========================*/

double NNTrain(NeuralNetwork* network, Matrix x, Matrix y, double learningRate); /*just for batch size 1, really only for testing*/

void TestSimpleSinNN(void); /*handy for debugging*/