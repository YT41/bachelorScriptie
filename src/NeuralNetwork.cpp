#include "NeuralNetwork.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "Matrix.hpp"
#include "MemArena.hpp"
#include "Random.hpp"


static double Identity(double x) { return x; }
static double Sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static double ReLU(double x) { return MAX(x, 0.0); }
static double DIdentityDx(double x) { return 1.0; }

static double DSigmoidDx(double x) { return 1.0 / (exp(x) + 2.0 + exp(-x)); }
static double DTanhDx(double x) { double tanhx = tanh(x); return (1.0 - (tanhx * tanhx)); }
static double DReLUDx(double x) { return MAX(1.0, 0.0); } /*does not actually exist everywhere*/

RealFn* activationFnLUT[] = 
{
    Identity,
    Sigmoid,
    tanh,
    ReLU
};

RealFn* activationFnDerivativeLUT[] = 
{
    DIdentityDx,
    DSigmoidDx,
    DTanhDx,
    DReLUDx
};


static double SquaredDistanceLoss(Matrix output, Matrix y)
{
    uint32_t d = output.rowCount;
    double sum = 0.0;
    for(uint32_t i = 0; i < d; i++)
    {
        double dif = (GetValueMatrix(output, i, 0) - GetValueMatrix(y, i, 0));
        sum += (dif * dif);
    }
    return sum;
}


/*b_l -= lr * a_{l+1} \circ (f')_l*/
static inline void UpdateBiasVectorGradientDescent(NeuralNetwork* network, uint32_t l)
{
    uint32_t d = (network->biasVectors[l].rowCount);
    for(uint32_t i = 0; i < d; i++)
        (network->biasVectors[l].data[GetIndex(i, 0, d)]) -= GetValueMatrix((network->layerVectors[l+1]), i, 0) * (network->learningRate);
}

/*W_l -= lr * a_{l+1} (a_l)^T*/
/*a_l = (f')_l (W_l)^T a_{l+1}*/
static inline void UpdateWeightMatrixGradientDescent(NeuralNetwork* network, uint32_t l)
{
    uint32_t m = (network->layerVectors[l].rowCount);
    uint32_t n = (network->layerVectors[l+1].rowCount);
    for(uint32_t i = 0; i < m; i++)
    {
        double sum = 0.0;
        for(uint32_t j = 0; j < n; j++)
        {
            sum += GetValueMatrix((network->layerVectors[l+1]), j, 0) * GetValueMatrix((network->weightMatrices[l]), j, i);

            /*gradient descent for weights*/
            (network->weightMatrices[l].data[GetIndex(j, i, n)]) -= GetValueMatrix((network->layerVectors[l+1]), j, 0) * GetValueMatrix((network->layerVectors[l]), i, 0) * (network->learningRate);
        }

        /*desired changes in the l'th layer*/
        (network->layerVectors[l].data[GetIndex(i, 0, m)]) = GetValueMatrix((network->activationFunctionDerivativeCache[l]), i, 0) * sum;
    }
}

static void BackPropagation(NeuralNetwork* network, Matrix y)
{
    /*TODO: make it work for general loss functions, right now this is squared error loss*/
    uint32_t hiddenlayerCount = (network->hiddenLayerCount);
    MatrixSubSelf((network->layerVectors[hiddenlayerCount+1]), y);
    MatrixHadamardSelf((network->layerVectors[hiddenlayerCount+1]), (network->activationFunctionDerivativeCache[hiddenlayerCount+1]));
    MatrixScaleSelf((network->layerVectors[hiddenlayerCount+1]), 2.0);

    for(int32_t l = hiddenlayerCount; l >= 0; l--)
    {
        UpdateBiasVectorGradientDescent(network, l);
        UpdateWeightMatrixGradientDescent(network, l);
    }
}


/*============================ public functions ============================*/

NeuralNetwork* NNCreate(uint32_t* neuronsPerLayer, ActivationFnID* activationFnPerLayer, uint32_t hiddenLayerCount, double learningRate)
{
    if((hiddenLayerCount > MAX_HIDDEN_LAYER_COUNT))
        return NULL;

    size_t allocSize = sizeof(NeuralNetwork);
    /*layer vector sizes*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 2); i++)
        allocSize += GetMatrixAllocSize(neuronsPerLayer[i], 1);

    /*cache vector sizes*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 2); i++)
        allocSize += GetMatrixAllocSize(neuronsPerLayer[i], 1);

    /*weight matrix sizes*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 1); i++)
        allocSize += GetMatrixAllocSize(neuronsPerLayer[i+1], neuronsPerLayer[i]);

    /*bias vector sizes*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 1); i++)
        allocSize += GetMatrixAllocSize(neuronsPerLayer[i+1], 1);

    MemArena arena = CreateMemArena(allocSize);
    NeuralNetwork* ret = (NeuralNetwork*)MemArenaAlloc(&arena, sizeof(NeuralNetwork));

    /*layer vector init*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 2); i++)
        ret->layerVectors[i] = CreateRandomMatrix(&arena, neuronsPerLayer[i], 1);

    /*cache vector init*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 2); i++)
        ret->activationFunctionDerivativeCache[i] = CreateRandomMatrix(&arena, neuronsPerLayer[i], 1);

    /*weight matrix init*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 1); i++)
        ret->weightMatrices[i] = CreateRandomMatrix(&arena, neuronsPerLayer[i+1], neuronsPerLayer[i]);

    /*bias vector init*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 1); i++)
        ret->biasVectors[i] = CreateRandomMatrix(&arena, neuronsPerLayer[i+1], 1);

    ret->arena = arena;
    ret->learningRate = learningRate;
    for(uint32_t i = 0; i < hiddenLayerCount + 1; i++)
        ret->activationFnPerLayer[i] = activationFnPerLayer[i];
    ret->hiddenLayerCount = hiddenLayerCount;

    return ret;
}

void NNDelete(NeuralNetwork* network)
{
    DeleteMemArena(&(network->arena));
}


// void NNSaveToFile(const NeuralNetwork* network, const char* fileName)
// {
//     FILE* file = fopen(fileName, "wb");
//     if(file != NULL)
//         fwrite((void*)network, sizeof(NeuralNetwork), 1, file);
//     fclose(file);
// }

// void NNLoadFromFile(NeuralNetwork* network, const char* fileName)
// {
//     FILE* file = fopen(fileName, "rb");
//     if(file != NULL)
//         fread((void*)network, sizeof(NeuralNetwork), 1, file);
//     fclose(file);
// }


// double NNLoss(const NeuralNetwork* network, const unsigned int correctOutputIndex)
// {
//     return pow((network->outputLayer[0] - ((double)correctOutputIndex / 38.0)), 2.0);
// }

void NNPredict(NeuralNetwork* network, Matrix x)
{
    CopyMatrixData((network->layerVectors[0]), x);

    for(uint32_t l = 0; l < ((network->hiddenLayerCount) + 1); l++)
    {
        MatrixMultiply(&(network->layerVectors[l+1]), (network->weightMatrices[l]), (network->layerVectors[l]));
        MatrixAddSelf((network->layerVectors[l+1]), (network->biasVectors[l]));

        ActivationFnID activationFnID = (network->activationFnPerLayer[l]);

        /*cache derivatives for use in backpropagation*/
        MatrixTransform(&(network->activationFunctionDerivativeCache[l+1]), (network->layerVectors[l+1]), activationFnDerivativeLUT[activationFnID]);

        MatrixTransformSelf((network->layerVectors[l+1]), activationFnLUT[activationFnID]);
    }
}

double NNTrain(NeuralNetwork* network, Matrix x, Matrix y)
{
    NNPredict(network, x);
    double loss = SquaredDistanceLoss((network->layerVectors[(network->hiddenLayerCount) + 1]), y);
    BackPropagation(network, y);

    return loss;
}