#include "NeuralNetwork.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "Matrix.hpp"
#include "MemArena.hpp"
#include "Random.hpp"
#include "MiscMath.hpp"


typedef void ActivationFn(Matrix z, uint32_t column); /*note: changes the matrix data in z*/
typedef void ActivationFnDerivative(Matrix dest, Matrix z, uint32_t column);


ActivationFn* activationFnLUT[] = 
{
    Identity,
    SigmoidElementWise,
    TanhElementWise,
    ReLUElementWise,
    Softmax
};

ActivationFnDerivative* activationFnDerivativeLUT[] = 
{
    IdentityDerivative,
    SigmoidDerivative,
    TanhDerivative,
    ReLUDerivative,
    SoftmaxDerivative
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
    uint32_t batchSize = NNGetBatchSize(network);

    for(uint32_t b = 0; b < batchSize; b++)
    {
        // for(uint32_t i = 0; i < d; i++)
        //     (network->biasVectors[l].data[GetIndex(i, 0, d)]) -= (GetValueMatrix((network->layerVectors[l+1]), i, b) * (network->learningRate));

        for(uint32_t i = 0; i < d; i++)
            (network->biasVectorGradientCaches[l].data[GetIndex(i, 0, d)]) += GetValueMatrix((network->layerVectors[l+1]), i, b);
    }
}

/*W_l -= lr * a_{l+1} (a_l)^T*/
/*a_l = (f')_l (W_l)^T a_{l+1}*/
static inline void UpdateWeightMatrixGradientDescent(NeuralNetwork* network, uint32_t l)
{
    uint32_t m = (network->layerVectors[l].rowCount);
    uint32_t n = (network->layerVectors[l+1].rowCount);
    uint32_t batchSize = NNGetBatchSize(network);

    for(uint32_t b = 0; b < batchSize; b++)
    {
        for(uint32_t i = 0; i < m; i++)
        {
            double sum = 0.0;
            for(uint32_t j = 0; j < n; j++)
            {
                sum += GetValueMatrix((network->layerVectors[l+1]), j, b) * GetValueMatrix((network->weightMatrices[l]), j, i);

                /*gradient descent for weights*/
                //(network->weightMatrices[l].data[GetIndex(j, i, n)]) -= GetValueMatrix((network->layerVectors[l+1]), j, b) * GetValueMatrix((network->layerVectors[l]), i, b) * (network->learningRate);
                (network->weightMatrixGradientCaches[l].data[GetIndex(j, i, n)]) += GetValueMatrix((network->layerVectors[l+1]), j, b) * GetValueMatrix((network->layerVectors[l]), i, b);
            }

            /*desired changes in the l'th layer*/
            /*TODO: this makes softmax only work in last layer (multiplication is seperate in last layer), which is fine for the purpose of this project*/
            (network->layerVectors[l].data[GetIndex(i, b, m)]) = GetValueMatrix((network->activationFunctionDerivativeCache[l]), i, b) * sum;
        }
    }
}


/*============================ public functions ============================*/

NeuralNetwork* NNCreate(uint32_t* neuronsPerLayer, ActivationFnID* activationFnPerLayer, uint32_t hiddenLayerCount, uint32_t batchSize)
{
    if((hiddenLayerCount > MAX_HIDDEN_LAYER_COUNT))
        return NULL;

    size_t allocSize = sizeof(NeuralNetwork);
    /*layer vector sizes*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 2); i++)
        allocSize += GetMatrixAllocSize(neuronsPerLayer[i], batchSize);

    /*derivative cache vector sizes*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 2); i++)
        allocSize += GetMatrixAllocSize(neuronsPerLayer[i], batchSize);

    /*weight matrix sizes (including gradient caches)*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 1); i++)
        allocSize += (GetMatrixAllocSize(neuronsPerLayer[i+1], neuronsPerLayer[i]) * 2);

    /*bias vector sizes (including gradient caches)*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 1); i++)
        allocSize += (GetMatrixAllocSize(neuronsPerLayer[i+1], 1) * 2);

    MemArena arena = CreateMemArena(allocSize);
    NeuralNetwork* ret = (NeuralNetwork*)MemArenaAlloc(&arena, sizeof(NeuralNetwork));

    /*layer vector init*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 2); i++)
        ret->layerVectors[i] = CreateRandomMatrix(&arena, neuronsPerLayer[i], batchSize);

    /*derivative cache vector init*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 2); i++)
        ret->activationFunctionDerivativeCache[i] = CreateRandomMatrix(&arena, neuronsPerLayer[i], batchSize);


    /*TODO: other initialization might be better for different layers based on activation function*/

    /*weight matrix init (including gradient caches)*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 1); i++)
    {
        ret->weightMatrices[i] = CreateRandomVariancePreservingMatrix(&arena, neuronsPerLayer[i+1], neuronsPerLayer[i]);
        ret->weightMatrixGradientCaches[i] = CreateMatrixAllVal(&arena, neuronsPerLayer[i+1], neuronsPerLayer[i], 0.0);
    }

    /*bias vector init (including gradient caches)*/
    for(uint32_t i = 0; i < (hiddenLayerCount + 1); i++)
    {
        ret->biasVectors[i] = CreateRandomVariancePreservingMatrix(&arena, neuronsPerLayer[i+1], 1);
        ret->biasVectorGradientCaches[i] = CreateMatrixAllVal(&arena, neuronsPerLayer[i+1], 1, 0.0);
    }

    ret->arena = arena;
    for(uint32_t i = 0; i < hiddenLayerCount + 1; i++)
        ret->activationFnPerLayer[i] = activationFnPerLayer[i];
    ret->hiddenLayerCount = hiddenLayerCount;

    return ret;
}

void NNDelete(NeuralNetwork* network)
{
    DeleteMemArena(&(network->arena));
}

void NNCopyParameters(NeuralNetwork* dest, const NeuralNetwork* src)
{
    for(uint32_t i = 0; i < ((dest->hiddenLayerCount) + 1); i++)
    {
        CopyMatrixData((dest->weightMatrices[i]), (src->weightMatrices[i]));
        CopyMatrixData((dest->biasVectors[i]), (src->biasVectors[i]));
    }
}

size_t NNGetParamCount(const NeuralNetwork* network)
{
    size_t paramSum = 0;
    for(uint32_t l = 0; l < (network->hiddenLayerCount + 1); l++)
    {
        paramSum += GetMatrixSize(network->weightMatrices[l]);
        paramSum += GetMatrixSize(network->biasVectors[l]);
    }
    return paramSum;
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

void NNSetLastLayer(NeuralNetwork* network, Matrix Y)
{
    CopyMatrixData(network->layerVectors[((network->hiddenLayerCount) + 1)], Y);
}

void NNBackwardPass(NeuralNetwork* network)
{
    uint32_t L = (network->hiddenLayerCount);
    
    for(int32_t l = L; l >= 0; l--)
    {
        UpdateBiasVectorGradientDescent(network, l);
        UpdateWeightMatrixGradientDescent(network, l);
    }
}

void NNGradientDescent(NeuralNetwork* network, double learningRate)
{
    uint32_t L = (network->hiddenLayerCount);
    for(uint32_t l = 0; l < (L + 1); l++)
    {
        /*weights*/
        MatrixScaleSelf((network->weightMatrixGradientCaches[l]), learningRate);
        MatrixSubSelf((network->weightMatrices[l]), (network->weightMatrixGradientCaches[l]));
        SetMatrix((network->weightMatrixGradientCaches[l]), 0.0); /*reset for next runs*/

        /*biases*/
        MatrixScaleSelf((network->biasVectorGradientCaches[l]), learningRate);
        MatrixSubSelf((network->biasVectors[l]), (network->biasVectorGradientCaches[l]));
        SetMatrix((network->biasVectorGradientCaches[l]), 0.0); /*reset for next runs*/
    }
}

Matrix NNPredict(NeuralNetwork* network, Matrix X)
{
    CopyMatrixData((network->layerVectors[0]), X);
    return NNPredictNoCopy(network);
}

Matrix NNPredictNoCopy(NeuralNetwork* network)
{
    uint32_t batchSize = NNGetBatchSize(network);
    uint32_t L = (network->hiddenLayerCount);
    for(uint32_t l = 0; l < L; l++)
    {
        MatrixAffineTransformColumnWiseC(&(network->layerVectors[l+1]), (network->weightMatrices[l]), (network->layerVectors[l]), (network->biasVectors[l]));

        ActivationFnID activationFnID = (network->activationFnPerLayer[l]);

        /*cache derivatives for use in backpropagation*/
        for(uint32_t j = 0; j < batchSize; j++)
        {
            activationFnDerivativeLUT[activationFnID]((network->activationFunctionDerivativeCache[l+1]), (network->layerVectors[l+1]), j);
            activationFnLUT[activationFnID]((network->layerVectors[l+1]), j);
        }
    }
    /*for last layer we do not cache derivatives, we just set the gradient to the right value in the last layer*/
    MatrixAffineTransformColumnWiseC(&(network->layerVectors[L+1]), (network->weightMatrices[L]), (network->layerVectors[L]), (network->biasVectors[L]));
    for(uint32_t j = 0; j < batchSize; j++)
        activationFnLUT[(network->activationFnPerLayer[L])]((network->layerVectors[L+1]), j);

    return (network->layerVectors[L+1]);
}

Matrix NNPredictSingleDataPoint(NeuralNetwork* network, uint32_t i, Matrix x)
{
    CopyMatrixData(GetColumnVectorMatrix(network->layerVectors[0], i), x);

    uint32_t L = (network->hiddenLayerCount);
    for(uint32_t l = 0; l < L; l++)
    {
        MatrixAffineTransform(GetColumnVectorMatrix((network->layerVectors[l+1]), i), (network->weightMatrices[l]),  GetColumnVectorMatrix((network->layerVectors[l]), i), (network->biasVectors[l]));

        ActivationFnID activationFnID = (network->activationFnPerLayer[l]);

        /*cache derivative for use in backpropagation, only for this single data point*/
        activationFnDerivativeLUT[activationFnID]((network->activationFunctionDerivativeCache[l+1]), (network->layerVectors[l+1]), i);
        activationFnLUT[activationFnID]((network->layerVectors[l+1]), i);
    }
    /*for last layer we do not cache derivatives, we just set the gradient to the right value in the last layer*/
    MatrixAffineTransform(GetColumnVectorMatrix((network->layerVectors[L+1]), i), (network->weightMatrices[L]),  GetColumnVectorMatrix((network->layerVectors[L]), i), (network->biasVectors[L]));
    activationFnLUT[(network->activationFnPerLayer[L])]((network->layerVectors[L+1]), i);

    return GetColumnVectorMatrix((network->layerVectors[L+1]), i);
}


/*TODO: make it work for general loss functions, right now this is squared error loss*/
double NNTrain(NeuralNetwork* network, Matrix x, Matrix y, double learningRate)
{
    double loss = SquaredDistanceLoss(NNPredict(network, x), y);

    uint32_t L = (network->hiddenLayerCount);
    MatrixSubSelf((network->layerVectors[L+1]), y);
    MatrixScaleSelf((network->layerVectors[L+1]), 2.0);

    NNBackwardPass(network);
    NNGradientDescent(network, learningRate);

    return loss;
}


/*========================== NN testing ==========================*/

void TestSimpleSinNN(void)
{
    uint32_t neuronsPerLayer[] = { 1, 16, 16, 1 };
    ActivationFnID activationFnPerLayer[] = { TANH, TANH, IDENTITY };
    NeuralNetwork* nn = NNCreate(neuronsPerLayer, activationFnPerLayer, 2, 1);

    MemArena arena = CreateMemArena(GetMatrixAllocSize(1, 1) * 2);

    FILE* file1 = fopen("res/NNTestLoss.txt", "w");

    Matrix x = CreateMatrix(&arena, 1, 1, NULL);
    Matrix y = CreateMatrix(&arena, 1, 1, NULL);
    for(uint64_t i = 0; i < 1000000; i++)
    {
        double r1 = UniformSim(-10.0, 10.0, true, true);

        SetValueMatrix(x, r1, 0, 0);
        SetValueMatrix(y, sin(r1), 0, 0);

        double loss = NNTrain(nn, x, y, 0.01);
        fprintf(file1, "%lu %f \n", i, loss);
    }

    FILE* file2 = fopen("res/NNTest.txt", "w");
    for(double input = -10.0; input < 10.0; input += 0.01)
    {
        SetValueMatrix(x, input, 0, 0);

        Matrix output = NNPredict(nn, x);

        fprintf(file2, "%f %f %f\n", input, sin(input), output.data[0]);
    }

    fclose(file1);
    fclose(file2);
    DeleteMemArena(&arena);
    NNDelete(nn);
}