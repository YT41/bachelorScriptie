#include "NeuralNetwork.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "Matrix.hpp"
#include "MemArena.hpp"
#include "Random.hpp"


typedef void ActivationFn(Matrix z); /*note: changes the matrix data in z*/
typedef void ActivationFnDerivative(Matrix dest, Matrix z);

static double Sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static double ReLU(double x) { return MAX(x, 0.0); }
static void SoftmaxHelperFunction(double* dest, Matrix z) /*z does not change*/
{
    uint32_t K = (z.rowCount);
    double m = GetValueMatrix(z, 0, 0);
    for(uint32_t i = 0; i < K; i++)
    {
        double zi = GetValueMatrix(z, i, 0);
        if(zi > m)
            m = zi;
    }

    double exponentialsCache[K];
    double sum = 0.0;
    for(uint32_t i = 0; i < K; i++)
    {
        exponentialsCache[i] = exp(GetValueMatrix(z, i, 0) - m);
        sum += exponentialsCache[i];
    }

    for(uint32_t i = 0; i < K; i++)
        dest[i] = (exponentialsCache[i] / sum);
}


static double DSigmoidDx(double x) { return 1.0 / (exp(x) + 2.0 + exp(-x)); }
static double DTanhDx(double x) { double tanhx = tanh(x); return (1.0 - (tanhx * tanhx)); }
static double DReLUDx(double x) { return (x < 0.0) ? 0.0 : 1.0; } /*does not actually exist everywhere*/


static void Identity(Matrix z) {  }
static void SigmoidElementWise(Matrix z) { MatrixTransformSelf(z, Sigmoid); }
static void TanhElementWise(Matrix z) { MatrixTransformSelf(z, tanh); }
static void ReLUElementWise(Matrix z) { MatrixTransformSelf(z, ReLU); }
static void Softmax(Matrix z) { SoftmaxHelperFunction((z.data), z); } /*based on safe softmax, prevents large exponentiation*/

/*dest must be a square matrix, z is the vector to evaluate derivative in, z is not changed*/
static void IdentityDerivative(Matrix dest, Matrix z) { SetMatrixIdentity(dest); }
static void SigmoidDerivative(Matrix dest, Matrix z) { SetMatrixDiagonal(dest, z.data); MatrixTransformDiagonalSelf(dest, DSigmoidDx); }
static void TanhDerivative(Matrix dest, Matrix z) { SetMatrixDiagonal(dest, z.data); MatrixTransformDiagonalSelf(dest, DTanhDx); }
static void ReLUDerivative(Matrix dest, Matrix z) { SetMatrixDiagonal(dest, z.data); MatrixTransformDiagonalSelf(dest, DReLUDx); }
static void SoftmaxDerivative(Matrix dest, Matrix z) 
{ 
    uint32_t K = (z.rowCount);
    double softmaxVals[K];
    SoftmaxHelperFunction(softmaxVals, z);

    for(uint32_t i = 0; i < K; i++)
    {
        for(uint32_t j = 0; j < K; j++)
            SetValueMatrix(dest, -(softmaxVals[i] * softmaxVals[j]), i, j);
    }

    /*set diagonal values correctly now*/
    for(uint32_t i = 0; i < K; i++)
        SetValueMatrix(dest, (softmaxVals[i] * (1.0 - softmaxVals[i])), i, i);
}



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
        /*TODO: this makes softmax only work in last layer (multiplication is seperate in last layer), which is fine for the purpose of this project*/
        (network->layerVectors[l].data[GetIndex(i, 0, m)]) = GetValueMatrix((network->activationFunctionDerivativeCache[l]), i, i) * sum;
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
        allocSize += GetMatrixAllocSize(neuronsPerLayer[i], neuronsPerLayer[i]);

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
        ret->activationFunctionDerivativeCache[i] = CreateRandomMatrix(&arena, neuronsPerLayer[i], neuronsPerLayer[i]);

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

void NNSetLastLayer(NeuralNetwork* network, Matrix y)
{
    CopyMatrixData(network->layerVectors[((network->hiddenLayerCount) + 1)], y);
}

void NNBackPropagation(NeuralNetwork* network)
{
    uint32_t L = (network->hiddenLayerCount);
    MatrixMultiply(&(network->layerVectors[L+1]), (network->activationFunctionDerivativeCache[L+1]), (network->layerVectors[L+1]));
    
    for(int32_t l = L; l >= 0; l--)
    {
        UpdateBiasVectorGradientDescent(network, l);
        UpdateWeightMatrixGradientDescent(network, l);
    }
}

Matrix NNPredict(NeuralNetwork* network, Matrix x)
{
    CopyMatrixData((network->layerVectors[0]), x);

    uint32_t L = (network->hiddenLayerCount);
    for(uint32_t l = 0; l < (L + 1); l++)
    {
        MatrixAffineTransform(&(network->layerVectors[l+1]), (network->weightMatrices[l]), (network->layerVectors[l]), (network->biasVectors[l]));

        ActivationFnID activationFnID = (network->activationFnPerLayer[l]);

        /*cache derivatives for use in backpropagation*/
        activationFnDerivativeLUT[activationFnID]((network->activationFunctionDerivativeCache[l+1]), (network->layerVectors[l+1]));

        activationFnLUT[activationFnID]((network->layerVectors[l+1]));
    }

    return (network->layerVectors[L+1]);
}

/*TODO: make it work for general loss functions, right now this is squared error loss*/
double NNTrain(NeuralNetwork* network, Matrix x, Matrix y)
{
    double loss = SquaredDistanceLoss(NNPredict(network, x), y);

    uint32_t L = (network->hiddenLayerCount);
    MatrixSubSelf((network->layerVectors[L+1]), y);
    MatrixScaleSelf((network->layerVectors[L+1]), 2.0);

    NNBackPropagation(network);

    return loss;
}


/*========================== NN testing ==========================*/

void TestSimpleSinNN(void)
{
    uint32_t neuronsPerLayer[] = { 1, 16, 16, 1 };
    ActivationFnID activationFnPerLayer[] = { TANH, TANH, IDENTITY };
    NeuralNetwork* nn = NNCreate(neuronsPerLayer, activationFnPerLayer, 2, 0.01);

    MemArena arena = CreateMemArena(GetMatrixAllocSize(1, 1) * 2);

    FILE* file1 = fopen("res/NNTestLoss.txt", "w");

    Matrix x = CreateMatrix(&arena, 1, 1, NULL);
    Matrix y = CreateMatrix(&arena, 1, 1, NULL);
    for(uint64_t i = 0; i < 1000000; i++)
    {
        double r1 = UniformSim(-10.0, 10.0, true, true);

        SetValueMatrix(x, r1, 0, 0);
        SetValueMatrix(y, sin(r1), 0, 0);

        double loss = NNTrain(nn, x, y);
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