#include "Attention.hpp"
#include "Matrix.hpp"
#include "MemArena.hpp"
#include "MiscMath.hpp"
#include <cstdio>
#include <math.h>
#include <cstdint>


AttentionMechanism* AMCreate(uint32_t tokenCount, uint32_t inputDim, uint32_t outputDim)
{
    size_t allocSize = sizeof(AttentionMechanism);
    /*Parameter matrix sizes*/
    allocSize += (GetMatrixAllocSize(outputDim, inputDim) * 3);

    /*=== cache sizes ===*/
    allocSize += (GetMatrixAllocSize(outputDim, inputDim) * 3);
    allocSize += GetMatrixAllocSize(inputDim, tokenCount); /*XA*/
    allocSize += GetMatrixAllocSize(tokenCount, tokenCount); /*AGradientCache*/
    allocSize += GetMatrixAllocSize(tokenCount, tokenCount); /*SGradientCache*/
    allocSize += (GetMatrixAllocSize(outputDim, tokenCount) * 2); /*QGradientCache, KGradientCache*/
    allocSize += GetMatrixAllocSize(inputDim, tokenCount); /*XGradientCache*/

    /*=== gradient accum matrix sizes ===*/
    allocSize += (GetMatrixAllocSize(outputDim, inputDim) * 3);

    allocSize += GetMatrixAllocSize(inputDim, tokenCount); /*X*/
    allocSize += (GetMatrixAllocSize(outputDim, tokenCount) * 3); /*Q, K, V*/
    allocSize += GetMatrixAllocSize(tokenCount, tokenCount); /*A*/
    allocSize += GetMatrixAllocSize(outputDim, tokenCount); /*O*/

    MemArena arena = CreateMemArena(allocSize);
    AttentionMechanism* ret = (AttentionMechanism*)MemArenaAlloc(&arena, sizeof(AttentionMechanism));

    /*=== parameters ===*/

    ret->queryWeights = CreateRandomVariancePreservingMatrix(&arena, outputDim, inputDim);
    ret->keyWeights = CreateRandomVariancePreservingMatrix(&arena, outputDim, inputDim);
    ret->valueWeights = CreateRandomVariancePreservingMatrix(&arena, outputDim, inputDim);

    /*=== caches ===*/

    ret->queryWeightsGradientCache = CreateMatrix(&arena, outputDim, inputDim, NULL);
    ret->keyWeightsGradientCache = CreateMatrix(&arena, outputDim, inputDim, NULL);
    ret->valueWeightsGradientCache = CreateMatrix(&arena, outputDim, inputDim, NULL);
    ret->XA = CreateMatrix(&arena, inputDim, tokenCount, NULL);
    ret->AGradientCache = CreateMatrix(&arena, tokenCount, tokenCount, NULL);
    ret->SGradientCache = CreateMatrix(&arena, tokenCount, tokenCount, NULL);
    ret->QGradientCache = CreateMatrix(&arena, outputDim, tokenCount, NULL);
    ret->KGradientCache = CreateMatrix(&arena, outputDim, tokenCount, NULL);
    ret->XGradientCache = CreateMatrix(&arena, inputDim, tokenCount, NULL);

    /*=== gradient accum matrices ===*/
    ret->queryWeightsGradientAccum = CreateMatrixAllVal(&arena, outputDim, inputDim, 0.0);
    ret->keyWeightsGradientAccum = CreateMatrixAllVal(&arena, outputDim, inputDim, 0.0);
    ret->valueWeightsGradientAccum = CreateMatrixAllVal(&arena, outputDim, inputDim, 0.0);


    ret->X = CreateMatrix(&arena, inputDim, tokenCount, NULL);

    ret->Q = CreateMatrix(&arena, outputDim, tokenCount, NULL);
    ret->K = CreateMatrix(&arena, outputDim, tokenCount, NULL);
    ret->V = CreateMatrix(&arena, outputDim, tokenCount, NULL);

    ret->A = CreateMatrix(&arena, tokenCount, tokenCount, NULL);
    ret->O = CreateMatrix(&arena, outputDim, tokenCount, NULL);

    ret->arena = arena;

    return ret;
}

void AMDelete(AttentionMechanism* a)
{
    DeleteMemArena(&(a->arena));
}

size_t AMGetParamCount(const AttentionMechanism* a)
{
    return (GetMatrixSize((a->queryWeights)) + GetMatrixSize((a->keyWeights)) + GetMatrixSize((a->valueWeights)));
}

void AMCopyParameters(AttentionMechanism* dest, const AttentionMechanism* src)
{
    CopyMatrixData((dest->queryWeights), (src->queryWeights));
    CopyMatrixData((dest->keyWeights), (src->keyWeights));
    CopyMatrixData((dest->valueWeights), (src->valueWeights));
}

Matrix AMGetMaskedAttentionWeightColumn(AttentionMechanism* a, uint32_t j, Matrix X)
{
    double attDimSqrt = sqrt((double)AMGetOutputDimension(a));

    for(uint32_t i = (j + 1); i < AMGetTokenCount(a); i++)
        SetValueMatrix((a->A), -INFINITY, i, j); /*mask*/

    /*TODO: optimize, we dont have to calculate all K_i again if we have already done a previous column run*/
    MatrixMultiply(GetColumnVectorMatrix((a->Q), j), (a->queryWeights), GetColumnVectorMatrix(X, j));
    for(uint32_t i = 0; i <= j; i++) /*because of mask we only have to calculate attention to tokens before and including i*/
    {
        MatrixMultiply(GetColumnVectorMatrix((a->K), i), (a->keyWeights), GetColumnVectorMatrix(X, i));

        double score = Dot(GetColumnVectorMatrix((a->Q), j), GetColumnVectorMatrix((a->K), i));
        SetValueMatrix((a->A), (score / attDimSqrt), i, j);
    }
    Softmax((a->A), j);

    return GetColumnVectorMatrix((a->A), j);
}

Matrix AMGetMaskedAttentionWeightMatrix(AttentionMechanism* a, Matrix X)
{
    for(uint32_t i = 0; i < AMGetTokenCount(a); i++)
        AMGetMaskedAttentionWeightColumn(a, i, X);

    return (a->A);
}

Matrix AMGetSingleMaskedAttention(AttentionMechanism* a, uint32_t j, Matrix X)
{
    Matrix attentionColumn = AMGetMaskedAttentionWeightColumn(a, j, X);
    /*TODO: optimize, we dont have to calculate V every time*/
    MatrixMultiply((a->V), (a->valueWeights), X);
    MatrixMultiply(GetColumnVectorMatrix((a->O), j), (a->V), attentionColumn);

    return GetColumnVectorMatrix((a->O), j);
}

Matrix AMGetMaskedAttention(AttentionMechanism* a, Matrix X)
{
    Matrix A = AMGetMaskedAttentionWeightMatrix(a, X);
    MatrixMultiply((a->V), (a->valueWeights), X);
    MatrixMultiply((a->O), (a->V), A);

    return (a->O);
}

void AMBackwardPass(AttentionMechanism* a)
{
    /*Gradient to W_V*/
    MatrixMultiply((a->XA), (a->X), (a->A));
    MatrixMultiplyTransposedB((a->valueWeightsGradientCache), (a->O), (a->XA));
    MatrixAddSelf((a->valueWeightsGradientAccum), (a->valueWeightsGradientCache));

    /*Gradient to S*/
    MatrixMultiplyTransposedA((a->AGradientCache), (a->V), (a->O));
    for(uint32_t j = 0; j < AMGetTokenCount(a); j++)
    {
        double dot = Dot(GetColumnVectorMatrix((a->AGradientCache), j), GetColumnVectorMatrix((a->A), j));
        for(uint32_t i = 0; i < AMGetTokenCount(a); i++)
            SetValueMatrix((a->SGradientCache), (GetValueMatrix((a->A), i, j) * (GetValueMatrix((a->AGradientCache), i, j) - dot)), i, j);
    }
    double attDimSqrt = sqrt((double)AMGetOutputDimension(a));
    MatrixScaleSelf((a->SGradientCache), (1.0 / attDimSqrt));

    /*Gradient to W_Q*/
    MatrixMultiply((a->QGradientCache), (a->K), (a->SGradientCache));
    MatrixMultiplyTransposedB((a->queryWeightsGradientCache), (a->QGradientCache), (a->X));
    MatrixAddSelf((a->queryWeightsGradientAccum), (a->queryWeightsGradientCache));

    /*Gradient to W_K*/
    MatrixMultiplyTransposedB((a->KGradientCache), (a->Q), (a->SGradientCache));
    MatrixMultiplyTransposedB((a->keyWeightsGradientCache), (a->KGradientCache), (a->X));
    MatrixAddSelf((a->keyWeightsGradientAccum), (a->keyWeightsGradientCache));

    /*Gradient to X*/
    MatrixMultiplyTransposedA((a->XGradientCache), (a->valueWeights), (a->O));
    MatrixMultiplyTransposedB((a->X), (a->XGradientCache), (a->A));

    MatrixMultiplyTransposedA((a->XGradientCache), (a->queryWeights), (a->QGradientCache));
    MatrixAddSelf((a->X), (a->XGradientCache));

    MatrixMultiplyTransposedA((a->XGradientCache), (a->keyWeights), (a->KGradientCache));
    MatrixAddSelf((a->X), (a->XGradientCache));
}

void AMGradientDescent(AttentionMechanism* a, double learningRate)
{
    MatrixScaleSelf((a->queryWeightsGradientAccum), learningRate);
    MatrixScaleSelf((a->keyWeightsGradientAccum), learningRate);
    MatrixScaleSelf((a->valueWeightsGradientAccum), learningRate);

    MatrixSubSelf((a->queryWeights), (a->queryWeightsGradientAccum));
    MatrixSubSelf((a->keyWeights), (a->keyWeightsGradientAccum));
    MatrixSubSelf((a->valueWeights), (a->valueWeightsGradientAccum));

    /*reset for next runs*/
    SetMatrix((a->queryWeightsGradientAccum), 0.0);
    SetMatrix((a->keyWeightsGradientAccum), 0.0);
    SetMatrix((a->valueWeightsGradientAccum), 0.0);
}