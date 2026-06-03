#include "Attention.hpp"
#include "Matrix.hpp"
#include "MemArena.hpp"
#include "MiscMath.hpp"
#include <cstdio>
#include <math.h>
#include <cstdint>


AttentionMechanism* SAMCreate(uint32_t tokenCount, uint32_t inputDim, uint32_t outputDim, double learningRate)
{
    size_t allocSize = sizeof(AttentionMechanism);
    /*Parameter matrix sizes*/
    allocSize += (GetMatrixAllocSize(outputDim, inputDim) * 3);

    /*cache vector sizes*/
    allocSize += (GetMatrixAllocSize(outputDim, 1) * 4);
    allocSize += GetMatrixAllocSize(inputDim, 1);
    allocSize += GetMatrixAllocSize(tokenCount, tokenCount);
    allocSize += GetMatrixAllocSize(tokenCount, 1);

    MemArena arena = CreateMemArena(allocSize);
    AttentionMechanism* ret = (AttentionMechanism*)MemArenaAlloc(&arena, sizeof(AttentionMechanism));

    /*=== parameters ===*/

    ret->queryWeights = CreateRandomVariancePreservingMatrix(&arena, outputDim, inputDim);
    ret->keyWeights = CreateRandomVariancePreservingMatrix(&arena, outputDim, inputDim);
    ret->valueWeights = CreateRandomVariancePreservingMatrix(&arena, outputDim, inputDim);

    /*=== caches ===*/

    ret->inputTokenCache = CreateMatrix(&arena, inputDim, 1, NULL);

    ret->queryVectorCache = CreateMatrix(&arena, outputDim, 1, NULL);
    ret->keyVectorCache = CreateMatrix(&arena, outputDim, 1, NULL);
    ret->valueVectorCache = CreateMatrix(&arena, outputDim, 1, NULL);

    ret->attentionMatrixCache = CreateMatrix(&arena, tokenCount, tokenCount, NULL);
    ret->attentionSingleColumnCache = CreateMatrix(&arena, tokenCount, 1, NULL);
    ret->outputVectorCache = CreateMatrix(&arena, outputDim, 1, NULL);

    ret->arena = arena;
    ret->learningRate = learningRate;

    return ret;
}

void SAMDelete(AttentionMechanism* a)
{
    DeleteMemArena(&(a->arena));
}

size_t SAMGetParamCount(const AttentionMechanism* a)
{
    return (GetMatrixSize((a->queryWeights)) + GetMatrixSize((a->keyWeights)) + GetMatrixSize((a->valueWeights)));
}

void SAMCopyParameters(AttentionMechanism* dest, const AttentionMechanism* src)
{
    CopyMatrixData((dest->queryWeights), (src->queryWeights));
    CopyMatrixData((dest->keyWeights), (src->keyWeights));
    CopyMatrixData((dest->valueWeights), (src->valueWeights));
}

Matrix GetMaskedAttentionColumn(AttentionMechanism* a, uint32_t i, Matrix X)
{
    double attDimSqrt = sqrt((double)SAMGetOutputDimension(a));

    for(uint32_t j = (i + 1); j < SAMGetTokenCount(a); j++)
        SetValueMatrix((a->attentionSingleColumnCache), -INFINITY, j, 0); /*mask*/

    MatrixMultiply(&(a->queryVectorCache), (a->queryWeights), GetColumnVectorMatrix(X, i));
    for(uint32_t j = 0; j <= i; j++) /*because of mask we only have to calculate attention to tokens before and including i*/
    {
        MatrixMultiply(&(a->keyVectorCache), (a->keyWeights), GetColumnVectorMatrix(X, j));

        double attention = Dot((a->queryVectorCache), (a->keyVectorCache));
        SetValueMatrix((a->attentionSingleColumnCache), (attention / attDimSqrt), j, 0);
    }
    Softmax((a->attentionSingleColumnCache), 0);

    return (a->attentionSingleColumnCache);
}

Matrix GetMaskedAttentionMatrix(AttentionMechanism* a, Matrix X)
{
    for(uint32_t i = 0; i < SAMGetTokenCount(a); i++)
        SetColumnVectorMatrix((a->attentionMatrixCache), GetMaskedAttentionColumn(a, i, X), i);

    return (a->attentionMatrixCache);
}

Matrix GetMaskedSelfAttention(AttentionMechanism* a, uint32_t i, Matrix X)
{
    Matrix attentionColumn = GetMaskedAttentionColumn(a, i, X);

    SetMatrix((a->outputVectorCache), 0.0);
    for(uint32_t j = 0; j <= i; j++)
    {
        /*calculates how relevant value vector of token j is to token i*/
        MatrixMultiply(&(a->valueVectorCache), (a->valueWeights), GetColumnVectorMatrix(X, j));
        MatrixScaleSelf((a->valueVectorCache), GetValueMatrix(attentionColumn, j, 0));
        
        MatrixAddSelf((a->outputVectorCache), (a->valueVectorCache));
    }

    return (a->outputVectorCache);
}