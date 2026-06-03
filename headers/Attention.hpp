#pragma once

#include <cstdint>

#include "Matrix.hpp"


/*QKV global self-attention mechanism*/
typedef struct AttentionMechanism
{
    MemArena arena;

    /*=== parameters ===*/

    Matrix queryWeights; /*W_Q*/
    Matrix keyWeights; /*W_K*/
    Matrix valueWeights; /*W_V*/

    /*=== caches ===*/

    Matrix inputTokenCache;

    Matrix queryVectorCache;
    Matrix keyVectorCache;
    Matrix valueVectorCache;

    Matrix attentionMatrixCache;
    Matrix attentionSingleColumnCache;
    Matrix outputVectorCache;

    /*might be better to move into backpropagation function*/
    double learningRate; //this determines how fast the backpropagation strides towards the local minimum, in a gradient descent sense
} AttentionMechanism;


AttentionMechanism* SAMCreate(uint32_t tokenCount, uint32_t inputDim, uint32_t outputDim, double learningRate);
void SAMDelete(AttentionMechanism* a);

static inline uint32_t SAMGetInputDimension(AttentionMechanism* a) { return (a->queryWeights.columnCount); }
static inline uint32_t SAMGetOutputDimension(AttentionMechanism* a) { return (a->queryWeights.rowCount); }
static inline uint32_t SAMGetTokenCount(AttentionMechanism* a) { return (a->attentionMatrixCache.rowCount); }

size_t SAMGetParamCount(const AttentionMechanism* a);
void SAMCopyParameters(AttentionMechanism* dest, const AttentionMechanism* src);

Matrix GetMaskedAttentionColumn(AttentionMechanism* a, uint32_t i, Matrix X);
Matrix GetMaskedAttentionMatrix(AttentionMechanism* a, Matrix X);

/*maskes attention with next data vector indices, only expects X to have the data vectors for j <= i, even column count of X is not expected to be above i*/
Matrix GetMaskedSelfAttention(AttentionMechanism* a, uint32_t i, Matrix X);

/*like GetMaskedSelfAttention but gets the entire transformed data matrix, is a bit faster than doing it individually, but expects full data matrix X*/
// Matrix GetFullMaskedSelfAttention(AttentionMechanism* a, Matrix X);

//void SAMBackpropagation();