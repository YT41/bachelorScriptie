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

    Matrix queryWeightsGradientCache; /*\nabla_{W_Q} L*/
    Matrix keyWeightsGradientCache; /*\nabla_{W_K} L*/
    Matrix valueWeightsGradientCache; /*\nabla_{W_V} L*/
    Matrix XA; /*set during gradient descent*/
    Matrix AGradientCache; /*\nabla_A L*/
    Matrix SGradientCache; /*\nabla_S L*/
    Matrix QGradientCache; /*\nabla_Q L*/
    Matrix KGradientCache; /*\nabla_K L*/
    Matrix XGradientCache; /*only for intermediate calculations*/

    /*=== for gradient descent ===*/

    Matrix queryWeightsGradientAccum;
    Matrix keyWeightsGradientAccum;
    Matrix valueWeightsGradientAccum;

    /*=== matrices ===*/

    Matrix X; /*input matrix*/

    Matrix Q; /*Q := W_Q X*/
    Matrix K; /*K := W_K X*/
    Matrix V; /*V := W_V X*/

    Matrix A; /*A := Softmax(S/sqrt(d_{att})), where S is the score matrix*/
    Matrix O; /*O := VA*/

    /*IMPORTANT NOTE: this attention mechanism is the transposed version of the usual version you see in papers, because it was easier to implement given the code I already had*/
} AttentionMechanism;


AttentionMechanism* AMCreate(uint32_t tokenCount, uint32_t inputDim, uint32_t outputDim);
void AMDelete(AttentionMechanism* a);

static inline uint32_t AMGetInputDimension(AttentionMechanism* a) { return (a->queryWeights.columnCount); }
static inline uint32_t AMGetOutputDimension(AttentionMechanism* a) { return (a->queryWeights.rowCount); }
static inline uint32_t AMGetTokenCount(AttentionMechanism* a) { return (a->A.rowCount); }
static inline Matrix AMGetA(AttentionMechanism* a) { return (a->A); }

size_t AMGetParamCount(const AttentionMechanism* a);
void AMCopyParameters(AttentionMechanism* dest, const AttentionMechanism* src);

Matrix AMGetMaskedAttentionWeightColumn(AttentionMechanism* a, uint32_t j, Matrix X);
Matrix AMGetMaskedAttentionWeightMatrix(AttentionMechanism* a, Matrix X);

/*maskes attention with next data vector indices, only expects X to have the data vectors for j <= i, even column count of X is not expected to be above i*/
Matrix AMGetSingleMaskedAttention(AttentionMechanism* a, uint32_t j, Matrix X);

/*like AMGetSingleMaskedAttention but gets the entire transformed data matrix, is a bit faster than doing it individually, but expects full data matrix X*/
Matrix AMGetMaskedAttention(AttentionMechanism* a, Matrix X);

/*assumes gradient of some function with respects to O has been put into O, propagates the gradient backwards into X*/
void AMBackwardPass(AttentionMechanism* a);
void AMGradientDescent(AttentionMechanism* a, double learningRate);