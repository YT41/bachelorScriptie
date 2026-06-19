#pragma once

#include <cstdint>

#include "Matrix.hpp"


/*QKV global self-attention mechanism, see section 3.2.1 of thesis report*/
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


/*Creates an instance of an attention mechanism, see section 3.2 of the thesis report

INPUTS:
uint32_t tokenCount: or sequence length it is sometimes called
uint32_t inputDim: d (see thesis report definition)
uint32_t outputDim: d_k (see thesis report definition)

RETURNS:
Heap-allocated pointer to AttentionMechanism structure, be sure to delete it with AMDelete whenever you're not using it anymore to prevent memory leaks.*/
AttentionMechanism* AMCreate(uint32_t tokenCount, uint32_t inputDim, uint32_t outputDim);
void AMDelete(AttentionMechanism* a);


static inline uint32_t AMGetInputDimension(AttentionMechanism* a) { return (a->queryWeights.columnCount); }
static inline uint32_t AMGetOutputDimension(AttentionMechanism* a) { return (a->queryWeights.rowCount); }
static inline uint32_t AMGetTokenCount(AttentionMechanism* a) { return (a->A.rowCount); }
static inline Matrix AMGetA(AttentionMechanism* a) { return (a->A); } /*gets the post-Softmax score matrix*/

size_t AMGetParamCount(const AttentionMechanism* a);
void AMCopyParameters(AttentionMechanism* dest, const AttentionMechanism* src);


/*Creates an instance of an attention mechanism, see section 3.2 of the thesis report

INPUTS:
AttentionMechanism* a: pointer to attention mechanism to get masked post-softmax score matrix column from
uint32_t j: column
Matrix X: input token matrix, expects column j of X to be set, dimension is expected to be n times d

RETURNS:
column j of post-Softmax score matrix*/
Matrix AMGetMaskedAttentionWeightColumn(AttentionMechanism* a, uint32_t j, Matrix X);
Matrix AMGetMaskedAttentionWeightMatrix(AttentionMechanism* a, Matrix X); /*just runs AMGetMaskedAttentionWeightColumn for all j, and then returns the entire post-softmax score matrix*/


/*gets masked attention for token j, only expects X to have the data vectors for i <= j

INPUTS:
AttentionMechanism* a: pointer to attention mechanism to get masked attention from
uint32_t j: column, so token index
Matrix X: input token matrix, expects all columns i <= j of X to be set, dimension is expected to be n times d

RETURNS:
attention vector for token j, this would be h_j in thesis report section 3.2.2*/
Matrix AMGetSingleMaskedAttention(AttentionMechanism* a, uint32_t j, Matrix X);
Matrix AMGetMaskedAttention(AttentionMechanism* a, Matrix X); /*like AMGetSingleMaskedAttention but gets the entire transformed data matrix, is a bit faster than doing it individually, but expects full data matrix X*/


void AMBackwardPass(AttentionMechanism* a); /*backpropagation, assumes gradient of some function with respects to O has been put into O, propagates the gradient backwards into X*/
void AMGradientDescent(AttentionMechanism* a, double learningRate); /*seperate function for updating the parameters of the attention mechanism according to gradient descent, where gradients will first be cached in AMBackwardPass call*/