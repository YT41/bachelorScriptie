#include "generalModelFunctions.hpp"
#include "Matrix.hpp"
#include "NeuralNetwork.hpp"
#include <math.h>
#include <stdint.h>


/*==================== General model Functions ====================*/

static inline void AppendValueVector(Matrix matrix, double val, size_t* size, uint32_t column)
{
    SetValueMatrix(matrix, val, *size, column);
    (*size)++;
}

/*prepare token i: X_i^{(0)} := (i, \psi(t), \eta_i, n_{<i})*/
void GetSingleInputToken(SRN* srn, Matrix X, uint32_t i, IntMatrix n, Matrix embeddedTime)
{
    uint32_t K = SRNGetReactionCount(srn);

    size_t tokenSize = 0;
    SetColumnMatrix(X, -1.0, i); /*padding is just set to -1, an invalid state*/

    AppendValueVector(X, (double)i, &tokenSize, i);

    for(uint32_t j = 0; j < (embeddedTime.rowCount); j++)
        AppendValueVector(X, GetValueMatrix(embeddedTime, j, 0), &tokenSize, i);

    for(uint32_t k = 0; k < K; k++)
    {
        AppendValueVector(X, (double)GetValueIntMatrix((srn->reactantMatrix), i, k), &tokenSize, i);
        AppendValueVector(X, (double)GetValueIntMatrix((srn->productMatrix), i, k), &tokenSize, i);
        AppendValueVector(X, (srn->reactionRates[k]), &tokenSize, i);
    }
    
    for(uint32_t j = 0; j < i; j++)
        AppendValueVector(X, (double)GetValueIntMatrix(n, j, 0), &tokenSize, i);
}

void GetInputTokens(SRN* srn, Matrix X, IntMatrix n, Matrix embeddedTime)
{
    uint32_t M = SRNGetReactionCount(srn);
    for(uint32_t i = 0; i < M; i++)
        GetSingleInputToken(srn, X, i, n, embeddedTime);
}

Matrix GetEmbeddedTime(NeuralNetwork* timeEmbedding, double t)
{
    /*sinusoidal time embedding (already experimented with)*/
    // for(uint32_t j = 1; j <= 16; j++)
    //     AppendValueVector(X, sin(2.0 * M_PI * (double)j * (t / T)), &tokenSize, i);

    const double eps = 1e-3;
    double timeInput[2] = { t, log(t + eps) };
    SetMatrixData(NNGetFirstLayer(timeEmbedding), timeInput);
    return NNPredictNoCopy(timeEmbedding);
}