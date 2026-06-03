#include "TCAM.hpp"
#include "Attention.hpp"


TCAM* TCAMCreate(SRN* srn, uint32_t attentionDim, uint32_t* neuronsPerHiddenLayer, uint32_t hiddenLayerCount, double learningRate)
{
    uint32_t M = SRNGetSpeciesCount(srn);
    uint32_t K = SRNGetReactionCount(srn);

    uint32_t inputTokenDim = 1 + 1 + (K * 3) + (M - 1);

    MemArena arena = CreateMemArena(
        sizeof(TCAM) + 
        GetMatrixAllocSize(inputTokenDim, 1) + 
        GetMatrixAllocSize(inputTokenDim, M)
    );

    TCAM* ret = (TCAM*)MemArenaAlloc(&arena, sizeof(TCAM));

    ret->tokenCache = CreateMatrix(&arena, inputTokenDim, 1, NULL);
    ret->X = CreateMatrix(&arena, inputTokenDim, M, NULL);

    ret->a = SAMCreate(M, inputTokenDim, attentionDim, learningRate);

    /*specify neurons per layer*/
    uint32_t neuronsPerLayer[hiddenLayerCount + 2];
    neuronsPerLayer[0] = attentionDim;
    for(uint32_t i = 1; i <= hiddenLayerCount; i++) { neuronsPerLayer[i] = neuronsPerHiddenLayer[i - 1]; }
    neuronsPerLayer[hiddenLayerCount + 1] = SRNGetMaxSpeciesCount(srn);

    /*specify activation functions for layers*/
    ActivationFnID activationFnPerLayer[hiddenLayerCount + 1];
    for(uint32_t i = 0; i < hiddenLayerCount; i++) { activationFnPerLayer[i] = TANH; }
    activationFnPerLayer[hiddenLayerCount] = SOFTMAX;

    ret->arena = arena;
    ret->nn = NNCreate(neuronsPerLayer, activationFnPerLayer, hiddenLayerCount, M, learningRate);
    ret->srn = srn;

    return ret;
}

void TCAMDelete(TCAM* m)
{
    SAMDelete((m->a));
    NNDelete((m->nn));
    DeleteMemArena(&(m->arena));
}


size_t TCAMGetParamCount(const TCAM* m)
{
    return (SAMGetParamCount((m->a)) + NNGetParamCount((m->nn)));
}

TCAM* TCAMCopy(const TCAM* m)
{
    uint32_t hiddenLayerCount = (m->nn->hiddenLayerCount);
    uint32_t neuronsPerHiddenLayer[hiddenLayerCount];
    for(uint32_t i = 0; i < hiddenLayerCount; i++)
        neuronsPerHiddenLayer[i] = (m->nn->layerVectors[i+1].rowCount);

    TCAM* ret = TCAMCreate((m->srn), SAMGetOutputDimension(m->a), neuronsPerHiddenLayer, hiddenLayerCount, (m->nn->learningRate));

    TCAMCopyParameters(ret, m);

    return ret;
}

void TCAMCopyParameters(TCAM* dest, const TCAM* src)
{
    SAMCopyParameters((dest->a), (src->a));
    NNCopyParameters((dest->nn), (src->nn));
}

// double TCAMTakeSample(TCAM* m, IntMatrix s, double t, Matrix desiredNudgesMLPOutput)
// double TCAMTakeSampleNoGradient(TCAM* m, IntMatrix s, double t);
// double TCAMPredict(TCAM* m, IntMatrix n, double t); /*directly gives P(n, t), so this is NOT a full-grid evaluation*/
// void TCAMGetFullProbabilityDistribution(TCAM* m, Tensor probabilities, double t);
// void TCAMTrain(TCAM* m, double T, double deltaT, double p, uint32_t B, uint32_t Q, uint64_t epochs, const char* fileName); /*every Q epochs parameters of target are updated*/
