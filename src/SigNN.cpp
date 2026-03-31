#include "SigNN.hpp"

#include "Matrix.hpp"
#include "MemArena.hpp"
#include "NeuralNetwork.hpp"
#include <cstdint>
#include <math.h>

/*iisignature*/
#include "SRN.hpp"
#include "iisignature/bch.hpp"
#include "iisignature/calcSignature.hpp"


SigNN* CreateSigNN(SRN* srn, uint32_t* neuronsPerHiddenLayer, ActivationFnID* activationFnPerLayer, uint32_t hiddenLayerCount, double learningRate, uint32_t truncationOrder)
{
    uint32_t M = SRNGetSpeciesCount(srn);
    uint32_t sigLen = calcSigTotalLength((M + 1), truncationOrder);

    MemArena arena = CreateMemArena(sizeof(SigNN) + GetMatrixAllocSize(sigLen, 1));

    SigNN* ret = (SigNN*)MemArenaAlloc(&arena, sizeof(SigNN));

    /*specify neurons per layer, the input will be the path signature of datapoints (n, t) in N^M x R*/
    uint32_t neuronsPerLayer[hiddenLayerCount + 2];
    neuronsPerLayer[0] = sigLen;
    for(uint32_t i = 1; i <= hiddenLayerCount; i++)
        neuronsPerLayer[i] = neuronsPerHiddenLayer[i - 1];
    neuronsPerLayer[hiddenLayerCount + 1] = M;

    ret->arena = arena;
    ret->nn = NNCreate(neuronsPerLayer, activationFnPerLayer, hiddenLayerCount, learningRate);
    ret->srn = srn;
    ret->signatureInput = CreateMatrix(&(ret->arena), sigLen, 1, NULL);

    ret->truncationOrder = truncationOrder;
    ret->sigLen = sigLen;

    return ret;
}   

void DeleteSigNN(SigNN* sigNN)
{
    NNDelete((sigNN->nn));
    DeleteMemArena(&(sigNN->arena));
}


double SigNNPredict(SigNN* sigNN, uint32_t* n, double t)
{
    uint32_t M = SRNGetSpeciesCount((sigNN->srn));

    double path[M + 1];
    for(uint32_t i = 0; i < M; i++)
        path[i] = (double)n[i];
    path[M] = t;

    /*TODO: dit werkt zo niet, het moet een pad zijn voor een signature, prima voor trainen, maar wat als ik gewoon P(n, t) wil hebben?*/
    CalcSignature::Signature sig;
    CalcSignature::calcSignature((M + 1), (sigNN->truncationOrder), 1, path, sig);
    sig.writeOut((sigNN->signatureInput.matrixData));
    PrintMatrix((sigNN->signatureInput));

    NNPredict((sigNN->nn), (sigNN->signatureInput));

    double conditionalProbabilityProduct = 1.0;
    for(uint32_t i = 0; i < M; i++)
        conditionalProbabilityProduct *= GetValueMatrix((sigNN->nn->layerVectors[(sigNN->nn->hiddenLayerCount) + 1]), i, 0);

    return conditionalProbabilityProduct;
}

void SigNNTrain(SigNN* sigNN, double T, uint64_t epochs)
{
}