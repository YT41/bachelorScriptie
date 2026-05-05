#include "BTCM.hpp"

#include "Matrix.hpp"
#include "MemArena.hpp"
#include "NeuralNetwork.hpp"
#include "Random.hpp"
#include "SRN.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>


/*============================ helper functions ============================*/

static inline double GetInitialConditionProbability(IntMatrix n, const SRN* srn)
{
    uint32_t M = SRNGetSpeciesCount(srn);
    for(uint32_t i = 0; i < M; i++)
    {
        if(GetValueIntMatrix(n, i, 0) != (int32_t)(srn->species[i].initialCount))
            return 0.0;
    }
    return 1.0;
}

/*return probability of having picking this sample (its always 1)*/
static inline double GetInitialConditionSample(IntMatrix s, const SRN* srn)
{
    uint32_t M = SRNGetSpeciesCount(srn);
    for(uint32_t i = 0; i < M; i++)
        SetValueIntMatrix(s, (srn->species[i].initialCount), i, 0);
    return 1.0;
}

/*prepare token i: x_i^{(0)} := (i, t, n_{<i})*/
static inline void PrepareInputToken(BTCM* m, uint32_t i, IntMatrix n, double t)
{
    /*prepare token i: x_i^{(0)} := (i, t, n_{<i})*/
    SetMatrix((m->tokenCache), 0.0); /*padding is just set to 0*/
    SetValueMatrix((m->tokenCache), (double)i, 0, 0);
    SetValueMatrix((m->tokenCache), t, 1, 0);
    for(uint32_t j = 0; j < i; j++)
        SetValueMatrix((m->tokenCache), GetValueIntMatrix(n, j, 0), (j + 2), 0);
}

static inline double GetTargetProbability(BTCM* m, IntMatrix sample, IntMatrix previousState, double t, double deltaT)
{
    /*we know that we can only enter the sample state from connected states, all other propensities are 0*/
    double enteringSampleStateProbability = 0.0;
    for(uint32_t k = 0; k < SRNGetReactionCount((m->srn)); k++)
    {
        double possiblePreviousStatePropensity = GetPreviousConnectedState((m->srn), sample, previousState, k);

        if(possiblePreviousStatePropensity != 0.0)
            enteringSampleStateProbability += (possiblePreviousStatePropensity * BTCMPredict(m, previousState, t));
    }
    double tSampleProbability = BTCMPredict(m, sample, t);
    /*note that this is just one step of explicit euler with the CME*/
    return (tSampleProbability + (deltaT * (enteringSampleStateProbability - (GetEscapeRate((m->srn), sample) * tSampleProbability))));
}


/*============================ public functions ============================*/

BTCM* BTCMCreate(SRN* srn, uint32_t* neuronsPerHiddenLayer, uint32_t hiddenLayerCount, double learningRate)
{
    uint32_t M = SRNGetSpeciesCount(srn);

    uint32_t inputTokenDim = M + 1; /*TODO: edit this to correct count, it should be higher than M + 1 eventually*/

    MemArena arena = CreateMemArena(sizeof(BTCM) + GetMatrixAllocSize(inputTokenDim, 1));

    BTCM* ret = (BTCM*)MemArenaAlloc(&arena, sizeof(BTCM));

    ret->tokenCache = CreateMatrix(&arena, inputTokenDim, 1, NULL);

    /*specify neurons per layer*/
    uint32_t neuronsPerLayer[hiddenLayerCount + 2];
    neuronsPerLayer[0] = inputTokenDim;
    for(uint32_t i = 1; i <= hiddenLayerCount; i++)
        neuronsPerLayer[i] = neuronsPerHiddenLayer[i - 1];
    neuronsPerLayer[hiddenLayerCount + 1] = SRNGetMaxSpeciesCount(srn);

    /*specify activation functions for layers*/
    ActivationFnID activationFnPerLayer[hiddenLayerCount + 1];
    for(uint32_t i = 0; i < hiddenLayerCount; i++)
        activationFnPerLayer[i] = TANH;
    activationFnPerLayer[hiddenLayerCount] = SOFTMAX;

    ret->arena = arena;
    ret->nn = NNCreate(neuronsPerLayer, activationFnPerLayer, hiddenLayerCount, learningRate);
    ret->srn = srn;

    return ret;
}   

void BTCMDelete(BTCM* m)
{
    NNDelete((m->nn));
    DeleteMemArena(&(m->arena));
}


/*doing it this way, it takes a sample and returns the probability of having taken this sample at the same time*/
double BTCMTakeSample(BTCM* m, IntMatrix s, double t, Matrix desiredNudgesMLPOutput)
{
    if(t == 0.0)
        return GetInitialConditionSample(s, (m->srn));

    uint32_t M = SRNGetSpeciesCount((m->srn));
    double finalProbability = 1.0;
    for(uint32_t i = 0; i < M; i++)
    {
        PrepareInputToken(m, i, s, t);

        Matrix out = NNPredict((m->nn), (m->tokenCache));

        /*TODO: debug: for checking MLP output*/
        // for(uint32_t j = 0; j < (out.rowCount); j++)
        //     printf("P(n_%u = %u | n_{<%u}, %.2f) = %.2f\n", i, j, i, t, GetValueMatrix(out, j, 0));

        /*simulate a count for species i using generated conditional probabilities*/
        uint32_t sim = PickUintWithChances(out.data, out.rowCount);
        SetValueIntMatrix(s, sim, i, 0);

        double conditionalProbability = GetValueMatrix(out, GetValueIntMatrix(s, i, 0), 0);

        /*increment desired nudges*/
        SetValueMatrix(desiredNudgesMLPOutput, (GetValueMatrix(desiredNudgesMLPOutput, sim, 0) + (1.0 / conditionalProbability)), sim, 0);

        finalProbability *= conditionalProbability;
    }

    return finalProbability;
}

double BTCMPredict(BTCM* m, IntMatrix n, double t)
{
    /*in this case we know the probabilities exactly*/
    if(t == 0.0)
        return GetInitialConditionProbability(n, (m->srn));

    uint32_t M = SRNGetSpeciesCount((m->srn));
    double finalProbability = 1.0;
    for(uint32_t i = 0; i < M; i++)
    {
        PrepareInputToken(m, i, n, t);

        Matrix out = NNPredict((m->nn), (m->tokenCache));
        finalProbability *= GetValueMatrix(out, GetValueIntMatrix(n, i, 0), 0);
    }

    return finalProbability;
}

void BTCMTrain(BTCM* m, double T, double deltaT, uint32_t B, uint64_t epochs)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(
        (GetIntMatrixAllocSize(M, 1) * 2) + 
        GetMatrixAllocSize(NNGetOutputDimension((m->nn)), 1)
    );

    IntMatrix sample = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix previousState = CreateBlankIntMatrix(&arena, M, 1);

    /*for gradient descent*/
    Matrix desiredNudgesMLPOutput = CreateMatrix(&arena, NNGetOutputDimension((m->nn)), 1, NULL);

    for(uint64_t e = 0; e < epochs; e++)
    {
        SetMatrix(desiredNudgesMLPOutput, 0.0);

        /*TODO: debug: t should be taken randomly in the future, in whatever way, this is just for testing*/
        //double t = UniformSim(0.0, T, false, false);
        double t = 0.0;

        double loss = 0.0; /*KL-divergence a.k.a. cross-entropy*/

        /*take B samples*/
        for(uint32_t b = 0; b < B; b++)
        {
            double sampleProbability = BTCMTakeSample(m, sample, (t + deltaT), desiredNudgesMLPOutput);
            double targetProbability = GetTargetProbability(m, sample, previousState, t, deltaT);

            /*TODO: debug: checks if either one is < 0 to prevent erros, 0 is allowerd, in this case we just get -inf*/
            if((sampleProbability < 0.0) || (targetProbability < 0.0))
                printf("error: trying to take log of non-positive value in trying to calculate cross-entropy\n");

            loss += (log(sampleProbability) - log(targetProbability));
        }
        loss /= (double)B;

        MatrixScaleSelf(desiredNudgesMLPOutput, (1.0 / (double)B));

        /*TODO: debug*/
        //PrintMatrix(desiredNudgesMLPOutput);

        NNSetLastLayer((m->nn), desiredNudgesMLPOutput);
        NNBackPropagation((m->nn));

        printf("loss of epoch %lu: %f\n", e, loss);
    }


    /*TODO: debug: for checking MLP output*/
    PrepareInputToken(m, 0, sample, 0.0 + deltaT);
    Matrix out = NNPredict((m->nn), (m->tokenCache));
    for(uint32_t j = 0; j < (out.rowCount); j++)
        printf("P(n_%u = %u | n_{<%u}, %.2f) = %f\n", 0, j, 0, 0.0 + deltaT, GetValueMatrix(out, j, 0));


    DeleteMemArena(&arena);
}