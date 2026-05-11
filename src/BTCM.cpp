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

/*return probability of having picked this sample in initial distribution (its always 1)*/
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
    /*note that this is just one step of explicit euler with the approximation of CME*/
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

BTCM* BTCMCopy(const BTCM* m)
{
    uint32_t hiddenLayerCount = (m->nn->hiddenLayerCount);
    uint32_t neuronsPerHiddenLayer[hiddenLayerCount];
    for(uint32_t i = 0; i < hiddenLayerCount; i++)
        neuronsPerHiddenLayer[i] = (m->nn->layerVectors[i+1].rowCount);

    BTCM* ret = BTCMCreate((m->srn), neuronsPerHiddenLayer, hiddenLayerCount, (m->nn->learningRate));

    BTCMCopyParameters(ret, m);

    return ret;
}

void BTCMCopyParameters(BTCM* dest, const BTCM* src)
{
    NNCopyParameters((dest->nn), (src->nn));
}


/*doing it this way, it takes a sample and returns the probability of having taken this sample at the same time*/
double BTCMTakeSample(BTCM* m, IntMatrix s, double t, Matrix desiredNudgesMLPOutput)
{
    // if(t == 0.0)
    //     return GetInitialConditionSample(s, (m->srn));

    SetMatrix(desiredNudgesMLPOutput, 0.0);

    uint32_t M = SRNGetSpeciesCount((m->srn));
    double conditionalProbabilityProduct = 1.0;
    for(uint32_t i = 0; i < M; i++)
    {
        PrepareInputToken(m, i, s, t);

        Matrix out = NNPredict((m->nn), (m->tokenCache));

        /*simulate a count for species i using generated conditional probabilities*/
        uint32_t sim = PickUintWithChances(out.data, out.rowCount);
        SetValueIntMatrix(s, sim, i, 0);

        double conditionalProbability = GetValueMatrix(out, GetValueIntMatrix(s, i, 0), 0);

        /*increment desired nudges*/
        SetValueMatrix(desiredNudgesMLPOutput, (GetValueMatrix(desiredNudgesMLPOutput, sim, 0) + (1.0 / conditionalProbability)), sim, 0);

        conditionalProbabilityProduct *= conditionalProbability;
    }

    return conditionalProbabilityProduct;
}

double BTCMTakeSampleNoGradient(BTCM* m, IntMatrix s, double t)
{
    // if(t == 0.0)
    //     return GetInitialConditionSample(s, (m->srn));

    uint32_t M = SRNGetSpeciesCount((m->srn));
    double conditionalProbabilityProduct = 1.0;
    for(uint32_t i = 0; i < M; i++)
    {
        PrepareInputToken(m, i, s, t);

        Matrix out = NNPredict((m->nn), (m->tokenCache));

        /*simulate a count for species i using generated conditional probabilities*/
        uint32_t sim = PickUintWithChances(out.data, out.rowCount);
        SetValueIntMatrix(s, sim, i, 0);

        conditionalProbabilityProduct *= GetValueMatrix(out, GetValueIntMatrix(s, i, 0), 0); 
    }

    return conditionalProbabilityProduct;
}

double BTCMPredict(BTCM* m, IntMatrix n, double t)
{
    /*in this case we know the probabilities exactly*/
    // if(t == 0.0)
    //     return GetInitialConditionProbability(n, (m->srn));

    uint32_t M = SRNGetSpeciesCount((m->srn));
    double conditionalProbabilityProduct = 1.0;
    for(uint32_t i = 0; i < M; i++)
    {
        PrepareInputToken(m, i, n, t);

        Matrix out = NNPredict((m->nn), (m->tokenCache));
        conditionalProbabilityProduct *= GetValueMatrix(out, GetValueIntMatrix(n, i, 0), 0);
    }

    return conditionalProbabilityProduct;
}

/*TODO: naive way of calculating full distribution, change if it ever takes too long*/
void BTCMGetFullProbabilityDistribution(BTCM* m, Tensor probabilities, double t)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(GetIntMatrixAllocSize(M, 1));

    IntMatrix n = CreateBlankIntMatrix(&arena, M, 1);
    SetIntMatrix(n, 0);
    do
    {
        SetValueTensor(probabilities, BTCMPredict(m, n, t), n);
        IncrementStateInStateSpace((m->srn), n);
    }
    while(!IntMatrixIsZero(n));

    DeleteMemArena(&arena);
}

void BTCMTrain(BTCM* m, double T, double deltaT, double lambda, uint32_t B, uint32_t Q, uint64_t epochs)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(
        (GetIntMatrixAllocSize(M, 1) * 2) + 
        (GetMatrixAllocSize(NNGetOutputDimension((m->nn)), 1) * 3)
    );

    BTCM* targetModelCopy = BTCMCopy(m);

    IntMatrix sample = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix previousState = CreateBlankIntMatrix(&arena, M, 1);

    double endT = 0.0;

    /*for gradient descent*/
    Matrix totalDesiredNudgesMLPOutput = CreateMatrix(&arena, NNGetOutputDimension((m->nn)), 1, NULL);
    Matrix desiredNudgesMLPOutput = CreateMatrix(&arena, NNGetOutputDimension((m->nn)), 1, NULL);
    Matrix zeroDesiredNudgesMLPOutput = CreateMatrix(&arena, NNGetOutputDimension((m->nn)), 1, NULL);
    for(uint64_t e = 0; e < epochs; e++)
    {
        double t = (StandardClosedUniformSim() * endT);
        double deltaTSim = deltaT; //(StandardClosedUniformSim() * deltaT); /*we can also vary deltaT, this might prevent overfitting*/
        if((e % 1000000) == 0) { endT += deltaT; } /*slowly increase interval to sample t uniformly from*/

        /*update parameters every Q epochs*/
        if(((e + 1) % Q) == 0) { BTCMCopyParameters(targetModelCopy, m); }

        double loss = 0.0; /*KL-divergence a.k.a. cross-entropy*/

        SetMatrix(totalDesiredNudgesMLPOutput, 0.0);

        /*take B samples*/
        for(uint32_t b = 0; b < B; b++)
        {
            double sampleProbability = BTCMTakeSample(m, sample, (t + deltaTSim), desiredNudgesMLPOutput);
            double targetProbability = GetTargetProbability(targetModelCopy, sample, previousState, t, deltaTSim);

            /*TODO: debug: checks if either one is < 0 to prevent erros, 0 is allowerd, in this case we just get -inf*/
            if((sampleProbability < 0.0) || (targetProbability < 0.0))
                printf("error: trying to take log of non-positive value in trying to calculate cross-entropy\n");

            double reward = (log(sampleProbability) - log(targetProbability));

            if((reward > -INFINITY) && (reward < INFINITY))
            {
                MatrixScaleSelf(desiredNudgesMLPOutput, reward);
                MatrixAddSelf(totalDesiredNudgesMLPOutput, desiredNudgesMLPOutput);
            }


            double zeroSampleProbability = BTCMTakeSample(m, sample, 0.0, zeroDesiredNudgesMLPOutput);
            double zeroTargetSampleProbability = GetInitialConditionProbability(sample, (m->srn));

            double zeroReward = (lambda * (log(zeroSampleProbability) - log(zeroTargetSampleProbability)));

            if((zeroReward > -INFINITY) && (zeroReward < INFINITY))
            {
                MatrixScaleSelf(zeroDesiredNudgesMLPOutput, zeroReward);
                MatrixAddSelf(totalDesiredNudgesMLPOutput, zeroDesiredNudgesMLPOutput);
            }

            loss += (reward + zeroReward);
        }
        loss /= (double)B;

        MatrixScaleSelf(totalDesiredNudgesMLPOutput, (1.0 / (double)B));
        NNSetLastLayer((m->nn), totalDesiredNudgesMLPOutput);
        NNBackPropagation((m->nn));

        if((e % 1000) == 0) { printf("loss of epoch %lu: %f\n", e, loss); }
    }


    /*TODO: debug: for checking MLP output*/
    PrepareInputToken(m, 0, sample, 0.0 + deltaT);
    Matrix out = NNPredict((m->nn), (m->tokenCache));
    for(uint32_t j = 0; j < (out.rowCount); j++)
        printf("P(n_%u = %u | n_{<%u}, %.2f) = %f\n", 0, j, 0, 0.0 + deltaT, GetValueMatrix(out, j, 0));


    BTCMDelete(targetModelCopy);
    DeleteMemArena(&arena);
}


/*==================== statistics ====================*/

void BTCMGetPerSpeciesMean(BTCM* m, Matrix mean, double t)
{
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena((GetIntMatrixAllocSize(M, 1) * 2));

    IntMatrix sample = CreateBlankIntMatrix(&arena, M, 1);
    IntMatrix sampleSum = CreateBlankIntMatrix(&arena, M, 1);
    SetIntMatrix(sampleSum, 0);

    const uint32_t sampleCount = 1000;
    for(uint32_t n = 0; n < sampleCount; n++)
    {
        BTCMTakeSampleNoGradient(m, sample, t);
        IntMatrixAddSelf(sampleSum, sample);
    }

    for(uint32_t i = 0; i < M; i++)
        SetValueMatrix(mean, ((double)GetValueIntMatrix(sampleSum, i, 0) / (double)sampleCount), i, 0);

    DeleteMemArena(&arena);
}

void BTCMGetPerSpeciesStandardDeviation(BTCM* m, Matrix std, double t)
{    
    uint32_t M = SRNGetSpeciesCount((m->srn));

    MemArena arena = CreateMemArena(GetMatrixAllocSize(M, 1) + GetIntMatrixAllocSize(M, 1));

    Matrix mean = CreateMatrix(&arena, M, 1, NULL);
    BTCMGetPerSpeciesMean(m, mean, t);

    IntMatrix sample = CreateBlankIntMatrix(&arena, M, 1);

    SetMatrix(std, 0.0);

    const uint32_t sampleCount = 1000;
    for(uint32_t n = 0; n < sampleCount; n++)
    {
        BTCMTakeSampleNoGradient(m, sample, t);

        for(uint32_t i = 0; i < M; i++)
            MatrixAddValue(std, pow((double)GetValueIntMatrix(sample, i, 0) - GetValueMatrix(mean, i, 0), 2.0), i, 0);
    }

    MatrixScaleSelf(std, (1.0 / (double)sampleCount));
    MatrixTransformSelf(std, sqrt);

    DeleteMemArena(&arena);
}