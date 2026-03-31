#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "Matrix.hpp"
#include "MemArena.hpp"
#include "Random.hpp"
#include "ReactionParser.hpp"
#include "SRN.hpp"
#include "SigNN.hpp"
#include "TrajectorySim.hpp"
#include "NeuralNetwork.hpp"


int main(int argc, char** argv)
{
    SetSeedRandU48(time(NULL));

    // SRN* srn = ParseSRN("res/GeneExpressionModel.txt");

    // PrintIntMatrix((srn->stoichiometricMatrix));

    // GillespieSRNTrajectorySim(3600.0, 20, srn, "res/trajectory.data");

    // uint32_t neuronsHiddenLayer = 32;
    // ActivationFnID activationFnPerLayer[] = { TANH, IDENTITY };
    // SigNN* sigNN = CreateSigNN(srn, &neuronsHiddenLayer, activationFnPerLayer, 1, 0.05, 3);

    // uint32_t n[2] = {1, 2};
    // SigNNPredict(sigNN, n, 0.0);

    // DeleteSigNN(sigNN);
    // DeleteSRN(srn);

    uint32_t neuronsPerLayer[] = { 1, 64, 1 };
    ActivationFnID activationFnPerLayer[] = { TANH, IDENTITY };
    NeuralNetwork* nn = NNCreate(neuronsPerLayer, activationFnPerLayer, 1, 0.01);

    MemArena arena = CreateMemArena(GetMatrixAllocSize(1, 1) * 2);

    FILE* file1 = fopen("res/NNTestLoss.txt", "w");

    Matrix x = CreateMatrix(&arena, 1, 1, NULL);
    Matrix y = CreateMatrix(&arena, 1, 1, NULL);
    for(uint64_t i = 0; i < 1000000; i++)
    {
        double r1 = UniformSim(-10.0, 10.0, true, true);

        SetValueMatrix(x, r1, 0, 0);
        SetValueMatrix(y, sin(r1), 0, 0);

        double loss = NNTrain(nn, x, y);
        fprintf(file1, "%lu %f \n", i, loss);
    }
    fclose(file1);

    FILE* file2 = fopen("res/NNTest.txt", "w");
    for(double input = -10.0; input < 10.0; input += 0.01)
    {
        SetValueMatrix(x, input, 0, 0);

        NNPredict(nn, x);

        fprintf(file2, "%f %f %f\n", input, sin(input), (nn->layerVectors[(nn->hiddenLayerCount) + 1].data[0]));
    }
    fclose(file2);


    DeleteMemArena(&arena);

    NNDelete(nn);

    return 0;
}