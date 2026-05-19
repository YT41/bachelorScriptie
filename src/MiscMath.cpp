#include "MiscMath.hpp"

#include "Matrix.hpp"
#include <cstdint>
#include <cstdio>


void LogFullDistribution(Tensor stateSpaceProbabilities, double t, FILE* logFile)
{
    uint32_t M = stateSpaceProbabilities.dimensionCount;
    MemArena arena = CreateMemArena(GetIntMatrixAllocSize(M, 1));

    IntMatrix n = CreateBlankIntMatrix(&arena, M, 1);
    SetIntMatrix(n, 0);
    do
    {
        for(uint32_t i = 0; i < M; i++)
            fprintf(logFile, "%d ", GetValueIntMatrix(n, i, 0));
        fprintf(logFile, "%f ", GetValueTensor(stateSpaceProbabilities, n));
        fprintf(logFile, "%f\n", t);
        IncrementTensorIndex(stateSpaceProbabilities, n);
    }
    while(!IntMatrixIsZero(n));
    fputs("\n", logFile); /*TODO: this is needed for heat plot but only for the x value for some reason, but it would be better to not have*/

    DeleteMemArena(&arena);
}

void LogDataPoint(double t, Matrix dataPoint, FILE* logFile)
{
    fprintf(logFile, "%f", t);
    for(uint32_t j = 0; j < (dataPoint.rowCount); j++)
        fprintf(logFile, " %f", GetValueMatrix(dataPoint, j, 0));
    fputs("\n", logFile);
}