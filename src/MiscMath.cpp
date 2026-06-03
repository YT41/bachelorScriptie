#include "MiscMath.hpp"

#include "Matrix.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>


/*====================== useful functions in machine learning ======================*/

static inline void SoftmaxHelperFunction(Matrix z, uint32_t column) /*z does not change*/
{
    uint32_t K = (z.rowCount);
    double m = GetValueMatrix(z, 0, column);
    for(uint32_t i = 1; i < K; i++)
    {
        double zi = GetValueMatrix(z, i, column);
        if(zi > m)
            m = zi;
    }

    double exponentialsCache[K];
    double sum = 0.0;
    for(uint32_t i = 0; i < K; i++)
    {
        exponentialsCache[i] = exp(GetValueMatrix(z, i, column) - m);
        sum += exponentialsCache[i];
    }

    for(uint32_t i = 0; i < K; i++)
        SetValueMatrix(z, (exponentialsCache[i] / sum), i, column);
}

double Sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double ReLU(double x) { return MAX(x, 0.0); }

double DSigmoidDx(double x) { double sigmoidx = Sigmoid(x); return (sigmoidx * (1.0 - sigmoidx)); }
double DTanhDx(double x) { double tanhx = tanh(x); return (1.0 - (tanhx * tanhx)); }
double DReLUDx(double x) { return (x < 0.0) ? 0.0 : 1.0; } /*does not actually exist everywhere*/

void Identity(Matrix z, uint32_t column) {  }
void SigmoidElementWise(Matrix z, uint32_t column) { MatrixColumnTransformSelf(z, column, Sigmoid); }
void TanhElementWise(Matrix z, uint32_t column) { MatrixColumnTransformSelf(z, column, tanh); }
void ReLUElementWise(Matrix z, uint32_t column) { MatrixColumnTransformSelf(z, column, ReLU); }
void Softmax(Matrix z, uint32_t column) { SoftmaxHelperFunction(z, column); }

/*dest must be same size as z, z is the vector to evaluate derivative in, z is not changed*/
void IdentityDerivative(Matrix dest, Matrix z, uint32_t column) { SetColumnMatrix(dest, column, 1.0); }
void SigmoidDerivative(Matrix dest, Matrix z, uint32_t column) { MatrixColumnTransform(&dest, z, column, DSigmoidDx); }
void TanhDerivative(Matrix dest, Matrix z, uint32_t column) { MatrixColumnTransform(&dest, z, column, DTanhDx); }
void ReLUDerivative(Matrix dest, Matrix z, uint32_t column) { MatrixColumnTransform(&dest, z, column, DReLUDx); }

/*TODO: rework or remove, does not work now, might be better this way as it gets very complicated otherwise*/
void SoftmaxDerivative(Matrix dest, Matrix z, uint32_t column)
{ 
    // uint32_t K = (z.rowCount);
    // double softmaxVals[K];
    // SoftmaxHelperFunction(softmaxVals, z);

    // for(uint32_t i = 0; i < K; i++)
    // {
    //     for(uint32_t j = 0; j < K; j++)
    //         SetValueMatrix(dest, -(softmaxVals[i] * softmaxVals[j]), i, j);
    // }

    // /*set diagonal values correctly now*/
    // for(uint32_t i = 0; i < K; i++)
    //     SetValueMatrix(dest, (softmaxVals[i] * (1.0 - softmaxVals[i])), i, i);
}


/*====================== logging functions ======================*/

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