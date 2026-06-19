#pragma once

#include <cstdio>
#include <stdint.h>

#include "Matrix.hpp"

#define MAX(a, b)       (((a) > (b)) ? (a) : (b))
#define MIN(a, b)       (((a) < (b)) ? (a) : (b))


static inline float Lerp(const float a, const float b, const float t) { return a + (t * (b - a)); }

/*====================== useful functions in machine learning ======================*/

double Sigmoid(double x);
double ReLU(double x);

double DSigmoidDx(double x);
double DTanhDx(double x);
double DReLUDx(double x); /*does not actually exist everywhere*/

void Identity(Matrix z, uint32_t column);
void SigmoidElementWise(Matrix z, uint32_t column);
void TanhElementWise(Matrix z, uint32_t column);
void ReLUElementWise(Matrix z, uint32_t column);
void Softmax(Matrix z, uint32_t column); /*based on safe softmax, prevents large exponentiation*/

/*dest must be same size as z, z is the vector to evaluate derivative in, z is not changed*/
void IdentityDerivative(Matrix dest, Matrix z, uint32_t column);
void SigmoidDerivative(Matrix dest, Matrix z, uint32_t column);
void TanhDerivative(Matrix dest, Matrix z, uint32_t column);
void ReLUDerivative(Matrix dest, Matrix z, uint32_t column);

void SoftmaxDerivative(Matrix dest, Matrix z, uint32_t column);


/*====================== logging functions ======================*/

/*for plotting purposes, works for all M, but is only really interpretable for for M=1,2*/
void LogFullDistribution(Tensor stateSpaceProbabilities, double t, FILE* logFile);

void LogDataPoint(double t, Matrix dataPoint, FILE* logFile);