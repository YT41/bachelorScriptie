#pragma once

#include <cstdio>
#include <stdint.h>

#include "Matrix.hpp"

#define MAX(a, b)       (((a) > (b)) ? (a) : (b))
#define MIN(a, b)       (((a) < (b)) ? (a) : (b))


/*TODO: implement if needed*/
/*return log(\sum_{i=1}^d e^{terms_i})*/
//static double LogSumExp(double* terms, uint32_t d);

/*for plotting purposes, works for all M, but is only really interpretable for for M=1,2*/
void LogFullDistribution(Tensor stateSpaceProbabilities, double t, FILE* logFile);

void LogDataPoint(double t, Matrix dataPoint, FILE* logFile);