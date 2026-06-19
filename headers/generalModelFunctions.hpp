#pragma once

#include "NeuralNetwork.hpp"
#include "SRN.hpp"


/*sets input token for species i in column i of X ( X_i := (i, \psi(t), \eta_i, n_{<i}) ), as defined in section 3.1.1 of the thesis report

INPUTS:
const SRN* srn: pointer to stochastic reaction network structure to get token for
Matrix X: data matrix to set column i of
uint32_t i: token index
IntMatrix n: the state
Matrix embeddedTime: \psi(t)*/
void GetSingleInputToken(SRN* srn, Matrix X, uint32_t i, IntMatrix n, Matrix embeddedTime);
void GetInputTokens(SRN* srn, Matrix X, IntMatrix n, Matrix embeddedTime); /*just repeats GetSingleInputToken for all indices i*/


/*evaluates \psi(t), as defined in section 3.1.1 of the thesis report

INPUTS:
NeuralNetwork* timeEmbedding: time embedding MLP
double t: time to evaluate in
double dropoutProbability: dropout probability for neurons within the time embedding MLP (not that important, just set it to 0 if you dont know what this is)

RETURNS:
output of \psi(t), in a vector*/
Matrix GetEmbeddedTime(NeuralNetwork* timeEmbedding, double t, double dropoutProbability);