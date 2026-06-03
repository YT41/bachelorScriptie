#pragma once

#include "NeuralNetwork.hpp"
#include "SRN.hpp"


/*prepare token i: X_i^{(0)} := (i, \psi(t), \eta_i, n_{<i})*/
void GetSingleInputToken(SRN* srn, Matrix X, uint32_t i, IntMatrix n, Matrix embeddedTime);
void GetInputTokens(SRN* srn, Matrix X, IntMatrix n, Matrix embeddedTime);

Matrix GetEmbeddedTime(NeuralNetwork* timeEmbedding, double t);