#pragma once

#include <cstdint>
#include <cstdio>
#include <stdint.h>

#include "Attention.hpp"
#include "Matrix.hpp"
#include "MemArena.hpp"
#include "NeuralNetwork.hpp"
#include "SRN.hpp"


/*time-conditioned model, MLP using KL-divergence, with time embedding and attention mechanism, see section 3 of the thesis report.
basic pipeline: time embedding --> x_i --> masked attention mechanism (optional) --> h_i (optional) --> MLP --> conditional probabilites*/
typedef struct TCM
{
    MemArena arena;

    Matrix batchCache;
    NeuralNetwork* MLP;
    NeuralNetwork* timeEmbedding;
    AttentionMechanism* a;
    
    SRN* srn;
} TCM;


/*Creates an instance of the time-conditioned model. 

INPUTS:
SRN* srn: A stochastic reaction network structure
uint32_t* neuronsPerHiddenLayer: a C-style array containing the neuron counts for the MLP, the first index is the amount of neurons in the first hidden layer, the second index is for the neuron count of the second layer, and so on
uint32_t hiddenLayerCount: element count of neuronsPerHiddenLayer
uint32_t timeEmbeddingDim: d_t, the output dimension of the time embedding
uint32_t attentionDim: d_k, the output dimension of the attention mechanism, set to 0 to disable

RETURNS:
Heap-allocated pointer to TCM (time-conditioned model) structure, be sure to delete it with TCMDelete whenever you're not using it anymore to prevent memory leaks.*/
TCM* TCMCreate(SRN* srn, uint32_t* neuronsPerHiddenLayer, uint32_t hiddenLayerCount, uint32_t timeEmbeddingDim, uint32_t attentionDim);
void TCMDelete(TCM* m); /*See comments for TCMCreate.*/


/*returns dim(\theta) of the model  m*/
size_t TCMGetParamCount(const TCM* m);


/*Creates a new instance of the time-conditioned model from the input m, and makes a completely seperate copy of m. 

INPUTS:
const TCM* m: time-condtioned model to copy

RETURNS:
Heap-allocated pointer to TCM (time-conditioned model) structure, copied from m, be sure to delete it with TCMDelete whenever you're not using it anymore to prevent memory leaks.*/
TCM* TCMCopy(const TCM* m);


/*copies all parameters (\theta) of the model from src into dest. 

INPUTS:
TCM* dest: pointer to time-conditioned model structure for which the parameters (\theta) are to be overwritten.
const TCM* src: pointer to time-conditioned model structure for which the parameters (\theta) are to be copied to dest.*/
void TCMCopyParameters(TCM* dest, const TCM* src);


/*Takes a sample from the model m, returns the probability of getting that sample, the sample is returned in s, the gradient of KL-loss with respects to last layer (pre-softmax) is set in desiredNudgesMLPOutput

INPUTS:
TCM* m: model to take sample from
IntMatrix s: integer vector, taken sample is set in the first column of s in function call
double t: time to take sample at
Matrix desiredNudgesMLPOutput: the gradient of KL-loss with respects to last layer (pre-softmax) is set in desiredNudgesMLPOutput in function call
double dropoutProbability: dropout probability for neurons within the MLP hidden layers (not that important, just set it to 0 if you dont know what this is)

RETURNS:
The probability of having taken the sample returned in s.*/
double TCMTakeSample(TCM* m, IntMatrix s, double t, Matrix desiredNudgesMLPOutput, double dropoutProbability);
double TCMTakeSampleNoGradient(TCM* m, IntMatrix s, double t, double dropoutProbability);


/*directly gives P(n, t) (joint probability distribution at time t), so assumes n is known

INPUTS:
TCM* m: pointer to model \hat{P}_\theta that predicts P(n, t)
IntMatrix n: integer vector to evaluate in
double t: time to evaluate in
double dropoutProbability: dropout probability for neurons within the MLP hidden layers (not that important, just set it to 0 if you dont know what this is)

RETURNS:
P(n, t)*/
double TCMPredict(TCM* m, IntMatrix n, double t, double dropoutProbability);


/*full grid evaluation of the joint probability distribution at time t (which is P(t))

INPUTS:
TCM* m: pointer to model \hat{P}_\theta that predicts P(t)
Tensor probabilities: Tensor in which P(t) is set in, make sure that it has the same dimensions as the state space
double t: time to evaluate in*/
void TCMGetFullProbabilityDistribution(TCM* m, Tensor probabilities, double t);


/*Trains the model m

INPUTS:
TCM* m: pointer to model \hat{P}_\theta to train
double T: last train time, so the model tries to fit the joint probability distributions for t within [0, T]
double deltaT: time step, see section 3.1.5 in the thesis report for a detailed explanation of what value you should pick for it
double p: probability that t is set to 0 every epoch (to train initial condition), see section 3.1.2 of the thesis report
uint32_t B: number of sampled states per epoch (used to approximate loss), make sure that it is big enough to sample enough rare states, try 1000
uint32_t Q: the frozen target copy is updated every Q epochs, see section 3.1.2 of the thesis report
uint64_t epochs: total number of epochs to train
double learningRate: learning rate, used in gradient descent, if your loss graph is chaotic, lower it
double dropoutProbability: dropout probability for neurons within the MLP hidden layers (not that important, just set it to 0 if you dont know what this is)
const char* logFile: C-string of path to file where loss data will be stored in, essential if you want to plot the loss*/
void TCMTrain(TCM* m, double T, double deltaT, double p, uint32_t B, uint32_t Q, uint64_t epochs, double learningRate, double dropoutProbability, const char* logFile);


/*==================== statistics ====================*/

/*Approximation of the species count means for every species

INPUTS:
TCM* m: pointer to model \hat{P}_\theta to approximate means for
Matrix mean: the means of every species are returned in the first column, obviously make sure it has at least M rows
double t: time to approximate means for
size_t sampleCount: amount of samples takes from m, more samples -> better approximation, 10000 should be more than enough*/
void TCMGetPerSpeciesMean(TCM* m, Matrix mean, double t, size_t sampleCount);


/*Approximation of the species count standard deviations for every species

INPUTS:
TCM* m: pointer to model \hat{P}_\theta to approximate standard deviations for
Matrix std: the standard deviations of every species are returned in the first column, obviously make sure it has at least M rows
double t: time to approximate standard deviations for
size_t sampleCount: amount of samples takes from m, more samples -> better approximation, 10000 should be more than enough*/
void TCMGetPerSpeciesStandardDeviation(TCM* m, Matrix std, double t, size_t sampleCount);


/*like TCMGetFullProbabilityDistribution, but prints it to the console instead, only use for debugging simple SRNs*/
void TCMPrintFullProbabilityDistribution(TCM* m, double t);


/*All of the following 4 functions are for plotting purposes, specifically comparing data of the time-conditioned model to that generated with the Gillespie algorithm
TCMLogMeanComparisonOverTime: for comparing means
TCMLogStdComparisonOverTime: for comparing standard deviations
TCMLogFullDistributionComparisonOverTime: for comparing joint probability distributions
TCMLogHellingerDistanceToGillespieOverTime: gets the Hellinger distance at each time sample

INPUTS:
TCM* m: pointer to model \hat{P}_\theta to compare to Gillespie
double T: end time, maximum time to compare for
double sampleStep: time step size between comparisons, always starts at t = 0, next for t = sampleStep, then t = 2 * sampleStep, etc
size_t sampleCount: amount of samples takes from m, more samples -> better approximation, 10000 should be more than enough
const char* fileName: path to file to dump data in*/
void TCMLogMeanComparisonOverTime(TCM* m, double T, double sampleStep, size_t sampleCount, const char* fileName);
void TCMLogStdComparisonOverTime(TCM* m, double T, double sampleStep, size_t sampleCount, const char* fileName);
void TCMLogFullDistributionComparisonOverTime(TCM* m, double T, double sampleStep, size_t sampleCount, const char* fileName);
void TCMLogHellingerDistanceToGillespieOverTime(TCM* m, double T, double sampleStep, size_t sampleCount, const char* fileName);


/*==================== experiments ====================*/

/*Simplest experiment, only tries to predict 1 time step, only handy for debugging, does not appear in the thesis report*/
void TCMSingleTimeStepExperiment(void);


/*Experiment to prove that the globally time-condtioned model works at all, using the birth-death process, the results in section 4.1 of the thesis report were generated with this experiment*/
void TCMGlobalTimeExperiment(void);


/*Experiment of the gene expression SRN, the results in section 4.2 of the thesis report were generated with this experiment*/
void TCMGeneExpressionExperiment(void);


/*Experiment of the linear signaling cascade SRN, the results in section 4.3 and 4.5 of the thesis report were generated with this experiment*/
void TCMSignalingCascadeExperiment(uint32_t M);


/*Experiment for the time embeddings, the results in section 4.4 of the thesis report were generated with this experiment*/
void TCMTimeEmbeddingExperiment(void);