#pragma once

#include <cstddef>
#include <stdio.h>
#include <stdint.h>

#include "SRN.hpp"


void NaiveSRNTrajectorySim(double deltaT, uint64_t timeStepCount, uint32_t epochs, const SRN* srn, const char* saveFileName);

void GillespieSRNTrajectorySim(double time, uint32_t epochs, const SRN* srn, const char* saveFileName);


/*========== statistics for the purpose of comparing it to some other model ==========*/

void GillespieSRNTrajectorySimGetFullDistribution(const SRN* srn, Tensor probabilities, double t, size_t sampleCount);
void GillespieSRNTrajectorySimGetPerSpeciesMean(const SRN* srn, Matrix mean, double t, size_t sampleCount);
void GillespieSRNTrajectorySimGetPerSpeciesStandardDeviation(const SRN* srn, Matrix std, double t, size_t sampleCount);

void GillespieSRNTrajectorySimLogFullDistribution(const SRN* srn, double T, double tStep, size_t sampleCount, FILE* logFile);
void GillespieSRNTrajectorySimLogPerSpeciesMean(const SRN* srn, double T, double tStep, size_t sampleCount, FILE* logFile);
void GillespieSRNTrajectorySimLogPerSpeciesStandardDeviation(const SRN* srn, double T, double tStep, size_t sampleCount, FILE* logFile);