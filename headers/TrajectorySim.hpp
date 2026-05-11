#pragma once

#include <stdint.h>

#include "SRN.hpp"


void NaiveSRNTrajectorySim(double deltaT, uint64_t timeStepCount, uint32_t epochs, const SRN* srn, const char* saveFileName);

void GillespieSRNTrajectorySim(double time, uint32_t epochs, const SRN* srn, const char* saveFileName);
void GetFullMarginalDistributionGillespieSRNTrajectorySim(const SRN* srn, Tensor probabilities, double t, uint32_t epochs);
void GillespieSRNTrajectorySimGetPerSpeciesMean(const SRN* srn, Matrix mean, double t);
void GillespieSRNTrajectorySimGetPerSpeciesStandardDeviation(const SRN* srn, Matrix std, double t);