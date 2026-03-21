#pragma once

#include <stdint.h>

#include "SRN.h"


void NaiveSRNTrajectorySim(double deltaT, uint64_t timeStepCount, uint32_t epochs, const SRN* srn, const char* saveFileName);

void GillespieSRNTrajectorySim(double time, uint32_t epochs, const SRN* srn, const char* saveFileName);