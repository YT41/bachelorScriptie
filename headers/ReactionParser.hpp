#pragma once

#include "SRN.hpp"


/*parses a file to produce SRN according to syntax explained in README.md under "importing your own SRNs".

INPUTS:
const char* fileName: path to file to parse

RETURNS:
Heap-allocated pointer to SRN structure, be sure to delete it with DeleteSRN whenever you're not using it anymore to prevent memory leaks.*/
SRN* ParseSRN(const char* fileName);