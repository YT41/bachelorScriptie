#include <time.h>

#include "Matrix.h"
#include "Random.h"
#include "ReactionParser.h"
#include "SRN.h"
#include "TrajectorySim.h"


int main(int argc, char** argv)
{
    SetSeedRandU48(time(NULL));

    SRN* srn = ParseSRN("res/GeneExpressionModel.txt");

    PrintMatrix((srn->stoichiometricMatrix));

    GillespieSRNTrajectorySim(3600.0, 20, srn, "res/trajectory.data");

    DeleteSRN(srn);

    return 0;
}