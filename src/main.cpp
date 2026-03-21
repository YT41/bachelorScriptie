#include <time.h>

/*iisignature*/
//#include "iisignature/bch.hpp"
//#include "iisignature/logsig.hpp"
//#include "iisignature/calcSignature.hpp"

#include "Matrix.hpp"
#include "Random.hpp"
#include "ReactionParser.hpp"
#include "SRN.hpp"
#include "TrajectorySim.hpp"


int main(int argc, char** argv)
{
    SetSeedRandU48(time(NULL));

    SRN* srn = ParseSRN("res/GeneExpressionModel.txt");

    PrintMatrix((srn->stoichiometricMatrix));

    GillespieSRNTrajectorySim(3600.0, 20, srn, "res/trajectory.data");

    DeleteSRN(srn);

    return 0;
}