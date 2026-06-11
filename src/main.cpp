#include "TCM.hpp"
#include "Attention.hpp"
#include "Matrix.hpp"
#include "MemArena.hpp"
#include "Random.hpp"
#include "SRN.hpp"
#include "TrajectorySim.hpp"
#include "ReactionParser.hpp"
#include <cstdint>
#include <cstdio>
#include <ctime>

int main(int argc, char** argv)
{
    //TCMSingleTimeStepExperiment();
    //TCMGlobalTimeExperiment();
    TCMGeneExpressionExperiment();
    //TCMSignallingCascadeExperiment(3);

    // SetSeedRandU48(time(NULL));

    // SRN* srn = ParseSRN("res/birthDeathModel.txt");

    // NaiveSRNTrajectorySim(0.1, 1000, 3, srn, "res/trajectory.data");

    // DeleteSRN(srn);

    // SetSeedRandU48(time(NULL));

    // uint32_t d = 3;
    // uint32_t dAtt = 2;
    // uint32_t tokenCount = 3;
    // AttentionMechanism* a = AMCreate(tokenCount, d, dAtt);

    // MemArena arena = CreateMemArena(10000);

    // double XVals[] = {  
    //     1.0, 4.0, 7.0,   
    //     2.0, 5.0, 8.0,
    //     3.0, 6.0, 9.0
    // };
    // Matrix X = CreateMatrix(&arena, d, tokenCount, XVals);
    // PrintMatrix(AMGetMaskedAttentionWeightMatrix(a, X)); printf("\n");
    // PrintMatrix(AMGetMaskedAttention(a, X)); printf("\n");

    

    // AMDelete(a);
    // DeleteMemArena(&arena);

    return 0;
}