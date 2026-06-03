#include "BTCM.hpp"
#include "Attention.hpp"
#include "Matrix.hpp"
#include "MemArena.hpp"
#include "Random.hpp"
#include <cstdint>
#include <cstdio>
#include <ctime>

int main(int argc, char** argv)
{
    //BTCMSingleTimeStepExperiment();
    //BTCMGlobalTimeExperiment();
    //BTCMGeneExpressionExperiment();
    BTCMSignallingCascadeExperiment(2);

    // SetSeedRandU48(time(NULL));

    // uint32_t d = 3;
    // uint32_t dAtt = 2;
    // uint32_t tokenCount = 2;
    // AttentionMechanism* a = SAMCreate(tokenCount, d, dAtt, 0.1);

    // MemArena arena = CreateMemArena(10000);

    // double XVals[] = {  
    //     1.0, 0.0, 0.0,   
    //     1.0, 1.0, 1.0 
    // };
    // Matrix X = CreateMatrix(&arena, d, tokenCount, XVals);

    // PrintMatrix(GetMaskedAttentionMatrix(a, X));

    // SAMDelete(a);
    // DeleteMemArena(&arena);

    return 0;
}