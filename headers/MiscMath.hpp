#pragma once

#include <stdint.h>

#define MAX(a, b)       (((a) > (b)) ? (a) : (b))
#define MIN(a, b)       (((a) < (b)) ? (a) : (b))


/*TODO: implement if needed*/
/*return log(\sum_{i=1}^d e^{terms_i})*/
//static double LogSumExp(double* terms, uint32_t d);