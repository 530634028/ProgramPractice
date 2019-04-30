



#ifndef cuda_ErodeAlg_h
#define cuda_ErodeAlg_h

#include "zImageIO.h"

__global__ void Morphology_erode_kernel(
	unsigned char* input,
	unsigned char* ouput,
	int nWidth,
	int nHeight,
	int nWdithStep);






#endif


