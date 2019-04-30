

#include "cuda_ErodeAlg.h"

// erode filter
__global__ void Morphology_erode_kernel(
	unsigned char* input,
	unsigned char* ouput,
	int nWidth,
	int nHeight,
	int nWdithStep) {

		const int ix = blockIdx.x*blockDim.x + threadIdx.x;
		const int iy = blockIdx.y*blockDim.y + threadIdx.y;

		const int ix_1 = max(0, ix - 1);
		const int ix1 = min(nWidth - 1, ix + 1);
		const int iy_1 = max(0, iy - 1);
		const int iy1 = min(nHeight - 1, iy + 1);

		if (ix < nWidth&&iy < nHeight) {
			if (input[iy*nWdithStep + ix]) {
				if (input[iy_1*nWdithStep + ix_1] == 0 ||
					input[iy_1*nWdithStep + ix] == 0 ||
					input[iy_1*nWdithStep + ix1] == 0 ||
					input[iy*nWdithStep + ix_1] == 0 ||
					input[iy*nWdithStep + ix1] == 0 ||
					input[iy1*nWdithStep + ix_1] == 0 ||
					input[iy1*nWdithStep + ix] == 0 ||
					input[iy1*nWdithStep + ix1] == 0
					) {
						ouput[iy*nWdithStep + ix] = 0;
				}
			}
		}
}

