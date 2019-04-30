


//#include "cuda_MeanFilterAlg.h"


//
//Median
//// ¾ùÖµÂË²¨
//__global__ void ImgFilter_3x3_m_Kernel(
//	unsigned char *input,
//	unsigned char *output,
//	int nWidth,
//	int nHeight,
//	int nWdithStep
//	) {
//		const int ix = blockIdx.x*blockDim.x + threadIdx.x;
//		const int iy = blockIdx.y*blockDim.y + threadIdx.y;
//
//		const int ix_1 = max(0, ix - 1);
//		const int ix1 = min(nWidth - 1, ix + 1);
//		const int iy_1 = max(0, iy - 1);
//		const int iy1 = min(nHeight - 1, iy + 1);
//
//		if (ix < nWidth&&iy < nHeight) {
//			int nTemp;
//			nTemp = input[iy_1*nWdithStep + ix_1];
//			nTemp += input[iy_1*nWdithStep + ix];
//			nTemp += input[iy_1*nWdithStep + ix1];
//			nTemp += input[iy*nWdithStep + ix_1];
//			nTemp += input[iy*nWdithStep + ix];
//			nTemp += input[iy*nWdithStep + ix1];
//			nTemp += input[iy1*nWdithStep + ix_1];
//			nTemp += input[iy1*nWdithStep + ix];
//			nTemp += input[iy1*nWdithStep + ix1];
//			nTemp /= 9;
//			output[iy_1*nWdithStep + ix] = nTemp;
//		}
//}
//