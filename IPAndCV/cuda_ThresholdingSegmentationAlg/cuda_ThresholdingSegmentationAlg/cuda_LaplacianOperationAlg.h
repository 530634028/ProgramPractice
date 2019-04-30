


#ifndef cuda_LaplacianOperationAlg_h
#define cuda_LaplacianOperationAlg_h

#include "zImageIO.h"
#include "helper_cuda.h"

void LaplacianFilter(Mat &imageInput, Mat &imageOutput);
//Laplacian operator for edge detection
__global__ void ImgFilter_3x3_Kernel(unsigned char *input, unsigned char *output, int nWidth,
	                                 int nHeight, int nWdithStep);









#endif
