



#include "cuda_LaplacianOperationAlg.h"

// detect edge by Laplacian operator, implemented by opencv 
void LaplacianFilter(Mat &imageInput, Mat &imageOutput)
{
	int kernel_size = 3;  //3
	int scale = 1;
	double delta = 0.5;
	int ddepth = CV_16S;

	Mat src_gray;
	/// Reduce noise by blurring with a Gaussian filter
	GaussianBlur(imageInput, imageInput, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//convert to gray
	cvtColor(imageInput, src_gray, COLOR_BGR2GRAY); 
	/// Apply Laplace function
	Mat abs_dst, dst;
	//laplacian
	Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	//convert
	convertScaleAbs(dst, imageOutput);
}












//// ËÄÁÚÓò2½×±ßÔµËã×Ó
//*
//0  1  0 
//1 -4  1
//0  1  0
//*/
//LaplacianËã×Ó±ßÔµ¼ì²â
__global__ void ImgFilter_3x3_Kernel(
	unsigned char *input,
	unsigned char *output,
	int nWidth,
	int nHeight,
	int nWdithStep
	) {
		const int ix = blockIdx.x*blockDim.x + threadIdx.x;
		const int iy = blockIdx.y*blockDim.y + threadIdx.y;

		const int ix_1 = max(0, ix - 1);
		const int ix1 = min(nWidth - 1, ix + 1);
		const int iy_1 = max(0, iy - 1);
		const int iy1 = min(nHeight - 1, iy + 1);

		if (ix < nWidth&&iy < nHeight) {
			int nTemp;
			nTemp = input[iy_1*nWdithStep + ix];
			nTemp += input[iy*nWdithStep + ix_1];
			nTemp -= input[iy*nWdithStep + ix]<<2;
			nTemp += input[iy*nWdithStep + ix1];
			nTemp += input[iy1*nWdithStep + ix1];
			nTemp =abs(nTemp);
			nTemp =min(255, nTemp);
			output[iy*nWdithStep + ix] = nTemp;
		}
}


