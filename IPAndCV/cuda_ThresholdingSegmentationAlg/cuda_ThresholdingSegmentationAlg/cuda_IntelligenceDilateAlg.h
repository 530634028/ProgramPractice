
/****************************************************
  Header file of Intelligence Dilate algorithm
 CUDA kernel for 2D image
 a:    zhonghy
 date: 2019-4-23
*****************************************************/

#ifndef cuda_IntelligenceDilateAlg_H
#define cuda_IntelligenceDilateAlg_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include "zImageIO.h"
#include "helper_cuda.h"

#include <string>
#include <stdio.h>

// for intelligence dilate algorithm
int cpu_IntelligenceDilate(const Mat &input, const Mat &mask, Mat &output, int lower, int upper, int radius);
cudaError_t cuda_IntelligenceDilate(const Mat &inputImage, const Mat &mask, Mat &outputImage, int lower, int upper, int radius);

//template <int X>
__global__ void cuda_IntelligenceDilateKernel(const unsigned char *dev_inputData, unsigned char *dev_mask, unsigned char *dev_outputData, 
	dim3 imageDim, int lower, int upper, int radius, int numOfStruct);

static void GenerateBallStructure2D(dim3 memDim, int radius, int &numOfStruct);



#endif
