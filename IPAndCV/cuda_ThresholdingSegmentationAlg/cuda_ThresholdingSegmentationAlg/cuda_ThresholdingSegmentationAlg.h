
/****************************************************
  Header file of threshold segmentation algorithm
 CUDA kernel.
 a:    zhonghy
 date: 2019-4-10
*****************************************************/

#ifndef cuda_ThresholdingSegmentationAlg_H
#define cuda_ThresholdingSegmentationAlg_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include "zImageIO.h"

// for thresholding algorithm
cudaError_t cuda_ThresholdingSegmentationAlg(const Mat &inputImage, Mat &outputImage, int thresold);
int cpu_ThresholdingSegmentationAlg(const Mat &input, Mat &output, int thresold);

__global__ void cuda_ThresholdingSegmentationAlg_Kernel(const unsigned char *input, unsigned char *output,
	int imageW, int imageH, int thresold);

// for intelligence dilate algorithm
int cpu_IntelligenceDilate(const Mat &input, const Mat &mask, Mat &output, int lower, int upper, int radius);
cudaError_t cuda_IntelligenceDilate(const Mat &inputImage, const Mat &mask, Mat &outputImage, int lower, int upper, int radius);

template <int X>
__global__ void cuda_IntelligenceDilate_Kernel(const unsigned char *dev_inputData, unsigned char *dev_mask, unsigned char *dev_outputData, 
	int imageWidth, int imageHeight, int lower, int upper, int radius);

static void GenerateBallStructure2D(dim3 memDim, int radius);



#endif