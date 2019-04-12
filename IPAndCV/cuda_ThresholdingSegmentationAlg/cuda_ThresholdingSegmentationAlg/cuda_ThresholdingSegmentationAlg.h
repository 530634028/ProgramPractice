
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

cudaError_t cuda_ThresholdingSegmentationAlg(const Mat &inputImage, Mat &outputImage, int thresold);
int cpu_ThresholdingSegmentationAlg(const Mat &input, Mat &output, int thresold);

__global__ void cuda_ThresholdingSegmentationAlg_Kernel(const unsigned char *input, unsigned char *output,
	int imageW, int imageH, int thresold);


#endif