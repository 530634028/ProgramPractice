
/****************************************************
  Implementation for threshold segmentation algorithm
 with CUDA.
 a:    zhonghy
 date: 2019-4-10
*****************************************************/

#include "cuda_ThresholdingSegmentationAlg.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

//const int thresold = 200;
const int MaxValue = 255;
const int MinValue = 0;

__global__ void cuda_ThresholdingSegmentationAlg_Kernel(const unsigned char *input, unsigned char *output,
	                                                    int imageW, int imageH, int thresold)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	int dataIndex = yIndex * imageW + xIndex;
	if(input[dataIndex] < thresold)
	{
		output[dataIndex] = MinValue;
	}
	else
	{
		output[dataIndex] = MaxValue;
	}
}

int cpu_ThresholdingSegmentationAlg(const Mat &input, Mat &output, int thresold)
{
	if(input.empty())
	{
		return -1;
	}
	int rows = input.rows;
	int cols = input.cols * input.channels();
	for(int i = 0; i < rows; ++i)
	{
		const unsigned char *inputDataPtr = input.ptr<unsigned char>(i);  // image access method of opencv
		unsigned char *outputDataPtr = output.ptr<unsigned char>(i);
		for(int j = 0; j < cols; ++j)
		{
			if(inputDataPtr[j] < thresold)
			{
				outputDataPtr[j] = MinValue;
			}
			else
			{
				outputDataPtr[j] = MaxValue;
			}
		}
	}

}

// Implementation for threshold segmentation algorithm using CUDA 
cudaError_t cuda_ThresholdingSegmentationAlg(const Mat &inputImage, Mat &outputImage, int thresold)
{
	unsigned char *dev_inputData = 0;
	unsigned char *dev_outputData = 0;
	int imageWidth = inputImage.cols;
	int imageHeight = inputImage.rows;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
	{
       fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
       return cudaStatus;
    }

	int size = imageWidth * imageHeight;
	cudaStatus = cudaMalloc((void**)&dev_inputData, size * sizeof(unsigned char));
	cudaStatus = cudaMalloc((void**)&dev_outputData, size * sizeof(unsigned char));

	cudaStatus = cudaMemcpy(dev_inputData, inputImage.data, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cuda_ThresholdingSegmentationAlg_Kernel<<<size / 512, 512 >>>(dev_inputData, dev_outputData, imageWidth, imageHeight, thresold);
	cudaStatus = cudaMemcpy(outputImage.data, dev_outputData, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

    cudaFree(dev_inputData);
    cudaFree(dev_outputData);

    return cudaStatus;
}
