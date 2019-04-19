
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
static __constant__ __device__ int conBallTable[520];
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



/////////////////////////////////////////////////////////////////////////////////////////
// implement 2D intelligence dilate
cudaError_t cuda_IntelligenceDilate(const Mat &inputImage, const Mat &mask, Mat &outputImage, int lower, int upper, int radius)
{
	unsigned char *dev_inputData = 0;
	unsigned char *dev_outputData = 0;
	unsigned char *dev_mask = 0;

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
	cudaStatus = cudaMalloc((void**)&dev_mask, size * sizeof(unsigned char));

	cudaStatus = cudaMemcpy(dev_inputData, inputImage.data, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_mask, mask.data, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

	// construct the ball structure
	dim3 imageDim;
	imageDim.x = imageWidth;
	imageDim.y = imageHeight;
	imageDim.z = 0;
	GenerateBallStructure2D(imageDim, radius);	
	switch(radius)
	{
		case 1:
			cuda_IntelligenceDilate_Kernel<6><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageWidth, imageHeight, lower, upper, radius);
			break;
		case 2:
			cuda_IntelligenceDilate_Kernel<32><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageWidth, imageHeight, lower, upper, radius);
			break;
		case 3:
			cuda_IntelligenceDilate_Kernel<122><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageWidth, imageHeight, lower, upper, radius);
			break;
		case 4:
			cuda_IntelligenceDilate_Kernel<256><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageWidth, imageHeight, lower, upper, radius);
			break;
		case 5:
			cuda_IntelligenceDilate_Kernel<514><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageWidth, imageHeight, lower, upper, radius);
			break;
		default:
			break;
	}
	//cuda_IntelligenceDilate_Kernel<<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageWidth, imageHeight, lower, upper, radius);

	cudaStatus = cudaMemcpy(outputImage.data, dev_outputData, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

    cudaFree(dev_inputData);
    cudaFree(dev_outputData);
	cudaFree(dev_mask);

    return cudaStatus;
}

// location of specified pixel's neighbors in the image
static void GenerateBallStructure2D(dim3 memDim, int radius)
{
	cudaError_t cudaStatus;
	int ballStructure[520];
	memset(ballStructure, 0, sizeof(ballStructure));
	int zbase = memDim.x * memDim.y;
	int ybase = memDim.x;
	int count = 0;

	for(int y = -radius; y <= radius; y++)
	{
		for(int x = -radius; x <= radius; x++)
		{
			int wz = y * ybase + x;
			if(wz == 0) continue;	// if the location of specified pixel
			if(sqrt(1.0 * ( x*x + y*y )) <= radius )
			{
				ballStructure[count++] = wz;
			}
		}
	}
	cudaStatus = (cudaMemcpyToSymbol(conBallTable, ballStructure, count * sizeof(conBallTable[0])));
	//if 
}

template <int X>
__global__ void cuda_IntelligenceDilate_Kernel(const unsigned char *dev_inputData, unsigned char *dev_mask, unsigned char *dev_outputData, 
	                                           int imageWidth, int imageHeight, int lower, int upper, int radius)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	int dataIndex = yIndex * imageWidth + xIndex;

	// eight neighbor region
	//int neighbor[8];
	int nx = imageWidth;
	int imageSize = imageWidth * imageHeight;

	// only for eight region
	//neighbor[0] = -1;
	//neighbor[1] = 1;

	//neighbor[2] = -nx;
	//neighbor[3] = nx;

	//neighbor[4] = nx - 1;
	//neighbor[5] = nx + 1;

	//neighbor[6] = -nx - 1;
	//neighbor[7] = -nx + 1;

	if(dev_mask[dataIndex])
	{
		dev_outputData[dataIndex] = dev_mask[dataIndex];
		// deal with pixels in specified pixel's neighbor region
		for(int i = 0; i < X; ++i)
		{
			int dataIndexTmp = dataIndex + conBallTable[i];  // + neighbor[i]; add the offset to specified pixel
			if(dataIndexTmp > 0 && dataIndexTmp < imageSize)
			{
				if(dev_inputData[dataIndexTmp] >= lower && dev_inputData[dataIndexTmp] <= upper)
				{
					dev_outputData[dataIndexTmp] = dev_mask[dataIndex];
				}
			}
		}
	}
}

int cpu_IntelligenceDilate(const Mat &input, const Mat &mask, Mat &output, int lower, int upper, int radius)
{
	if(input.empty() || mask.empty())
	{
		return -1;
	}
	int rows = input.rows;
	int cols = input.cols * input.channels();

	for(int x = 0; x < rows; ++x)
	{
		//const unsigned char *inputDataPtr = input.ptr<unsigned char>(x);  // image access method of opencv
		const unsigned char *maskPtr = mask.ptr<unsigned char>(x);
		unsigned char *outputDataPtr = output.ptr<unsigned char>(x);

		for(int y = 0; y < cols; ++y)
		{
			if(maskPtr[y])
			{
				outputDataPtr[y] = maskPtr[y];
				for (int i = x - radius; i <= x + radius; i++)
				{
					for (int j = y - radius; j <= y + radius; j++)
					{
						if(i >= 0 && i <= rows && j >= 0 && j <= cols )
						{
							int pixelValueRoot = input.at<unsigned char>(i, j);  // the type must be matched
							if (static_cast<int>(pixelValueRoot) <= upper && static_cast<int>(pixelValueRoot) >= lower) //&& !outputDataPtr[y])
							{
								output.at<unsigned char>(i, j) = MaxValue;
							}
						}
					}
				}
			}	
		} // for y
      }// for x
}