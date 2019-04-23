
/****************************************************
  Implementation for threshold segmentation algorithm
 with CUDA.
 a:    zhonghy
 date: 2019-4-10
*****************************************************/

#include "cuda_ThresholdingSegmentationAlg.h"

//const int thresold = 200;
static __constant__ __device__ int conBallTable[520];
const int MaxValue = 255;
const int MinValue = 0;

// print necessary information into file, log_print or log_printf is already defined in CUDA???
void log_print(const char *filename, const char *str)   //__declspec(dllexport) 
{
	FILE *fp = fopen(filename,"a");//"log_gpu.txt"
	fprintf(fp,"%s",str);
	fclose(fp);
}

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
	return 1;
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
	cudaDeviceSynchronize();

	//for 2D image, if 3D image, value is 6 32 122 256 514
	switch(radius)
	{
		case 1:
			cuda_IntelligenceDilate_Kernel<8><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageDim, lower, upper, radius);  // it is not 6, is 8; 
			break;
		case 2:
			cuda_IntelligenceDilate_Kernel<24><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageDim, lower, upper, radius);
			break;
		case 3:
			cuda_IntelligenceDilate_Kernel<48><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageDim, lower, upper, radius);
			break;
		case 4:
			cuda_IntelligenceDilate_Kernel<80><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageDim, lower, upper, radius);
			break;
		case 5:
			cuda_IntelligenceDilate_Kernel<120><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageDim, lower, upper, radius);
			break;
		default:
			break;
	}

	//int rectangularLen  = 1 + 2 * radius;
	//rectangularLen *= rectangularLen;
	//rectangularLen -= 1;
	//const int len = rectangularLen;
	//cuda_IntelligenceDilate_Kernel<len><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageDim, lower, upper, radius);

	//cuda_IntelligenceDilate_Kernel<<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageWidth, imageHeight, lower, upper, radius);
	cudaDeviceSynchronize();

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

	FILE *fp = fopen("cudaBallStructure.txt","a");//"log_gpu.txt"
	for(int y = -radius; y <= radius; y++)
	{
		for(int x = -radius; x <= radius; x++)
		{
			int neighborPixelOffset = y * ybase + x;
			if(neighborPixelOffset == 0) continue;	// if the location of specified pixel
			/*if(sqrt(1.0 * ( x*x + y*y )) <= radius ) //if use this code, the structure is circle region.
			{*/
			ballStructure[count++] = neighborPixelOffset;
			/*}*/

			// write the table into txt file
			std::string str = std::to_string(long double(neighborPixelOffset)) + " "; 
			fprintf(fp,"%s",str.c_str());
		}
		std::string anthorLine = std::string("\n");
		fprintf(fp,"%s", anthorLine.c_str());
	}
	fclose(fp);

	cudaStatus = (cudaMemcpyToSymbol(conBallTable, ballStructure, count * sizeof(conBallTable[0])));
	//if 
}

template <int X>
__global__ void cuda_IntelligenceDilate_Kernel(const unsigned char *dev_inputData, unsigned char *dev_mask, unsigned char *dev_outputData, 
	                                           dim3 imageDim, int lower, int upper, int radius)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	int dataIndex = yIndex * imageDim.x + xIndex;

	// eight neighbor region
	//int neighbor[8];
	int nx = imageDim.x; //imageWidth;
	int imageSize = imageDim.x * imageDim.y;//imageWidth * imageHeight;

	// only for eight region, full
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
			int dataIndexNeighbor = dataIndex + conBallTable[i];  // + neighbor[i]; add the offset to specified pixel
			if(dataIndexNeighbor >= 0 && dataIndexNeighbor < imageSize)
			{
				if(dev_inputData[dataIndexNeighbor] >= lower && dev_inputData[dataIndexNeighbor] <= upper)
				{
					dev_outputData[dataIndexNeighbor] = dev_mask[dataIndex];
				}
			}
		} // for
	}
}

int cpu_IntelligenceDilate(const Mat &input, const Mat &mask, Mat &output, int lower, int upper, int radius)
{
	if(input.empty() || mask.empty())
	{
		return -1;
	}
	int rows = input.rows;
	int cols = input.cols; //* input.channels();

	// write the offset into txt file
	FILE *fp = fopen("cpuBallStructure.txt","a");//"log_gpu.txt"
	int flag = 0;

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
						if(i >= 0 && i < rows && j >= 0 && j < cols )
						{

							int indexTmp = j * input.cols + i;
							int indexOrigin = y * input.cols + x;
							int offset = indexTmp - indexOrigin;
							// write the table into txt file
			                std::string str = std::to_string(long double(offset)) + " "; 
			                fprintf(fp,"%s",str.c_str());
							
							if(i == x && j == y) continue;

							//std::cout << i << "-" << j << " ";
							unsigned char pixelValueRoot = input.at<unsigned char>(i, j);  // the type must be matched
							if (static_cast<int>(pixelValueRoot) <= upper && static_cast<int>(pixelValueRoot) >= lower) //&& !outputDataPtr[y])                         
							{
								output.at<unsigned char>(i, j) = maskPtr[y]; // MaxValue;
							}
						}
					}
					std::string anthorLine = std::string("\n");
		            fprintf(fp,"%s", anthorLine.c_str());
				}
			}	
		} // for y
      }// for x
	return 1;
}