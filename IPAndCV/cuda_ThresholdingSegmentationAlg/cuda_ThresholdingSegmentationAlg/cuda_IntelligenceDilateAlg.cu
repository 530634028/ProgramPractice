
/****************************************************
  Implementation file of Intelligence Dilate algorithm
 CUDA kernel for 2D image
 a:    zhonghy
 date: 2019-4-23
*****************************************************/

#include "cuda_IntelligenceDilateAlg.h"

// generate location of specified pixel's neighbors in 2D image, resprsented by row
// the region is rectangle in which size is (2 * radius + 1) * (2 * radius + 1)
static void GenerateRectStructure2D(dim3 memDim, int radius, int &numOfStruct)
{
	cudaError_t cudaStatus;
	int ballStructure[520];
	memset(ballStructure, 0, sizeof(ballStructure));
	int zbase = memDim.x * memDim.y;
	int ybase = memDim.x;
	//int count = 0;

	for(int y = -radius; y <= radius; y++)
	{
		for(int x = -radius; x <= radius; x++)
		{
			int neighborPixelOffset = y * ybase + x;
			if(neighborPixelOffset == 0) continue;	// if the location of specified pixel
			ballStructure[numOfStruct++] = neighborPixelOffset;

		}
	}

	cudaStatus = (cudaMemcpyToSymbol(constStructTable, ballStructure, numOfStruct * sizeof(constStructTable[0])));
}

// the region is ball whose radius is radius
static void GenerateBallStructure2D(dim3 memDim, int radius, int &numOfStruct)
{
	cudaError_t cudaStatus;
	int ballStructure[520];
	memset(ballStructure, 0, sizeof(ballStructure));
	int zbase = memDim.x * memDim.y;
	int ybase = memDim.x;	

	for(int y = -radius; y <= radius; y++)
	{
		for(int x = -radius; x <= radius; x++)
		{
			int neighborPixelOffset = y * ybase + x;
			if(neighborPixelOffset == 0) continue;	// if the location of specified pixel
			if(sqrt(1.0 * ( x*x + y*y )) <= radius ) //if use this code, the structure is circle region.
			{
				ballStructure[numOfStruct++] = neighborPixelOffset;
			}
		}
	}

	cudaStatus = (cudaMemcpyToSymbol(constStructTable, ballStructure, numOfStruct * sizeof(constStructTable[0])));
}


// template <int X>
__global__ void cuda_IntelligenceDilateKernel(const unsigned char *dev_inputData, unsigned char *dev_mask, unsigned char *dev_outputData, 
	                                           dim3 imageDim, int lower, int upper, int radius, int numOfStruct)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
	int dataIndex = yIndex * imageDim.x + xIndex;

	// eight neighbor region
	int nx = imageDim.x;   // imageWidth;
	int imageSize = imageDim.x * imageDim.y;  // imageWidth * imageHeight;

	if(dev_mask[dataIndex])
	{
		dev_outputData[dataIndex] = dev_mask[dataIndex];
		// deal with pixels in specified pixel's neighbor region
		for(int i = 0; i < numOfStruct; ++i)   // X
		{
			int dataIndexNeighbor = dataIndex + constStructTable[i];  // + neighbor[i]; add the offset to specified pixel
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
	int numOfStruct = 0;

	GenerateRectStructure2D(imageDim, radius, numOfStruct);
	//cudaDeviceSynchronize();

	////for 2D image, if 3D image, value is 6 32 122 256 514
	//switch(radius)
	//{
	//	case 1:
	//		cuda_IntelligenceDilate_Kernel<8><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageDim, lower, upper, radius);  // it is not 6, is 8; 
	//		break;
	//	case 2:
	//		cuda_IntelligenceDilate_Kernel<24><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageDim, lower, upper, radius);
	//		break;
	//	case 3:
	//		cuda_IntelligenceDilate_Kernel<48><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageDim, lower, upper, radius);
	//		break;
	//	case 4:
	//		cuda_IntelligenceDilate_Kernel<80><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageDim, lower, upper, radius);
	//		break;
	//	case 5:
	//		cuda_IntelligenceDilate_Kernel<120><<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageDim, lower, upper, radius);
	//		break;
	//	default:
	//		break;
	//}

	cuda_IntelligenceDilateKernel<<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageDim, lower, upper, radius, numOfStruct);
	//cuda_IntelligenceDilate_Kernel<<<size / 512, 512 >>>(dev_inputData, dev_mask, dev_outputData, imageWidth, imageHeight, lower, upper, radius);
	cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(outputImage.data, dev_outputData, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();

    cudaFree(dev_inputData);
    cudaFree(dev_outputData);
	cudaFree(dev_mask);

    return cudaStatus;
}

int cpu_IntelligenceDilate(const Mat &input, const Mat &mask, Mat &output, int lower, int upper, int radius)
{
	if(input.empty() || mask.empty())
	{
		return -1;
	}
	int rows = input.rows;
	int cols = input.cols; //* input.channels();

	//// write the offset into txt file
	//FILE *fp = fopen("cpuBallStructure.txt","a");//"log_gpu.txt"
	//int flag = 0;

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

							//int indexTmp = j * input.cols + i;
							//int indexOrigin = y * input.cols + x;
							//int offset = indexTmp - indexOrigin;
							//// write the table into txt file
			    //            std::string str = std::to_string(long double(offset)) + " "; 
			    //            fprintf(fp,"%s",str.c_str());
							
							if(i == x && j == y) continue;

							//std::cout << i << "-" << j << " ";
							unsigned char pixelValueRoot = input.at<unsigned char>(i, j);  // the type must be matched
							if (static_cast<int>(pixelValueRoot) <= upper && static_cast<int>(pixelValueRoot) >= lower) //&& !outputDataPtr[y])                         
							{
								output.at<unsigned char>(i, j) = maskPtr[y]; // MaxValue;
							}
						}
					}
					//std::string anthorLine = std::string("\n");
		   //         fprintf(fp,"%s", anthorLine.c_str());
				}
			}	
		} // for y
      }// for x
	return 1;
}
