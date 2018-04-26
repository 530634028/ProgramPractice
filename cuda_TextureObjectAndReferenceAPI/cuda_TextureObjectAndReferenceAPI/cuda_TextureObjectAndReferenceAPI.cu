
/******************************************************
*
* Program used to test texture object and reference API
*
* a   : zhonghy
* date: 2018-4-24
********************************************************/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>



//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


//Simple transformtion kernel
__global__ void transformKernel(float *output, cudaTextureObject_t texObj,
	                            int width, int height, float theta)
{
	//Calculate normalized texture coordinates
	//2D block
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = x / (float)width;
	float v = y / (float)height;

	//transform coordinates
	u -= 0.5f;
	v -= 0.5f;
	float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
	float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

	//Read from texture and write to global memory
	output[y * width + x] = tex2D<float>(texObj, tu, tv);
}


int main(int argc, char *argv[])
{
	//Allocate CUDA array in device memory
	int width = 1024;
	int height = 1024;
	int dataSize = width * height;
	float *h_data;
	h_data = (float*)malloc(dataSize * sizeof(float));


	cudaChannelFormatDesc channelDesc = 
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *cuArray;
	checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, width, height));

	//Copy to device memory some data located at address h_data
	//in host memory
	checkCudaErrors(cudaMemcpyToArray(cudaArray, 0, 0, h_data,
		size, cudaMemcpyHostToDevice));

	//Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	//Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0]   = cudaAddressModeWrap;
	texDesc.addressMode[1]   = cudaAddressModeWrap;
	texDesc.filterMode       = cudaFilterModeLinear;
	texDesc.readMode         = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	//Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	//Allocate result of transformation in device memory
	float *output;
	checkCudaErrors(cudaMalloc(&output, width * height * sizeof(float)));

	//Invoke kernel
	dim3 dimBlock(16, 16);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, 
		         (height + dimBlock.y - 1) / dimBlock.y);
	transformKernel<<<dimGrid, dimBlock>>>(output, 
		      texObj, width, height, angle);

	//Destory texture object
	cudaDestroyTextureObject(texObj);

	//Free device memory
    cudaFreeArray(cuArray);
    cudaFree(output);
    
    return EXIT_SUCCESS;
}
