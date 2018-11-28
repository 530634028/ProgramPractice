
/******************************************************
*
* Program used to test surface object and reference API
*
* a   : zhonghy
* date: 2018-4-27
********************************************************/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>


//declaration for functions
int print(float *, int, int);
__global__ void copyKernel(cudaSurfaceObject_t, cudaSurfaceObject_t,
	                       int, int, float *);
__global__ void copyKernelRef(int, int, float *);


int print(float *mat, int width, int height)
{
	if(!mat)
	{
		return 0;
	}
	for(int i = 0; i < width; ++i)
	{
		for(int j = 0; j < height; ++j)
		{
			std::cout << mat[j * width + i] << " ";
		}
		std::cout << std::endl;
	}

}

//Simple copy kernel
__global__ void copyKernel(cudaSurfaceObject_t inputSurfObj, 
	                       cudaSurfaceObject_t outputSurfObj,
	                       int width, int height, float *output)
{
	//Calculate normalized surface coordinates
	//2D block
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x < width && y < height)
	{
		float data;
		//read from input surface
		surf2Dread(&data, inputSurfObj, x * 4, y);
		//write to output surface
		surf2Dwrite(data, outputSurfObj, x * 4, y);

		surf2Dread(&data, outputSurfObj, x * 4, y);
		output[y * width + x] = data;
	}
}

//2D surfaces
surface<void, 2> inputSurfRef;
surface<void, 2> outputSurfRef;

//Simple copy kernel
__global__ void copyKernelRef(int width, int height, float *output)
{
	//Calculate surface coordinates
	//2D block
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x < width && y < height)
	{
		float data;
		//read from input surface
		surf2Dread(&data, inputSurfRef, x * 4, y);
		//write to output surface
		surf2Dwrite(data, outputSurfRef, x * 4, y);

		surf2Dread(&data, outputSurfRef, x * 4, y);
		output[y * width + x] = data;
	}
}


int main(int argc, char *argv[])
{
	/**********************surface object*********************/
	//Allocate CUDA array in device memory
	int width = 256;
	int height = 256;
	int size = width * height;
	float *h_data;
	h_data = new float[size];
	//h_data = (float*)malloc(size * sizeof(float));
	for(int i = 0; i < width; ++i)
	{
		for(int j = 0; j < height; ++j)
		{
			h_data[j * width + i] = (i + j + 2) / 2;
		}
	}
	print(h_data, 5 ,5);
	std::cout << std::endl;

	cudaChannelFormatDesc channelDesc = 
		cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	cudaArray *cuInputArray;
	checkCudaErrors(cudaMallocArray(&cuInputArray, &channelDesc,
		width, height, cudaArraySurfaceLoadStore));

	cudaArray *cuOutputArray;
	checkCudaErrors(cudaMallocArray(&cuOutputArray, &channelDesc,
		width, height, cudaArraySurfaceLoadStore));

	//Copy to device memory some data located at address h_data
	//in host memory
	checkCudaErrors(cudaMemcpyToArray(cuInputArray, 0, 0, h_data,
		size, cudaMemcpyHostToDevice));

	//Allocate result of transformation in device memory
	float *output;
	checkCudaErrors(cudaMalloc(&output, width * height * sizeof(float)));

	/***********************surface object*****************************/
	////Specify surface
	//struct cudaResourceDesc resDesc;
	//memset(&resDesc, 0, sizeof(resDesc));
	//resDesc.resType = cudaResourceTypeArray;
	////create the surface objects
	//resDesc.res.array.array = cuInputArray;
	//cudaSurfaceObject_t inputSurfObj = 0;
	//checkCudaErrors(cudaCreateSurfaceObject(&inputSurfObj, &resDesc));
	//resDesc.res.array.array = cuOutputArray;
	//cudaSurfaceObject_t outputSurfObj = 0;
	//checkCudaErrors(cudaCreateSurfaceObject(&outputSurfObj, &resDesc));
	////Invoke kernel
	//dim3 dimBlock(16, 16);
	//dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, 
	//	         (height + dimBlock.y - 1) / dimBlock.y);
	//copyKernel<<<dimGrid, dimBlock>>>(inputSurfObj, outputSurfObj,
	//	                              width, height, output);


	/************************surfce reference*******************/
	//Bind the arrays to the surface references
	checkCudaErrors(cudaBindSurfaceToArray(inputSurfRef, cuInputArray));
	checkCudaErrors(cudaBindSurfaceToArray(outputSurfRef, cuOutputArray));
	dim3 dimBlock(16, 16);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, 
		(height + dimBlock.y - 1) / dimBlock.y);
	copyKernelRef<<<dimGrid, dimBlock>>>(width, height, output);

	//read result from device
	float *output_h;
	//output_h = (float*)malloc(size * sizeof(float));
	output_h = new float[size];
	checkCudaErrors(cudaMemcpy(output_h, output, size, cudaMemcpyDeviceToHost));
    print(output_h, 256, 10);

	////Destory Surface object
	//checkCudaErrors(cudaDestroySurfaceObject(inputSurfObj));
	//checkCudaErrors(cudaDestroySurfaceObject(outputSurfObj));

	//Free device memory
    cudaFreeArray(cuInputArray);
	cudaFreeArray(cuOutputArray);
    cudaFree(output);
	free(h_data);
	delete[] output_h;







	/*********************surface reference*********************/
	////Allocate CUDA array in device memory
	//int width = 256;
	//int height = 256;
	//int size = width * height;
	//float *h_data;
	//h_data = (float*)malloc(size * sizeof(float));
	//for(int i = 0; i < width; ++i)
	//{
	//	for(int j = 0; j < height; ++j)
	//	{
	//		h_data[j * width + i] = (i + j + 2) / 2;
	//	}
	//}
	//print(h_data, 5 ,5);

	//cudaChannelFormatDesc channelDesc = 
	//	cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	//cudaArray *cuInputArray;
	//checkCudaErrors(cudaMallocArray(&cuInputArray, &channelDesc,
	//	width, height, cudaArraySurfaceLoadStore));

	//cudaArray *cuOutputArray;
	//checkCudaErrors(cudaMallocArray(&cuOutputArray, &channelDesc,
	//	width, height, cudaArraySurfaceLoadStore));

	////Copy to device memory some data located at address h_data
	////in host memory
	//checkCudaErrors(cudaMemcpyToArray(cuInputArray, 0, 0, h_data,
	//	size, cudaMemcpyHostToDevice));

	////Bind the arrays to the surface references
	//checkCudaErrors(cudaBindSurfaceToArray(inputSurfRef, cuInputArray));
	//checkCudaErrors(cudaBindSurfaceToArray(outputSurfRef, cuOutputArray));

	////Allocate result of transformation in device memory
	//float *output;
	//checkCudaErrors(cudaMalloc(&output, width * height * sizeof(float)));

	////Invoke kernel
	//dim3 dimBlock(16, 16);
	//dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, 
	//	         (height + dimBlock.y - 1) / dimBlock.y);
	//copyKernelRef<<<dimGrid, dimBlock>>>(width, height);

	////read result from device
	//float *output_h;
	////output_h = (float*)malloc(size * sizeof(float));
	//output_h = new float[size];
	//checkCudaErrors(cudaMemcpy(output_h, output, size, cudaMemcpyDeviceToHost));
 //   print(output_h, 256, 10);

	////Destory Surface object
	//checkCudaErrors(cudaDestroySurfaceObject(inputSurfObj));
	//checkCudaErrors(cudaDestroySurfaceObject(outputSurfObj));

	////Free device memory
 //   cudaFreeArray(cuInputArray);
	//cudaFreeArray(cuOutputArray);
 //   cudaFree(output);
	//free(h_data);
	//delete[] output_h;

    return EXIT_SUCCESS;
}
