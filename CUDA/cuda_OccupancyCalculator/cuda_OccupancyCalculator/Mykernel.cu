
/********************************************

  This is test for cuda, used to calculate the
  occupancy.It then reports the occupancy level 
  with the ratio between concurrent warps versus
  maximum warps per multiprocessor.
  a:zhonghy
*********************************************/


#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include "helper_cuda.h"

#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>

//Device code
__global__ void MyKernel(int *d, int *a, int *b)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx];
}

//Device code2
__global__ void MyKernelNew(int *array, int arrayCount)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < arrayCount)
	{
		array[idx] *= array[idx];
	}
}

//Host code2
int launchMyKernelNew(int *array, int arrayCount)  //wrong write array not *array so get error  MSB3721
{
	int blockSize; //The launch configurator returned block size
	int minGridSize; //The minimum grid size needed to achieve the
	                 //maximum occupancy for a full device launch
	int gridSize;     //The actual grid size needed, basedd on input size

	cudaOccupancyMaxPotentialBlockSize( 
		&minGridSize, &blockSize, (void*)MyKernelNew, 0, arrayCount);

	//std::cout << blockSize << std::endl;
	//Round up according to array size


	//need to add code: cudaMalloc() cudaMemcpy(),copy array to device
	//gridSize = (arrayCount + blockSize - 1) / blockSize;
	////std::cout << gridSize << std::endl;
	//MyKernelNew<<<gridSize, blockSize>>>(array, arrayCount);
	//cudaDeviceSynchronize();


	//calculate occupany
	int maxActiveBlocks;
	checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&maxActiveBlocks, MyKernelNew, blockSize, 0));

	int device;
	cudaDeviceProp props;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);

	float occupancy = (maxActiveBlocks * blockSize / props.warpSize) /
		(float)(props.maxThreadsPerMultiProcessor / props.warpSize);

	std::cout << "MyKernelNew: " << std::endl;
	std::cout << "Launched blocks of size: " << blockSize << std::endl <<
		"Theoretical occupancy:" << occupancy << std::endl;
	std::cout << "maxActiveBlocks: " << maxActiveBlocks << std::endl;
	std::cout << "prop.warpSize: " << props.warpSize << std::endl;
	std::cout << "prop.maxThreadsPerMultiProcessor: " << props.maxThreadsPerMultiProcessor << std::endl;
	//If interested, the occupancy can be calculated with
	//cudaOccupancyMaxActiveBlockPerMultiprocessor
	return 0;
}

//Host code
int main(int argc, char *argv[])
{
	int numBlocks;   //Occupancy in terms of active blocks
	int blockSize = 32;

	//Thest variables are used to convert occupancy to warps
	int device;
	cudaDeviceProp prop;
	int activeWarps;
	int maxWarps;

	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));

	checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
		MyKernel, blockSize, 0));

	std::cout << "numBlocks: " << numBlocks << std::endl;
	std::cout << "blockSize: " << blockSize << std::endl;
	std::cout << "prop.warpSize: " << prop.warpSize << std::endl;
	std::cout << "prop.maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;

	//know maxThreadsPerMulProcessor = numBlocks * blockSize
	activeWarps = numBlocks * blockSize / prop.warpSize;
	maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

	std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" <<
		std::endl;



    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

	//occupancy-based kernel launch
	const int num = 100000;
	int a[num] = {0};
	launchMyKernelNew(a, num);

    return 0;
}