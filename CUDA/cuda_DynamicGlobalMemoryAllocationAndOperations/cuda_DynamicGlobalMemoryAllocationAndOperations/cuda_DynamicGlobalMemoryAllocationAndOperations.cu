
/************************************************
*
*  Test for DynamicGlobalMemoryAllocationAndOperations function!
*  date:2018-5-16
*  a   :zhonghy
*
*************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define NUM_BLOCKS 20

//per thread allocation
__global__ void mallocTestPerThread()
{
	size_t size = 123;
	char *ptr = (char*)malloc(size);
	memset(ptr, 0, size);
	printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
	free(ptr);
}

//per Thread Block Allocation
__global__ void mallocTestPerBlock()
{
	__shared__ int *data;
	
	//The first thread in the block does the allocation and initialization
	//and then shares the pointer with all other threads through shared memory
	//so that access can easily be coalesced.
	//64 bytes per thread are allocated.
	if(threadIdx.x == 0)
	{
		size_t size = blockDim.x * 64; //per every thread?
		data = (int *)malloc(size);
		memset(data, 0, size);
	}
	__syncthreads();

	//Check for failure
	if(data == NULL)
		return;

	//Threads index into the memory, ensuring coalescence
	int *ptr = data;
	for(int i = 0; i < 64; ++i)
	{
		ptr[i * blockDim.x + threadIdx.x] = threadIdx.x;
	    printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
		printf("\n");
	}

	//Ensure all threads complete before freeing
	__syncthreads();

	//Only one thread may free the memory
	if(threadIdx.x == 0)
		free(ptr);
}

//Allocation persisting between kernel launches
__device__ int* dataptr[NUM_BLOCKS]; //Per-block pointer?

__global__ void allocmem()
{
	//only the first thread in the block does the allocation
	//since we want only one allocation per block.
	if(threadIdx.x == 0)
		dataptr[blockIdx.x] = (int *)malloc(blockDim.x * sizeof(int)); //4
    __syncthreads();

	//chech for failure
	if(dataptr[blockIdx.x] == NULL)
		return;

	//zero the data with all threads in parallel
	dataptr[blockIdx.x][threadIdx.x] = 0;
}

//simple example: store thread ID into each element
__global__ void usemem()
{
	int *ptr = dataptr[blockIdx.x];
	if(ptr != NULL)
		ptr[threadIdx.x] += threadIdx.x;
}

//print the content of the buffer before freeing it
__global__ void freemem()
{
	int *ptr = dataptr[blockIdx.x];
	if(ptr != NULL)
		printf("Block %d, Thread %d: final value = %d\n",
		        blockIdx.x, threadIdx.x, ptr[threadIdx.x]);

	//only free from one thread!
	if(threadIdx.x == 0)
		free(ptr);
}



int main()
{
	/*per thread*/
 //  //Set a heap size of 128 megabytes. Note that that is must
 //  //be done before any kernel is launched
	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
	//mallocTestPerThread<<<1, 5>>>();
	//cudaDeviceSynchronize();

	/*per block*/
	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
	//mallocTestPerBlock<<<10, 128>>>();
	//cudaDeviceSynchronize();

	/*Allocation Persisting Between Kernel Launches*/
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
	//Allocate memory
	allocmem<<< NUM_BLOCKS, 10 >>>();
	//Use memory
	usemem<<< NUM_BLOCKS, 10 >>>();
	usemem<<< NUM_BLOCKS, 10 >>>();
	usemem<<< NUM_BLOCKS, 10 >>>();
	//Free memory
	freemem<<< NUM_BLOCKS, 10 >>>();

	cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}
