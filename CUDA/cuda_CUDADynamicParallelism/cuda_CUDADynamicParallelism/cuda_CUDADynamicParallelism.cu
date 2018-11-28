
/****************************************
*
*  date: 2018-5-23
*  a   : zhonghy
*
*
*
*****************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


//simple Hello World program incorporating dynamic parallelism
__global__ void childKernel()
{
	printf("Hello ");
}

__global__ void parentKernel()
{
	//launch child
	childKernel<<<1, 1>>>();
	if(cudaSuccess != cudaGetLastError())
	{
		return;
	}

	//wait for child to complete
	if(cudaSuccess != cudaDeviceSynchronize())
	{
		return;
	}

	printf("World!\n");
}

int main(int argc, char *argv[])
{
	//launch parent
	parentKernel<<<1, 1>>>();
	if(cudaSuccess != cudaGetLastError())
	{
		return 1;
	}

	//wait for parent to complete
	if(cudaSuccess != cudaDeviceSynchronize())
	{
		return 2;
	}

    return EXIT_SUCCESS;
}
