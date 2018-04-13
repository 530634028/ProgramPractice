






//#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdio.h>
#include <iostream>


__global__ void MyKernel(float *devPtr, size_t pitch, int width, int height)
{

}


int main(int argc, char *argv[])
{
	int width = 64, height = 64;
	float *devPtr;
	size_t pitch;
	cudaMallocPitch(&devPtr, &pitch, 
		width * sizeof(float), height);
	MyKernel<<<100, 512>>>(float *devPtr, size_t pitch, int width, int height);


	return 0;
}
