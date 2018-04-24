/****************************************
*
*  Program used for stream priority and
*  Multi-Device system.
*
*  a   : zhonghy
*  date: 2018-4-24
*****************************************/

//CUDA head files
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"       //maybe raise errors, because use v8.0 helper_cuda.h?

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>


//cudaError_t, remember
__global__ void MyKernel(float *p)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	p[tid] *= p[tid];  
}

int main(int argc, char *argv[])
{
   ////stream priorities
   ////get the range of stream priorities for this device
   //int priority_high, priority_low;
   //checkCudaErrors(cudaDeviceGetStreamPriorityRange(&priority_high, &priority_low));
   ////create streams with highest and lowest available priorities
   //cudaStream_t st_high, st_low;
   //checkCudaErrors(cudaStreamCreateWithPriority(&st_high, 
	  // cudaStreamNonBlocking, priority_high));
   //checkCudaErrors(cudaStreamCreateWithPriority(&st_low,
	  // cudaStreamNonBlocking, priority_low));
   //std::cout << priority_high << " " << priority_low << std::endl;

   //checkCudaErrors(cudaStreamDestroy(st_high));
   //checkCudaErrors(cudaStreamDestroy(st_low));


	/**********************************************************/
	//Multi-Device
	int deviceCount;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));
	int device;
	for(device = 0; device < deviceCount; ++device)
	{
		cudaDeviceProp deviceProp;
		checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device));
		std::cout << "Device " << device << " has compute capability "
			<< deviceProp.major << "." << deviceProp.minor << "." << std::endl;
	}

	size_t size = 1024 * sizeof(float);
	checkCudaErrors(cudaSetDevice(0));            //set device 0 as current
	cudaStream_t s0;
	checkCudaErrors(cudaStreamCreate(&s0));
	float *p0;
	float *p0_h;
	p0_h = (float*)malloc(size);
	for(int i = 0; i < 1024; ++i)
	{
		p0_h[i] = i;
	}
	checkCudaErrors(cudaMalloc(&p0, size));       //Allocate memeory on device 0
	checkCudaErrors(cudaMemcpy(p0, p0_h, size, cudaMemcpyHostToDevice));
	MyKernel<<<1000, 128, 0, s0>>>(p0); //Launch kernel on device 0
	checkCudaErrors(cudaMemcpy(p0_h, p0, size, cudaMemcpyDeviceToHost));

	//for p-p access
	checkCudaErrors(cudaSetDevice(1));
	checkCudaErrors(cudaDeviceEnablePeerAccess(0, 0));
	MyKernel<<<1000, 128>>>(p0); //Launch kernel on device 0

	for(int i = 0; i < 5; ++i)
	{
		std::cout << p0_h[i] << " ";
	}
	std::cout << std::endl;
	std::cout << cudaGetLastError() << std::endl;



	////if have other device 1, for test
	checkCudaErrors(cudaSetDevice(1));            //Set device 1 as current
	cudaStream_t s1;
	checkCudaErrors(cudaStreamCreate(&s1));
	float *p1;
	float *p1_h;
	p1_h = (float*)malloc(size);
	for(int i = 0; i < 1024; ++i)
	{
		p1_h[i] = i;
	}
	checkCudaErrors(cudaMalloc(&p1, size));       //Allocate memeory on device 0
	checkCudaErrors(cudaMemcpy(p1, p1_h, size, cudaMemcpyHostToDevice));
	MyKernel<<<1000, 128, 0, s1>>>(p1); //Launch kernel on device 1

	//This kernel launch will fail: Launch kernel on device 1 in s0
	MyKernel<<<1000, 128, 0, s0>>>(p1);

	checkCudaErrors(cudaMemcpy(p1_h, p1, size, cudaMemcpyDeviceToHost));
	for(int i = 0; i < 5; ++i)
	{
		std::cout << p1_h[i] << " ";
	}
	std::cout << std::endl;

	checkCudaErrors(cudaFree(p0));
	checkCudaErrors(cudaFree(p1));
	free(p0_h);
	free(p1_h);
	checkCudaErrors(cudaStreamDestroy(s0));
	checkCudaErrors(cudaStreamDestroy(s1));

	/*****************************************************************/
	//p-p memory copy
	checkCudaErrors(cudaSetDevice(0));   //set device 0 as current
	float *p2;
	size_t size1 = 1024 * sizeof(float);
	checkCudaErrors(cudaMalloc(&p2, size));  //Allocate memory on device 0
	checkCudaErrors(cudaSetDevice(1));       //set device 1 as current
	float *p3;
	checkCudaErrors(cudaMalloc(&p3, size));  //Allocate memory on device 1
	checkCudaErrors(cudaSetDevice(0));       //set device 0 as current
	MyKernel<<<1000, 128>>>(p2);             //Launch kernel on device 0

	checkCudaErrors(cudaSetDevice(1));       //set device 1 as current
	checkCudaErrors(cudaMemcpyPeer(p3, 1, p2, 0, size)); //Copy p2 to p3, memory copy from device to device
	MyKernel<<<1000, 128>>>(p3);             //Launch kernel on device 1

    return 0;
}
