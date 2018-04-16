/*******************************************
*
*  Program for 2D/3D array allocations of CUDA
*  a   :zhonghy
*  date:2018-4-16
*
********************************************/

//#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdio.h>
#include <iostream>

//access type of global variables 
__constant__ float constData[256];
__device__ float devData;
__device__ float *devPointer;


//Device code, access for 2D array, will be padded
//pitch is actual size of row after padded? inerval across each rows
__global__ void MyKernel2D(float *devPtr, size_t pitch, int width, int height)
{
	for(int r = 0; r < height; r++)
	{
		float *row = (float*)((char*)devPtr + pitch); //first pointer of row
			for(int c = 0; c < width; c++)
			{
				float element = row[c];
			}
	}
}

//Device code, access for 3D array, will be padded
__global__ void MyKernel3D(cudaPitchedPtr devPitchedPtr,
	                       int width, int height, int depth)
{
	char *devPtr = (char *)devPitchedPtr.ptr;   //add type transfrom?
	size_t pitch = devPitchedPtr.pitch;
	size_t slicePitch = pitch * height;
	for(int z = 0; z < depth; ++z)
	{
		float *slice = (float*)(devPtr + z * slicePitch);
		for(int y = 0; y < height; ++y)
		{
			float *row = (float*)(devPtr + y * pitch); //poir to calucate first pointer
			for(int x = 0; x < width; ++x)
			{
				float element = row[x];
			}
		}
	}
}

int main(int argc, char *argv[])
{
	//CUDA allocate width x height 2D array
	int width = 64, height = 64;
	float *devPtr;
	size_t pitch;
	cudaMallocPitch(&devPtr, &pitch,    //cudaMallocPitch()
		width * sizeof(float), height);

	MyKernel2D<<<100, 512>>>(devPtr, pitch, width, height);
	
	std::cout << "pitch of 2D array: " << pitch << std::endl;
	std::cout << "width of 2D array is :" << width << std::endl;

	cudaFree(devPtr);


	//CUDA allocate witdth x height x depth 3D array
	int depth = 64;
	cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
	cudaPitchedPtr devPitchedPtr;
	cudaMalloc3D(&devPitchedPtr, extent);

	MyKernel3D<<<100, 512>>>(devPitchedPtr, width, height, depth);

	std::cout << std::endl;
	std::cout << "pitch of 2D array: " << devPitchedPtr.pitch << std::endl;
	std::cout << "width of 2D array is :" << devPitchedPtr.xsize << std::endl;
	std::cout << "y of 2D array is :" << devPitchedPtr.ysize << std::endl;

	cudaFree(devPitchedPtr.ptr);



	//access type of global variables 
	//__constant__ float constData[256];
	float data[256];
	cudaMemcpyToSymbol(constData, data, sizeof(data));
	cudaMemcpyFromSymbol(data, constData, sizeof(data));

	//__device__ float devData;
	float value = 3.14f;
	cudaMemcpyToSymbol(&devData, &value, sizeof(float)); //wrong in document

	//__device__ float *devPointer;
	float *ptr;
	cudaMalloc(&ptr, 256 * sizeof(float));
	cudaMemcpyToSymbol(devPointer, ptr, sizeof(ptr));  //it is like reference?

	return 0;
}