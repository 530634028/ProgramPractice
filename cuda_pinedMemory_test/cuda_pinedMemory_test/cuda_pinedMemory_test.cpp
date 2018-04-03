
/***************************
*
*
*  Test for cudaHostAlloc()
*
*
***************************/


#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "helper_cuda.h"

int main(int argc, char *argv[])
{
	int dev = 0;
	cudaSetDevice(dev);

	unsigned int isize = 1 << 22;
	unsigned int nbytes = isize * sizeof(float);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	if(!deviceProp.canMapHostMemory)
	{
		printf("Device %d does not support mapping cpu host memory!\n", dev);
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}

	printf("%s starting at ", argv[0]);
	printf("device %d: %s memory size %d nbyte %5.2fMB canMap %d\n", dev,
		deviceProp.name, isize, nbytes / (1024.0f * 1024.0f),
		deviceProp.canMapHostMemory);

	float *h_a;
	cudaMallocHost((float **)&h_a, nbytes);

	float *d_a;
	cudaMalloc((float **)&d_a, nbytes);

	memset(h_a, 0, nbytes);

	for(int i = 0; i < isize; ++i ) h_a[i] = 100.10f;

	cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);

	cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFreeHost(h_a);

	cudaDeviceReset();

	return EXIT_SUCCESS;
}