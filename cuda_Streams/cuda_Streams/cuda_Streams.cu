
/********************************************************
*
*
* This program is used to test Streams for CUDA performance
* improvement!
* a:zhonghy
*
*
*********************************************************/


//#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <helper_cuda.h>
#include <helper_functions.h>

#include <vector>
#include <string>
#include <stdio.h>
#include <math.h>

const int N = (1024*1024);
const int FULL_DATA_SIZE  = N * 20;



//for test kernel,c is output
__global__ void testKernel(int *c, const int *a, const int *b)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < N)
	{
		c[idx] = (a[idx] + b[idx]) / 2;
	}
}

//unuse stream(default)
void UnUsedStreams()
{
	//start event clock
	cudaEvent_t start, stop;
	float elapsedTime;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	int *host_a, *host_b, *host_c;
	int *dev_a, *dev_b, *dev_c;

	//alloc memory in GPU
	checkCudaErrors(cudaMalloc((void **)dev_a, FULL_DATA_SIZE * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)dev_b, FULL_DATA_SIZE * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)dev_c, FULL_DATA_SIZE * sizeof(int)));

	//alloc memory in CPU
	host_a = (int *)malloc(FULL_DATA_SIZE * sizeof(int));
	host_b = (int *)malloc(FULL_DATA_SIZE * sizeof(int));
	host_c = (int *)malloc(FULL_DATA_SIZE * sizeof(int));

	//assign vlaues in cpu
	for(int i = 0; i < FULL_DATA_SIZE; ++i)
	{
		host_a[i] = i;
		host_b[i] = FULL_DATA_SIZE -i;
	}

	//copy data from CPU to GPU
	checkCudaErrors(cudaMemcpy(dev_a, host_a, FULL_DATA_SIZE * sizeof(int),
		cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_b, host_b, FULL_DATA_SIZE * sizeof(int),
		cudaMemcpyHostToDevice));

	testKernel<<<FULL_DATA_SIZE / 1024, 1024>>>(dev_c, dev_a, dev_b);

	checkCudaErrors(cudaMemcpy(host_c, dev_c, FULL_DATA_SIZE * sizeof(int),
		cudaMemcpyDeviceToHost));

	//end of clock
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	std::cout << "Time consume: " << elapsedTime << std::endl;

	//show output
	for(int i = 0; i < 10; ++i)
	{
		std::cout << host_c[i] << std::endl;
	}

	getchar();


	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
}

int UsedStreams()
{
	//attain device properties
	cudaDeviceProp prop;
	int deviceID;
	checkCudaErrors(cudaGetDevice(&deviceID));
	checkCudaErrors(cudaGetDeviceProperties(&prop, deviceID));

	//test if have overlap
	if(!prop.deviceOverlap)
	{
		std::cout << "No device will handle overlaps." << endl;
		return 0;
	}

	//start event clock
	cudaEvent_t start, stop;
	float elapsedTime;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	//create streams
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	int *host_a, *host_b, *host_c;
	int *dev_a, *dev_b, *dev_c;

	//alloc memory in GPU
	checkCudaErrors(cudaMalloc((void **)dev_a, N * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)dev_b, N * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)dev_c, N * sizeof(int)));

	//alloc memory in CPU, must pined memory
	checkCudaErrors(cudaHostAlloc((void **)&host_a, FULL_DATA_SIZE * sizeof(int),
		cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void **)&host_b, FULL_DATA_SIZE * sizeof(int),
		cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc((void **)&host_c, FULL_DATA_SIZE * sizeof(int),
		cudaHostAllocDefault));

	//assign vlaues in cpu
	for(int i = 0; i < FULL_DATA_SIZE; ++i)
	{
		host_a[i] = i;
		host_b[i] = FULL_DATA_SIZE -i;
	}

	for(int i= 0; i < FULL_DATA_SIZE; i +=N)
	{
		checkCudaErrors(cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int),
			cudaMemcpyHostToDevice, stream));
		checkCudaErrors(cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int),
			cudaMemcpyHostToDevice, stream));

		testKernel<<<N /1024, 1024, 0, stream>>>(dev_c, dev_a, dev_b);

		checkCudaErrors(cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int),
			cudaMemcpyDeviceToHost, stream));
	}

	//end of clock
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	std::cout << "Time consume: " << elapsedTime << std::endl;

	//show output
	for(int i = 0; i < 10; ++i)
	{
		std::cout << host_c[i] << std::endl;
	}

	getchar();

	checkCudaErrors(cudaFreeHost(host_a));
	checkCudaErrors(cudaFreeHost(host_b));
	checkCudaErrors(cudaFreeHost(host_c));

	checkCudaErrors(cudaFree(dev_a));
	checkCudaErrors(cudaFree(dev_b));
	checkCudaErrors(cudaFree(dev_c));

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	return 0;
}


int main(int argc, char *argv[])
{
    UnUsedStreams();
	UsedStreams();
    return 0;
}