
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
#include <iostream>

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

//add callback functions 2018-4-23 zhong
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *data)
{
	std::cout << "Inside callback " << (size_t)data << std::endl;
}

//unuse stream(default)
int UnUsedStreams()
{
	//start event clock
	cudaEvent_t start, stop;
	float elapsedTime;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	int *host_a = 0, *host_b = 0, *host_c = 0;
	int *dev_a = 0, *dev_b = 0, *dev_c = 0;

	//alloc memory in GPU
	checkCudaErrors(cudaMalloc((void **)&dev_a, FULL_DATA_SIZE * sizeof(int)));  //remember dev_a need &
	checkCudaErrors(cudaMalloc((void **)&dev_b, FULL_DATA_SIZE * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&dev_c, FULL_DATA_SIZE * sizeof(int)));

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

	std::cout << "UnUsedStreams Time consume: " << elapsedTime << std::endl;

	//show output
	for(int i = 0; i < 10; ++i)
	{
		std::cout << host_c[i] << std::endl;
	}

	//getchar();


	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	return 0;
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
		//just write like endl, get error MSB3721: The command ""C:\Program Files\NVIDIA GPU Computing
		std::cout << "No device will handle overlaps." << std::endl; 
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

	int *host_a = 0, *host_b = 0, *host_c = 0;
	int *dev_a = 0, *dev_b = 0, *dev_c = 0;

	//alloc memory in GPU
	checkCudaErrors(cudaMalloc((void **)&dev_a, N * sizeof(int))); //remember dev_a need &
	checkCudaErrors(cudaMalloc((void **)&dev_b, N * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&dev_c, N * sizeof(int)));

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
		host_b[i] = FULL_DATA_SIZE - i;
	}

	for(int i= 0; i < FULL_DATA_SIZE; i += N)
	{
		checkCudaErrors(cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int),
			cudaMemcpyHostToDevice, stream));
		checkCudaErrors(cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int),
			cudaMemcpyHostToDevice, stream));

		testKernel<<<N / 1024, 1024, 0, stream>>>(dev_c, dev_a, dev_b);

		checkCudaErrors(cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int),
			cudaMemcpyDeviceToHost, stream));
	}

	//wait until gpu execution finish
	checkCudaErrors(cudaStreamSynchronize(stream));  //later add 


	//end of clock
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	std::cout << "UsedStreams Time consume: " << elapsedTime << std::endl;

	//show output
	for(int i = 0; i < 10; ++i)
	{
		std::cout << host_c[i] << std::endl;
	}

	//getchar();

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

//staged concurrent copy and execute
int UsedStagedStreams() //something wrong, why faster than UsedStreams
{
	//std::cout << "Used Staged Strems: " << std::endl; 
	//attain device properties
	const int nStreams = 8; //number of used non-null stream
	cudaDeviceProp prop;
	int deviceID;
	checkCudaErrors(cudaGetDevice(&deviceID));
	checkCudaErrors(cudaGetDeviceProperties(&prop, deviceID));

	//test if have overlap
	if(!prop.deviceOverlap)
	{
		//just write like endl, get error MSB3721: The command ""C:\Program Files\NVIDIA GPU Computing
		std::cout << "No device will handle overlaps." << std::endl; 
		return 0;
	}

	//start event clock
	cudaEvent_t start, stop;
	float elapsedTime;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	//create streams
	//cudaStream_t stream;  //error arise--repetition next
	//cudaStreamCreate(&stream);

	int *host_a = 0, *host_b = 0, *host_c = 0;
	int *dev_a = 0, *dev_b = 0, *dev_c = 0;


	//alloc memory in GPU
	checkCudaErrors(cudaMalloc((void **)&dev_a, FULL_DATA_SIZE * sizeof(int))); //remember dev_a need &
	checkCudaErrors(cudaMalloc((void **)&dev_b, FULL_DATA_SIZE * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&dev_c, FULL_DATA_SIZE * sizeof(int)));

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
		host_b[i] = FULL_DATA_SIZE - i;
	}


	int sizeOfStream = FULL_DATA_SIZE * sizeof(int) / nStreams;

	cudaStream_t stream[nStreams];
	for(int i = 0; i < nStreams; i++)
	{
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}


	////add callback functions 
	//for(int i= 0; i < nStreams; i++)
	//{
	//	int offset = i * FULL_DATA_SIZE / nStreams; //it is different from sizeOfStream
	//	checkCudaErrors(cudaMemcpyAsync(dev_a + offset, host_a + offset, sizeOfStream,
	//		cudaMemcpyHostToDevice, stream[i]));
	//	checkCudaErrors(cudaMemcpyAsync(dev_b + offset, host_b + offset, sizeOfStream,
	//		cudaMemcpyHostToDevice, stream[i]));

	//	testKernel<<<FULL_DATA_SIZE / (nStreams * 1024) , 1024, 0, stream[i]>>>(dev_c + offset, dev_a + offset, dev_b + offset);

	//	checkCudaErrors(cudaMemcpyAsync(host_c + offset, dev_c + offset, sizeOfStream,
	//		cudaMemcpyDeviceToHost, stream[i]));

	//	//add callback functions
	//	checkCudaErrors(cudaStreamAddCallback(stream[i], MyCallback, (void*)i, 0));
	//}

	//another way to run zhonghy-2018-4-19 added, in 3.2.5.5 of guide
	for(int i= 0; i < nStreams; i++)
	{
		int offset = i * FULL_DATA_SIZE / nStreams; //it is different from sizeOfStream
		checkCudaErrors(cudaMemcpyAsync(dev_a + offset, host_a + offset, sizeOfStream,
			cudaMemcpyHostToDevice, stream[i]));
	}

	for(int i= 0; i < nStreams; i++)
	{
		int offset = i * FULL_DATA_SIZE / nStreams; //it is different from sizeOfStream
		checkCudaErrors(cudaMemcpyAsync(dev_b + offset, host_b + offset, sizeOfStream,
			cudaMemcpyHostToDevice, stream[i]));
	}

	for(int i= 0; i < nStreams; i++)
	{
		int offset = i * FULL_DATA_SIZE / nStreams; //it is different from sizeOfStream
		testKernel<<<FULL_DATA_SIZE / (nStreams * 1024) , 1024, 0, stream[i]>>>(dev_c + offset, dev_a + offset, dev_b + offset);
	}

	for(int i= 0; i < nStreams; i++)
	{
		int offset = i * FULL_DATA_SIZE / nStreams; //it is different from sizeOfStream
		checkCudaErrors(cudaMemcpyAsync(host_c + offset, dev_c + offset, sizeOfStream,
			cudaMemcpyDeviceToHost, stream[i]));
	}
	


	for(int i = 0; i < nStreams; ++i)
	{
				//wait until gpu execution finish
		checkCudaErrors(cudaStreamSynchronize(stream[i]));  //later add 
	}

   
	//end of clock
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	std::cout << "UsedStagedStreams Time consume: " << elapsedTime << std::endl;

	//show output
	for(int i = 0; i < 10; ++i)
	{
		std::cout << host_c[i] << std::endl;
	}

	//getchar();

	for(int i = 0; i < nStreams; ++i)
	{
		//wait until gpu execution finish
		checkCudaErrors(cudaStreamDestroy(stream[i]));  //later add 
	}

	//free all memory 
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
    UnUsedStreams(); // nonStreams
	UsedStreams();//20 streams
	UsedStagedStreams();//4 Streams
	//why cann't runs, because of the getchar() function
    return 0;
}