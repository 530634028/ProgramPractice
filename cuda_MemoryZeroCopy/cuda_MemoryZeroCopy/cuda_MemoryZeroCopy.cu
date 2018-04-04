
/********************************************
*
* Usage: Test for cudaHostAlloc() function
* a: zhy
*
* https://www.cnblogs.com/zhangshuwen/p/7349267.html
*
*********************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include <stdio.h>

#define imin(a, b) (a<b?a:b);

const int N = 33 * 1024 *1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(int size, float *a, float *b, float *c)
{
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float temp = 0;
	while(tid < size)
	{
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	//set the cache values
	cache[cacheIndex] = temp;

	//synchronize threads in this block
	__syncthreads();

    //for reductions, threadsPerBlock must be a power of 2
	//because of the following code
	int i = blockDim.x /2;
	while(i != 0)
	{
		if(cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if(cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}


//cudaMalloc memory version
float malloc_test(int size)
{
	cudaEvent_t start, stop;
	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;
	float elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	//alloc in cpu memory
	a = (float *)malloc(size * sizeof(float));
	b = (float *)malloc(size * sizeof(float));
	partial_c = (float *)malloc(blocksPerGrid * sizeof(float));
	
	//alloc in GPU memory
	checkCudaErrors(cudaMalloc((void **)&dev_a, size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&dev_b, size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&dev_partial_c, blocksPerGrid * sizeof(float)));

	//filling CPU memory alloced with data
	for(int i = 0; i < size; ++i)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	checkCudaErrors(cudaEventRecord(start, 0));
	checkCudaErrors(cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice));

	dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

	//copy data from GPU to CPU
	checkCudaErrors(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	//exit cal in cpu
	c = 0;
	for(int i = 0; i < blocksPerGrid; i++)
	{
		c += partial_c[i];
	}

	//delete pointer
	checkCudaErrors(cudaFree(dev_a));
	checkCudaErrors(cudaFree(dev_b));
	checkCudaErrors(cudaFree(dev_partial_c));

	free(a);
	free(b);
	free(partial_c);

	//release the event
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	printf("Value calculated: %f\n", c);

	return elapsedTime;
}


//zero copy version
float cuda_host_alloc_test(int size)
{
	cudaEvent_t start, stop;
	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;
	float elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	//alloc in cpu memory
	checkCudaErrors(cudaHostAlloc((void **)&a, size * sizeof(float), //cudaHostAlloc();
		cudaHostAllocWriteCombined | cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc((void **)&b, size * sizeof(float),
		cudaHostAllocWriteCombined | cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc((void **)&partial_c, blocksPerGrid * sizeof(float),
		cudaHostAllocWriteCombined | cudaHostAllocMapped));

	//filling CPU memory alloced with data
	for(int i = 0; i < size; ++i)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	checkCudaErrors(cudaHostGetDevicePointer(&dev_a, a, 0));
	checkCudaErrors(cudaHostGetDevicePointer(&dev_b, b, 0));               //cudaHostGetDevicePointer()
	checkCudaErrors(cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0));

	for(int i = 0; i < size; i++)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	checkCudaErrors(cudaEventRecord(start, 0));

	dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

	checkCudaErrors(cudaThreadSynchronize());
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	//exit cal in cpu
	c = 0;
	for(int i = 0; i < blocksPerGrid; i++)
	{
		c += partial_c[i];
	}

	//free pointer
	checkCudaErrors(cudaFreeHost(a));    //cudaFreeHost()
	checkCudaErrors(cudaFreeHost(b));
	checkCudaErrors(cudaFreeHost(partial_c));

	//free the event
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	printf("Value calculated: %f\n", c);

	return elapsedTime;
}


int main(int argc, char *argv[])
{
   cudaDeviceProp prop;
   int whichDevice;
   checkCudaErrors(cudaGetDevice(&whichDevice));
   checkCudaErrors(cudaGetDeviceProperties(&prop, whichDevice));

   if(prop.canMapHostMemory != 1)
   {
	   printf("Device can not map memory.\n");
	   return 0;
   }

   float elapsedTime;
   checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
    
   //try it with malloc
   elapsedTime = malloc_test(N);
   printf("Time using cudaMalloc:   %3.lf ms\n", elapsedTime);


   //try it with cudaHostAlloc
   elapsedTime = cuda_host_alloc_test(N);
   printf("Time using cudaHostAlloc:  %3.lf ms\n", elapsedTime);

}
