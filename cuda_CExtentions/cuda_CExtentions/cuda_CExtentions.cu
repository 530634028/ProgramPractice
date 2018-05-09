/******************************************
*
* Test for C language extension of CUDA
* date:2018-5-9
* a:   zhonghy
*
******************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>

extern __shared__ float shared[];
extern __shared__ float array[];
short array0[128];
float array1[64];
int   array2[256];

/****************************************************/
//right version, declare and initialize way, offsets
__device__ void func()
{
	short *array0 = (short*)array;
	float *array1 = (float*)&array0[128];
	int   *array2 = (int*)&array1[64];
}

//incorrect version
__device__ void funcE()
{
	short *array0 = (short *)array;
	float *array1 = (float *)&array0[127];
}


/****************************************************/
void foo(const float *a, const float *b, float *c)
{
	c[0] = a[0] * b[0];
	c[1] = a[0] * b[0];
	c[2] = a[0] * b[0] * a[1];
	c[3] = a[0] * b[1];
	c[4] = a[0] * b[0];
	c[5] = b[0];
}

void foo1(const float* __restrict__ a,
	     const float* __restrict__ b, 
		 float* __restrict__ c)
{
	float t0 = a[0];
	float t1 = b[0];
	float t2 = t0 * t2;
	float t3 = a[1];
	c[0] = t2;
	c[1] = t2;
	c[4] = t2;
	c[2] = t2 * t3;
	c[3] = t0 * t3;
	c[5] = t1;
}



/****************************************************/
__device__ volatile int x = 1, y = 2;
__device__ void WriteXY()
{
	x = 10;
	__threadfence();
	y = 20;
}

__device__ void ReadXY(int *A, int *B)
{
	*A = x;
	__threadfence();
	*B = y;
}


/****************************************************/
__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;

__device__ float calculatePartialSum(const float *array, int N)
{
	float result = 0;
	for(int i = 0; i < N; ++i)
	{
		result += array[blockIdx.x * N + i];
	}
	return result;
}

__device__ float calculateTotalSum(volatile float *result)
{
	float r = 0;
	for(int i =0; i < gridDim.x; ++i)
	{
		r += result[i];
	}
	return r;
}

__global__ void sum(const float *array, unsigned int N, 
	                volatile float *result, float *totalSum)
{
	float partialSum = calculatePartialSum(array, N);

	if(threadIdx.x == 0)
	{
		result[blockIdx.x] = partialSum;
        
		__threadfence();

		unsigned int value = atomicInc(&count, gridDim.x);

		isLastBlockDone = (value == (gridDim.x - 1));
	}

	__syncthreads();

	if(isLastBlockDone)
	{
		*totalSum = calculateTotalSum(result);

		if(threadIdx.x == 0)
		{
			result[0] = *totalSum;
			count = 0;
		}
	}
}



__global__ void Test(int *A, int *B)
{
	WriteXY();
	ReadXY(A, B);
}



int main()
{
	////define of built-in vector types
	//int2 a = make_int2(1, 2);
	//std::cout << a.x << a.y << std::endl;


	////Test for __threadfence()
	//float *b = new float[10];
	////b = (int *)malloc(10 * sizeof(int))
	//for(int i = 0; i < 10; ++i)
	//{
	//	b[i] = i;
	//}
	//float *d_b;
	//float *d_r;
	//float *d_totalSum;
	//float *c = new float;
	//checkCudaErrors(cudaMalloc((void **)&d_b, 10 * sizeof(float)));
	//checkCudaErrors(cudaMalloc((void **)&d_r, 1 * sizeof(float)));
	//checkCudaErrors(cudaMalloc((void **)&d_totalSum, 1 * sizeof(float)));
	//checkCudaErrors(cudaMemcpy(d_b, b, sizeof(float) * 10, cudaMemcpyHostToDevice));
	//sum<<<5, 1>>>(d_b, 2, d_r, d_totalSum);

	//checkCudaErrors(cudaMemcpy(c, d_totalSum, 1 * sizeof(float), cudaMemcpyDeviceToHost));
	//std::cout << *c << std::endl;



	////for test. not appear errors
	int *A = new int;
	int *B = new int;
	*A = 0;
	*B = 0;
	int *d_A;
	int *d_B;
	checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_B, sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_A, A, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, B, sizeof(int), cudaMemcpyHostToDevice));

	Test<<<1024, 512>>>(d_A, d_B);

	checkCudaErrors(cudaMemcpy(A, d_A, 1 * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(B, d_B, 1 * sizeof(int), cudaMemcpyDeviceToHost));

	std::cout << *A << " " << *B << std::endl;



    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
