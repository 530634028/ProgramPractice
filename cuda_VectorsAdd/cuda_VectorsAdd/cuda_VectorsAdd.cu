/***********************************************
*
*  This program is used for vector add 
*  date:2018-5-2
*  a:zhonghy
**********************************************/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"


#include <stdio.h>
#include <iostream>
#include <string>

//?
//#include <malloc.h>

const int N = 32;

//cudaError_t addWithCuda1D(int *c, const int *a, const int *b, unsigned int size);
//cudaError_t addWithCuda1D2D(int c[N][N], const int a[N][N], const int b[N][N], unsigned int size);
//cudaError_t addWithCuda2D2D(int c[N][N], const int a[N][N], const int b[N][N], unsigned int size);

////1D blocksPerGrid 1D threadsPerBlock
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

//1D blocksPerGrid 2D threadsPerBlock matrix add
__global__ void MatAdd(int *C, int *A, int *B)   //wrong here C, A, B write to A B C
{
	int i = threadIdx.x;
	int j = threadIdx.y;

	//int index = blockDim.x * threadIdx.y + threadIdx.x;
	//C[index] = A[index] + B[index];    //here wrong write C[j * blockDim.x + i] = B[j * blockDim.x + i] + C(need to mend A)[j * blockDim.x + i];

	C[j * blockDim.x + i] = A[j * blockDim.x + i] + B[j * blockDim.x + i];
}

//2D blocksPerGrid 2D threadsPerBlock matrix add
__global__ void MatAdd_2D(int *C, int *A, int *B)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	C[j * blockDim.x * gridDim.x + i] = B[j * blockDim.x * gridDim.x + i] + A[j * blockDim.x * gridDim.x + i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda1D(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    //// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda1D2D(int *c, int *a, int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

//#ifndef DEBUG
//	for(int i = 0; i < 5; ++i)
//	{
//		std::cout << b[i] << " ";
//	}
//
//#endif

    //// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


	int numBlocks = 1;
	dim3 threadsPerBlock(32, 32);
    // Launch a kernel on the GPU with one thread for each element.
    MatAdd<<<numBlocks, threadsPerBlock>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	//for(int i = 0; i < 5; ++i)
	//{
	//	std::cout << c[i] << " ";
	//}


Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;

}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda2D2D(int *c, int *a, int *b, unsigned int size)
{
    int *dev_a = NULL;
    int *dev_b = NULL;
    int *dev_c = NULL;
    cudaError_t cudaStatus;

    //// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int ));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    // Launch a kernel on the GPU with one thread for each element.
    MatAdd_2D<<<numBlocks, threadsPerBlock>>>(dev_c, dev_a, dev_b);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;

}

int main(int argc, char *argv[])
{
	//For 1D vector add
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda1D(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	std::cout << std::endl;



	/******************************************************/
	//For 2D matrix add
	int *A; 
	int *B;
	int *C;
	A = (int*)malloc(N * N * sizeof(int)); 
	B = (int*)malloc(N * N * sizeof(int));
	C = (int*)malloc(N * N * sizeof(int));

	//for(int i = 0; i < N; ++i)
	//{
	//  A[i] = (int*)malloc(sizeof(int) * N);
	//  B[i] = (int*)malloc(sizeof(int) * N);
	//  C[i] = (int*)malloc(sizeof(int) * N);
	//  //std::cout << i << std::endl;
	//}

	//for(int i = 0; i < N; ++i)
	//{
	//	for(int j = 0; j < N; ++j)
	//	{
	//		A[j * N + i] = i + j + 1;
	//		B[j * N + i] = i + j + 1;
	//	}
	//}

	for(int i = 0; i < N * N; ++i)
	{
		A[i] = i  + 1;
		B[i] = i  + 1;
		//std::cout << A[i] << " ";
		//if(i % 5 == 0)
		//	std::cout << std::endl;
	}


	// Add vectors in parallel.
    cudaStatus = addWithCuda1D2D(C, A, B, N * N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    printf("A + B = {%d,%d,%d}\n",
        C[0], C[1], C[100]);
	std::cout << std::endl;

	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }


	// Add vectors in parallel.
    cudaStatus = addWithCuda2D2D(C, A, B, N * N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    printf("A + B = {%d,%d,%d}\n",
        C[0], C[1], C[20]);



    return 0;
}
