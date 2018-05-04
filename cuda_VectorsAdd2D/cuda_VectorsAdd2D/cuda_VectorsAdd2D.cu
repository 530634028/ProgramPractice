/***********************************************
*
*  This program is used for 2D vector add 
*  date:2018-5-4
*  a:zhonghy
**********************************************/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"


#include <stdio.h>
#include <iostream>
#include <string>

const int N = 32;

//1D blocksPerGrid 2D threadsPerBlock matrix add
__global__ void MatAdd(int **A, int **B, int **C)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	C[i][j] = A[i][j] + B[i][j];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda1D2D(int **A, int **B, int **C, unsigned int size)
{
    int **dev_a = NULL;
    int **dev_b = NULL;
    int **dev_c = NULL;
    cudaError_t cudaStatus;

#ifndef DEBUG
	for(int i = 0; i < 5; ++i)
	{
		for(int j = 0; j < 5; ++j)
		   std::cout << A[i][j] << " ";
	}

#endif

    //// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, N * sizeof(int *));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, N * sizeof(int *));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, N * sizeof(int *));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy((void *)dev_a, (void*)A, N * sizeof(int *), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy((void*)dev_b, (void*)B, N * sizeof(int *), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


	int numBlocks = 1;
	dim3 threadsPerBlock(N, N);
    // Launch a kernel on the GPU with one thread for each element.
    MatAdd<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c);

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
    cudaStatus = cudaMemcpy((void*)C, (void*)dev_c, N * sizeof(int *), cudaMemcpyDeviceToHost);
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

int main(int argc, char *argv[])
{
	//Host memory allocation
	int **A = (int**)malloc(N * sizeof(int*)),
	int	**B = (int**)malloc(N * sizeof(int*)), 
	int	**C = (int**)malloc(N * sizeof(int*));
	int *dataA = (int*)malloc(N * N *sizeof(int));
	int *dataB = (int*)malloc(N * N *sizeof(int));
	int *dataC = (int*)malloc(N * N *sizeof(int));

	//Device memory allocation
	int **dev_A;
	int **dev_B;
	int **dev_C;
    int *dev_dataA;
	int *dev_dataB;
	int *dev_dataC;

	cudaMalloc((void**)(&dev_A), N * sizeof(int *));
	cudaMalloc((void**)(&dev_dataA), N * N * sizeof(int));
	cudaMalloc((void**)(&dev_B), N * sizeof(int *));
	cudaMalloc((void**)(&dev_dataB), N * N * sizeof(int));
	cudaMalloc((void**)(&dev_C), N * sizeof(int *));
	cudaMalloc((void**)(&dev_dataC), N * N * sizeof(int));

	for(int i = 0; i < N * N; ++i)
	{
		dataA[i] = i + 1;
		dataB[i] = i + 1;
		dataC[i] = 0;
	}

	cudaMemcpy((void *)dev_dataA, (void *)dataA, N * N * sizeof(int), cudaMemcpyHostToDevice);  //not sizeof(int)
	cudaMemcpy((void *)dev_dataB, (void *)dataB, N * N * sizeof(int), cudaMemcpyHostToDevice);

	for(int i = 0; i < N; ++i)
	{
		A[i] = dev_dataA + N * i;   //why? A is host, but dev_dataA is device?
		B[i] = dev_dataB + N * i;
		C[i] = dev_dataC + N * i;
	}

	cudaMemcpy((void *)dev_A, (void *)A, N * sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev_B, (void *)B, N * sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev_C, (void *)C, N * sizeof(int *), cudaMemcpyHostToDevice);

	int numBlocks = 1;
	dim3 threadsPerBlock(N, N);
    // Launch a kernel on the GPU with one thread for each element.
    MatAdd<<<numBlocks, threadsPerBlock>>>(dev_A, dev_B, dev_C);

	cudaMemcpy((void *)dataC, (void *)dev_dataC, N * N * sizeof(int), cudaMemcpyDeviceToHost);

	for(int i = 0; i < 5; ++i)
	{
       std::cout << dataC[i] << " ";

	}

	cudaFree((void *)dev_dataC);
	cudaFree((void *)dev_C);
	free(C);
	free(dataC);

	cudaFree((void *)dev_dataB);
	cudaFree((void *)dev_B);
	free(B);
	free(dataB);

	cudaFree((void *)dev_dataA);
	cudaFree((void *)dev_A);
	free(A);
	free(dataA);

    //cudaError_t cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    return 0;
}
