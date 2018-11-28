/*********************************************
*
*  Use to test shared memory in matrix multiplication.
* 
*  a   :zhonghy
*  date:2018-4-17
*
**********************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"   //for checkCudaErrors() function

#include <iostream>
#include <string>
#include <stdio.h>


//Matrices are stored in row-major order:
//M(row, col) = *(M.elements + row * M.width + col) ->M.elements is
//first pointer?

typedef struct {
	int width;
	int height;
	int stride;
	float *elements;
} Matrix;

//Thread block size
#define BLOCK_SIZE 16

/*   without shared memory  */
//Forward declaration of the matrix multiplication kernel
//Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C)
{
	//Each thread computes one element of C
	//by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y; //the threads'(y,x) direction is opposite to matrix(x,y)
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for(int e = 0; e < A.width; ++e )
	{
		Cvalue += A.elements[row * A.width + e]
		* B.elements[e * B.width + col];
		C.elements[row * C.width + col] = Cvalue;
	}

}

//Matrix multiplication - Host code
//Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void Matmul(const Matrix A, const Matrix B, Matrix C)
{
	//Load A and B to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	checkCudaErrors(cudaMalloc(&d_A.elements, size));
	checkCudaErrors(cudaMemcpy(d_A.elements, A.elements, size, 
		cudaMemcpyHostToDevice));

	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	checkCudaErrors(cudaMalloc(&d_B.elements, size));
	checkCudaErrors(cudaMemcpy(d_B.elements, B.elements, size, 
		cudaMemcpyHostToDevice));

	//Allocate C in device memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	checkCudaErrors(cudaMalloc(&d_C.elements, size));

	//Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y); //?
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	//Read C from device memory
	checkCudaErrors(cudaMemcpy(C.elements, d_C.elements, size, 
		cudaMemcpyDeviceToHost));

	//free device memory
	checkCudaErrors(cudaFree(d_A.elements));
	checkCudaErrors(cudaFree(d_B.elements));
	checkCudaErrors(cudaFree(d_C.elements));
}



/*  with shared memory */
//Get a matrix element, excute on device (__device__ qualifier)
__device__ float GetElement(const Matrix A, int row, int col)
{
	return A.elements[row * A.stride + col];
}

//Set A matrix element , stride represent the width of Csub?
__device__ float SetElement(Matrix A, int row, int col, float value)
{
	A.elements[row * A.stride + col] = value;
}

//Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
//loacted col sub-matrices to the right and row sub-matrices down
//from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
	Matrix Asub;
	Asub.width    = BLOCK_SIZE;
	Asub.height   = BLOCK_SIZE;
	Asub.stride   = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return Asub;
}

// Matrix multiplication kernel called by MatMulWithSharedMem
__global__ void MatMulSharedMemKernel(Matrix A, Matrix B, Matrix C)
{
	//Block row and column
	int blockRow = blockIdx.y;  //wrong here, data is opsite to block zhonghy
	int blockCol = blockIdx.x;

	//Each thread block computes one sub-matrix Csub of C
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);  //move to the current block zhonghy

	//Each thread computes one element of Csub
	//by accumulating results into Cvalue
	float Cvalue = 0;

	//Thread row and column within Csub
	int row = threadIdx.y;
	int col = threadIdx.x;

	//Loop pver all the sub-matrices of A and B that are
	//required to compute Csub
	//Multiply each pair of sub-matrices together
	//and accumulate the results
	for(int m = 0; m < (A.width / BLOCK_SIZE); ++m) //m is number of Asub(divided by blocksize zhonghy)
	{
		//Get sub-matrix Asub of A
		Matrix Asub = GetSubMatrix(A, blockRow, m);

		//Get sub-matrix Bsub of B ?
		Matrix Bsub = GetSubMatrix(B, m, blockCol);

		//Shared memory used to store Asub and Bsub respectively
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		//Load Asub and Bsub from device memory to shared memory
		//Each thread loads one element of each sub-matrix(important)
		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);

		//Synchronize to make sure the sub-matrices aire loaded
		//before starting the computation
		__syncthreads();

		//Multiply Asub and Bsub together
		for(int e = 0; e < BLOCK_SIZE; ++e)
			Cvalue += As[row][e] * Bs[e][col];

		//Synchronize to make sure that preceding
		//computation is done before loading two new 
		//sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	//Write Csub to device memory
	//Each thread writes one element
	SetElement(Csub, row, col, Cvalue);
}

//Matrix multiplication - Host code
//Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMulWithSharedMem(const Matrix A, const Matrix B, Matrix C)
{
	//Load A and B to device memory
	Matrix d_A;
	d_A.width = d_A.stride = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	checkCudaErrors(cudaMalloc(&d_A.elements, size));
	checkCudaErrors(cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice));

	Matrix d_B;
	d_B.width = d_B.stride = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	checkCudaErrors(cudaMalloc(&d_B.elements, size));
	checkCudaErrors(cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice));

	//Allocate C in device memory
	Matrix d_C;
	d_C.width = d_C.stride = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	checkCudaErrors(cudaMalloc(&d_C.elements, size));

	//invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / BLOCK_SIZE, A.height / BLOCK_SIZE); //number of blocks
	MatMulSharedMemKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	//Read C from device memory
	checkCudaErrors(cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_A.elements));
	checkCudaErrors(cudaFree(d_B.elements));
	checkCudaErrors(cudaFree(d_C.elements));
}




int main(int argc, char *argv[])
{
	//Host allocate for A B C
	Matrix A;
	A.width = 1024;    //A.width less than B.width. program work? 
	A.height = 1024;
	//A.elements = (float*)malloc(A.width * A.height * sizeof(float));
	A.elements = new float[A.width * A.height];
	for(int i = 0; i < A.width; ++i)
	{
		float *p = A.elements + i * A.height;
		for(int j = 0; j < A.height; ++j)
		{
			*(p + j) = 1;
		}
	}
	
	Matrix B;
	B.width = 1024;
	B.height = 1024;
	//B.elements = (float*)malloc(B.width * B.height * sizeof(float));
	B.elements = new float[B.width * B.height];
	for(int i = 0; i < B.width; ++i)
	{
		float *p = B.elements + i * B.height;
		for(int j = 0; j < B.height; ++j)
		{
			*(p + j) = 2;
		}
	}

	Matrix C;
	C.width = B.width;  //wrong here previous,written like C.width = A.height
	C.height = A.height;
	//C.elements = (float*)malloc(C.width * C.height * sizeof(float));
	C.elements = new float[C.width * C.height];

	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));
	//invoke MatMul function, without shared memory
	Matmul(A, B, C);
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));  //if have not this line, program get wrong result
	float time;
	checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
	//checkCudaErrors(cudaEventDestroy(start));
	//checkCudaErrors(cudaEventDestroy(stop));
	std::cout << "Time consumption without shared memory is: " <<
		time << "ms" << std::endl;


	for(int i = 0; i < 5; ++i)
	{
		float *p = C.elements + i * C.height;
		for(int j = 0; j < C.height; ++j)
		{
			std::cout << *(p + j) << " ";
		}
		std::cout << std::endl;
	}
	std:: cout << C.width << "  " << C.height << std::endl;


	std::cout << std::endl;
    //invoke function with shared memory
	checkCudaErrors(cudaEventRecord(start, 0));
	MatMulWithSharedMem(A, B, C);
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));  //if have not this line, program get wrong result
	//float time;
	checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	std::cout << "Time consumption with shared memory is: " <<
		time << "ms" << std::endl;


	for(int i = 0; i < 5; ++i)
	{
		float *p = C.elements + i * C.height;
		for(int j = 0; j < C.height; ++j)
		{
			std::cout << *(p + j) << " ";
		}
		std::cout << std::endl;
	}
	std:: cout << C.width << "  " << C.height << std::endl;


	//free(A.elements);
	//free(B.elements);
	//free(C.elements);
	delete[] A.elements;
	delete[] B.elements;
	delete[] C.elements;
    return 0;
}