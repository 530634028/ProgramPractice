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
	float * elements;
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


int main(int argc, char *argv[])
{
	//Host allocate for A B C
	Matrix A;
	A.width = 4;    //A.width less than B.width. program work?
	A.height = 4;
	A.elements = (float*)malloc(A.width * A.height * sizeof(float));
	for(int i = 0; i < A.width; ++i)
	{
		float *p = A.elements + i * A.height;
		for(int j = 0; j < A.height; ++j)
		{
			*(p + j) = 1;
		}
	}
	
	Matrix B;
	B.width = 4;
	B.height = 4;
	B.elements = (float*)malloc(B.width * B.height * sizeof(float));
	for(int i = 0; i < B.width; ++i)
	{
		float *p = B.elements + i * B.height;
		for(int j = 0; j < B.height; ++j)
		{
			*(p + j) = 2;
		}
	}

	Matrix C;
	C.width = A.height;
	C.height = B.width;
	C.elements = (float*)malloc(C.width * C.height * sizeof(float));

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
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
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



	//invoke function with shared memory












	free(A.elements);
	free(B.elements);
	free(C.elements);
    return 0;
}
