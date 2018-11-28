
/************************************************
*
*  Test for printf function!
*  date:2018-5-16
*  a   :zhonghy
*
*************************************************/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


__global__ void helloCUDA(float f)
{
	printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}

__global__ void helloCUDAZero(float f)
{
	if( 0 == threadIdx.x)
		printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}

int main(int argc, char *argv[])
{

  //helloCUDA<<<1, 5>>>(1.2345f);
  //cudaDeviceSynchronize();

  helloCUDAZero<<<1, 5>>>(1.2345f);
  cudaDeviceSynchronize();

  return EXIT_SUCCESS;
}
