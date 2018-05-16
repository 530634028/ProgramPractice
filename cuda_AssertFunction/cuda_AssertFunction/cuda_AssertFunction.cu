
/************************************************
*
*  Test for assert function!
*  date:2018-5-16
*  a   :zhonghy
*
*************************************************/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include <assert.h>

#include <stdio.h>
#include <iostream>

__global__ void testAssert(void)
{
	int is_one = 1;
	int should_be_one = 0;

	//This will have no affect
	assert(is_one);

	//This will halt kernel execution
	assert(should_be_one);
}

int main(int argc, char *argv[])
{
    testAssert<<<1,1>>>();
	cudaDeviceSynchronize();
    return EXIT_SUCCESS;
}
