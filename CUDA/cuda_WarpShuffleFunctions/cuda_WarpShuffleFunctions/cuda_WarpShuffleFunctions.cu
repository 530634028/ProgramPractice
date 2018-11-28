/*****************************
*
* Examples of CUDA warp shuffle functions
     __shfl(), __shfl_up(), __shfl_down()
* date: 2018-5-15
* a   : zhonghy
*
******************************/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <stdio.h>
#include <iostream>


//use __shfl() implement broadcast of a single value across a warp
__global__ void bcast(int arg)
{
	int laneId = threadIdx.x &0x1f;
	int value;
	if(laneId == 0 ) //note unused variable for
		value = arg; // all threads except lane 0
	value = __shfl(value, 0); //Get "value" from lane 0
	if(value != arg)
		printf("Thread %d failed.\n", threadIdx.x);
	//printf("Thread %d Succeed.%d\n", threadIdx.x, value);
}

//use __shfl_up() implement inclusive plus-scan across
//sub-partitions of 8 threads
__global__ void scan4()
{
	int laneId = threadIdx.x & 0x1f;
	//Seed sample starting value(inverse of lane ID)
	int value = 31 - laneId;

	//Loop to accumulate scan within my partition.
	//Scan requires log2(n) == 3 steps for 8 threads
	//It works by an accumulated sum up the warp 
	//by 1, 2, 4, 8 etc. steps
	for(int i = 1; i <= 4; i *= 2)
	{
		//Note: shfl requires all threads being
		//accessed to be active.Therefore we do
		//the __shfl unconditionally so that we
		//can read even from threads which won't do a
		//sum, and then conditionally assign thre result
		int n = __shfl_up(value, i, 8);
		if(laneId >= i)
			value += n;
	}
	printf("Thread %d final value = %d\n", threadIdx.x, value);
}

//use __shfl_down() to implement reduction across warp
__global__ void warpReduce()
{
	int laneId = threadIdx.x & 0x1f;
	int value = 31 - laneId;

	//Use XOR mode to perform butterfly reduction
	for(int i = 16; i >= 1; i /= 2)
	{
		value += __shfl_xor(value, i, 32);
	}
	printf("Thread %d final value = %d\n", threadIdx.x, value);
}



int main()
{
	////bcast
	//bcast<<<1, 32>>>(1234);
	//cudaDeviceSynchronize();

	////scan4
	//scan4<<<1, 32>>>();
	//cudaDeviceSynchronize();

	//warpReduce
	warpReduce<<<1, 32>>>();
	cudaDeviceSynchronize();

    return 0;
}
