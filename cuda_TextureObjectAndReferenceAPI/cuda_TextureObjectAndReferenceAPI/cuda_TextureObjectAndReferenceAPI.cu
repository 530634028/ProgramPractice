
/******************************************************
*
* Program used to test texture object and reference API
*
* a   : zhonghy
* date: 2018-4-24
********************************************************/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>



//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void transformKernel(float *output, cudaTextureObject_t texObj,
	                            int width, int height, float theta)
{

}

int main(int argc, char *argv[])
{


    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
