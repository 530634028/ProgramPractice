/****************************
*
* Used to Unified Memory Programming
* a: zhonghy
* date:2018-6-26
*
******************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <stdio.h>
#include <iostream>


//__global__ void AplusB(int *ret, int a, int b)
//{
//	ret[threadIdx.x] = a + b + threadIdx.x;
//}

//managed variables
//__device__ __managed__ int ret[1000];
//__global__ void AplusB(int a, int b)
//{
//	ret[threadIdx.x] = a + b + threadIdx.x;
//}

//__global__ void printme(char *str)
//{
//	printf(str);
//}

//J 2.1.2
//__device__ __managed__ int x[2];
//__device__ __managed__ int y;
//__global__ void kernel()
//{
//	x[1] = x [0] + y;
//}

//J 2.2.1
__device__ __managed__ int x, y = 2;
__global__ void kernel()
{
	x = 10;
}



int main()
{
	/*1***without of unified memory****/ 
	//int *ret;
	//cudaMalloc(&ret, 1000 * sizeof(int));
	//AplusB<<<1, 1000>>>(ret, 10, 100);
	//int *host_ret = (int *)malloc(1000 * sizeof(int));
	//cudaMemcpy(host_ret, ret, 1000 * sizeof(int),
	//	cudaMemcpyDeviceToHost);
	//for(int i = 0; i < 1000; ++i)
	//{
	//	printf("%d: A+B = %d\n", i, host_ret[i]);
	//}
	//free(host_ret);
	//cudaFree(ret);

	/*2***with managed memory****/
	//2.1cudaMallocManaged allocate, need which comput capability
	//int *ret;
	//cudaMallocManaged(&ret, 1000 * sizeof(int));  //remind
	//AplusB<<<1, 1000>>>(ret, 10, 100);
	//cudaDeviceSynchronize();
	//for(int i = 0; i < 1000; ++i)
	//{
	//	printf("%d: A+B = %d\n", i, ret[i]);
	//}
	//cudaFree(ret);
	
	//2.2managed variables
	//AplusB<<<1, 1000>>>(10, 100);
	//cudaDeviceSynchronize();
	//for(int i = 0; i < 1000; ++i)
	//{
	//	printf("%d: A+B = %d\n", i, ret[i]);
	//}

	/*J 2.1.1***used managed memory****/
	//// Allocate 100 bytes of memory, accessible to both host and device
	//char *s;
	//cudaMallocManaged(&s, 100);
	//// Note direct Host-code use of "s"
	//strncpy(s, "Hello Unified Memory\n", 99);
	//// Here we pass "s" to a kernel without explicitly copying
	//printme<<<1, 1>>>(s);
	//cudaDeviceSynchronize();
	//// Free as for normal CUDA allocations
	//cudaFree(s);

	//J 2.1.2
	//x[0] = 3;
	//y = 5;
	//kernel<<<1, 1>>>();
	//cudaDeviceSynchronize();
	//printf("result = %d\n", x[1]);

	//J 2.2.1
	//kernel<<<1, 1>>>();
	//cudaDeviceSynchronize();
	//y = 20;
	////cudaDeviceSynchronize();

	//J 2.2.2
	//cudaStream_t stream1, stream2;
	//cudaStreamCreate(&stream1);
	//cudaStreamCreate(&stream2);
	//int *non_managed, *managed, *also_managed;
	//cudaMalloc(&non_managed, 4);
	//cudaMallocManaged(&managed, 4);
	//cudaMallocManaged(&also_managed, 4);
 //   // Point1: CPU can access non-managed data
	//kernel<<<1, 1, 0, stream1>>>(managed);
	//*non_managed = 1;
	//// Point2: CPU cannot access any managed data while GPU is busy,
	//// unless concurrentManagedAccess = 1
	//// Note we have not yet synchronized, so "kernel" is still active
	//*also_managed = 2; 
	//// Point3: Concurrent GPU kernels can access the same data
	//kernel<<<1, 1, 0, stream2>>>(managed);
	//// Point4: Multi-GPU concurrent access is also premitted
	//cudaSetDevice(1);
	//kernel<<<1, 1>>>(managed);

	//J 2.2.4
	//cudaStream_t stream1;
	//cudaStreamCreate(&stream1);
	//cudaStreamAttachMemAsync(stream1, &y, 0, cudaMemAttachHost);
	//cudaDeviceSynchronize();      // Wait for Host attachment to occur
 //   kernel<<<1, 1, 0, stream1>>>();// Note: Launches into stream1
	//y = 20;                        // Success - a kernel is running but y
	//                              // has been associated with no stream

	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	cudaStreamAttachMemAsync(stream1, &x);
	cudaDeviceSynchronize();
	kernel<<<1, 1, 0, stream1>>>();
	y = 20;                          //error: y is still associated globally
	                                 //with all streams by default

    return EXIT_SUCCESS;
}
