
/****************************************************
    Header file of Sobel Operation algorithm
  CUDA kernel for 2D image
  a: zhonghy  
  date: 2019-4-26
*****************************************************/

#ifndef cuda_SobelOperationAlg_H
#define cuda_SobelOperationAlg_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <device_functions.h>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;

__global__ void cuda_SobelOperationKernel(unsigned char *, unsigned char *, int , int);
void cpu_SobelOperation(Mat , Mat , int , int );
void cuda_SobelOperation(Mat srcImg, Mat dstImg, int imgHeight, int imgWidth);

#endif
