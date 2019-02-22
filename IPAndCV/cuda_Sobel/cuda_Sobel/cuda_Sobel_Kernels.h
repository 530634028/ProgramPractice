

#ifndef CUDA_SOBEL_KERNELS_H
#define CUDA_SOBEL_KERNELS_H

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

__global__ void sobelInCuda(unsigned char *, unsigned char *, int , int);
void sobel(Mat , Mat , int , int );
void sobelCuda(Mat srcImg, Mat dstImg, int imgHeight, int imgWidth);

#endif
