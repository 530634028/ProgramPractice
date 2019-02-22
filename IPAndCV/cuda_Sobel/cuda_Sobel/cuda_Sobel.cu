
#include "cuda_Sobel_Kernels.h"

//Sobel算子边缘检测CPU函数
void sobel(Mat srcImg, Mat dstImg, int imgHeight, int imgWidth)
{
    int Gx = 0;
    int Gy = 0;
    for (int i = 1; i < imgHeight - 1; i++)
    {
        uchar *dataUp = srcImg.ptr<uchar>(i - 1);
        uchar *data = srcImg.ptr<uchar>(i);
        uchar *dataDown = srcImg.ptr<uchar>(i + 1);
        uchar *out = dstImg.ptr<uchar>(i);
        for (int j = 1; j < imgWidth - 1; j++)
        {
            Gx = (dataUp[j + 1] + 2 * data[j + 1] + dataDown[j + 1]) - (dataUp[j - 1] + 2 * data[j - 1] + dataDown[j - 1]);
            Gy = (dataUp[j - 1] + 2 * dataUp[j] + dataUp[j + 1]) - (dataDown[j - 1] + 2 * dataDown[j] + dataDown[j + 1]);
            out[j] = (abs(Gx) + abs(Gy)) / 2;
        }
    }
}

//CUDA version of Sobel Operator for edge detection
void sobelCuda(Mat srcImg, Mat dstImg, int imgHeight, int imgWidth)
{
	//创建GPU内存
	unsigned char *d_in;
	unsigned char *d_out;

	cudaMalloc((void**)&d_in, imgHeight * imgWidth * sizeof(unsigned char));
	cudaMalloc((void**)&d_out, imgHeight * imgWidth * sizeof(unsigned char));

	//将高斯滤波后的图像从CPU传入GPU
	cudaMemcpy(d_in, srcImg.data, imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(64, 64);   //32 32
	dim3 blocksPerGrid((imgWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

	//调用核函数
	sobelInCuda <<< blocksPerGrid, threadsPerBlock >>>(d_in, d_out, imgHeight, imgWidth);

	//将图像传回GPU
	cudaMemcpy(dstImg.data, d_out, imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	//释放GPU内存
	cudaFree(d_in);
	cudaFree(d_out);
}


//Sobel算子边缘检测核函数
__global__ void sobelInCuda(unsigned char *dataIn, unsigned char *dataOut, int imgHeight, int imgWidth)
{
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int index = yIndex * imgWidth + xIndex;
    int Gx = 0;
    int Gy = 0;

    if (xIndex > 0 && xIndex < imgWidth - 1 && yIndex > 0 && yIndex < imgHeight - 1)
    {
        Gx = dataIn[(yIndex - 1) * imgWidth + xIndex + 1] + 2 * dataIn[yIndex * imgWidth + xIndex + 1] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]
            - (dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[yIndex * imgWidth + xIndex - 1] + dataIn[(yIndex + 1) * imgWidth + xIndex - 1]);
        Gy = dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex - 1) * imgWidth + xIndex] + dataIn[(yIndex - 1) * imgWidth + xIndex + 1]
            - (dataIn[(yIndex + 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex + 1) * imgWidth + xIndex] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]);
        dataOut[index] = (abs(Gx) + abs(Gy)) / 2;
    }
}
