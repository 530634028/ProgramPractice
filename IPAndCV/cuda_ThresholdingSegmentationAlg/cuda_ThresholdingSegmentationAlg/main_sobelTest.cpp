
/****************************************************
  

 a:   
 date: 
*****************************************************/

#include "cuda_SobelOperationAlg.h"
#include "time.h"


int main()
{
	Mat grayImg = imread("F:/ProgramPractice/IPAndCV/data/test02.jpg", 0);

	int imgHeight = grayImg.rows;  // how much rows in the image
	int imgWidth = grayImg.cols;   // how much cols in the image

	Mat gaussImg;
	//高斯滤波
	GaussianBlur(grayImg, gaussImg, Size(3, 3), 0, 0, BORDER_DEFAULT);

	clock_t timeStart, timeEnd;
	double timeComsume;
	timeStart = clock();

	//Sobel算子CPU实现
	Mat dst(imgHeight, imgWidth, CV_8UC1, Scalar(0));
	sobel(gaussImg, dst, imgHeight, imgWidth);
	timeEnd = clock();
	std::cout << "CPU Use Time: " <<(double)(timeEnd - timeStart)/CLOCKS_PER_SEC << std::endl;

	//CUDA实现后的传回的图像
	clock_t cudaTimeStart, cudaTimeEnd;
	//double timeComsume;
	cudaTimeStart = clock();
	Mat dstImg(imgHeight, imgWidth, CV_8UC1, Scalar(0));
	sobelCuda(gaussImg, dstImg, imgHeight, imgWidth);
	cudaTimeEnd = clock();
	std::cout << "GPU Use Time: " <<(double)(cudaTimeEnd - cudaTimeStart)/CLOCKS_PER_SEC << std::endl;

	Mat tmp = dst - dstImg;

	cv::imshow("original image", grayImg);
	cv::imshow("sobel cuda", dstImg);
	cv::imshow("sobel cpu", dst);
	cv::imshow("sobel cpu - cuda", tmp);
	cv::waitKey();

	return 0;
}