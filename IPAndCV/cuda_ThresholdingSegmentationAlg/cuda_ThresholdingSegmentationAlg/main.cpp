
/****************************************************
  Test for various version of threshold segmentation 
 algorithm with CUDA.
 a:    zhonghy
 date: 2019-4-10
*****************************************************/

#include "cuda_ThresholdingSegmentationAlg.h"
#include "cuda_IntelligenceDilateAlg.h"
#include "cuda_SobelOperationAlg.h"
#include "cuda_LaplacianOperationAlg.h"
#include "zImageIO.h"

#include <iostream>
#include <string>
#include <time.h>


int main(int argc, char **argv)
{
	Mat src, src_gray, result;
	// load
	zImageIO *imageIO = new zImageIO;
	string fileName;
	fileName =  "F:/ProgramPractice/IPAndCV/data/test02.jpg";
	imageIO->ReadImageData(fileName);
	src = imageIO->GetImageData();
	result = Mat::zeros(src.rows, src.cols, CV_8UC1);
	Mat cpuResult = Mat::zeros(src.rows, src.cols, CV_8UC1);

	cvtColor(src, src_gray, COLOR_BGR2GRAY); 

	clock_t timeStart, timeEnd;
	double timeComsume;
	timeStart = clock();

	int erroflag = cuda_ThresholdingSegmentationAlg(src_gray, result, 230);
	//erroflag = cpu_ThresholdingSegmentationAlg(src_gray, cpuResult, 230);

	Mat cpuDilateResult =  Mat::zeros(src.rows, src.cols, CV_8UC1);
	Mat gpuDilateResult = Mat::zeros(src.rows, src.cols, CV_8UC1);
	erroflag = cuda_IntelligenceDilate(src_gray, result, gpuDilateResult, 150, 250, 2);
	erroflag = cpu_IntelligenceDilate(src_gray, result, cpuDilateResult, 150, 250, 2);

	timeEnd = clock();
	std::cout << "GPU Use Time: " <<(double)(timeEnd - timeStart)/CLOCKS_PER_SEC << std::endl;

	Mat tmp = cpuDilateResult - gpuDilateResult;  // result
	//Mat tmp = gpuDilateResult - result;

	Mat lapResult; //= Mat::zeros(src.rows, src.cols, CV_8UC1);  if use CV_8UC1, is not match CV_16S
	LaplacianFilter(src, lapResult);

	cv::imshow("original image", src_gray);
	cv::imshow("thresold_cuda", result);
	cv::imshow("dilate_cuda", tmp);  //dilateResult  tmp gpuDilateResult
	//cv::imshow("sobel cpu", dst);
	cv::imshow("LaplacianFilter for edge detection", lapResult);
	cv::waitKey();

	return EXIT_SUCCESS;
}


////////////////////test for corner detection///////////////////
//#include <iostream>
//#include <string>
//#include <vector>
//#include <map>
//#include<iostream>
//using namespace std;
//using namespace cv;
//
//void main()
//{
//	Mat img = imread("F:/ProgramPractice/IPAndCV/ImagePreprocess/test02.jpg");
//	imshow("src", img);
//	Mat result = img.clone();
//	Mat gray, dst , corner_img;//corner_img存放检测后的角点图像
//	cvtColor(img, gray, COLOR_BGR2GRAY);
//
//	cornerHarris(gray, corner_img, 2, 3, 0.04);//cornerHarris角点检测
//	//imshow("corner", corner_img);
//
//	double minv = 0.0, maxv = 0.0;
//	//double* minp = &minv;
//	//double* maxp = &maxv;
//	cv::minMaxIdx(corner_img, &minv, &maxv);
//	float thresd = 0.01 * maxv;  //max(corner_img.rows)
//
//	threshold(corner_img, dst, thresd, 255, THRESH_BINARY);
//	imshow("dst", dst);
//
//	int rowNumber = gray.rows;  //获取行数
//	int colNumber = gray.cols;  //获取每一行的元素
//	cout << rowNumber << endl;
//	cout << colNumber << endl;
//	cout << dst.type() << endl;
//
//	for (int i = 0; i<rowNumber; i++)
//	{
//		for (int j = 0; j<colNumber; j++)
//		{
//			if (dst.at<float>(i, j) == 255)//二值化后，灰度值为255为角点
//			{
//				circle(result, Point(j, i),3, Scalar(0, 0, 255), 2, 8);
//			}
//		}
//	}
//
//	imshow("result", result);
//	waitKey(0);
//}

///////////////// test for LaplacianFilter ////////////////
//int main(int argc, char *argv[])
//{
//	Mat src, src_gray, dst;
//	// load
//	zImageIO* imageIO = new zImageIO;
//	string fileName;
//	fileName =  "F:/ProgramPractice/IPAndCV/data/test02.jpg";
//	imageIO->ReadImageData(fileName);
//	//imageIO.ReadImageData(fileName);
//	//src = imageIO.GetImageData();
//	src = imageIO->GetImageData();
//
//	if (src.empty())
//	{
//		return -1;
//	}
//
//	LaplacianFilter(src, dst);
//	//CornerHarrisFilter(src, dst);
//
//	//![display]
//	const char* window_name = "Laplace Demo";
//	const char* window = "Origin";
//	imshow(window, src);
//	imshow(window_name, dst);  //abs_dst
//	waitKey();
//
//	delete imageIO;
//	return EXIT_SUCCESS;
//}



/////////////// test for soble operation /////////////////////////
//int main()
//{
//	Mat grayImg = imread("F:/ProgramPractice/IPAndCV/data/test02.jpg", 0);
//
//	int imgHeight = grayImg.rows;  // how much rows in the image
//	int imgWidth = grayImg.cols;   // how much cols in the image
//
//	Mat gaussImg;
//	//高斯滤波
//	GaussianBlur(grayImg, gaussImg, Size(3, 3), 0, 0, BORDER_DEFAULT);
//
//	clock_t timeStart, timeEnd;
//	double timeComsume;
//	timeStart = clock();
//
//	//Sobel算子CPU实现
//	Mat dst(imgHeight, imgWidth, CV_8UC1, Scalar(0));
//	cpu_SobelOperation(gaussImg, dst, imgHeight, imgWidth);
//	timeEnd = clock();
//	std::cout << "CPU Use Time: " <<(double)(timeEnd - timeStart)/CLOCKS_PER_SEC << std::endl;
//
//	//CUDA实现后的传回的图像
//	clock_t cudaTimeStart, cudaTimeEnd;
//	//double timeComsume;
//	cudaTimeStart = clock();
//	Mat dstImg(imgHeight, imgWidth, CV_8UC1, Scalar(0));
//	cuda_SobelOperation(gaussImg, dstImg, imgHeight, imgWidth);
//	cudaTimeEnd = clock();
//	std::cout << "GPU Use Time: " <<(double)(cudaTimeEnd - cudaTimeStart)/CLOCKS_PER_SEC << std::endl;
//
//	Mat tmp = dst - dstImg;
//
//	cv::imshow("original image", grayImg);
//	cv::imshow("sobel cuda", dstImg);
//	cv::imshow("sobel cpu", dst);
//	cv::imshow("sobel cpu - cuda", tmp);
//	cv::waitKey();
//
//	return 0;
//}