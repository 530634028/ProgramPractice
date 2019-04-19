
/****************************************************
  Test for various version of threshold segmentation 
 algorithm with CUDA.
 a:    zhonghy
 date: 2019-4-10
*****************************************************/

#include <iostream>
#include <string>
#include "zImageIO.h"
#include "cuda_ThresholdingSegmentationAlg.h"

#include <time.h>

int main(int argc, char **argv)
{
	Mat src, src_gray, result;
	// load
	zImageIO* imageIO = new zImageIO;
	string fileName;
	fileName =  "F:/ProgramPractice/IPAndCV/data/test02.jpg";
	imageIO->ReadImageData(fileName);
	src = imageIO->GetImageData();
	result = Mat::zeros(src.rows, src.cols, CV_8UC1);

	cvtColor(src, src_gray, COLOR_BGR2GRAY); 

	clock_t timeStart, timeEnd;
	double timeComsume;
	timeStart = clock();

	int erroflag = cuda_ThresholdingSegmentationAlg(src_gray, result, 230);
	//int erroflag = cpu_ThresholdingSegmentationAlg(src_gray, result);
	Mat cpuDilateResult =  Mat::zeros(src.rows, src.cols, CV_8UC1);
	Mat gpuDilateResult = Mat::zeros(src.rows, src.cols, CV_8UC1);
	erroflag = cpu_IntelligenceDilate(src_gray, result, cpuDilateResult, 150, 250, 3);
	erroflag = cuda_IntelligenceDilate(src_gray, result, gpuDilateResult, 150, 250, 3);


	timeEnd = clock();
	std::cout << "GPU Use Time: " <<(double)(timeEnd - timeStart)/CLOCKS_PER_SEC << std::endl;

	Mat tmp = cpuDilateResult - gpuDilateResult;  // result

	cv::imshow("original image", src_gray);
	cv::imshow("thresold_cuda", result);
	cv::imshow("dilate_cuda", tmp);  //dilateResult
	//cv::imshow("sobel cpu", dst);
	cv::waitKey();

	return EXIT_SUCCESS;
}
