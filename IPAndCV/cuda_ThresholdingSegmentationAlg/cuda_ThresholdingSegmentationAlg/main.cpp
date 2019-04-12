
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

	int erroflag = cuda_ThresholdingSegmentationAlg(src_gray, result, 200);
	//int erroflag = cpu_ThresholdingSegmentationAlg(src_gray, result);

	timeEnd = clock();
	std::cout << "GPU Use Time: " <<(double)(timeEnd - timeStart)/CLOCKS_PER_SEC << std::endl;

	cv::imshow("original image", src_gray);
	cv::imshow("thresold_cuda", result);
	//cv::imshow("sobel cpu", dst);
	cv::waitKey();

	return EXIT_SUCCESS;
}
