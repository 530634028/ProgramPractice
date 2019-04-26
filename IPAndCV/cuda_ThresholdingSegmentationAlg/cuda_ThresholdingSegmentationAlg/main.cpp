
/****************************************************
  Test for various version of threshold segmentation 
 algorithm with CUDA.
 a:    zhonghy
 date: 2019-4-10
*****************************************************/
#include "cuda_ThresholdingSegmentationAlg.h"
#include "cuda_IntelligenceDilateAlg.h"
#include "zImageIO.h"

#include <iostream>
#include <string>
#include <time.h>


//// write information into specified file
//void log_print(const char *filename, const char *str)   //__declspec(dllexport) 
//{
//	FILE *fp = fopen(filename,"a");//"log_gpu.txt"
//	fprintf(fp,"%s",str);
//	fclose(fp);
//}

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

	cv::imshow("original image", src_gray);
	cv::imshow("thresold_cuda", result);
	cv::imshow("dilate_cuda", tmp);  //dilateResult  tmp gpuDilateResult
	//cv::imshow("sobel cpu", dst);
	cv::waitKey();

	return EXIT_SUCCESS;
}
