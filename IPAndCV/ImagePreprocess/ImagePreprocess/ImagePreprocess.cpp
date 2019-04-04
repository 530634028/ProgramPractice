
/************************************************************************/
/* 



*/
/************************************************************************/



#include "zImageIO.h"


// detect edge by Laplacian operator
void LaplacianFilter(Mat &imageInput, Mat &imageOutput)
{
	int kernel_size = 3;  //3
	int scale = 1;
	double delta = 0.5;
	int ddepth = CV_16S;

	Mat src_gray;
	/// Reduce noise by blurring with a Gaussian filter
	GaussianBlur(imageInput, imageInput, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//convert to gray
	cvtColor(imageInput, src_gray, COLOR_BGR2GRAY); 
	/// Apply Laplace function
	Mat abs_dst, dst;
	//laplacian
	Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	//convert
	convertScaleAbs(dst, imageOutput);
}

void CornerHarrisFilter(Mat &imageInput, Mat &imageOutput)
{
	double thresh = 0.01; // current threshold
	Mat harris_dst, harris_dst_r;
	Mat src_gray;                      // because src_gray is gray image
	Mat result = imageInput.clone();

	cvtColor(imageInput, src_gray, COLOR_BGR2GRAY);
	cornerHarris(src_gray, harris_dst, 2, 3, 0.04, BORDER_DEFAULT);

	double minv = 0.0, maxv = 0.0;
	cv::minMaxIdx(harris_dst, &minv, &maxv);
	double thresd = 0.01 * maxv;       // max(corner_img.rows)
	threshold(harris_dst, harris_dst_r, thresd, 255, THRESH_BINARY);
 
	int rowNumber = src_gray.rows;  
	int colNumber = src_gray.cols;  
	for (int i = 0; i < rowNumber; i++)
	{
		for (int j = 0; j < colNumber; j++)
		{
			if (harris_dst_r.at<float>(i, j) == 255)
			{
				circle(result, Point(j, i), 3, cv::Scalar(0, 0, 255), 2, 8);
			}
		}
	}
	imageOutput = result;
}


int main(int argc, char *argv[])
{
	Mat src, src_gray, dst;
	// load
	zImageIO* imageIO = new zImageIO;
	string fileName;
	fileName =  "F:/ProgramPractice/IPAndCV/ImagePreprocess/test02.jpg";
	imageIO->ReadImageData(fileName);
	//imageIO.ReadImageData(fileName);
	//src = imageIO.GetImageData();
	src = imageIO->GetImageData();

	if (src.empty())
	{
		return -1;
	}

	LaplacianFilter(src, dst);
	//CornerHarrisFilter(src, dst);

	//![display]
	const char* window_name = "Laplace Demo";
	const char* window = "Origin";
	imshow(window, src);
	imshow(window_name, dst);  //abs_dst
	waitKey();

	delete imageIO;
	return EXIT_SUCCESS;
}

//#include "opencv2/imgproc.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//
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