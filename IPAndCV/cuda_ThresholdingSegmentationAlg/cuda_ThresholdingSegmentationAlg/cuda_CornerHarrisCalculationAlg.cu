





#include "cuda_CornerHarrisCalculationAlg.h"

void cpu_CornerHarrisFilter(Mat &imageInput, Mat &imageOutput)
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

