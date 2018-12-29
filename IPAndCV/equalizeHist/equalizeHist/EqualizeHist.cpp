#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

/**  @function main */
int main( int argc, char** argv )
{
	Mat src, dst;

	char* source_window = "Source image";
	char* equalized_window = "Equalized Image";

	/// 加载源图像
	std::string  filePath = "F:/ProgramPractice/IPAndCV/ConnectedComponentAnalysis-Labeling/test02.jpg";
	src = imread( filePath.c_str(), 1 );

	if( !src.data )
	{ cout<<"Usage: ./Histogram_Demo <path_to_image>"<<endl;
	return -1;}

	/// 转为灰度图
	cvtColor( src, src, COLOR_BGR2GRAY );

	/// 应用直方图均衡化
	equalizeHist( src, dst );

	/// 显示结果
	namedWindow( source_window, WINDOW_AUTOSIZE );
	namedWindow( equalized_window, WINDOW_AUTOSIZE );

	imshow( source_window, src );
	imshow( equalized_window, dst );

	/// 等待用户按键退出程序
	waitKey(0);

	return 0;
}





