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

	/// ����Դͼ��
	std::string  filePath = "F:/ProgramPractice/IPAndCV/ConnectedComponentAnalysis-Labeling/test02.jpg";
	src = imread( filePath.c_str(), 1 );

	if( !src.data )
	{ cout<<"Usage: ./Histogram_Demo <path_to_image>"<<endl;
	return -1;}

	/// תΪ�Ҷ�ͼ
	cvtColor( src, src, COLOR_BGR2GRAY );

	/// Ӧ��ֱ��ͼ���⻯
	equalizeHist( src, dst );

	/// ��ʾ���
	namedWindow( source_window, WINDOW_AUTOSIZE );
	namedWindow( equalized_window, WINDOW_AUTOSIZE );

	imshow( source_window, src );
	imshow( equalized_window, dst );

	/// �ȴ��û������˳�����
	waitKey(0);

	return 0;
}





