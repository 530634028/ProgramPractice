
/*****************************************
  This program is used to Read and Write 
  images.
  author: zhonghy
  date:   2018-12-5

******************************************/

#ifndef __ZIMAGEIO_H__
#define __ZIMAGEIO_H__

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <map>

using namespace cv;
using namespace std;

typedef struct ImageInfo
{
	int xDim;
	int yDim;
	int sliceDim;
	enum   imageType{JPEG, DICOM, TIFF};
}ImageDim;

class zImageIO
{

public:
	zImageIO()
	{
	   m_imageInfo.xDim = 0;
	   m_imageInfo.yDim = 0;
	   m_imageInfo.sliceDim = 0;
	   //m_imageData = Mat::zeros(m_imageInfo.xDim, m_imageInfo.yDim, CV_16S);
	};
	~zImageIO(){};
	int SetImageData(cv::Mat &src);
    int ReadImageData(std::string fileName);
	Mat &GetImageData(){ return m_imageData; };

private:
	cv::Mat   m_imageData;
	ImageDim  m_imageInfo;
};




#endif

