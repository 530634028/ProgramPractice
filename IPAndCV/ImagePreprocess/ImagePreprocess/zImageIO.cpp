
/*****************************************
  Class zImageIO implement file. 
  images.
  author: zhonghy
  date:   2018-12-5

******************************************/

#include "zImageIO.h"

int zImageIO::ReadImageData(std::string fileName)
{
	m_imageData = imread(fileName, IMREAD_COLOR);
	if (m_imageData.empty())
	{
		return -1;
	}
	else
	{
	   return 1;
	}

}

