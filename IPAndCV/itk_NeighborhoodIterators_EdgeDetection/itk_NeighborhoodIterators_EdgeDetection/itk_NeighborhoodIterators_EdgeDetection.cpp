/************************************************************************/
/* 
      This program used to implement a simple Sobel edge detection
   algorithm
*/
/**********************************************************************/


#include "itkConstNeighborhoodIterator.h"
#include "itkImageRegionIterator.h" 
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImage.h"

#include "itkImageIOFactory.h"         //necessary, if omitted, program can't create IO factory
#include "itkBMPImageIOFactory.h"   
#include "itkJPEGImageIOFactory.h"
#include "itkPNGImageIOFactory.h"
#include "itkImageIOBase.h"

//using namespace itk;

int main(int argc, char *argv[])
{
	typedef float                             PixelType ;   //float or unsigned char; with different type, we have different result
	typedef itk::Image<PixelType, 2>          ImageType ;
	typedef itk::ImageFileReader< ImageType > ReaderType;

	typedef itk::ConstNeighborhoodIterator< ImageType > NeighborhoodIteratorType;
	typedef itk::ImageRegionIterator< ImageType >       IteratorType;

	typedef unsigned char                          WritePixelType;
	typedef itk::Image< WritePixelType, 2>         WriteImageType;
	typedef itk::ImageFileWriter< WriteImageType > WriterType;

	typedef itk::RescaleIntensityImageFilter< ImageType, WriteImageType> RescaleFilterType;
	//itk::ObjectFactoryBase::RegisterFactory(itk::PNGImageIOFactory::New());
	itk::BMPImageIOFactory::RegisterOneFactory();   
	itk::JPEGImageIOFactory::RegisterOneFactory();
	itk::PNGImageIOFactory::RegisterOneFactory();

	ReaderType::Pointer reader = ReaderType::New();
	char *fileNname = "F:\\ProgramPractice\\IPAndCV\\itk_NeighborhoodIterators_EdgeDetection\\BrainT1Slice.png";
	reader->SetFileName(fileNname);
	//reader->SetImageIO();

	//typedef itk::GDCMImageIO ImageIOType;
	//ImageIOType::Pointer gdcmImageIO = ImageIOType::New();
	//reader->SetImageIO( gdcmImageIO );

	try
	{
		reader->Update();
	}
	catch (itk::ExceptionObject &err)
	{
		std::cerr << "ExceptionObject caught !" << std::endl;
		std::cerr << err << std::endl;
		return EXIT_FAILURE;
	}

	NeighborhoodIteratorType::RadiusType radius;
	radius.Fill(1);
	NeighborhoodIteratorType it(radius, reader->GetOutput(),
		reader->GetOutput()->GetRequestedRegion());

	ImageType::Pointer output = ImageType::New();
	output->SetRegions(reader->GetOutput()->GetRequestedRegion());
	output->Allocate();

	IteratorType out(output, reader->GetOutput()->GetRequestedRegion());

	NeighborhoodIteratorType::OffsetType offset1 = {{-1, -1}};
	NeighborhoodIteratorType::OffsetType offset2 = {{1, -1}};
	NeighborhoodIteratorType::OffsetType offset3 = {{-1, 0}};
	NeighborhoodIteratorType::OffsetType offset4 = {{1, 0}};
	NeighborhoodIteratorType::OffsetType offset5 = {{-1, 1}};
	NeighborhoodIteratorType::OffsetType offset6 = {{1, 1}};

	for (it.GoToBegin(), out.GoToBegin(); !it.IsAtEnd(); ++it, ++out )
	{
		float sum;
		sum = it.GetPixel(offset2) - it.GetPixel(offset1);
		sum += 2 * it.GetPixel(offset4) - 2 * it.GetPixel(offset3);
		sum += it.GetPixel(offset6) - it.GetPixel(offset5);
		out.Set(sum);
	}

	RescaleFilterType::Pointer rescaler = RescaleFilterType::New();
    rescaler->SetOutputMinimum( 0 );
	rescaler->SetOutputMaximum( 255 );
	rescaler->SetInput(output);

	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName("result.jpg");
	writer->SetInput(rescaler->GetOutput());

	try
	{
		writer->Update();
	}
	catch (itk::ExceptionObject &err)
	{
		std::cerr << "ExceptionObject caught !" << std::endl;
		std::cerr << err << std::endl;
		return EXIT_FAILURE;
	}
}


//#include "itkImage.h"
//#include <iostream>
//
//int main()
//{
//	typedef itk::Image< unsigned short, 3 > ImageType;
//
//	ImageType::Pointer image = ImageType::New();
//
//	std::cout << "ITK Hello World !" << std::endl;
//
//	return 0;
//}
