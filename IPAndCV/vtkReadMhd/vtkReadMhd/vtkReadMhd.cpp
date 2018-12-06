//
//#include <vtkSmartPointer.h>
//#include <vtkMetaImageReader.h>
//#include <vtkImageViewer2.h>
//#include <vtkRenderer.h>
//#include <vtkRenderWindow.h>
//#include "vtkRenderWindowInteractor.h"
//#include "vtkAutoInit.h" 
////VTK_MODULE_INIT(vtkRenderingOpenGL2); // VTK was built with vtkRenderingOpenGL2
////VTK_MODULE_INIT(vtkInteractionStyle);
//
//int main(int argc, char* argv[])
//
//{
//
//	vtkSmartPointer<vtkMetaImageReader> reader =
//
//		vtkSmartPointer<vtkMetaImageReader>::New();
//
//	reader->SetFileName("F:\\data\\dataTest.mhd");
//
//	reader->Update();
//
//	vtkSmartPointer<vtkImageViewer2> viewer =
//
//		vtkSmartPointer<vtkImageViewer2>::New();
//
//	viewer->SetInputConnection(reader->GetOutputPort());
//
//	//设置基本属性
//
//	viewer->SetSize(640, 480);
//
//	viewer->SetColorLevel(500);
//
//	viewer->SetColorWindow(2000);
//
//	viewer->SetSlice(40);
//
//	viewer->SetSliceOrientationToXY();
//
//	viewer->Render();
//
//	viewer->GetRenderer()->SetBackground(1, 1, 1);
//
//	viewer->GetRenderWindow()->SetWindowName("ImageViewer2D");
//
//	vtkSmartPointer<vtkRenderWindowInteractor> rwi =
//
//		vtkSmartPointer<vtkRenderWindowInteractor>::New();
//
//	//设置交互属性
//
//	viewer->SetupInteractor(rwi);
//
//	rwi->Start();
//
//	return 0;
//
//}


#include <vtkSmartPointer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkImageData.h>
#include <vtkImageMapper3D.h>
#include <vtkImageCast.h>
#include <vtkMetaImageWriter.h>
#include <vtkMetaImageReader.h>
#include <vtkImageMandelbrotSource.h>
#include <vtkImageActor.h>

int main(int, char *[])
{
	std::string filePath = "F:\\data\\dataTest.mhd";
	std::string filePathRaw = "F:\\data\\dataTest.raw";
	//vtkSmartPointer<vtkImageMandelbrotSource> source =   vtkSmartPointer<vtkImageMandelbrotSource>::New();

	//vtkSmartPointer<vtkImageCast> castFilter = 
	//	vtkSmartPointer<vtkImageCast>::New();
	//castFilter->SetOutputScalarTypeToUnsignedChar();
	//castFilter->SetInputConnection(source->GetOutputPort());
	//castFilter->Update();

	//vtkSmartPointer<vtkMetaImageWriter> writer =
	//	vtkSmartPointer<vtkMetaImageWriter>::New();
	//writer->SetInputConnection(castFilter->GetOutputPort());
	//writer->SetFileName(filePath.c_str());
	//writer->SetRAWFileName(filePathRaw.c_str());
	//writer->Write();

	vtkSmartPointer<vtkMetaImageReader> reader = 
		vtkSmartPointer<vtkMetaImageReader>::New();
	reader->SetFileName(filePath.c_str());
	reader->Update();

	vtkSmartPointer<vtkImageActor> actor =
		vtkSmartPointer<vtkImageActor>::New();
	actor->GetMapper()->SetInputConnection(reader->GetOutputPort());

	vtkSmartPointer<vtkRenderer> renderer =
		vtkSmartPointer<vtkRenderer>::New();
	vtkSmartPointer<vtkRenderWindow> renderWindow =
		vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
	vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
		vtkSmartPointer<vtkRenderWindowInteractor>::New();
	renderWindowInteractor->SetRenderWindow(renderWindow); 
	renderer->AddActor(actor);
	renderer->SetBackground(.2, .3, .4);
	renderWindow->Render();
	renderWindowInteractor->Start();
	return 0;
}
