
// VTK-test.cpp : 定义控制台应用程序的入口点。
//


#include <vtkVersion.h>
#include <vtkPlaneSource.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
//解决no override found for vtkpolydatamapper
//#include <vtkAutoInit.h>
//VTK_MODULE_INIT(vtkRenderingOpenGL2)

//#include <iostream>
#include <vtkAutoInit.h>   
//VTK_MODULE_INIT(vtkRenderingOpenGL);  //自己的PCL1.8.0安装后产生的是OpenGL2,所以这一句需要改成OpenGL2.
VTK_MODULE_INIT(vtkRenderingOpenGL2);  //
VTK_MODULE_INIT(vtkInteractionStyle); //外部依赖项添加opengl32.lib（我自己的还要添加vfw32.lib你的不知道要不要）
VTK_MODULE_INIT(vtkRenderingFreeType);
VTK_MODULE_INIT(vtkRenderingVolumeOpenGL2)





int main(int argc, char *argv[])
{
	// Create a plane
	vtkSmartPointer<vtkPlaneSource> planeSource =
		vtkSmartPointer<vtkPlaneSource>::New();
	planeSource->SetCenter(1.0, 0.0, 0.0);
	planeSource->SetNormal(1.0, 0.0, 1.0);
	planeSource->Update();

	vtkPolyData* plane = planeSource->GetOutput();

	// Create a mapper and actor
	vtkSmartPointer<vtkPolyDataMapper> mapper =
		vtkSmartPointer<vtkPolyDataMapper>::New();
#if VTK_MAJOR_VERSION <= 5
	mapper->SetInput(plane);
#else
	mapper->SetInputData(plane);
#endif

	vtkSmartPointer<vtkActor> actor =
		vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	// Create a renderer, render window and interactor
	vtkSmartPointer<vtkRenderer> renderer =
		vtkSmartPointer<vtkRenderer>::New();
	vtkSmartPointer<vtkRenderWindow> renderWindow =
		vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
	vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
		vtkSmartPointer<vtkRenderWindowInteractor>::New();
	renderWindowInteractor->SetRenderWindow(renderWindow);

	// Add the actors to the scene
	renderer->AddActor(actor);
	renderer->SetBackground(.1, .2, .3); // Background color dark blue

										 // Render and interact
	renderWindow->Render();
	renderWindowInteractor->Start();

	return EXIT_SUCCESS;
}
