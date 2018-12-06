




#include <stdio.h>
#include <malloc.h>

int main()
{
	int **a;
	int i, j;
	a = (int**)malloc(sizeof(int*)*3);//为二维数组分配3行
	for (i = 0; i < 3; ++i){//为每列分配4个大小空间
		a[i] = (int*)malloc(sizeof(int)*4);
	}
	//初始化
	for (i = 0; i < 3; ++i){
		for (j = 0; j < 4; ++j){
			a[i][j] = i+j;
		}
	}
	//输出测试
	for (i = 0; i < 3; ++i){
		for (j = 0; j < 4; ++j){
			printf ("%d ", a[i][j]);
		}
		printf ("\n");
	}
	//释放动态开辟的空间
	for (i = 0; i < 3; ++i){
		free(a[i]);
	}
	free(a);
	return 0;
}