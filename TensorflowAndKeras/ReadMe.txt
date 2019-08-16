
1、AttributeError: module 'tensorflow.python.keras.datasets.fashion_mnist' has no attribute 'load_data'
   版本问题，需要1.9.0以上？

2、https://www.cnblogs.com/sench/p/9683905.html 有关numpy.random的介绍

3、该文件夹下程序是tf以及keras官网教程中所使用的程序

4、问题描述： 利用keras的Sequential堆叠layer时出现了TypeError: The added layer must be an instance of class Layer
解决方案： 检查keras的导入，如果出现使用tensorflow.python.keras方式引用和keras引用混合就会出现这个问题。

统一使用：

from tensorflow.python.keras.datasets import mnist

                from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten

                from tensorflow.python.keras.models import Sequential

                from tensorflow.python.keras.utils import np_utils
1
2
3
4

或者使用这个：

from keras.datasets import mnist

                      from keras.layers.core import Dense, Dropout, Activation, Flatten

                      from keras.models import Sequential

                      from keras.utils import np_utils

5、下面总结一下防止神经网络出现过拟合的最常见方法：

获取更多训练数据。
降低网络容量。
添加权重正则化。
添加丢弃层。
还有两个重要的方法在本指南中没有介绍：数据增强和批次归一化。

6、keras datasets下载的数据默认放在C:\Users\Administrator\.keras\datasets目录下 （注意）


