1、没有标签数据：利用Scipy库进行mat数据（annotates数据，从官网下载的）读取

2、并不是添加了struct_as_record=True在scio.loadmat（）中才可以读取struct；（注意：是mat数据中有cell以及struct的元组）
     而是：mat文件中有两个--一个cell一个struct（名字为：class，annotates）  ---注意---消耗两个小时

3、numpy.narry是什么东西；代码中读取的结构是[[ ][ ], [ ][ ]]

4、返回值怎么能够写在for循环中（error）

5、python im2rec.py F:\data\cars\lists\train.lst F:\data\cars\rec --resize=256 --encoding=.jpg --quality=100  --num-thread
 4
     Q：存在问题，读取图像的路径是错误的？？？？
     Solutions：在配置文件中将car_ims路径写成了cars_ims，导致在读取图像时错误

6、理解im2rec.py工具的使用  ---注意---

7、使用的也是迁移学习--通过自定义FC层进行再训练

8、训练命令：python fine_tune_cars.py --vgg F:\data\cars\vgg16\vgg16  --checkpoints F:\data\cars\checkpoints --prefix vggnet  
     ---注意---F:\data\cars\vgg16\vgg16其中最后的vgg16是用于作为识别目录文件的前缀

9、python fine_tune_cars.py --vgg F:\data\cars\vgg16\vgg16  --checkpoints F:\data\cars\checkpoints --prefix vggnet  --start-epoch 45

10、SGD_le4-5_30_45   --->理解命名的含义，优化器、lr、numepoch（对应前面的lr）

11、调整代码，不需要每个epoch都进行保存checkpoints；

12、test_cars: python test_cars.py --checkpoints E:\data\cars\checkpoints\ --prefix vggnet --epoch 55

