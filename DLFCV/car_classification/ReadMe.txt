1��û�б�ǩ���ݣ�����Scipy�����mat���ݣ�annotates���ݣ��ӹ������صģ���ȡ

2�������������struct_as_record=True��scio.loadmat�����вſ��Զ�ȡstruct����ע�⣺��mat��������cell�Լ�struct��Ԫ�飩
     ���ǣ�mat�ļ���������--һ��cellһ��struct������Ϊ��class��annotates��  ---ע��---��������Сʱ

3��numpy.narry��ʲô�����������ж�ȡ�Ľṹ��[[ ][ ], [ ][ ]]

4������ֵ��ô�ܹ�д��forѭ���У�error��

5��python im2rec.py F:\data\cars\lists\train.lst F:\data\cars\rec --resize=256 --encoding=.jpg --quality=100  --num-thread
 4
     Q���������⣬��ȡͼ���·���Ǵ���ģ�������
     Solutions���������ļ��н�car_ims·��д����cars_ims�������ڶ�ȡͼ��ʱ����

6�����im2rec.py���ߵ�ʹ��  ---ע��---

7��ʹ�õ�Ҳ��Ǩ��ѧϰ--ͨ���Զ���FC�������ѵ��

8��ѵ�����python fine_tune_cars.py --vgg F:\data\cars\vgg16\vgg16  --checkpoints F:\data\cars\checkpoints --prefix vggnet  
     ---ע��---F:\data\cars\vgg16\vgg16��������vgg16��������Ϊʶ��Ŀ¼�ļ���ǰ׺

9��python fine_tune_cars.py --vgg F:\data\cars\vgg16\vgg16  --checkpoints F:\data\cars\checkpoints --prefix vggnet  --start-epoch 45

10��SGD_le4-5_30_45   --->��������ĺ��壬�Ż�����lr��numepoch����Ӧǰ���lr��

11���������룬����Ҫÿ��epoch�����б���checkpoints��

12��test_cars: python test_cars.py --checkpoints E:\data\cars\checkpoints\ --prefix vggnet --epoch 55

