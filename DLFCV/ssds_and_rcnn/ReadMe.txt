1��Error��google.protobuf.text_format.ParseError: 106:24 : String missing ending quote: '"F:/data/lisa/experiments/training/\r'
    Solution��
    fine_tune_checkpoint:  "F:/data/lisa/experiments/training/faster_rcnn_resnet101_coco_2018_01_28/model.ckpt.data-00000-of-00001"
    ����д����
    fine_tune_checkpoint: "F:/data/lisa/experiments/training/
    faster_rcnn_resnet101_coco_2018_01_28/model.ckpt.data-00000-of-00001"


2��company��python F:\ExternalLib\models\research\object_detection\train.py --logtostderr --pipeline_config_path=F:\data\lisa\experiments\training\faster_rcnn_lisa.config --train_dir=F:\data\lisa\experiments\training
     home��python E:\ExternalLib\models\research\object_detection\train.py --logtostderr --pipeline_config_path=config\faster_rcnn_lisa.config --train_dir=E:\data\lisa\experiments\training

3��error��ValueError: Argument must be a dense tensor: range(0, 3) - got shape [3], but wanted [].

    solutions��I believe it is the same Python3 incompatibility that has crept up before (see #3443 ). The issue is with models/research/object_detection/utils/learning_schedules.py lines 167-169. Currently it is
                   rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                      range(num_boundaries), [0] * num_boundaries))
   Wrap list() around the range() like this:
                   rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                      list(range(num_boundaries)), [0] * num_boundaries))

4��company����ʱ��������Դ�ľ����� -- CPU�ڴ治��������

5���Լ�����faster_rcnn_resnet101_coco_2018_01_28.tarģ���Լ����ݼ�

6��ע��ע��---��ʵ������Ҫinstall the tensorflow obejection API (��װ�����ҽ̳�)

7��no modual "deployment" ���������models�Ļ�������·��û�����ú�--Ӧ�÷���system��

8��evaluate����������python E:\ExternalLib\models\research\object_detection\eval.py --logtostderr --pipeline_config_path=config\faster_rcnn_lisa.config --checkpoint_dir=E:\data\lisa\experiments\training --eval_dir=E:\data\lisa\experiments\evaluation 
     Q:ModuleNotFoundError: No module named 'pycocotools'
     Solution:��װ pycocotools
     Q����װpycocotoolsʱ�����⣺cl : Command line error D8021 : invalid numeric argument '/Wno-cpp'
     Solutions����Ҫ��װpycocotools��ʹ�����ϵĽ̳�

9����pip install pycocotools
     Q��raise ReadTimeoutError(self._pool, None, 'Read timed out.')
     Solutions��������װԴ�����ɹ��ڵģ����°�װ��pip install -i https://pypi.douban.com/simple pycocotools  
     ע�⣺�����ж��Դ��������Դ
               ���� http://pypi.douban.com/ ����

����
                      http://pypi.hustunique.com/ ��������ѧ

����
                      http://pypi.sdutlinux.org/ ɽ������ѧ

����
                      http://pypi.mirrors.ustc.edu.cn/ �й���ѧ������ѧ

10��pip install ��װʱ�����غ��ŵ���ʱλ�ã�C:\\Users\\mk\\AppData\\Local\\Temp\\pip-install-jelyq_cq\\

11��python��װ�ĵ��������ڣ�C:\Users\mk\AppData\Roaming\Python\Python36\site-packages������������win10������

12��'tensorboard' �����ڲ����ⲿ���Ҳ���ǿ����еĳ���
     ���������ļ���
     s���ҵ�tensorboard.exe�ļ�·����������ӵ�system--path���������У����磺    C:\Users\mk\AppData\Roaming\Python\Python36\Scripts
     Ϊʲô��c/programfile����python36������C:\Users\mk\AppData\Roaming\Python\Python36Ҳ��36���ǰ�װ��ʱ��������𣿣������������������ڴ�ŵ������⣿����

13��a��ѵ����֮��python E:\ExternalLib\models\research\object_detection\export_inference_graph.py --input_type image_tensor --pipeline_config_path config\faster_rcnn_lisa.config --trained_checkpoint_prefix E:\data\lisa\experiments\training\model.ckpt-50000 --output_directory E:\data\lisa\experiments\exported_model    ���mdel
       b����ʹ�ã�python predict.py --model E:\data\lisa\experiments\exported_model\frozen_inference_graph.pb --labels E:\data\lisa\records\classes.pbtxt --image
E:\data\image4.png --num-classes 3
         �����ݼ������ͼ�����ݽ���Ԥ��








        









