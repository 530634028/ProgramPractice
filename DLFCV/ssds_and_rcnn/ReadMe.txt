1��Error��google.protobuf.text_format.ParseError: 106:24 : String missing ending quote: '"F:/data/lisa/experiments/training/\r'
    Solution��
fine_tune_checkpoint:  "F:/data/lisa/experiments/training/faster_rcnn_resnet101_coco_2018_01_28/model.ckpt.data-00000-of-00001"
����д����
fine_tune_checkpoint: "F:/data/lisa/experiments/training/
faster_rcnn_resnet101_coco_2018_01_28/model.ckpt.data-00000-of-00001"


2��python F:\ExternalLib\models\research\object_detection\train.py --logtostderr --pipeline_config_path=F:\data\lisa\experiments\training\faster_rcnn_lisa.config --train_dir=F:\data\lisa\experiments\training

3��error��ValueError: Argument must be a dense tensor: range(0, 3) - got shape [3], but wanted [].

solutions��I believe it is the same Python3 incompatibility that has crept up before (see #3443 ). The issue is with models/research/object_detection/utils/learning_schedules.py lines 167-169. Currently it is


rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                      range(num_boundaries),
  
                [0] * num_boundaries))

Wrap list() around the range() like this:


rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                     list(range(num_boundaries)),

               [0] * num_boundaries))
4��company����ʱ��������Դ�ľ����� -- CPU�ڴ治��������
5���Լ�����faster_rcnn_resnet101_coco_2018_01_28.tarģ���Լ����ݼ�









