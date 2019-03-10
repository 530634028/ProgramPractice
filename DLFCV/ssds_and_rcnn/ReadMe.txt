1、Error：google.protobuf.text_format.ParseError: 106:24 : String missing ending quote: '"F:/data/lisa/experiments/training/\r'
    Solution：
    fine_tune_checkpoint:  "F:/data/lisa/experiments/training/faster_rcnn_resnet101_coco_2018_01_28/model.ckpt.data-00000-of-00001"
    错误写法：
    fine_tune_checkpoint: "F:/data/lisa/experiments/training/
    faster_rcnn_resnet101_coco_2018_01_28/model.ckpt.data-00000-of-00001"


2、company：python F:\ExternalLib\models\research\object_detection\train.py --logtostderr --pipeline_config_path=F:\data\lisa\experiments\training\faster_rcnn_lisa.config --train_dir=F:\data\lisa\experiments\training
     home：python E:\ExternalLib\models\research\object_detection\train.py --logtostderr --pipeline_config_path=config\faster_rcnn_lisa.config --train_dir=E:\data\lisa\experiments\training

3、error：ValueError: Argument must be a dense tensor: range(0, 3) - got shape [3], but wanted [].

    solutions：I believe it is the same Python3 incompatibility that has crept up before (see #3443 ). The issue is with models/research/object_detection/utils/learning_schedules.py lines 167-169. Currently it is
                   rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                      range(num_boundaries), [0] * num_boundaries))
   Wrap list() around the range() like this:
                   rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                      list(range(num_boundaries)), [0] * num_boundaries))

4、company运行时，出现资源耗尽错误 -- CPU内存不够？？？

5、自己下载faster_rcnn_resnet101_coco_2018_01_28.tar模型以及数据集

6、注意注意---本实例中需要install the tensorflow obejection API (安装网上找教程)

7、no modual "deployment" 这个问题是models的环境变量路径没有设置好--应该放在system中

8、evaluate（评估）：python E:\ExternalLib\models\research\object_detection\eval.py --logtostderr --pipeline_config_path=config\faster_rcnn_lisa.config --checkpoint_dir=E:\data\lisa\experiments\training --eval_dir=E:\data\lisa\experiments\evaluation 
     Q:ModuleNotFoundError: No module named 'pycocotools'
     Solution:安装 pycocotools
     Q：安装pycocotools时的问题：cl : Command line error D8021 : invalid numeric argument '/Wno-cpp'
     Solutions：需要安装pycocotools，使用网上的教程

9、在pip install pycocotools
     Q：raise ReadTimeoutError(self._pool, None, 'Read timed out.')
     Solutions：更换安装源，换成国内的；如下安装：pip install -i https://pypi.douban.com/simple pycocotools  
     注意：可以有多个源还有其他源
               　　 http://pypi.douban.com/ 豆瓣

　　
                      http://pypi.hustunique.com/ 华中理工大学

　　
                      http://pypi.sdutlinux.org/ 山东理工大学

　　
                      http://pypi.mirrors.ustc.edu.cn/ 中国科学技术大学

10、pip install 安装时库下载后存放的临时位置：C:\\Users\\mk\\AppData\\Local\\Temp\\pip-install-jelyq_cq\\

11、python安装的第三方库在：C:\Users\mk\AppData\Roaming\Python\Python36\site-packages？？？？？（win10环境）

12、'tensorboard' 不是内部或外部命令，也不是可运行的程序
     或批处理文件。
     s：找到tensorboard.exe文件路径，将其添加到system--path环境变量中，例如：    C:\Users\mk\AppData\Roaming\Python\Python36\Scripts
     为什么在c/programfile中有python36，而在C:\Users\mk\AppData\Roaming\Python\Python36也有36，是安装的时候的问题吗？？？？？？？后者用于存放第三方库？？？

13、a、训练完之后：python E:\ExternalLib\models\research\object_detection\export_inference_graph.py --input_type image_tensor --pipeline_config_path config\faster_rcnn_lisa.config --trained_checkpoint_prefix E:\data\lisa\experiments\training\model.ckpt-50000 --output_directory E:\data\lisa\experiments\exported_model    输出mdel
       b、再使用：python predict.py --model E:\data\lisa\experiments\exported_model\frozen_inference_graph.pb --labels E:\data\lisa\records\classes.pbtxt --image
E:\data\image4.png --num-classes 3
         对数据集外面的图像数据进行预测








        









