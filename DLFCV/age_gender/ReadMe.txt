1������agegemderhelper�����е�(userID, imagePath, faceID, age, gender) = row[:5]��row[��5]���ܴ��ڿ��У�����Ҫ����������䣺
              if len(row[:5]) != 5:
                    continue
2��ִ���������python im2rec.py F:\data\adience\lists\age_test.lst F:\data\adience\rec\ --resize=256 --encoding=.jpg --quality=100  --num-thread 4
     ������.rec�ļ���������Ϊʲô���ɵ�.rec�ļ�����F:\data\adience\rec\Ŀ¼�£�������ǰһ��Ŀ¼�� ��

3��Q��imperative_utils.h:90: GPU support is disabled. Compile MXNet with USE_CUDA=1 to enable GPU support.
     Solutions��install MXNet GPU version, for example: pip install mxnet-cu90, (cu90 is refered to CUDA 9.0)

4�� Q�� Check failed: exec_ctx.dev_id < device_count_ (2 vs. 1)
 Invalid GPU Id: 2, Valid device id should be less than device_count: 1
      Solutions��
# compile the model
model = mx.model.FeedForward(
    ctx=[mx.gpu(0)], # mx.gpu(3)], # by your need
��������ϵͳ��GPU�������޸ģ�Ŀǰ�޸�Ϊmx.gpu(0)��---ע��---

5��ִ�е�����(lr=le-3)��python train.py --checkpoints F:\data\adience\checkpoints\age --prefix agenet
     (le-5):python train.py --checkpoints F:\data\adience\checkpoints\age  --prefix agenet --start-epoch 110 (Ҫ����checkpoints������)

6��Q:Ϊʲô��120��ʼѵ����ʱ��û�н��������
     Solutions����Ϊnum_epoch = 40 ��start_epoch���õ�Ϊ120����˼Ҫ���Ϊ��start_epoch��ʼѵ����num_epoch���������ò�������
7��python plot_log.py --network AgeNet --dataset AgeData

     
     




