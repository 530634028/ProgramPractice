1、其中agegemderhelper函数中的(userID, imagePath, faceID, age, gender) = row[:5]，row[：5]可能存在空行，所以要加上如下语句：
              if len(row[:5]) != 5:
                    continue
2、执行如下命令：python im2rec.py F:\data\adience\lists\age_test.lst F:\data\adience\rec\ --resize=256 --encoding=.jpg --quality=100  --num-thread 4
     来生成.rec文件（？但是为什么生成的.rec文件不在F:\data\adience\rec\目录下，而是在前一个目录下 ）

3、Q：imperative_utils.h:90: GPU support is disabled. Compile MXNet with USE_CUDA=1 to enable GPU support.
     Solutions：install MXNet GPU version, for example: pip install mxnet-cu90, (cu90 is refered to CUDA 9.0)

4、 Q： Check failed: exec_ctx.dev_id < device_count_ (2 vs. 1)
 Invalid GPU Id: 2, Valid device id should be less than device_count: 1
      Solutions：
# compile the model
model = mx.model.FeedForward(
    ctx=[mx.gpu(0)], # mx.gpu(3)], # by your need
根据自身系统的GPU数进行修改，目前修改为mx.gpu(0)，---注意---

5、执行的命令(lr=le-3)：python train.py --checkpoints F:\data\adience\checkpoints\age --prefix agenet
     (le-5):python train.py --checkpoints F:\data\adience\checkpoints\age  --prefix agenet --start-epoch 110 (要看你checkpoints的名字)

6、Q:为什么从120开始训练的时候没有结果？？？
     Solutions：因为num_epoch = 40 而start_epoch设置的为120；意思要理解为从start_epoch开始训练到num_epoch，所以设置参数错误
7、python plot_log.py --network AgeNet --dataset AgeData

     
     




