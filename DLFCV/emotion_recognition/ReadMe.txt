1、python运算符：
+：　加法运算
-：　减法运算
*：　乘法运算
**：  幂运算
/:  　除法运算（如果有小数则返回结果为小数，如果都为整数则返回结果为整数）
//： 整除，取整数部分
%： 取余

2、python emotion_detector.py --cascade F:\ProgramPractice\DLFCV\pyimagesearch\sharedresource\haarcascade_frontalface_default.xml --mode
l F:\data\fer2013\output\checkpoints\weights-071.hdf5 --video F:\data\IMG_4509.MOV
    emotion_detector.py 运行没有结果 --注意--

3、Q：无结果
   Sloution：因为rects的长度为0，说明为检测到人脸，所以要调整检测人脸的detector的区域size？？？

4、已验证，可用，精度还需提高