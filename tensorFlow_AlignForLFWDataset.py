#
# Align for LFW dataset
# date: 2018-5-18
# a   : zhonghy
#
#

for N in {1..4}:
 do python F:\ProgramDevelopment\facenet\src\align\align_dataset_mtcnn.py F:\d
ata\lfw\raw F:\data\lfw\lfw_mtcnnpy_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 & done





