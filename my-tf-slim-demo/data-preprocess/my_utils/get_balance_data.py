#coding=utf-8
'''
随机的从文件夹复制图片
得到0,1,2,3各自数据量500:500:500:500
对于每一类
我们需要每次手动指定from_dir，to_dir两变量

'''

import os
import shutil
import random

#不平衡数据集每一类数据的目录路径
from_dir = './3'
image_list = os.listdir(from_dir)


#随机采样图片，直到数量足够
gap = 500

#平衡数据集每一类数据目录
to_dir = './balance_dataset/vali/3'

for i in xrange(gap):
    #选出文件
    random_image = random.choice(image_list)
    file_name, file_extend = os.path.splitext(random_image)

    # 新名字
    new_name = file_name + ('_%s' % str(i)) + file_extend

    #新旧路径
    file_path = os.path.join(from_dir, random_image)
    newfile_path = os.path.join(to_dir, new_name)

    shutil.copyfile(file_path, newfile_path)
    print i
