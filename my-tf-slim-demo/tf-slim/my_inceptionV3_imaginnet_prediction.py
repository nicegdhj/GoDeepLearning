#coding=utf-8

'''
对于imgnet 数据集
加载inceptionV3  checkpoints 后输出一张图片预测值
参考
http://www.open-open.com/lib/view/open1480403665286.html
http://stackoverflow.com/questions/33648322/tensorflow-image-reading-display
'''

from matplotlib import pyplot as plt
import utils

import numpy as np
import os
import tensorflow as tf

#对应的网络的和预处理
from nets import inception_v3
from preprocessing import inception_preprocessing


#inceptionV3-imagenet checkpoints path
checkpoints_dir = '/home/nicehija/PycharmProjects/beijingproject-inceptionV3-slim0.12/slim/tmp/checkpoints'
#test photo path
photo_path1 = '/home/nicehija/PycharmProjects/images_test/general/car.jpg'


slim = tf.contrib.slim

# 网络模型的输入图像有默认的尺寸
# 因此，我们需要先调整输入图片的尺寸
image_size = inception_v3.inception_v3.default_image_size #299
with tf.Graph().as_default():

    filenames = [photo_path1]
    filename_queue = tf.train.string_input_producer(filenames)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_jpeg(value, channels=3) #是jpg格式的decodeer
    # 对图片做缩放操作，保持长宽比例不变，裁剪得到图片中央的区域
    # 裁剪后的图片大小等于网络模型的默认尺寸
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)

    # 可以批量导入图像
    # 第一个维度指定每批图片的张数
    # 我们每次只导入一张图片
    processed_images  = tf.expand_dims(processed_image, 0)

    # 创建模型，使用默认的arg scope参数
    # arg_scope是slim library的一个常用参数
    # 可以设置它指定网络层的参数，比如stride, padding 等等。

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        ## num_classes=1001是对于imagenet而言的 至于为什么不是1000它的原网页有解释,如果是你的3类 写3就好了
        logits, _ = inception_v3.inception_v3(processed_images, num_classes=1001, is_training=False)

    # 我们在输出层使用softmax函数，使输出项是概率值
    probabilities = tf.nn.softmax(logits)

    # 创建一个函数，从checkpoint读入网络权值
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
        slim.get_variables_to_restore())

    with tf.Session() as sess:
        # 加载权值
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        init_fn(sess)

        # 图片经过缩放和裁剪，最终以numpy矩阵的格式传入网络模型
        np_image, network_input, probabilities, = sess.run([image,
                                                           processed_image,
                                                           probabilities, ])

        probabilities = probabilities[0, 0:]

        coord.request_stop()
        coord.join(threads)


    #
    # plt.figure()
    #
    # # 显示下载的图片及其预处理之后的对比图像
    # plt.subplot(1, 2, 1)
    # plt.imshow(np_image.astype(np.uint8))
    # plt.title("image")
    # plt.axis('off')
    #
    # # 显示最终传入网络模型的图片
    # # 图像的像素值做了[-1, 1]的归一化
    # plt.subplot(1, 2, 2)
    # plt.imshow(network_input / (network_input.max() - network_input.min()) )
    # plt.title("Resized, Cropped and Mean-Centered input to network")
    # plt.axis('off')
    #
    # plt.show()

    #show results
    utils.print_prob(probabilities, './synset.txt', 5)
