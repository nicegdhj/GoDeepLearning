#coding=utf-8
'''
显示图片进入网络之前的预处理过程

参考：
http://www.open-open.com/lib/view/open1480403665286.html
http://stackoverflow.com/questions/33648322/tensorflow-image-reading-display
'''
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf

from preprocessing import vgg_preprocessing
from preprocessing import inception_preprocessing
from preprocessing import my_inception_preprocessing


slim = tf.contrib.slim


photo_path ='/home/nicehija/PycharmProjects/images_test/general/car.jpg'


# 网络模型的输入图像有默认的尺寸
image_size =299         #inceptionV3
# image_size = 224      #vgg

with tf.Graph().as_default():

    filenames = [photo_path]
    filename_queue = tf.train.string_input_producer(filenames)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_jpeg(value, channels=3)
    # 对图片做缩放操作，保持长宽比例不变，裁剪得到图片中央的区域
    # 裁剪后的图片大小等于网络模型的默认尺寸
    processed_image = my_inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_train = my_inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=True)

    with tf.Session() as sess:
        # 加载权值
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(tf.global_variables_initializer())

        # 图片经过缩放和裁剪，最终以numpy矩阵的格式传入网络模型
        np_image, network_input, image_train = sess.run([image, processed_image, processed_train ])
        coord.request_stop()
        coord.join(threads)



    plt.figure()

    # 显示下载的图片
    plt.subplot(1, 3, 1)
    plt.imshow(np_image.astype(np.uint8))
    plt.title("image ")
    plt.axis('off')

    # 显示最终传入网络模型的图片
    plt.subplot(1, 3, 2)
    plt.title(" inception preprocess train_data")
    plt.imshow(image_train)
    plt.axis('off')

    # plt.imshow(network_input)
    plt.subplot(1, 3, 3)
    plt.title(" inception preprocess vali")
    plt.imshow(network_input)
    plt.axis('off')

    plt.show()

