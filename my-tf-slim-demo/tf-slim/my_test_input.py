#coding=utf-8
'''
加载 tf-slim vgg16 checkpoints 后输出一张图片预测值

参考
http://www.open-open.com/lib/view/open1480403665286.html
http://stackoverflow.com/questions/33648322/tensorflow-image-reading-display

'''

from matplotlib import pyplot as plt

import numpy as np
import os
import tensorflow as tf

from my_model import vgg
from preprocessing import vgg_preprocessing
import utils


checkpoints_dir = '/home/nicehija/PycharmProjects/tf_slim_vgg16/checkpoints'
photo_path = '/home/nicehija/PycharmProjects/tf_slim_vgg16/test_image/car.jpg'
slim = tf.contrib.slim

# 网络模型的输入图像有默认的尺寸
# 因此，我们需要先调整输入图片的尺寸
image_size = vgg.vgg_16_default_image_size

with tf.Graph().as_default():

    filenames = [photo_path]
    filename_queue = tf.train.string_input_producer(filenames)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_jpeg(value, channels=3)
    # 对图片做缩放操作，保持长宽比例不变，裁剪得到图片中央的区域
    # 裁剪后的图片大小等于网络模型的默认尺寸
    processed_image = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)

    # 可以批量导入图像
    # 第一个维度指定每批图片的张数
    # 我们每次只导入一张图片
    processed_images  = tf.expand_dims(processed_image, 0)

    # 创建模型，使用默认的arg scope参数
    # arg_scope是slim library的一个常用参数
    # 可以设置它指定网络层的参数，比如stride, padding 等等。

    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, _, _ = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)


    # 我们在输出层使用softmax函数，使输出项是概率值
    probabilities = tf.nn.softmax(logits)

    # 创建一个函数，从checkpoint读入网络权值
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_variables_to_restore())

    # 初始化image_tf
    # init_op = tf.variables_initializer([image_tf])

    with tf.Session() as sess:
        # 加载权值
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        init_fn(sess)

        # 图片经过缩放和裁剪，最终以numpy矩阵的格式传入网络模型
        np_image, network_input, probabilities = sess.run([image,
                                                           processed_image,
                                                           probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                            key=lambda x:x[1])]

        coord.request_stop()
        coord.join(threads)



    plt.figure()

    # 显示下载的图片
    plt.subplot(1, 2, 1)
    plt.imshow(np_image.astype(np.uint8))
    plt.title("image")
    plt.axis('off')

    # 显示最终传入网络模型的图片
    # 图像的像素值做了[-1, 1]的归一化
    plt.subplot(1, 2, 2)
    plt.imshow(network_input / (network_input.max() - network_input.min()) )
    plt.title("Resized, Cropped and Mean-Centered input to network")
    plt.axis('off')

    plt.show()

    #show
    utils.print_prob(probabilities, './synset.txt', 5)
