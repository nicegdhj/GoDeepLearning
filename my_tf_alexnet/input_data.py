# coding=utf-8
"""
使用queue传递训练数据
"""
import os
import numpy as np
import tensorflow as tf


def data_augmentation(image, reshape_size):
    '''
    每张图片都必须经历的步骤: 归一化
    之后data augmentation
    ————————————
    :param reshape_size: 是一个list，[height, width, channel]
    '''

    image = tf.random_crop(image, reshape_size)  # reshape_size代表原图裁剪后的尺寸如原图是32.那么随机裁剪到24？
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    return image


def label_to_one_hot(label_batch, batch_size, n_classes):
    """
    将label变为one hot形式
    计算loss时：
    tf.nn.sparse_softmax_cross_entropy_with_logits，不需要对label做one-hot变化
    tf.nn.softmax_cross_entropy_with_logits(), 需要one-hot
    """

    label_batch = tf.one_hot(label_batch, depth=n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])
    return label_batch

def input_images_from_tfdata(tfrecords_file, batch_size, num_classes, is_shuffle=True, label_one_hot=True, is_training=True):
    '''
    读取tfrecord 并且解码， batch
    image: 4D tensor - [batch_size, width, height, channel]
    label: 1D tensor - [batch_size]
    '''
    # 第一步，指定读入的queue
    # 当训练时,则循环读入多个epochs, 其实val不需要只要使用较大量的数据即可,也不用严格的一个epoch
    filename_queue = tf.train.string_input_producer([tfrecords_file])


    # 第二步，指定一个reader
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # 第三步decode过程
    # tf.parse_single_example解析器，可以将Example协议内存块(protocol buffer)解析为张量
    feature = {'label': tf.FixedLenFeature([], tf.int64),
               'image_raw': tf.FixedLenFeature([], tf.string)}
    img_features = tf.parse_single_example(serialized_example, features=feature)

    label = tf.cast(img_features['label'], tf.int32)

    image = tf.decode_raw(img_features['image_raw'], tf.uint8) # RGB图像应该是8bit，像素值0~255
    image = tf.cast(image, tf.float32) # 转化到tf.float32
    image = tf.reshape(image, [32, 32, 3])
    if is_training:
        #训练时，输入数据需要做数据增广
        image = data_augmentation(image, [24, 24, 3]) # 可以选择data augmentation

    #测试时，无需对图像做标准化以外的增强数据处理
    image = tf.image.per_image_standardization(image)  # 归一化
    image = tf.image.resize_images(image, [227, 227])

    # queue中图片数量的设定
    # min_fraction_of_examples_in_queue =0.4 对于cifar10 32*32 示例代码这么写的
    #min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *min_fraction_of_examples_in_queue)
    min_queue_examples = 1000
    if is_shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=64,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

    else:
        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size=batch_size,
                                                  num_threads=64,
                                                  capacity=min_queue_examples + 3 * batch_size)


    if label_one_hot:
        label_batch = label_to_one_hot(label_batch, batch_size, n_classes=num_classes)
    else:
        label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch

