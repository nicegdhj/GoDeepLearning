# coding=utf-8
"""
在测试集上输出
"""
import tensorflow as tf
import cv2
import random
import numpy as np

import alexnet
import input_data



# test accracy
# def test_acc():
#
#     tf_data_path = '/home/njuciairs/HeJia/myProjects/Limu_LearnDeepLearning/data/tf_cifar10_train/' \
#                    'train_images_0.8percent/cifar10_val.tfrecords'
#     batch_size = 10
#     images_batch, labels_batch = input_data.input_images_from_tfdata(tf_data_path, batch_size, label_one_hot=True)
#     logits = alexnet.alexnet(images_batch, 10, is_training=True)
#     correct, accuracy = alexnet.accray(logits, labels_batch)
#
#     with tf.Session() as sess:
#
#         i = 0
#         sess.run(tf.global_variables_initializer())
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#
#         try:
#             while not coord.should_stop() and i < 3:
#                 # just plot one batch size
#                 corr, acc = sess.run([correct, accuracy])
#                 print corr, acc
#                 i += 1
#
#         except tf.errors.OutOfRangeError:
#             print('done!')
#         finally:
#             coord.request_stop()
#             coord.join(threads)
#####################################################################################################
# test one-hot
"""
检查one-hot 编码有没有对应上
"""
def test_build():
    """
    测试alexnet结构的是否搭建成功
    """
    batch_size = 5
    height, width = 227, 227
    channel = 3
    num_classes = 1000

    inputs = tf.random_uniform((batch_size, height, width, channel))
    logits = alexnet.alexnet(inputs, num_classes)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a = sess.run(logits)
        for v in tf.trainable_variables():
            print v


def test_one_hot():
    tf_data_path = '/home/njuciairs/HeJia/myProjects/Limu_LearnDeepLearning/data/tf_cifar10_train/train_images_0.8percent/cifar10_train.tfrecords'
    batch_size = 10

    images_batch1, labels_batch = input_data.input_images_from_tfdata(tf_data_path, batch_size,
                                                                      num_classes=10,
                                                                      is_shuffle=None,
                                                                      label_one_hot=None)
    images_batch2, labels_batch_onehot = input_data.input_images_from_tfdata(tf_data_path, batch_size,
                                                                             num_classes=10,
                                                                             is_shuffle=None,
                                                                             label_one_hot=True)
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i < 3:
                # just plot one batch size
                label1, label2 = sess.run([labels_batch, labels_batch_onehot])
                print label1
                print '######'
                print label2
                i +=1

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
            coord.join(threads)

################################################################################################
# test val queue

# def check_decode_images(images, labels, batch_size, images_dir):
#     '''
#     检查tfrecord文件是否能够被正确decode，以及对应上标签
#     ————————————
#     输入是一个batch
#     '''
#
#
#     for i in range(batch_size):
#         index = random.randint(0, 10000)
#         image = images[i, :, :, :]
#         label = labels[i]
#         label_name = label_dict[int(label)]
#         cv2.imwrite(images_dir + '/' + label_name + '_' + str(index) + '.png', image)
#
#
#
# BATCH_SIZE = 64
# tf_file_path = '/home/njuciairs/HeJia/myProjects/Limu_LearnDeepLearning/data/tf_cifar10_train/train_images_0.8percent' \
#                '/cifar10_val.tfrecords'
# check_data_dir = '/home/njuciairs/HeJia/myProjects/Limu_LearnDeepLearning/data/check_data'
#
# image_batch, label_batch = input_data.input_images_from_tfdata(tfrecords_file=tf_file_path,
#                                                                batch_size=BATCH_SIZE,
#                                                                num_classes=10,
#                                                                is_shuffle=None,
#                                                                label_one_hot=None)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     # usage of num_epochs means you apparently have to use initialize_local_variables()
#     sess.run(tf.local_variables_initializer())
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     batches = int(10000/BATCH_SIZE)
#     try:
#         for batch in range(batches):
#             if coord.should_stop():
#                 break
#             # just plot one batch size
#             image, label = sess.run([image_batch, label_batch])
#             check_decode_images(image, label, BATCH_SIZE, check_data_dir)
#             print 'batch_num:', batch+1
#
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#     coord.join(threads)
####################################################################################################
# '''
# 检查 网络的acc和loss 有没有被正确计算
# '''
# #
# def check_decode_images(images, labels, logits ,batch_size, images_dir):
#     '''
#     检查tfrecord文件是否能够被正确decode，以及对应上标签
#     ————————————
#     输入是一个batch
#     '''
#     label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
#                   8: 'ship', 9: 'truck'}
#
#     for i in range(batch_size):
#         index = random.randint(0, 10000)
#         image = images[i, :, :, :]
#         label = labels[i, :] # 还是onehot形式
#         pre = logits[i, :]
#         # 从one_hot 转化为 正常
#         label_name = label_dict[np.argmax(label)]
#         predict_name = label_dict[np.argmax(pre)]
#         cv2.imwrite(images_dir + '/' + label_name + '_' +predict_name+str(index) + '.png', image)
#
#
#
# tf_data_path = '/home/nicehija/my-projects/remote_GPU/data/cifar10_train_0to9.tfrecords'
# check_images ='/home/nicehija/my-projects/remote_GPU/data/check_images'
# batch_size = 20
# images_batch, labels_batch = input_data.input_images_from_tfdata(tf_data_path, batch_size,num_classes=10,label_one_hot=True, is_shuffle=True)
#
# logits = alexnet.alexnet(images_batch, num_classes=10 , is_training=True)
# acc = alexnet.accray(logits, labels_batch)
# # loss = alexnet.loss(logits, labels_batch)
#
# with tf.Session() as sess:
#
#     i = 0
#     sess.run(tf.global_variables_initializer())
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     try:
#         while not coord.should_stop() and i < 1:
#             # just plot one batch size
#             logt, im_b, la_b, corr = sess.run([logits, images_batch, labels_batch, acc])
#             check_decode_images(im_b,la_b ,logt, batch_size, check_images) #注意第二项是logt,检查有没有预测对
#             print corr
#
#             i += 1
#
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#         coord.join(threads)
test_one_hot()