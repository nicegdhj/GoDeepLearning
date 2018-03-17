# coding=utf-8
"""
训练集和验证集上训练
---------
两个input queue，一个给train，一个给vali
train和vali的log分别存放



"""
import tensorflow as tf
import os
import numpy as np

import alexnet
import input_data



# 数据集相关
IMG_W = 227
IMG_H = 227
CHANNAL = 3
DATASET_SIZE = {'train': 40000, 'val': 10000}

# 超参
N_CLASSES = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MAX_STEP = 30000


#
tf_train_data = '/home/njuciairs/HeJia/myProjects/Limu_LearnDeepLearning/data/tf_cifar10_train/train_images_0.8percent/cifar10_train.tfrecords'
tf_vali_data = '/home/njuciairs/HeJia/myProjects/Limu_LearnDeepLearning/data/tf_cifar10_train/train_images_0.8percent/cifar10_val.tfrecords'
train_log_dir = '/home/njuciairs/HeJia/myProjects/Limu_LearnDeepLearning/log/log_myalexnet/log5/train_log'
val_log_dir = '/home/njuciairs/HeJia/myProjects/Limu_LearnDeepLearning/log/log_myalexnet/log5/val_log'
model_checkpoints = '/home/njuciairs/HeJia/myProjects/Limu_LearnDeepLearning/log/log_myalexnet/log5/checkpoints'




def train():

    # data input
    with tf.name_scope('input'):
        train_images_batch, train_labels_batch = input_data.input_images_from_tfdata(tf_train_data, BATCH_SIZE,
                                                                                     num_classes=N_CLASSES,
                                                                                     is_shuffle=True,
                                                                                     label_one_hot=True,)

        val_images_batch, val_labels_batch = input_data.input_images_from_tfdata(tf_vali_data, BATCH_SIZE,
                                                                                 num_classes=N_CLASSES,
                                                                                 is_shuffle=True,
                                                                                 label_one_hot=True,
                                                                                 is_training=False)


    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_H, IMG_W, CHANNAL])
    y = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])

    logits = alexnet.alexnet(x, num_classes=10, is_training=True)
    loss = alexnet.loss(logits, y)
    accuracy = alexnet.accray(logits, y)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = alexnet.optimize(loss, learning_rate=LEARNING_RATE, global_step=my_global_step)

    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        tf.get_variable_scope().reuse_variables()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break

                tra_images, tra_labels = sess.run([train_images_batch, train_labels_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy], feed_dict={x: tra_images,
                                                                                       y: tra_labels})


                if step % 20 == 0 or (step + 1) == MAX_STEP:
                    print ('Step: %d, train_batch_loss: %.4f, train_batch_acc: %.4f%%' % (step, tra_loss, tra_acc))
                    summary_str = sess.run(summary_op, feed_dict={x: tra_images, y: tra_labels})
                    tra_summary_writer.add_summary(summary_str, step)

                # 验证val 数据集的整个epoch
                if step % 100 == 0 or (step + 1) == MAX_STEP:
                        val_images, val_labels = sess.run([val_images_batch, val_labels_batch])
                        val_loss, val_acc = sess.run([loss, accuracy],
                                                     feed_dict={x: val_images, y: val_labels})
                        print('**  Step %d, val_batch_loss = %.2f, val_batch_accuracy = %.2f%%  **' % (step, val_loss,
                                                                                                       val_acc))

                        summary_str = sess.run(summary_op, feed_dict={x: val_images, y: val_labels})
                        val_summary_writer.add_summary(summary_str, step)

                if step+1 % 10000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(model_checkpoints, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

train()





###############################
# 验证val 数据集的整个epoch
# if (step + 1) % 200 == 0 or (step + 1) == MAX_STEP:
#
#     batches = int(DATASET_SIZE['val'] / BATCH_SIZE)
#     val_epoch_loss_list = []
#     val_epoch_acc_list = []
#     for batch in range(batches):
#         val_images, val_labels = sess.run([val_images_batch, val_labels_batch])
#         val_loss, val_acc = sess.run([loss, accuracy],
#                                      feed_dict={x: val_images, y: val_labels})
#         val_epoch_loss_list.append(val_loss)
#         val_epoch_acc_list.append(val_acc)
#
#     val_epoch_acc = np.mean(val_epoch_acc_list)
#     val_epoch_loss = np.mean(val_epoch_loss_list)
#
#     print('Step %d, val loss = %.2f, val_epoch_acc = %.2f%%  *********' % (step + 1,
#                                                                            val_epoch_loss,
#                                                                            val_epoch_acc))
#     summary_str = sess.run(summary_op, feed_dict={x: val_images, y: val_labels})
#     val_summary_writer.add_summary(summary_str, step)