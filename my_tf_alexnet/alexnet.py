# coding=utf-8

"""
Note:
      LRN 被去掉了
      所有的random_normal_initializer 被xavier_initializer替代
      图片入口是 227*227*3
      参考的是tfslim的写法
      pading=SAME表示前后map的size大小是一致的
to do：
     fc层用全局conv替代
     加入bn
"""

import tensorflow as tf

from utils import layers


def alexnet_bn():
    pass

def alexnet(inputs, num_classes, is_training=True):
    """
    inputs: 4-D tensor，[batch_size, height, width, channels]
    输入图片size 227*227*3
    """
    with tf.name_scope('AlexNet'):
        dropout_prob = tf.where(is_training, 0.5, 1.0)
        # 227*227*3，conv1，pad = 0
        conv_1 = layers.conv_bn(inputs=inputs, kernel_size=[11, 11], num_outputs=64, name='conv_1', stride_size=[4, 4],
                             padding='VALID')
        # 55*55*64
        pool_1 = layers.maxpool(inputs=conv_1, kernel_size=[3, 3], name='pool_1', stride_size=[2, 2], padding='VALID')
        # 27*27*64，conv2
        conv_2 = layers.conv_bn(inputs=pool_1, kernel_size=[5, 5], num_outputs=192, name='conv_2', stride_size=[1, 1])
        # 27*27*192，
        pool_2 = layers.maxpool(inputs=conv_2, kernel_size=[3, 3], name='pool_2', stride_size=[2, 2], padding='VALID')
        # 13*13*192
        conv_3 = layers.conv_bn(inputs=pool_2, kernel_size=[3, 3], num_outputs=384, name='conv_3', stride_size=[1, 1])
        # 13*13*384
        conv_4 = layers.conv_bn(inputs=conv_3, kernel_size=[3, 3], num_outputs=384, name='conv_4', stride_size=[1, 1])
        # 13*13*384
        conv_5 = layers.conv_bn(inputs=conv_4, kernel_size=[3, 3], num_outputs=256, name='conv_5', stride_size=[1, 1])
        # 13*13*256
        pool_5 = layers.maxpool(inputs=conv_5, kernel_size=[3, 3], name='pool_5', stride_size=[2, 2], padding='VALID')
        # 6*6*256

        # Flatten 全连接层要每张图要展平
        flatten = tf.reshape(pool_5, [-1, 6*6*256])  # -1那个维度是batchsize
        fc6 = layers.fully_connected(flatten, 4096, name='fc6')
        fc6_dropout = layers.dropout(fc6, dropout_prob, name="fc6_dropout")
        fc7 = layers.fully_connected(fc6_dropout, 4096, name='fc7')
        fc7_dropout = layers.dropout(fc7, dropout_prob, name='fc7_dropout')
        # 最后一层fc之后没有激活函数
        fc8 = layers.fully_connected(fc7_dropout, num_classes, name='fc8', activation_fn=None)
        return fc8

# fc8的输出logits有两个用途，一个是用于计算度量指标（准确率），一个是用于计算loss(交叉熵或其他)
# fc8-->softmax——>loss 计算交叉熵损失函数
# fc8-->accracy

def loss(fc8_logits, labels):
    with tf.name_scope('loss'):
        cost = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=fc8_logits)
        cost = tf.reduce_mean(cost, name='loss') # 怎么求平均的？
        tf.summary.scalar('cross_entropy_loss', cost)
    return cost

def accray(logits, labels):
    """
       logits: Logits tensor, float - [batch_size, NUM_CLASSES].
       labels: Labels tensor,
    """
    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar('accuracy', accuracy)
    return accuracy

def num_correct_prediction(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Return:
      the number of correct predictions
  """
  correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
  correct = tf.cast(correct, tf.int32)
  n_correct = tf.reduce_sum(correct)
  return n_correct


def optimize(loss, learning_rate, global_step):
    '''optimization, use Gradient Descent as default
    '''
    with tf.name_scope('optimizer'):
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
