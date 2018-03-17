#coding=utf-8

import utils
import os
import numpy as np
import tensorflow as tf
import pickle

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops



from my_model import vgg
from preprocessing import vgg_preprocessing


checkpoints_dir = '/home/nicehija/PycharmProjects/tf_slim_vgg16/checkpoints'
photo_path = '/home/nicehija/PycharmProjects/lung_cancer/analysis/useful_images/maybe_images/218978肺腺癌10x40倍-1.jpg'

slim = tf.contrib.slim

# label = np.array([1, 0])  # 1-hot result for Boxer
label = np.array([1 if i == 825 else 0 for i in range(1000)])  # 1-hot result for Boxer.
image_size = vgg.vgg_16_default_image_size


@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):

    #guided bp
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))
    # deconvnet
    # return gen_nn_ops._relu_grad(grad, op.outputs[0])
# Create tensorflow graph for evaluation
eval_graph = tf.Graph()
with eval_graph.as_default():

    with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
        #load pictures
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
        processed_images = tf.expand_dims(processed_image, 0)

        # 创建模型，使用默认的arg scope参数
        # arg_scope是slim library的一个常用参数
        with slim.arg_scope(vgg.vgg_arg_scope()):
            #target_conv_layer = vgg.pool5

            logits, _, target_conv_layer = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)

        # 我们在输出层使用softmax函数，使输出项是概率值
        probabilities = tf.nn.softmax(logits)


        # Get last convolutional layer gradient for generating gradCAM visualization
        cost = tf.reduce_sum((probabilities - label) ** 2)
        target_conv_layer_grad = tf.gradients(cost, target_conv_layer)[0]

        # Guided backpropagtion back to input layer
        gb_grad = tf.gradients(cost, processed_image)[0]

        # Normalizing the gradients
        target_conv_layer_grad_norm = tf.div(target_conv_layer_grad,
                                             tf.sqrt(tf.reduce_mean(tf.square(target_conv_layer_grad))) + tf.constant(
                                                 1e-5))


        # 创建一个函数，从checkpoint读入网络权值
        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'vgg_16.ckpt'),  #vgg_16.ckpt
            slim.get_variables_to_restore())

with tf.Session(graph=eval_graph) as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 加载权值
    init_fn(sess)
    probabilities, gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run(
        [ probabilities, gb_grad, target_conv_layer, target_conv_layer_grad_norm])


    # 对每一张图片
    # print (processed_image)             [224,224,3]
    # print (target_conv_layer)           [1,7,7,512]
    # print (target_conv_layer_grad_norm) [1,7,7,512]
    # print (gb_grad)                     [224,224,3]

    probabilities = probabilities[0, 0:]
    # convert [1,7,7,512] to [7,7,512]
    target_conv_layer_value = target_conv_layer_value[0]
    target_conv_layer_grad_value = target_conv_layer_grad_value[0]


    utils.print_prob(probabilities, './synset.txt', 5)
    utils.visualize(processed_image, target_conv_layer_value, target_conv_layer_grad_value, gb_grad_value)

    coord.request_stop()
    coord.join(threads)