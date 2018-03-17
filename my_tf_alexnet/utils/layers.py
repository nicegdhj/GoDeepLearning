# coding=utf-8
import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def conv(inputs, kernel_size, num_outputs, name,
        stride_size=[1, 1], padding='SAME', activation_fn=tf.nn.relu):
    """
    Convolution layer followed by activation fn:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        kernel_size: List, filter size [height, width]
        num_outputs: Integer, number of convolution filters
        name: String, scope name
        stride_size: List, convolution stide [height, width]
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height+-, width+-, num_outputs]
    """

    with tf.variable_scope(name) as scope:
        num_filters_in = inputs.get_shape()[-1].value
        kernel_shape = [kernel_size[0], kernel_size[1], num_filters_in, num_outputs]
        stride_shape = [1, stride_size[0], stride_size[1], 1]

        weights = tf.get_variable('weights', kernel_shape, tf.float32, xavier_initializer())
        bias = tf.get_variable('bias', [num_outputs], tf.float32, tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(inputs, weights, stride_shape, padding=padding, name=scope.name)
        outputs = tf.nn.bias_add(conv, bias)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

def conv_bn(inputs, kernel_size, num_outputs, name,
        is_training=True, stride_size=[1, 1], padding='SAME', activation_fn=tf.nn.relu):
    """
    Convolution layer followed by batch normalization then activation fn:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        kernel_size: List, filter size [height, width]
        num_outputs: Integer, number of convolution filters
        name: String, scope name
        is_training: Boolean, in training mode or not
        stride_size: List, convolution stide [height, width]
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height, width, num_outputs]
    """

    with tf.variable_scope(name) as scope:
        num_filters_in = inputs.get_shape()[-1].value
        kernel_shape = [kernel_size[0], kernel_size[1], num_filters_in, num_outputs]
        stride_shape = [1, stride_size[0], stride_size[1], 1]

        weights = tf.get_variable('weights', kernel_shape, tf.float32, xavier_initializer())
        bias = tf.get_variable('bias', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(inputs, weights, stride_shape, padding=padding, name=scope.name)
        outputs = tf.nn.bias_add(conv, bias)
        outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, is_training=is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

def fully_connected(inputs, num_outputs, name, activation_fn=tf.nn.relu):
    """
    Fully connected layer:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        num_outputs: Integer, number of output neurons
        name: String, scope name
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, num_outputs]
    """

    with tf.variable_scope(name):
        num_filters_in = inputs.get_shape()[-1].value

        weights = tf.get_variable('weights', [num_filters_in, num_outputs], tf.float32, xavier_initializer())
        bias = tf.get_variable('bias', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        outputs = tf.matmul(inputs, weights)
        outputs = tf.nn.bias_add(outputs, bias)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def maxpool(inputs, kernel_size, name, stride_size=[1, 1], padding='SAME'):
    """
    Max pooling layer:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        kernel_size: List, filter size [height, width]
        name: String, scope name
        stride_size: List, convolution stide [height, width]
        padding: String, input padding

    Returns:
        outputs: Tensor, [batch_size, height / kernelsize[0], width/kernelsize[1], channels]
    """

    kernel_shape = [1, kernel_size[0], kernel_size[1], 1]
    stride_shape = [1, stride_size[0], stride_size[1], 1]
    outputs = tf.nn.max_pool(inputs, ksize=kernel_shape,
                             strides=stride_shape, padding=padding, name=name)
    return outputs

def dropout(inputs, prob, name):
    """"dropout"""
    return tf.nn.dropout(x=inputs, keep_prob=prob, name=name)


def batch_norm(x):
    '''
    Batch normlization(I didn't include the offset and scale)
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x