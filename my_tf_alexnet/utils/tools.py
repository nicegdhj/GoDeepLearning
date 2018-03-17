# coding=utf-8
'''
一些调试使用的工具
'''

import tensorflow as tf

def print_all_variables(train_only=True):
    """Print all trainable and non-trainable variables
    without tl.layers.initialize_global_variables(sess)

    Parameters
    ----------
    train_only : boolean
        If True, only print the trainable variables, otherwise, print all variables.
    """
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
        print("  [*] printing trainable variables")
    else:
        try: # TF1.0
            t_vars = tf.global_variables()
        except: # TF0.12
            t_vars = tf.all_variables()
        print("  [*] printing global variables")
    for idx, v in enumerate(t_vars):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))


