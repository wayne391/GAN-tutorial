
import tensorflow as tf
import numpy as np

NO_OPS = "NO_OPS"


def label_to_onehot(labels, num_class):
    labels = np.array(labels)
    onehots = np.zeros((len(labels), num_class), dtype=bool)
    for idx, lab in enumerate(labels):
        onehots[idx, lab] = True
    return onehots

def scope_has_variables(scope):
  return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0

def vec_cond_concat(x, y):
    return tf.concat([x, y], 1)

def conv_cond_concat(x, y):
    x_shapes = x.get_shape().as_list()
    shape_num = len(x_shapes)
    y = tf.reshape(y, [-1] + [1]*(shape_num-2) +[y.get_shape()[-1]])
    y_shapes = y.get_shape().as_list()
    return tf.concat([x, y*tf.ones(x_shapes[:-1] + [y_shapes[-1]])], 3)

def conv2d_sn(tensor_in, out_channels, kernels, strides, stddev=0.02, name='conv2d',
                 padding='valid', update_collection=None):

    with tf.variable_scope(name) as scope:
        if scope_has_variables(scope):
            scope.reuse_variables()


        if isinstance(kernels, int):
            kernels = [kernels, kernels]

        if isinstance(strides, int):
            strides = [strides, strides]

        # init vars
        weight = tf.get_variable('weight',
                    kernels+[tensor_in.get_shape()[-1], out_channels],
                    initializer=tf.truncated_normal_initializer(stddev=stddev))

        bias = tf.get_variable('bias',
                    [out_channels],
                    initializer=tf.constant_initializer(0.0))

        # spectral norm
        print(' *spectral norm activated!')
        weight, sigma = spectral_norm_weight(weight, update_collection=update_collection)

        # conv
        conv = tf.nn.conv2d(tensor_in, weight, strides=[1]+strides+[1], padding=padding.upper())

        out_shape = tf.stack([tf.shape(tensor_in)[0]]+list(conv.get_shape()[1:]))
        output = tf.reshape(tf.nn.bias_add(conv, bias), out_shape)
        print(output.get_shape())

        return output, sigma

def dense_sn(tensor_in, output_size, stddev=0.02, name='dense', update_collection=None):

    with tf.variable_scope(name) as scope:
        if scope_has_variables(scope):
            scope.reuse_variables()

        # init vars
        weight = tf.get_variable('weight',
                    [tensor_in.get_shape()[1], output_size], tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=stddev))

        bias = tf.get_variable('bias',
                    [output_size],
                    initializer=tf.constant_initializer(0.0))

        # spectral norm
        print(' *spectral norm activated!')
        weight, sigma = spectral_norm_weight(weight, update_collection=update_collection)

        # dense
        lin = tf.nn.bias_add(tf.matmul(tensor_in, weight), bias)

        return lin, sigma


def batch_norm(tensor_in, apply_mode=True):
    '''
    apply_mode:
        True:
        False:
        None:
    '''
    if apply_mode is None:
        print('*bypass bn')
        return tensor_in
    else:
        return tf.contrib.layers.batch_norm(tensor_in, decay=0.9,
                                                    epsilon=1e-5,
                                                    updates_collections=None,
                                                    scale=True,
                                                    is_training=apply_mode)

def flattern(tensor_in):
    return tf.reshape(tensor_in,
         [-1, np.product([s.value for s in tensor_in.get_shape()[1:]])])

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm_weight(W, u=None, num_iters=1, update_collection=None, name='weights_SN'):

    print(' ** spectral norm cell,', update_collection)

    with tf.variable_scope(name):

        W_shape = W.shape.as_list()
        W_reshaped = tf.reshape(W, [-1, W_shape[-1]])

        if u is None:
            u = tf.get_variable("u", [1, W_shape[-1]],
                     initializer=tf.truncated_normal_initializer(), trainable=False)

        def power_iteration(i, u_i, v_i):
            v_ip1 = l2_norm(tf.matmul(u_i, tf.transpose(W_reshaped)))
            u_ip1 = l2_norm(tf.matmul(v_ip1, W_reshaped))
            return i + 1, u_ip1, v_ip1

        _, u_final, v_final = tf.while_loop(
            cond=lambda i, _1, _2: i < num_iters,
            body=power_iteration,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                    u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]])))

        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        W_bar = W_reshaped / sigma

        if update_collection is None:
            with tf.control_dependencies([u.assign(u_final)]):
                W_bar = tf.reshape(W_bar, W_shape)
        else:
            W_bar = tf.reshape(W_bar, W_shape)
            if update_collection != NO_OPS:
                tf.add_to_collection(update_collection, u.assign(u_final))

    return W_bar, sigma
