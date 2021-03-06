"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
Modified by: Vinay Ramesh
"""
import numpy as np
import tensorflow as tf
from CapsComplete.models.utils import reduce_sum
from CapsComplete.models.utils import softmax
from CapsComplete.models.utils import get_shape

epsilon = 1e-9

class PrimaryCaps(object):
    def __init__(self, num_outputs,
                       vec_len, 
                       kernel_size,
                       stride):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.kernel_size = kernel_size
        self.stride = stride 
        self.conv_out = tf.keras.layers.Conv2D(self.num_outputs*self.vec_len,
                                          self.kernel_size,
                                          strides=self.stride,
                                          padding="valid",
                                          name='primary_caps_conv')
        
    def __call__(self, inputs):
        capsules_not_flat = self.conv_out(inputs)
        # fix this
        dim = tf.reduce_prod(capsules_not_flat.get_shape()[1:])
        dim = tf.div(dim, self.vec_len)
        capsules = tf.reshape(capsules_not_flat, [-1, dim, self.vec_len, 1])
        self.poses = squash(capsules)
        self.activations = tf.sqrt(tf.reduce_sum(tf.square(self.poses), -2, keepdims=True) + epsilon)
        return self.poses

class ClassCaps(object):
    def __init__(self, num_outputs,
                       vec_len):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.dense_out = tf.keras.layers.Dense(self.num_outputs, 
                                               activation=None)
    
    def __call__(self, inputs):
        in_vec_len = inputs.get_shape().as_list()[-2]
        dim = tf.reduce_prod(inputs.get_shape()[1:])
        dim = tf.div(dim, in_vec_len)
        self.input = tf.reshape(inputs, shape=(-1, dim, 1, in_vec_len, 1))

        with tf.variable_scope('routing'):
            # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
            print(self.input.get_shape())
            zeros_dims = tf.stack([tf.shape(self.input)[0], dim, self.num_outputs, 1, 1])
            b_IJ = tf.fill(zeros_dims, 0.0)
            # tf.zeros([self.input.get_shape()[0], dim, self.num_outputs, 1, 1], dtype=tf.dtypes.float32)
            capsules = routing(self.input, b_IJ, num_outputs=self.num_outputs, num_dims=self.vec_len)
            self.poses = tf.squeeze(capsules, axis=1)
            self.activations = tf.sqrt(reduce_sum(tf.square(self.poses),
                                               axis=2, keepdims=True) + epsilon)
        return self.poses

def routing(input, b_IJ, num_outputs=10, num_dims=16, iter_routing=3):
    ''' The routing algorithm.
    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
        num_outputs: the number of output capsules.
        num_dims: the number of dimensions for output capsule.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''
    stddev = 0.01
    # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
    input_shape = get_shape(input)
    W = tf.get_variable('Weight', shape=[1, input_shape[1], num_dims * num_outputs] + input_shape[-2:],
                        dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
    biases = tf.get_variable('bias', shape=(1, 1, num_outputs, num_dims, 1))

    # Eq.2, calc u_hat
    # Since tf.matmul is a time-consuming op,
    # A better solution is using element-wise multiply, reduce_sum and reshape
    # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
    # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
    # reshape to [a, c]
    input = tf.tile(input, [1, 1, num_dims * num_outputs, 1, 1])
    # assert input.get_shape() == [cfg.batch_size, 1152, 160, 8, 1]

    u_hat = reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, input_shape[1], num_outputs, num_dims, 1])
    # assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = softmax(b_IJ, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                # assert s_J.get_shape() == [cfg.batch_size, 1, num_outputs, num_dims, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                # assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                u_produce_v = reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                # assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return(v_J)


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)