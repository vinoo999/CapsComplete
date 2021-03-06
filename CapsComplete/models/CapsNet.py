"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
Modified: Vinay Ramesh
"""

import tensorflow as tf

from CapsComplete.models.utils import softmax
from CapsComplete.models.utils import reduce_sum
from CapsComplete.models import CapsLayer

epsilon = 1e-9

class CapsNet(object):
    def __init__(self, height=28, width=28, channels=1, num_label=10, is_training=True, **kwargs):
        """
        Args:
            height: Integer, the height of inputs.
            width: Integer, the width of inputs.
            channels: Integer, the channels of inputs.
            num_label: Integer, the category number.
        """
        self.height = height
        self.width = width
        self.channels = channels
        self.num_label = num_label
        self.mask_with_y = is_training
        
        # Loss Parameters
        self.m_plus = kwargs.get('m_plus', 0.9)
        self.m_minus = kwargs.get('m_minus', 0.1)
        self.lambda_val = kwargs.get('lambda_val', 0.5)
        self.regularization_scale = kwargs.get('lambda', 0.392)

        self.graph = tf.Graph()

        with self.graph.as_default():
            if is_training:
                self.inputs = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.channels))
                self.labels = tf.placeholder(tf.int32, shape=(None,))
                self.Y = tf.one_hot(self.labels, depth=self.num_label, axis=1, dtype=tf.float32)

                self.build_arch()
                self.loss()
                self._summary()

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
            else:
                self.inputs = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.channels))
                self.labels = tf.placeholder(tf.int32, shape=(None, ))
                self.Y = tf.one_hot(self.labels, depth=self.num_label, axis=1, dtype=tf.float32)
                self.build_arch()
                self.loss()
                correct_prediction = tf.equal(tf.to_int32(self.labels), self.predictions)
                self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

        tf.logging.info('Seting up the main structure')

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            # Conv1, return tensor with shape [batch_size, 20, 20, 256]
            conv1 = tf.keras.layers.Conv2D(256,
                                           kernel_size=9, 
                                           strides=1,
                                           padding='valid')(self.inputs)

        # Primary Capsules layer, return tensor with shape [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            primary_caps = CapsLayer.PrimaryCaps(num_outputs=32, 
                                                vec_len=8, 
                                                kernel_size=9, stride=2)(conv1)

        # DigitCaps layer, return shape [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            self.digit_caps = CapsLayer.ClassCaps(num_outputs=self.num_label, vec_len=16)(primary_caps)
            
        # Decoder structure in Fig. 2
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
            self.v_length = tf.sqrt(reduce_sum(tf.square(self.digit_caps),
                                               axis=2, keepdims=True) + epsilon)
            self.probs = softmax(self.v_length, axis=1)
            # assert self.probs.get_shape() == [cfg.batch_size, self.num_label, 1, 1]

            # b). pick out the index of max softmax val of the 10 caps
            # [batch_size, 10, 1, 1] => [batch_size] (index)
            self.predictions = tf.to_int32(tf.argmax(self.probs, axis=1))
            # assert self.predictions.get_shape() == [cfg.batch_size, 1, 1]
            self.predictions = tf.reshape(self.predictions, shape=(-1, ))

            # Method 1.
            if not self.mask_with_y:
                # c). indexing
                self.masked_v = tf.gather(self.digit_caps, self.predictions, axis=1)[:,0,:,:]
            # Method 2. masking with true label, default mode
            else:
                self.masked_v = tf.gather(self.digit_caps, self.labels, axis=1)[:,0,:,:]
                print(self.digit_caps.get_shape())
                print(self.masked_v.get_shape())
                self.v_length = tf.sqrt(reduce_sum(tf.square(self.digit_caps), axis=2, keepdims=True) + epsilon)

        # 2. Reconstructe the MNIST images with 3 FC layers
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        with tf.variable_scope('Decoder'):
            vector_j = self.masked_v[:,:,0]
            fc1 = tf.keras.layers.Dense(512, activation='relu')(vector_j)
            fc2 = tf.keras.layers.Dense(1024, activation='relu')(fc1)
            self.decoded = tf.keras.layers.Dense(self.height * self.width * self.channels,
                                                             activation=tf.sigmoid)(fc2)
            self.recons = tf.reshape(self.decoded, (-1, self.height, self.width, self.channels))

    def loss(self):
        # 1. The margin loss

        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., self.m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - self.m_minus))
        # assert max_l.get_shape() == [cfg.batch_size, self.num_label, 1, 1]

        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(-1, self.num_label))
        max_r = tf.reshape(max_r, shape=(-1, self.num_label))

        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.Y
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + self.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
        
        self.classification_loss = tf.losses.softmax_cross_entropy(self.Y, tf.squeeze(self.v_length))
        # 2. The reconstruction loss
        dim = tf.reduce_prod(tf.shape(self.inputs)[1:])
        orgin = tf.reshape(self.inputs, shape=(-1, dim))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*784=0.392
        self.total_loss = self.margin_loss + self.regularization_scale * self.reconstruction_err

    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        recon_img = tf.reshape(self.decoded, shape=(-1, self.height, self.width, self.channels))
        train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        self.train_summary = tf.summary.merge(train_summary)

        correct_prediction = tf.equal(tf.to_int32(self.labels), self.predictions)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
