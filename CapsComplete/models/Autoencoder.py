import tensorflow as tf
import numpy as np

class AutoEncoder(object):
    def __init__(self, height=28, width=28, channels=1, num_label=10, is_training=True):
        '''
        Inspired from with significant changes
        https://towardsdatascience.com/autoencoders-introduction-and-implementation-3f40483b0a85
        '''
        self.height = height
        self.width = width
        self.channels = channels
        self.num_label = num_label

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_network()
            self._build_loss()
            if is_training:
                self._setup_train()
    
    def create_network(self):
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channels])
        self.labels = tf.placeholder(tf.int32, shape=[None,])
        with tf.variable_scope("Encoder"):
            conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                                        padding='same',
                                        activation='relu', name='conv1')(self.inputs)                    
            pool1 = tf.keras.layers.MaxPooling2D(padding='same', 
                                                 pool_size=(2,2),
                                                 name='pool1')(conv1)

            conv2 = tf.keras.layers.Conv2D(16,
                                            kernel_size=(3, 3),
                                            padding='same',
                                            activation='relu', 
                                            name='conv2')(pool1)
            pool2 = tf.keras.layers.MaxPooling2D(padding='same', 
                                                 pool_size=(2,2),
                                                name='pool2')(conv2)
            
            flat = tf.keras.layers.Flatten()(pool2)

            fc1 = tf.keras.layers.Dense(256,
                                        activation='relu',name='fc1')(flat)
            fc1 = tf.keras.layers.Dropout(0.2)(fc1)
            fc2 = tf.keras.layers.Dense(128,
                                        activation='relu',name='fc2')(fc1)
            fc2 = tf.keras.layers.Dropout(0.2)(fc2)

        self.logits = tf.keras.layers.Dense(self.num_label, name='classification_layer')(fc2)
        self.probs = tf.nn.softmax(self.logits, axis=1)
        self.predictions = tf.to_int32(tf.argmax(self.probs, axis=1))
        with tf.variable_scope("decoder"):
            fc3 = tf.keras.layers.Dense(256, activation='relu', name='fc3')(fc2)
            fc3 = tf.keras.layers.Dropout(0.2)(fc3)

            deconv_input = tf.reshape(fc3, shape=[-1, 4, 4, 16])
            deconv1 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2), name='deconv1')(deconv_input)
            deconv2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2), name='deconv2')(deconv1)
            flat2 = tf.keras.layers.Flatten()(deconv2)
            num_outputs = self.height * self.width * self.channels
            recon = tf.keras.layers.Dense(num_outputs, activation=tf.sigmoid)(flat2)
            self.recons = tf.reshape(recon, shape=[-1, self.height, self.width, self.channels], name='reconstruction') 
        
        with tf.variable_scope('accuracy'):
            logits_idx = tf.to_int32(tf.argmax(tf.nn.softmax(self.logits, axis=1), axis=1))
            correct_prediction = tf.equal(tf.to_int32(self.labels), logits_idx)
            correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            self.accuracy = tf.reduce_mean(correct / tf.cast(tf.shape(self.logits)[0], tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
    
    def train(self, *args, **kwargs):
        self._build_loss()
        self._setup_train()
        return (self.total_loss, self.train_op, self.summary_ops)
    
    def _build_loss(self, reg=0.392):
        with tf.variable_scope("loss"):
            self.reconstruction_loss = tf.nn.l2_loss(self.recons - self.inputs, name='reconstruction_loss')
            one_hot = tf.one_hot(self.labels, depth=self.num_label)
            self.classification_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(one_hot, self.probs))
            
            self.total_loss = reg*self.reconstruction_loss + self.classification_loss
            tf.summary.scalar("loss", self.total_loss)
    
    def _setup_train(self):
        optimizer = tf.train.RMSPropOptimizer(0.001)
        self.train_op = optimizer.minimize(self.total_loss)
        self.train_summary = tf.summary.merge_all()