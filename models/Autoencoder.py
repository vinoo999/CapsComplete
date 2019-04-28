import tensorflow as tf
import numpy as np

class AutoEncoder(object):
    def __init__(self, height=28, width=28, channels=1, num_label=10):
        '''
        Inspired from with significant changes
        https://towardsdatascience.com/autoencoders-introduction-and-implementation-3f40483b0a85
        '''
        self.height = height
        self.width = width
        self.channels = channels
        self.num_label = num_label

    def create_network(self, inputs, labels):
        self.image_input = inputs
        self.labels = labels

        with tf.variable_scope("Encoder"):
            inputs = tf.reshape(self.image_input, shape=[-1, self.height, self.width, self.channels])
            conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                                        padding='same',
                                        activation='relu', name='conv1')(inputs)                    
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

        # Output layers: separate outputs for the weather and the ground labels
        self.classification_layer = tf.keras.layers.Dense(self.num_label, name='classification_layer')(fc2)

        with tf.variable_scope("decoder"):
            fc3 = tf.keras.layers.Dense(256, activation='relu', name='fc3')(fc2)
            fc3 = tf.keras.layers.Dropout(0.2)(fc3)

            deconv_input = tf.keras.layers.Reshape((self.height//4,self.width//4,16))(fc3)
            deconv1 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2), name='deconv1')(deconv_input)
            deconv2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3,3), activation='relu', padding='same', strides=(2,2), name='deconv2')(deconv1)
            self.recon = tf.keras.layers.Conv2D(1, kernel_size=(3, 3),
                                        padding='same',
                                         name='output')(deconv2) 
    
    def train(self, *args, **kwargs):
        self._build_loss()
        self._setup_train()
        return (self.loss, self.train_ops, self.summary_ops)
    
    def _build_loss(self, reg=0.392):
        with tf.variable_scope("loss"):
            originals = tf.reshape(self.image_input, shape=[-1, self.height, self.width, self.channels])
            reconstruction_loss = tf.nn.l2_loss(self.recon - originals, name='reconstruction_loss')
            one_hot = tf.one_hot(self.labels, depth=self.num_label)
            classification_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(one_hot, self.classification_layer))
            self.accuracy = tf.metrics.accuracy(labels=self.labels, predictions=tf.argmax(self.classification_layer))
            
            self.loss = reg*reconstruction_loss + classification_loss
    
    def _setup_train(self):
        optimizer = tf.train.RMSPropOptimizer(0.001)
        self.train_ops = optimizer.minimize(self.loss)
        self.summary_ops = tf.summary.merge_all()