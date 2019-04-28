import numpy as np
#import random as rn
#import os
#os.environ['PYTHONHASHSEED'] = '0'
#np.random.seed(42)
#rn.seed(12345)
import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth=True
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
K.set_image_data_format('channels_last')

import sys
import random
import imageio

#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#from keras import layers, models, optimizers
from keras.utils import to_categorical
#import matplotlib.pyplot as plt
#from utils import combine_images, plot_log
#from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
#from keras.layers.normalization import BatchNormalization
#from keras.preprocessing.image import ImageDataGenerator
#import densenet

def occlude(arr):
    out_dir = "" # Absolute or relative path to output directory
    occlusion_prob = .1 # 80% chance of occlusion of any given pixel


    # Track progress in subjecting dataset to probabilistic occlusion
    it_count = 0.0
    tot = float(len(arr) * len(arr[0]) * len(arr[0][0]))

    # Iterate over the numpy arrays representing the input images
    for img_index in range(0, len(arr)):
        for row_index in range(0, len(arr[img_index])):
            for col_index in range(0, len(arr[img_index][row_index])):
                it_count += 1.0

                # For each pixel in the image, there is an occlusion_prob chance
                # that it will be set to zero (black)
                if random.random() < occlusion_prob:
                    arr[img_index][row_index][col_index] = 0

                sys.stdout.write('\r')
                sys.stdout.write("{}%".format(round(it_count*100.0/tot, 2)))
                sys.stdout.flush()

    # Save the numpy array representing the first image in the dataset (after
    # subjecting it to probabilistic occlusion) as a PNG for testing purposes.

    imageio.imwrite(out_dir + 'sample_occ.png', arr[0])

    return arr


def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        if x.ndim == 3:
            # 'RGB'->'BGR'
            x = x[::-1, ...]
            # Zero-center by mean pixel
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x = x[:, ::-1, ...]
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
        # Zero-center by mean pixel
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    x *= 0.017 # scale values

    return x

def load_dataset():
    # Load the dataset from Keras
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(np.shape(x_test))
    print(x_test[0])
    x_train, x_test = occlude(x_train), occlude(x_test)

    # Preprocessing the dataset
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train= preprocess_input(x_train)
    x_test= preprocess_input(x_test)
    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32')
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32')
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_dataset()
print(np.shape(x_train))
print(np.shape(x_train[0]))
