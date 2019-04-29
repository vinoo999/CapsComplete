import os
import scipy
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def load_mnist(batch_size, is_training=True, quantity=-1):
    path = os.path.join('data', 'mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        if quantity == -1:
            size = 60000
        else:
            size = min(60000, abs(quantity))
        origX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)
        trainX = origX[:size,:,:,:]
        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        origY = loaded[8:].reshape((60000)).astype(np.int32)
        trainY = origY[:size]
        
        val_boundary = int(0.9*size)
        trX = trainX[:val_boundary] / 255.
        trY = trainY[:val_boundary]

        valX = trainX[val_boundary:, ] / 255.
        valY = trainY[val_boundary:]

        num_tr_batch = val_boundary // batch_size
        num_val_batch = (size-val_boundary) // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
        if quantity == -1:
            size = 10000
        else:
            size = min(10000, abs(quantity))
        teX = teX[:size,:,:,:]

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)
        teY = teY[:size]

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_fashion_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'fashion-mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_data(dataset, batch_size, is_training=True, one_hot=False, quantity=-1):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training, quantity=quantity)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size, num_threads):
    if dataset == 'mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist(batch_size, is_training=True)
    elif dataset == 'fashion-mnist':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_fashion_mnist(batch_size, is_training=True)
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return(shape)

def save_to(results_dir, is_training):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if is_training:
        loss = os.path.join(results_dir,'loss.csv')
        train_acc = os.path.join(results_dir , 'train_acc.csv')
        val_acc = os.path.join(results_dir, 'val_acc.csv')

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        return fd_train_acc, fd_loss, fd_val_acc
    else:
        test_acc = os.path.join(results_dir, 'test_acc.csv')
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return fd_test_acc

def train(model, num_label, 
          dataset='mnist', 
          batch_size=128, n_epochs=20,
          results_dir='./results/', log_dir='./logs/',
          train_sum_freq=100,
          val_sum_freq=500,
          save_freq=3,
          num_samples=-1):
    
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(dataset, batch_size, is_training=True, quantity=num_samples)
    # Y = valY[:num_val_batch * batch_size].reshape((-1, 1))

    fd_train_acc, fd_loss, fd_val_acc = save_to(results_dir, True)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=model.graph) as sess:
        with tf.Session() as temp_sess:
            temp_sess.run(tf.global_variables_initializer())
        sess.run(tf.variables_initializer(model.graph.get_collection('variables')))
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        print("\nNote: all of results will be saved to directory: " + results_dir)

        batch_num = 0
        for epoch in range(n_epochs):
            print("Training for epoch %d/%d:" % (epoch, n_epochs))
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
                start = step * batch_size
                end = start + batch_size
                global_step = epoch * num_tr_batch + step
                X_batch, Y_batch, batch_num = get_batch(trX, trY, batch_size, batch_num)
                if global_step % train_sum_freq == 0:
                    _, loss, train_acc, summary_str = sess.run([model.train_op, model.total_loss, model.accuracy, model.train_summary], feed_dict={model.X: X_batch, model.labels: Y_batch})
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(model.train_op, feed_dict={model.X: X_batch, model.labels: Y_batch})

                if val_sum_freq != 0 and (global_step) % val_sum_freq == 0:
                    val_acc = 0
                    for i in range(num_val_batch):
                        start = i * batch_size
                        end = start + batch_size
                        acc = sess.run(model.accuracy, {model.X: valX[start:end], model.labels: valY[start:end]})
                        val_acc += acc
                    val_acc = val_acc / (batch_size * num_val_batch)
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()
            if (epoch + 1) % save_freq == 0:
                tf.logging.info('Model Saving')
                saver.save(sess, os.path.join(log_dir, '/model_epoch_{}'.format(str(epoch))))

        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()

def evaluation(model, num_label, dataset='mnist', 
               batch_size=128,
               results_dir='./results/', log_dir='./logs/'):
    teX, teY, num_te_batch = load_data(dataset, batch_size, is_training=False)
    fd_test_acc = save_to(results_dir, False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=model.graph) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(log_dir))
        tf.logging.info('Model restored!')

        test_acc = 0
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * batch_size
            end = start + batch_size
            acc = sess.run(model.accuracy, {model.X: teX[start:end], model.labels: teY[start:end]})
            test_acc += acc
        test_acc = test_acc / (batch_size * num_te_batch)
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + results_dir + '/test_acc.csv')

def get_batch(X, Y, batch_size, batch_num):
    """
    Return minibatch of samples and labels
    :param X, y: samples and corresponding labels
    :parma batch_size: minibatch size
    :returns: X_batch
    """
    new_start = batch_size * batch_num
    if new_start >= X.shape[0]:
        new_start = 0
        batch_num = 0
    new_end = new_start + batch_size
    batch_num += 1
    if new_end >= X.shape[0]:
        new_end = X.shape[0]
        batch_num = 0
    X_batch = X[np.arange(new_start, new_end), ...]
    Y_batch = Y[np.arange(new_start, new_end)]
    return X_batch, Y_batch, batch_num