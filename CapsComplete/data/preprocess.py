import os
import numpy as np

def load_mnist(batch_size, is_training=True, quantity=-1, path='./data/mnist',
               occlusion=0, ox_dim=0, oy_dim=0, occ_prob=0):
    if path is None:
        path = './data/mnist'
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        if quantity == -1:
            size = 60000
        else:
            size = min(60000, abs(quantity))
        origX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)
        trainX = origX[:size,:,:,:]
        trainX = occlude(trainX, occlusion=occlusion, ox_dim=ox_dim, oy_dim=oy_dim, occ_prob=occ_prob)
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
        teX = occlude(teX, occlusion=occlusion, ox_dim=ox_dim, oy_dim=oy_dim, occ_prob=occ_prob)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)
        teY = teY[:size]

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_fashion_mnist(batch_size, is_training=True, quantity=-1, path='./data/fashion-mnist', occlusion=0, ox_dim=0, oy_dim=0, occ_prob=0):
    if path is None:
        path = './data/fashion-mnist'
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        if quantity == -1:
            size = 60000
        else:
            size = min(60000, abs(quantity))
        origX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)
        trainX = origX[:size,:,:,:]
        trainX = occlude(trainX, occlusion=occlusion, ox_dim=ox_dim, oy_dim=oy_dim, occ_prob=occ_prob)
        

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        origY = loaded[8:].reshape((60000)).astype(np.int32)
        trainY = origY[:size]

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
        if quantity == -1:
            size = 10000
        else:
            size = min(10000, abs(quantity))
        teX = teX[:size,:,:,:]
        teX = occlude(teX, occlusion=occlusion, ox_dim=ox_dim, oy_dim=oy_dim, occ_prob=occ_prob)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)
        teY = teY[:size]

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_data(dataset, batch_size, is_training=True, one_hot=False, quantity=-1,
              occlusion=0, ox_dim=0, oy_dim=0, occ_prob=0, path=None):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training, quantity=quantity,
                          occlusion=occlusion, ox_dim=ox_dim, oy_dim=oy_dim, occ_prob=occ_prob, path=path)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training, quantity=quantity,
                          occlusion=occlusion, ox_dim=ox_dim, oy_dim=oy_dim, occ_prob=occ_prob,path=path)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)

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

def occlude(imgs, occlusion=1, ox_dim=14, oy_dim=14, occ_prob=.7):
    num_imgs = imgs.shape[0]
    imgs = np.copy(imgs)
    # Deterministic occlusion
    if occlusion == 1:
        for img_index in range(num_imgs):
            # Randomize location
            upper_left_x = np.random.randint(0,28-ox_dim)
            upper_left_y = np.random.randint(0,28-oy_dim)
            for i in range(upper_left_x, upper_left_x + ox_dim):
                imgs[img_index, i, upper_left_y:upper_left_y+oy_dim] = 0

    # Probabilistic occlusion
    elif occlusion == 2:
        rands = np.random.random(imgs.shape)
        bools = rands > occ_prob
        imgs = imgs * bools

    return imgs