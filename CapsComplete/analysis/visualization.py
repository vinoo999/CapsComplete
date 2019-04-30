from matplotlib import pyplot as plt
import numpy as np
import scipy

def plot_imgs(imgs, cols=1):
    fig = plt.figure()
    n_images = imgs.shape[0]
    for n, image in enumerate(imgs):
        _ = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        image = image.reshape((28,28))
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        plt.axis('off')
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
    # fig.set_size_inches(np.array(fig.get_size_inches())[0])
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

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