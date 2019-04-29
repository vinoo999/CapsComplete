from matplotlib import pyplot as plt
import numpy as np
def plot_imgs(imgs):
    fig = plt.figure()
    n_images = imgs.shape[0]
    cols = 2
    for n, image in enumerate(imgs):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        image = image.reshape((28,28))
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
    fig.set_size_inches(np.array(fig.get_size_inches()))
    plt.show()