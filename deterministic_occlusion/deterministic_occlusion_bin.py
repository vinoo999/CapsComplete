from mlxtend.data import loadlocal_mnist
import sys
import numpy as np
import random
import imageio

out_dir = "" # Absolute or relative path to output directory
mnist_data_dir = "../probabilistic_occlusion/mnist_bin/" # Absolute or relative path to directory
                              # containing binary MNIST dataset files

occlusion_x_dim = 14
occlusion_y_dim = 14

# Load the data into numpy arrays
X, y = loadlocal_mnist(
        images_path= mnist_data_dir + 't10k-images-idx3-ubyte',
        labels_path= mnist_data_dir + 't10k-labels-idx1-ubyte')

# Check the dimensions of the arrays for accuracy
print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('Digits:  0 1 2 3 4 5 6 7 8 9')
print('labels: %s' % np.unique(y))
print('Class distribution: %s' % np.bincount(y))

# Track progress in subjecting dataset to probabilistic occlusion
it_count = 0.0
tot = float(X.shape[0])

# Iterate over the numpy arrays representing the input images
for img_index in range(0, len(X)):
    it_count += 1.0

    square = np.split(X[img_index], 28)

    # Randomize location
    upper_left_x = random.randint(0,28-occlusion_x_dim)
    upper_left_y = random.randint(0,28-occlusion_y_dim)

    for i in range(upper_left_x, upper_left_x + occlusion_x_dim):
        for j in range(upper_left_y, upper_left_y + occlusion_y_dim):
            square[i][j] = 0#255

    X[img_index] = np.concatenate(square, axis=None)

    sys.stdout.write('\r')
    sys.stdout.write("{}%".format(round(it_count*100.0/tot, 2)))
    sys.stdout.flush()

# Save the numpy array representing the first image in the dataset (after
# subjecting it to probabilistic occlusion) as a PNG for testing purposes.
imageio.imwrite(out_dir + 'sample1.png', np.split(X[0], 28))
imageio.imwrite(out_dir + 'sample2.png', np.split(X[1], 28))
imageio.imwrite(out_dir + 'sample3.png', np.split(X[2], 28))

# Output the transformed images in CSV format. Note: this is for demo purposes
# only, in practice we will use the transformed numpy array directly
np.savetxt(fname= out_dir + 'transformed_images.csv',
           X=X, delimiter=',', fmt='%d')
np.savetxt(fname= out_dir + 'labels.csv',
           X=y, delimiter=',', fmt='%d')
