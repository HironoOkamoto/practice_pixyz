import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.datasets.base import get_data_home
from sklearn.model_selection import train_test_split
import urllib.request

import numpy as np
from scipy.ndimage.interpolation import rotate
from skimage.transform import resize

from os.path import join, exists
import os

from tqdm import tqdm


mnist_path = join(get_data_home(), "mldata/mnist-original.mat")
if not exists(os.path.dirname(mnist_path)):
    os.makedirs(os.path.dirname(mnist_path))
mnist_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
urllib.request.urlretrieve(mnist_url, mnist_path)
mnist = fetch_mldata('MNIST original', data_home=get_data_home())
mnist_data = mnist["data"]/255.
mnist_target = mnist["target"].astype(int)

train_X, test_X, train_y, test_y = train_test_split(mnist_data, mnist_target, random_state=42, test_size=10000)
test_X, valid_X, test_y, valid_y = train_test_split(test_X, test_y, random_state=42, test_size=2000)


def change_scale(image, scale):
    image = resize(image, (scale, scale), mode='constant')
    pad_size = int((28-scale)/2)
    image = np.pad(image, (pad_size, pad_size), 'constant')
    return image

def change_rotation(image, angle):
    image = rotate(image, angle)
    image = resize(image, (28, 28), mode='constant')
    return image


mnist_A_path = "../data/MNIST_A/"
for p in ["train_X", "test_X", "valid_X"]:
    path = join(mnist_A_path, p)
    if not exists(path):
        os.makedirs(path)

np.random.seed(42)

def create_mnist_A(data_X, data_y, data_kind, iteration_num):
    pos_x_list = [0, 36]
    pos_y_list = [0]
    angle_list = [0]
    scale_list = [28] 
    mnist_size = 28

    label_list = []
    for m_, m in enumerate([1, 2]):
        data_X_subset = data_X[data_y==m].reshape(-1, mnist_size, mnist_size)
        size = data_X_subset.shape[0]
        range_size = range(size)
        for l in tqdm(range(iteration_num)):
            for k2, pos_y in enumerate(pos_y_list):
                for k1, pos_x in enumerate(pos_x_list):
                    for j, angle in enumerate(angle_list):
                        theta = np.pi * angle/180
                        sin = np.sin(theta).round(3)
                        cos = np.cos(theta).round(3)
                        for i, scale in enumerate(scale_list):
                            n =1+k1+2*l+2*iteration_num*m_
                            index = np.random.choice(range_size)
                            sample = data_X_subset[index]
                            sample_ = change_scale(sample, scale)
                            sample_ = change_rotation(sample_, angle)
                            black = np.zeros((64, 64))
                            black[pos_y:pos_y+mnist_size, pos_x:pos_x+mnist_size] = sample_
                            plt.imsave(join(mnist_A_path, "{}_X/{}.png".format(data_kind, n)), black, cmap=plt.cm.gray)
                            label_list.append([scale, sin, cos, pos_x, pos_y, m])
    mnist_A_label = np.array(label_list)
    np.save(join(mnist_A_path,"{}_y.npy".format(data_kind)), mnist_A_label)

print("creating train data...")
create_mnist_A(train_X, train_y, "train", 20000)
print("creating test data...")
create_mnist_A(test_X, test_y, "test", 2000)
print("creating valid data...")
create_mnist_A(valid_X, valid_y, "valid", 2000)
