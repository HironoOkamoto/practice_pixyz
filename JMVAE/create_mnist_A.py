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


top_left = (0, 0)
top_right = (0, 36)
bottom_left = (36, 0)
bottom_right = (36, 36)
position = [top_left, top_right, bottom_left, bottom_right]


def change_rotation(image, angle_index):
    h, w = image.shape
    angle_list = [45, 0, -45]
    angle = angle_list[angle_index]
    if angle:
        image = rotate(image, angle)
        image = resize(image, (36, 36), mode='constant')
        image = image[4:32, 4:32]
    return image


def change_scale(image, scale_index):
    scale_size = [28, 16]
    scale = scale_size[scale_index]
    image = resize(image, (scale, scale), mode='constant')
    if scale_index:
        image = np.pad(image, (6, 6), 'constant')
    return image


mnist_A_path = "../data/MNIST_A/"
for p in ["train_X", "test_X", "valid_X"]:
    path = join(mnist_A_path, p)
    if not exists(path):
        os.makedirs(path)

np.random.seed(42)

def create_mnist_A(data_X, data_y, data_kind, iteration):
    label_num = 10
    position_num = 4
    rotation_num = 3
    scale_num = 2
    mnist_size = 28

    label_list = []
    for m in tqdm(range(10)):
        data_X_subset = data_X[data_y==m].reshape(-1, mnist_size, mnist_size)
        size = data_X_subset.shape[0]
        range_size = range(size)
        for l in range(iteration):
            for k in range(position_num):
                for j in range(rotation_num):
                    for i in range(scale_num):
                        n =1+i+2*j+6*k+24*l+24*iteration*m
                        index = np.random.choice(range_size)
                        sample = data_X_subset[index]
                        sample_ = change_scale(sample, i)
                        sample_ = change_rotation(sample_, j)
                        black = np.zeros((64, 64))
                        pos_x, pos_y = position[k]
                        black[pos_x:pos_x+mnist_size, pos_y:pos_y+mnist_size] = sample_
                        plt.imsave(join(mnist_A_path, "{}_X/{}.png".format(data_kind, n)), black, cmap=plt.cm.gray)
                        label_list.append([i, j, k, m])
    mnist_A_label = np.array(label_list)
    np.save(join(mnist_A_path,"{}_y.npy".format(data_kind)), mnist_A_label)

print("creating train data...")
create_mnist_A(train_X, train_y, "train", 1000)
print("creating test data...")
create_mnist_A(test_X, test_y, "test", 100)
print("creating valid data...")
create_mnist_A(valid_X, valid_y, "valid", 100)
