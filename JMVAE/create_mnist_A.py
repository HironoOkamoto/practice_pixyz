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
import argparse

parser = argparse.ArgumentParser()  

parser.add_argument("-n", "--name", default='MNIST_A_default')
parser.add_argument("-i", "--iteration", type=int, default=1000)
tp = lambda x:list(map(int, x.split(',')))
parser.add_argument("-nl", '--number_list', type=tp, default="0,1,2,3,4,5,6,7,8,9")
parser.add_argument("-pxl", '--pos_x_list', type=tp, default="0,36")
parser.add_argument("-pyl", '--pos_y_list', type=tp, default="0,36")
parser.add_argument("-al", '--angle_list', type=tp, default="-45,0,45")
parser.add_argument("-sl", '--scale_list', type=tp, default="28,16")

args = parser.parse_args()   

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


mnist_A_path = "../data/{}/".format(args.name)
for p in ["train_X", "test_X", "valid_X"]:
    path = join(mnist_A_path, p)
    if not exists(path):
        os.makedirs(path)

np.random.seed(42)

def create_mnist_A(data_X, data_y, data_kind, iteration_num):
    mnist_size = 28
    
    number_list = args.number_list
    pos_x_list = args.pos_x_list
    pos_y_list = args.pos_y_list
    angle_list = args.angle_list
    scale_list = args.scale_list

    nl = len(number_list)
    pxl = len(pos_x_list)
    pyl = len(pos_y_list)
    al = len(angle_list)
    sl = len(scale_list)

    label_list = []
    for m_, m in enumerate(number_list):
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
                            n =1+i+sl*j+sl*al*k1+sl*al*pxl*k2+sl*al*pxl*pyl*l+sl*al*pxl*pyl*iteration_num*m_
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

iteration_num = args.iteration
print("creating train data...")
create_mnist_A(train_X, train_y, "train", iteration_num)
print("creating test data...")
create_mnist_A(test_X, test_y, "test", int(iteration_num/10))
print("creating valid data...")
create_mnist_A(valid_X, valid_y, "valid", int(iteration_num/10))
