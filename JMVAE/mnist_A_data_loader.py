from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch
import os
import numpy as np


class MNIST_A(data.Dataset):
    """Dataset class for the mnist_A dataset."""

    def __init__(self, image_dir, attr_path, transform, select_attr, onehot):
        """Initialize and preprocess the MNIST-A dataset."""
        self.image_dir = image_dir
        self.labels = np.load(attr_path)
        self.transform = transform
        self.select_attr = select_attr
        self.onehot = onehot
        self.select_attribute()

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        index = self.indices[index]
        label = self.labels[index]
        if self.onehot:
            label = self.label2onehot(label)
        filename = "{}.png".format(index+1)
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images
    
    def select_attribute(self):
        """Preprocess the MNIST-A attribute file."""

        self.indices = []
        for attr in self.select_attr:
            self.indices.extend(np.where((self.labels == attr).sum(1) == 4)[0])
        self.num_images = len(self.indices)
        
    def label2onehot(self, label):
        I_3 = np.eye(3)
        I_4 = np.eye(4)
        I_10 = np.eye(10)
        return np.hstack((label[0], I_3[label[1]], I_4[label[2]], I_10[label[3]]))


def get_mnist_A_loader(image_dir, attr_path, select_attr=False, batch_size=128, mode='train', onehot=True):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Grayscale(num_output_channels=1))
    transform.append(T.ToTensor())
    transform = T.Compose(transform)

    dataset = MNIST_A(image_dir, attr_path, transform, select_attr, onehot)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'))
    return data_loader