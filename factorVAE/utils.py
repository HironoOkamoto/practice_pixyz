import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encoder_plot(data_loader, E, D):
    E.eval()
    D.eval()
    
    images = iter(data_loader).next()
    
    images = images.to(device)
    z = E.sample_mean({"x": images})
    samples = D.sample_mean({"z": z})
    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1).squeeze()
    
    print("↓generate")
    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=0., hspace=0.)
        plt.imshow(samples[i], cmap=plt.cm.gray)
    images = images.cpu().data.numpy().transpose(0, 2, 3, 1).squeeze()
    for i in range(10):
        plt.subplot(2, 10, i+11)
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=0., hspace=0.)
        plt.imshow(images[i], cmap=plt.cm.gray)
    plt.show()
    print("↑true")
    
def traverse_plot(data_loader, E, D, n, m = 11, scale = 5, z_dim=10):
    E.eval()
    D.eval()

    images = iter(data_loader).next()
    images = images[n:n+1].to(device)
    z = E.sample_mean({"x": images})
    samples = D.sample_mean({"z": z})
    images_ = images.cpu().data.numpy().transpose(0, 2, 3, 1).squeeze()
    samples_ = samples.cpu().data.numpy().transpose(0, 2, 3, 1).squeeze()
    plt.subplot(121)
    plt.title("original")
    plt.imshow(images_, plt.cm.gray)
    plt.subplot(122)
    plt.title("reconstruction")
    plt.imshow(samples_, plt.cm.gray)
    plt.show()

    zeros = torch.zeros(z_dim*m, z_dim)
    scales = np.linspace(-scale, scale, m)
    for j in range(z_dim):
        for i, v1 in enumerate(scales):
            zeros[i+m*j][j] = v1

    z_repeat = z.squeeze().repeat(z_dim*m).view(-1, z_dim)
    z_ = z_repeat + zeros.to(device)
    samples = D(z_)
    samples_ = samples["probs"].cpu().data.numpy().transpose(0, 2, 3, 1).squeeze()

    plt.figure(figsize=(m, z_dim))
    for i in range(z_dim*m):
        plt.subplot(z_dim, m, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=0., hspace=0.)
        if i < m:
            plt.title("{:+}".format(int(scales[i])), fontsize=15)
        if i%m == 0:
            plt.ylabel("z{}".format(int(i/m+1)), fontsize=15)
        plt.imshow(samples_[i], cmap=plt.cm.gray)

    plt.show()
