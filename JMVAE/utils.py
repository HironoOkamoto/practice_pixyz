import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def label2onehot(y):
    y[:, 3] = y[:, 3] == 36
    y[:, 5] = y[:, 5] - 1
    y = y[:, [3, 5]]
    return y

def encoder_plot(data_loader, E, D, conditional=True):
    images, labels = iter(data_loader).next()
    images = images.to(device)
    labels = label2onehot(labels).to(device)
    
    if conditional:
        z = E.sample_mean({"x": images, "y": labels})
        samples = D.sample_mean({"z": z, "y": labels})
    else:
        z = E.sample_mean({"x": images})
        samples = D.sample_mean({"z": z})
    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1).squeeze()

    print("↓generate")
    plt.figure(figsize=(14, 4))
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=0., hspace=0.)
        plt.title(labels[i].detach().cpu().numpy())
        plt.imshow(samples[i], plt.cm.gray)
    images = images.cpu().data.numpy().transpose(0, 2, 3, 1).squeeze()
    for i in range(10):
        plt.subplot(2, 10, i+11)
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=0., hspace=0.)
        plt.imshow(images[i], plt.cm.gray)
    plt.show()
    print("↑true")

