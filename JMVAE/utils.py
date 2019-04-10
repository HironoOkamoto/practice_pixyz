import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encoder_plot(data_loader, E, D, conditional=True):
#     E.eval()
#     D.eval()

    images, labels = iter(data_loader).next()
    images = images.to(device)
    labels = labels.to(device)
    
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
    
def label2onehot(label):
    I_3 = np.eye(3)
    I_4 = np.eye(4)
    I_10 = np.eye(10)
    return np.hstack((label[:, 0][:, None], I_3[label[:, 1]], I_4[label[:, 2]], I_10[label[:, 3]]))

def onehot2label(onehot):
    return np.hstack(((onehot[:, 0]>0.5)[:, None], onehot[:, 1:4].argmax(1)[:, None],
                      onehot[:, 4:8].argmax(1)[:, None], onehot[:, 8:].argmax(1)[:, None]))


def elbo(x, q, p):
    #1. sample from q(z|x) 
    samples = q.sample(x)
    
    #2. caluculate the lower bound (log p(x,z) - log q(z|x))
    lower_bound = p.log_likelihood(samples) - q.log_likelihood(samples)

    loss = -torch.mean(lower_bound)

    return loss